import os
import math
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import weightwatcher as ww
from operator import itemgetter

from lib.utils import *
from lib.neuroal import *

def get_neuroal_sparsity_distribution(args, model, tokenizer, device, layers):
    """
    Get non-uniform sparsity distribution using neuroal method for both block and row steps w.r.t. the selected approach.
    """
    best_lambda_group, best_lambda_neuron = None, None
    block_sparsity = {}
    layer_names = []
    for i in range(len(layers)):
        block_sparsity[f'{i}'] = None
        subset = find_layers(layers[i])
        for name in subset:
            layer_names.append(f'{name}-{i}')

    #set linear schedule for block sparsity
    count = 1
    for key in block_sparsity:
        block_sparsity[key] = count
        count += 1  

    args.nsamples = 8
    calib_data_8, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    
    # BLOCK STEP
    if args.neuroal_step in ['block', 'both']:
        if args.sparsity_ratio < 0.8:
            lambda_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.25]
        else:
            lambda_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2]
        

        all_dist = {}
        
        #get block-wise sparsity distribution for different lambda values
        for l in lambda_values:
            dist = get_distribution_block(block_sparsity, layer_names, args.sparsity_ratio, l)
            all_dist[l] = dist

        #evaluate block-wise sparsity distribution for different lambda values
        results = {}
        with torch.no_grad():
            for dist_key, dist_value in tqdm(all_dist.items(), desc="Evaluating Block Step"):
                if args.prune_method != 'sparsegpt':
                    all_act, error, values_diff = get_neuroal_block(args, model, calib_data_8, dist_value, device)
                elif args.prune_method == 'sparsegpt':
                    all_act, error, values_diff = get_neuroal_block_sparsegpt(args, model, calib_data_8, dist_value, device)
                results[dist_key] = (all_act, error, values_diff)

        #get the best block-wise sparsity distribution
        c = []
        cc = []
        for key, value in results.items():
            c.append(value[0])
            cc.append(value[2])
        
        dist = all_dist[lambda_values[c.index(min(c))]]
        difference_values = cc[c.index(min(c))]
        print('Best block-wise lambda:', lambda_values[c.index(min(c))])
        best_lambda_group = lambda_values[c.index(min(c))]
        print('*' * 30)
        granularity = None

    # ROW STEP
    if args.neuroal_step in ['row', 'both'] and args.prune_method != 'sparsegpt':
        #if block step is not performed, get the difference values for row step using uniform block sparsity
        if args.neuroal_step == 'row':
            dist = get_distribution_block(block_sparsity, layer_names, args.sparsity_ratio, 0)
            _, _, difference_values = get_neuroal_block(args, model, calib_data_8, dist, device)

        lambda_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.25]
        all_dist = {}

        #get row-wise sparsity distribution for different lambda values
        print("Running row step...")
        for l in lambda_values:
            all_dist[l] = get_distribution_row(args, model, dist, l, difference_values)

        #evaluate row-wise sparsity distribution for different lambda values
        results = {}
        for dist_key, dist_value in tqdm(all_dist.items(), desc="Evaluating Row Step"):
            all_act, act_dict = get_neuroal_row(args, model, calib_data_8, dist_value, device)
            results[dist_key] = (all_act, act_dict)

        #get the best row-wise sparsity distribution
        c = []
        cc = []
        for key, value in results.items():
            c.append(value[0])
            cc.append(value[1])

        dist = all_dist[lambda_values[c.index(min(c))]]
        print('Best row-wise lambda:', lambda_values[c.index(min(c))])
        best_lambda_neuron = lambda_values[c.index(min(c))]

        granularity = 'neuron'


    return dist, granularity, best_lambda_group, best_lambda_neuron


### Codebase based on https://github.com/luuyin/OWL ######
def get_owl_sparsity_distribution(args, model, tokenizer, device, calib_data):
    """
    Get non-uniform sparsity distribution using OWL method for blocks.
    """
    with torch.no_grad():
        print("Pruning starts")

        if args.model_type == "opt":
            activations_dense, owl_values = run_activation_opt(
                args, model, tokenizer, device, calib_data, owl=True, out_ratio=args.M
            )
        elif args.model_type in ["phi", "llama", "mistral"]:
            activations_dense, owl_values = run_activation(
                args, model, tokenizer, device, calib_data, owl=True, out_ratio=args.M
            )
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        layer_names = list(activations_dense.keys())
        dist = get_sparsity_owl(owl_values, layer_names, args.sparsity_ratio, args.lam)

    return dist, None

def get_uniform_sparsity_distribution(args, layers):
    """
    Get uniform sparsity distribution across blocks.
    """
    dist = {}
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            dist[f'{name}-{i}'] = args.sparsity_ratio
    return dist, None



###### Codebase based on https://github.com/haiquanlu/AlphaPruning ######
def get_alpha_sparsity_distribution(args, model, device,model_name, layers):
    if not os.path.exists(f"alpha_data/{model_name}-alpha_peak.npy"):
        metric_values = get_esd_metrics(args, model, "alpha_peak")
        np.save(f"alpha_data/{model_name}-alpha_peak.npy", metric_values)

    if args.sparsity_ratio <= 0.7:
        s1 = 1.0 - args.epsilon
        s2 = 1.0 + args.epsilon
    else:
        s1 = 1.0 - args.epsilon
        s2 = 1 + args.epsilon
        #s1 = 1.0 - (1.0 - args.sparsity_ratio) 
        #s2 = 1.0 + (1.0 - args.sparsity_ratio) 
    all_layer_ratio = ww_sparsity(args, model, device, s1, s2, model_name)
    
    dist = {}
    count = 0
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            dist[f'{name}-{i}'] = all_layer_ratio[count]
            count += 1

    return dist, None


def ww_sparsity(args, model, device=torch.device("cuda:0"), s1=0.8, s2=1.2, model_name=None):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers
    
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())

    layer_num_in_block = int(len(prunables) / len(blocks))

    metrics = np.load(f"alpha_data/{model_name}-alpha_peak.npy")
    
    block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) for i in range(0, len(metrics), layer_num_in_block)]
    metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    
    print("metric values:", metrics)
            
    scores = torch.tensor(metrics)
    prunables = torch.tensor(prunables)

    # linear mapping
    max = torch.max(scores)
    min = torch.min(scores)
    
    layerwise_pruning_ratios = (((scores - min) / (max - min)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables) * args.sparsity_ratio / (torch.sum(prunables * layerwise_pruning_ratios))  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    
    return layerwise_pruning_ratios


def get_esd_metrics(args, model, metric):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers
    
    if metric == 'alpha_mid':
        metrics = net_esd_estimator(blocks,
            fix_fingers='xmin_mid'
        )
        metrics = metrics['alpha']
    elif metric == 'alpha_peak':
        metrics = net_esd_estimator(blocks,
            fix_fingers='xmin_peak'
        )
        metrics = metrics['alpha']
    else:
        watcher = ww.WeightWatcher(model=blocks)
        details = watcher.analyze(mp_fit=True, randomize=True)
        
        if args.WW_metric == 'entropy':
            metrics = np.array(details.entropy)
        elif args.WW_metric == 'alpha':
            metrics = np.array(details.alpha)
        elif args.WW_metric == 'mp_softrank':
            metrics = np.array(details.mp_softrank)
        elif args.WW_metric == 'stable_rank':
            metrics = np.array(details.stable_rank)
        elif args.WW_metric == 'random_distance':
            metrics = np.array(details.rand_distance)
        elif args.WW_metric == 'log_norm':
            metrics = np.array(details.log_norm)
        elif args.WW_metric == 'log_spectral_norm':
            metrics = np.array(details.log_spectral_norm)
        elif args.WW_metric == 'alpha_weighted':
            metrics = np.array(details.alpha_weighted)
        elif args.WW_metric == 'log_alpha_norm':
            metrics = np.array(details.log_alpha_norm)
        elif args.WW_metric == 'spectral_norm':
            metrics = np.array(details.spectral_norm)
    
    return metrics    

def net_esd_estimator(
            net=None,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers=None,
            xmin_pos=2,
            conv_norm=0.5, 
            filter_zeros=False):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha':[],
        'spectral_norm': [],
        'D': [],
        'longname':[],
        'eigs':[],
        'norm':[],
        'alphahat': []
        }
    print("=================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print("=================================")
    # iterate through layers
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone().cpu()
            # i have checked that the multiplication won't affect the weights value
            # normalization and tranpose Conv2d
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            matrix = matrix.to(torch.float32)
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            # ascending order 
            eigs, _ = torch.sort(eigs, descending=False)
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()
            
            if filter_zeros:
                nz_eigs = eigs[eigs > EVALS_THRESH]
                N = len(nz_eigs)
                # somethines N may equal 0, if that happens, we don't filter eigs
                if N == 0:
                    nz_eigs = eigs
                    N = len(nz_eigs)
            else:
                nz_eigs = eigs
                N = len(nz_eigs)
            log_nz_eigs  = torch.log(nz_eigs)

            if fix_fingers == 'xmin_mid':
                i = int(len(nz_eigs) / xmin_pos)    
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n)
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                        ))
            else:
                alphas = torch.zeros(N-1)
                Ds     = torch.ones(N-1)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                    boundaries = torch.linspace(min_e, max_e, bins + 1)
                    h = counts, boundaries
                    ih = torch.argmax(h[0])  # 
                    xmin2 = 10 ** h[1][ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2
                
                for i, xmin in enumerate(nz_eigs[:-1]):
                    if fix_fingers == 'xmin_peak':
                        if xmin < xmin_min:
                            continue
                        if xmin > xmin_max:
                            break

                    n = float(N - i)
                    seq = torch.arange(n)
                    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    alphas[i] = alpha
                    if alpha > 1:
                        Ds[i] = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                        ))

                min_D_index = torch.argmin(Ds)
                final_alpha = alphas[min_D_index]
                final_D = Ds[min_D_index]
            
            final_alpha = final_alpha.item()
            final_D = final_D.item()
            final_alphahat=final_alpha*math.log10(spectral_norm)

            results['spectral_norm'].append(spectral_norm)
            results['alphahat'].append(final_alphahat)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(eigs.detach().cpu().numpy())

    return results