import torch
import numpy as np
from lib.utils import *


def get_distribution_block(values,layer_name, sparsity_ratio,l):
    """
    Define block-wise sparsity distribution based lamnda value.
    """

    group_sparsity = []
    for key, value in values.items():
        group_sparsity.append(value)
    group_sparsity=np.array(group_sparsity)
    group_sparsity = ((group_sparsity - group_sparsity.min()) * (1/(group_sparsity.max() - group_sparsity.min()) * l*2))
    group_sparsity=group_sparsity-np.mean(group_sparsity)+(1-sparsity_ratio)
    group_sparsity = sorted(group_sparsity, reverse=True)
    dist = {}

    for key in layer_name:
        if 'classifier' in key: continue
        layer_id = key.split('-')[1]
        dist[key] = 1 - group_sparsity[int(layer_id)]
        if dist[key] <= 0:
            dist[key] = 0.01
        elif dist[key] >= 1:
            dist[key] = 0.99

    return dist



def get_distribution_row(args, model, dist_base, l, alignment, device='cuda:0'):
    """
    Define row-wise sparsity distribution based on the alignment values.
    """
    layers = model.model.decoder.layers if args.model_type == "opt" else model.model.layers
    dist_final = {}
    new_alignment = {}


    for layer_id_to_check in range(len(layers)-1):          
        layer = layers[layer_id_to_check]
        subset = find_layers(layer)
        layer_next = layers[layer_id_to_check+1]
        subset_next = find_layers(layer_next)

        layers_name = [name for name in subset]
        for idx_s, name in enumerate(subset):
            if idx_s <= 2: continue
            if idx_s == 3:
                new_alignment[f'{layers_name[0]}-{layer_id_to_check}'] = alignment[f'{layers_name[3]}-{layer_id_to_check}']
                new_alignment[f'{layers_name[1]}-{layer_id_to_check}'] = alignment[f'{layers_name[3]}-{layer_id_to_check}']
                new_alignment[f'{layers_name[2]}-{layer_id_to_check}'] = alignment[f'{layers_name[3]}-{layer_id_to_check}']
            elif idx_s == 4:
                new_alignment[f'{layers_name[idx_s-1]}-{layer_id_to_check}'] = alignment[f'{layers_name[idx_s]}-{layer_id_to_check}']
            elif idx_s > 4:
                if args.model_type == 'opt' or args.model_type == 'phi':
                    new_alignment[f'{layers_name[idx_s-1]}-{layer_id_to_check}'] = alignment[f'{layers_name[idx_s]}-{layer_id_to_check}']
                else:
                    if idx_s == len(subset) - 1:
                        new_alignment[f'{layers_name[idx_s-1]}-{layer_id_to_check}'] = alignment[f'{layers_name[idx_s]}-{layer_id_to_check}']
                        new_alignment[f'{layers_name[idx_s-2]}-{layer_id_to_check}'] = alignment[f'{layers_name[idx_s]}-{layer_id_to_check}']

        for idx_s, name in enumerate(subset_next):
            if idx_s == 0:
                new_alignment[f'{layers_name[-1]}-{layer_id_to_check}'] = alignment[f'{name}-{layer_id_to_check+1}']
                break


    for layer_id_to_check, layer in enumerate(layers):
        subset = find_layers(layer)
        if layer_id_to_check == len(layers) - 1:
            for name in subset:
                dist_final[f'{name}-{layer_id_to_check}'] = [dist_base[f'{name}-{layer_id_to_check}'] for _ in range(subset[name].weight.shape[0])]
            break
        else:
            for name in subset:
                alignment_values = new_alignment.get(f'{name}-{layer_id_to_check}', None)
                if alignment_values is None:
                    continue
            
                s_single = alignment_values.cpu().tolist()
                tot = alignment_values.sum().cpu()

                #1-check if the alignment values are zero in the layer, if so, keep the same distribution as the base
                #2-Mistral model has some q_proj,k_proj,v_proj with different dimensions, so we keep the same distribution as the base only for the q_proj,k_proj,v_proj in row-wise case
                if tot == 0 or (args.model_type == 'mistral' and any(x in name for x in ['q_proj', 'k_proj', 'v_proj'])):
                    dist_final[f'{name}-{layer_id_to_check}'] = [dist_base[f'{name}-{layer_id_to_check}']] * len(s_single)
                else:
                    lambda_param = l
                    group_sparsity = np.array(s_single, dtype=float)
                    group_sparsity_min, group_sparsity_max = group_sparsity.min(), group_sparsity.max()
                    group_sparsity = ((group_sparsity - group_sparsity_min) / (group_sparsity_max - group_sparsity_min)) * lambda_param * 2
                    group_sparsity -= np.mean(group_sparsity) - (1 - dist_base[f'{name}-{layer_id_to_check}'])
                    
                    a = 1 - group_sparsity
                    dist_final[f'{name}-{layer_id_to_check}'] = np.clip(a, 0.01, 0.99)

    return dist_final




def get_neuroal_block(args, model, dataloader, dist_base, device='cuda:0'):
    """
    Get NeuroAL metric for block-wise sparsity distribution.
    """
    if args.model_type == "opt":
        layers = model.model.decoder.layers
        with torch.no_grad():
            inps, outs, attention_mask = prepare_calibration_input_opt(args, model, dataloader, device)
    else:
        layers = model.model.layers
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)


    neuroal_, alignment_ = {}, {}

    # Iterate over each layer, computing activations and metrics
    for layer_id, layer in enumerate(layers):  
        subset = find_layers(layer)
        
        if args.model_type == 'opt':
            activations_dense, output_dense, inps_dense, outs_dense = get_forward(
                subset, inps, outs, attention_mask, args, layer
            )
        else:
            activations_dense, output_dense, inps_dense, outs_dense = get_forward(
                subset, inps, outs, attention_mask, args, layer, position_ids
            )

        weights = {name: subset[name].weight.data.clone() for name in subset}
        
        # Vectorized metric computation and mask application
        for name, weight in weights.items():
            if args.prune_method == 'magnitude':
                W_metric = torch.abs(weight)
            elif args.prune_method == "wanda":
                W_metric = torch.abs(weight) * activations_dense[name].reshape((1, -1))
            elif args.prune_method == "multiflow":
                actn_norm = activations_dense[name].reshape((1, -1))
                importance_per_output = (torch.abs(weight) * actn_norm).mean(dim=1)
                importance_per_input = (torch.abs(weight) * actn_norm).mean(dim=0)
                W_metric = torch.abs(weight) * torch.ger(importance_per_output, importance_per_input)

            # Apply pruning mask
            W_mask = compute_mask(W_metric, 'row', dist_base[f'{name}-{layer_id}'])
            subset[name].weight.data[W_mask] = 0.

        # Compute sparse activations
        if args.model_type == 'opt':
            activations_sparse, output_sparse, inps_sparse, outs_sparse = get_forward(
                subset, inps, outs, attention_mask, args, layer
            )
        else:
            activations_sparse, output_sparse, inps_sparse, outs_sparse = get_forward(
                subset, inps, outs, attention_mask, args, layer, position_ids
            )

        # Update inputs and outputs for the next layer
        inps, outs = inps_sparse, outs_sparse

        # Calculate neuroal_ metrics and reset weights
        for name in subset:
            subset[name].weight.data = weights[name]
            dense, sparse = activations_dense[name], activations_sparse[name]
            difference = (dense / dense.sum()) - (sparse / sparse.sum())
            neuroal_[f'{name}-{layer_id}'] = torch.norm(difference, p=2).item() / dense.numel()
            alignment_[f'{name}-{layer_id}'] = dense - sparse

    # Clear unused variables and cache
    del inps, outs, activations_dense, activations_sparse, output_dense, output_sparse
    torch.cuda.empty_cache()

    neuroal = sum(neuroal_.values())
    return neuroal, neuroal_, alignment_


def get_neuroal_row(args,model, dataloader, dist_base, device='cuda:0'):
    """
    Get NeuroAL metric for row-wise sparsity distribution.
    """
    if args.model_type == "opt":
        layers = model.model.decoder.layers
        with torch.no_grad():
            inps, outs, attention_mask = prepare_calibration_input_opt(args, model, dataloader, device)
    else:
        layers = model.model.layers
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)
    
    
    neuroal_ = {}
    for layer_id_to_check in range(len(layers)):  
        layer = layers[layer_id_to_check]
        subset = find_layers(layer)
        if args.model_type == 'opt':
            activations_dense, output_dense , inps_dense , outs_dense = get_forward(subset, inps, outs, attention_mask, args, layer)
        else:
            activations_dense, output_dense , inps_dense , outs_dense = get_forward(subset, inps, outs, attention_mask, args, layer, position_ids)
    

        weights = {}
        for name in subset:
            weights[name] = subset[name].weight.data.clone()
            if args.prune_method == 'magnitude':
                W_metric = torch.abs(subset[name].weight.data)
            elif args.prune_method == "wanda":
                W_metric = torch.abs(subset[name].weight.data) * activations_dense[name].reshape((1, -1))

            elif args.prune_method == "multiflow":
                actn_norm = activations_dense[name].reshape((1,-1))
                importance_per_output = (torch.abs(subset[name].weight.data) * actn_norm).mean(dim=1)
                importance_per_input = (torch.abs(subset[name].weight.data) * actn_norm).mean(dim=0)
                W_metric = torch.abs(subset[name].weight.data) * torch.ger(importance_per_output, importance_per_input) 
            
            W_mask = compute_mask(W_metric, 'neuron', dist_base[f'{name}-{layer_id_to_check}'])
            subset[name].weight.data[W_mask] = 0. 
            
        if args.model_type == 'opt':
            activations_sparse, output_sparse, inps_sparse,out_sparse = get_forward(subset, inps, outs, attention_mask, args, layer)
        else:
            activations_sparse, output_sparse, inps_sparse,out_sparse = get_forward(subset, inps, outs, attention_mask, args, layer, position_ids)

        inps = inps_sparse.clone()
        outs = out_sparse.clone()

        for idx_h, name in enumerate(subset):
            subset[name].weight.data = weights[name]
            dense = activations_dense[name]
            sparse = activations_sparse[name]

            
            dense = dense / dense.sum()
            sparse = sparse / sparse.sum()
            difference = dense - sparse
            norm_difference_norm_act = torch.norm(difference, p=2)

            neuroal_[f'{name}-{layer_id_to_check}'] = norm_difference_norm_act.item() / len(dense)

        del inps_sparse, out_sparse, activations_dense, activations_sparse, output_sparse, output_dense, weights
        torch.cuda.empty_cache()
        
    del inps, outs
    torch.cuda.empty_cache()
    neuroal = 0

    neuroal = sum(neuroal_.values())

    return neuroal,neuroal_




def get_neuroal_block_sparsegpt(args,model, dataloader, dist_base, device='cuda:0'):
    if args.model_type == "opt":
        layers = model.model.decoder.layers
        with torch.no_grad():
            inps, outs, attention_mask = prepare_calibration_input_opt(args, model, dataloader, device)
    else:
        layers = model.model.layers
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)
    
    neuroal_ = {}
    alignment_ = {}
    for layer_id_to_check in range(len(layers)):  
        layer = layers[layer_id_to_check]
        subset = find_layers(layer)
        if args.model_type == 'opt':
            activations_dense, output_dense , _ , _, gpts = get_forward_sparsegpt(subset, inps, outs, attention_mask, args, layer)
        else:
            activations_dense, output_dense , _ , _, gpts = get_forward_sparsegpt(subset, inps, outs, attention_mask, args, layer, position_ids)
    

        weights = {}
        for name in subset:
            weights[name] = subset[name].weight.data.clone()
            gpts[name].fasterprune(
                    dist_base[f'{name}-{layer_id_to_check}'],
                    prune_n=0,
                    prune_m=0,
                    percdamp=0.01,
                    blocksize=128,
                )  
            gpts[name].free()

        if args.model_type == 'opt':
            activations_sparse, output_sparse, inps_sparse,out_sparse, _ = get_forward_sparsegpt(subset, inps, outs, attention_mask, args, layer)
        else:
            activations_sparse, output_sparse, inps_sparse,out_sparse, _ = get_forward_sparsegpt(subset, inps, outs, attention_mask, args, layer, position_ids)

        inps = inps_sparse.clone()
        outs = out_sparse.clone()

        for idx_h, name in enumerate(subset):
            subset[name].weight.data = weights[name]
            dense = activations_dense[name].clone()
            sparse = activations_sparse[name].clone()
            difference_neuron = dense - sparse
            dense = dense / dense.sum()
            sparse = sparse / sparse.sum()
            difference = dense - sparse
            norm_difference_norm_act = torch.norm(difference, p=2)
            neuroal_[f'{name}-{layer_id_to_check}'] =  norm_difference_norm_act.item()/len(dense)
            alignment_[f'{name}-{layer_id_to_check}'] = difference_neuron


    del inps, outs, inps_sparse, out_sparse, activations_dense, activations_sparse, output_sparse, output_dense, weights
    torch.cuda.empty_cache()
    neuroal = 0
    neuroal = sum(neuroal_.values())


    return neuroal,neuroal_, alignment_




