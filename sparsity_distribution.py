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
        lambda_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.25]
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