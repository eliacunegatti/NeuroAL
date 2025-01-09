# Standard libraries
import torch
import numpy as np
from torch import nn
from tqdm import tqdm

# Local module imports
from .data import get_loaders
from .layerwrapper import WrappedGPT
from lib.layerwrapper import WrappedGPT
from lib.sparsegpt import SparseGPT
from ops.mask_gen.functions.mask_gen import mask_gen


##### NeuroAL Utils #####

def prepare_calibration_input(args, model, dataloader, device):
    """
    Prepare input for calibration for LLama/Phi/Mistral models.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers


    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)

    attention_mask = cache["attention_mask"]
    if attention_mask != None:
        if attention_mask.shape[2] != model.seqlen:    
            attention_mask = attention_mask[:, :, :model.seqlen, :model.seqlen]

    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def prepare_calibration_input_opt(args, model, dataloader, device):
    """
    Prepare input for calibration for OPT models.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers


    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask

def get_forward(subset, input, output, attention_mask, args, layer, position_ids=None):
    """
    Compute activations and output values for a given layer.
    """
    input = input.clone()
    output = output.clone()
    
    wrapped_layers = {}
    for name in subset:
        wrapped_layers[name] = WrappedGPT(subset[name])
    def add_batch(name):
        def tmp(_, inp, out):
            wrapped_layers[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in wrapped_layers:
        handles.append(subset[name].register_forward_hook(add_batch(name)))

    if args.model_type == 'opt':
        for j in range(args.nsamples):
            with torch.no_grad():
                output[j] = layer(
                    input[j].unsqueeze(0),
                    attention_mask=attention_mask,
                )[0]                
    else:
        for j in range(args.nsamples):
            with torch.no_grad():
                output[j] = layer(
                    input[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

    for h in handles:
        h.remove()

    activations = {}
    output_values = {}
    for name in wrapped_layers:
        activations[name] = torch.sqrt(wrapped_layers[name].scaler_row)
        output_values[name] = None
    
    
    del wrapped_layers
    torch.cuda.empty_cache()
    return activations, output_values, output, input

def get_forward_sparsegpt(subset, input, output, attention_mask, args, layer, position_ids=None):
    """
    Compute activations and output values for a given layer when SparseGPT is selected as pruning algorithm.
    """
    input = input.clone()
    output = output.clone()
    
    wrapped_layers = {}
    for name in subset:
        wrapped_layers[name] = SparseGPT(subset[name])
    def add_batch(name):
        def tmp(_, inp, out):
            wrapped_layers[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in wrapped_layers:
        handles.append(subset[name].register_forward_hook(add_batch(name)))

    if args.model_type == 'opt':
        for j in range(args.nsamples):
            with torch.no_grad():
                output[j] = layer(
                    input[j].unsqueeze(0),
                    attention_mask=attention_mask,
                )[0]                
    else:
        for j in range(args.nsamples):
            with torch.no_grad():
                output[j] = layer(
                    input[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

    for h in handles:
        h.remove()

    activations = {}
    output_values = {}
    for name in wrapped_layers:
        activations[name] = torch.sqrt(wrapped_layers[name].scaler_row)
        output_values[name] = None
    
    torch.cuda.empty_cache()
    return activations, output_values, output, input, wrapped_layers



###### OWL's Utils ######

def check_outlier_mean(mask,threshold):
    """
    Compute OWL's outlier ratio for a given threshold M.
    """
    W = mask
    count = 0 
    total_params = 0
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()
    outlier_ratio=float(count)/total_params*100
    return outlier_ratio


def get_sparsity_owl(owl_values, layer_name, sparsity_ratio,l):
    """
    Given the OWL values, compute the sparsity distribution for a given lamnda value.
    """
    group_sparsity = []
    for key, value in owl_values.items():
        group_sparsity.append(value)

    group_sparsity=np.array(group_sparsity)
    group_sparsity = ((group_sparsity - group_sparsity.min()) * (1/(group_sparsity.max() - group_sparsity.min()) * l*2))
    group_sparsity=group_sparsity-np.mean(group_sparsity)+(1-sparsity_ratio)
        
    dist = {}

    for key in layer_name:
        if 'classifier' in key: continue
        layer_id = key.split('-')[1]
        dist[key] = 1 - group_sparsity[int(layer_id)]
        #dist[key] = group_sparsity[int(layer_id)]
        if dist[key] <= 0:
            dist[key] = 0.01
        elif dist[key] >= 1:
            dist[key] = 0.99

    return dist


def run_activation_opt(
    args, model, tokenizer, device=torch.device("cuda:0"), dataloader=None, owl=False, out_ratio=None):

    """
    Run a forward pass to compute activations for OWL's outliers for OPT models.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input_opt(args, model, dataloader, device)


    layers = model.model.decoder.layers
    activations , owl_values, output_values = {}, {}, {}
    for i in tqdm(range(len(layers)), desc="Computing OWL's Activations"):
        layer = layers[i]
        subset = find_layers(layer)



        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                )[0]
        for h in handles:
            h.remove()
        layer_wmetric = []

        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            activations[f'{name}-{i}'] = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            if owl:
                layer_wmetric.append(W_metric)  
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                )[0]
        inps, outs = outs, inps

        if owl:
            layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            owl_values[f'{i}'] = out_ratio_layer
            del layer_wmetric
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return activations, owl_values

def run_activation(
    args, model, tokenizer, device=torch.device("cuda:0"), dataloader=None, owl=False, out_ratio=None):

    """
    Run a forward pass to compute activations for OWL's outliers for LLama/Phi/Mistral models.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False


    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)


    layers = model.model.layers
    activations = {}
    owl_values = {}
    output_values = {}
    for i in tqdm(range(len(layers)), desc="Computing OWL's Activations"):
        layer = layers[i]
        subset = find_layers(layer)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )


            activations[f'{name}-{i}'] = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            if owl:
                layer_wmetric.append(W_metric)    

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        inps, outs = outs, inps
        if owl:
            layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            owl_values[f'{i}'] = out_ratio_layer

            del layer_wmetric
            
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return activations, owl_values


####################################


##### DsNoT Utils #####

def return_reorder_indice(input_tensor):
    """
    For instance:
    [[1., -2., 3.],
    [-2, 2., -4],
    [5., 6., -7],
    [-6, -7, -4]]
    return indices of
    [[-2.,  3.,  1.],
    [-2., -4.,  2.],
    [-7.,  6.,  5.],
    [-6., -7., -4.]]
    Description: The relative order in the positive number remains unchanged, and the relative order in the negative number is flipped.
    """
    positive_tensor = input_tensor.clone()
    negative_tensor = input_tensor.clone()

    positive_mask = positive_tensor > 0
    negative_mask = negative_tensor < 0

    positive_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )
    negative_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )

    positive_indices[~positive_mask] = float("inf")
    negative_indices[~negative_mask] = float("inf")

    positive_value, _ = torch.sort(positive_indices, dim=1)
    negative_value, _ = torch.sort(negative_indices, dim=1)

    positive_value = torch.flip(positive_value, dims=[1])

    negative_value[negative_value == float("inf")] = 0
    positive_value[positive_value == float("inf")] = 0

    reorder_indice = (positive_value + negative_value).to(torch.int64)

    return reorder_indice


####################################


#################### Base Utils ####################


def compute_mask(W_metric, prune_granularity, sparsity):
    """
    Given the W_metric, compute the mask for a given sparsity ratio and granularity.
    """
    if prune_granularity == "layer":
        thres = torch.sort(W_metric.flatten().cuda())[0][int(W_metric.numel() * sparsity)].cpu()
        W_mask = (W_metric <= thres)
        return W_mask 
    elif prune_granularity == "row":
        W_mask = (torch.zeros_like(W_metric)==1)
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity)]
        W_mask.scatter_(1, indices, True)
        return W_mask 
    elif prune_granularity == 'neuron':
        #code base on row-wise pruning https://github.com/OpenGVLab/LLMPrune-BESA 
        sort_indices = torch.sort(W_metric, dim=-1,stable=True, descending=True)[1]
        sparsity = [1-x for x in sparsity]
        sparsity_levels = torch.tensor(sparsity, device=W_metric.device)
        row_block_prune_num = (sparsity_levels * W_metric.shape[1]).to(dtype=torch.long)
        row_prune_num = row_block_prune_num.reshape(-1, 1).repeat(1, 1).reshape(-1)
        W_mask = mask_gen(sort_indices, W_metric.shape, row_prune_num).to(dtype=torch.bool)
        return W_mask




def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def check_sparsity(model):
    """
    Check the sparsity of a given model.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data.clone()
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"Block {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params



def check_sparsity_opt(model):
    """
    Check the sparsity of a given OPT model.
    """

    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.decoder.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
        print(f"Block {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 



