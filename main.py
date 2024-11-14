import os
import pickle
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.data import get_loaders
from lib.eval import eval_ppl, eval_zero_shot
from lib.prune import (
    prune_DSnoT, prune_wanda, prune_magnitude, prune_sparsegpt, 
    check_sparsity, prune_multiflow
)
from lib.neuroal import *
from lib.prune_opt import (
    check_sparsity_opt, prune_DSnoT_opt, prune_magnitude_opt, 
    prune_wanda_opt, prune_sparsegpt_opt, prune_multiflow_opt
)
from sparsity_distribution import (
    get_neuroal_sparsity_distribution, get_owl_sparsity_distribution, 
    get_uniform_sparsity_distribution
)


from config import access_token as access_token
import warnings
warnings.filterwarnings("ignore")

def get_llm(model, cache_dir="llm_weights", token=None):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto",
        token=token
    )

    model.seqlen = 2048
    return model

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Model selection
    parser.add_argument('--model', type=str, default="facebook/opt-125m",
                        choices=["facebook/opt-125m", "facebook/opt-6.7B", "microsoft/phi-2",
                                 "baffo32/decapoda-research-llama-7B-hf", "dfurman/LLaMA-13B",
                                 "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b-hf",
                                 "meta-llama/Llama-2-13b-hf"], 
                        help='LLM model to be pruned')

    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity ratio')

    # Pruning methods
    parser.add_argument("--prune_method", type=str,
                        choices=["wanda", "sparsegpt", "magnitude", "dense", "multiflow"],
                        help="Pruning method")
    
    # Top-up options
    parser.add_argument("--top_up", type=str, default='neuroal',
                        choices=["neuroal", "owl", "DSnoT", "uniform"], help="Top-up method")
    parser.add_argument("--cache_dir", type=str, default="llm_weights", help="Cache directory for model weights")

    # NeuroAl Hyperparameters
    parser.add_argument("--neuroal_step", type=str, default="both",
                        choices=["block", "row", "both"], help="NeuroAL step")

    # OWL Hyperparameters
    parser.add_argument("--M", type=int, default=5, help="M parameter for OWL")
    parser.add_argument("--lam", type=float, default=0.08, help="Lambda parameter for OWL")

    # DSnoT Hyperparameters
    parser.add_argument('--max_cycle_time', type=int, default=50, help='Max cycle time for DSnoT')
    parser.add_argument('--without_DSnoT', action="store_true", help="Disable DSnoT")
    parser.add_argument('--update_threshold', type=float, default=0.1, help='Update threshold for DSnoT')
    parser.add_argument('--pow_of_var_regrowing', type=float, default=1, help='Variance power for regrowing')
    parser.add_argument('--pow_of_var_pruning', type=float, default=1, help='Variance power for pruning')
    parser.add_argument("--skip_layer", type=str, default="mlp",
                        choices=["no_skip", "mlp", "self_attn"], help="Layers to skip")
    parser.add_argument("--skip_sub_layer", type=str, default="no_skip",
                        choices=["no_skip", "q_proj", "k_proj", "v_proj", "o_proj", 
                                 "gate_proj", "down_proj", "up_proj", "fc1", "fc2", "out_proj"], 
                        help="Sub-layers to skip")
    parser.add_argument('--without_same_sign', type=bool, default=True, help="Disable same-sign constraint")

    # Task Hyperparameters
    parser.add_argument('--tasks', type=str, default='LM', choices=["LM", "zero-shot", "both"],
                        help="Evaluation tasks for the pruned model")

    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def determine_model_type(args):
    model_lower = args.model.lower()
    if any(substring in model_lower for substring in ["llama"]):
        return "llama"
    elif "opt" in model_lower:
        return "opt"
    elif "phi" in model_lower:
        return "phi"
    elif "mistral" in model_lower:
        return "mistral"
    else:
        print("Warning: Model type not specified from model path, please specify manually")
    return args.model_type

def main():
    args = parse_arguments()
    set_seed(args.seed)
    args.model_type = determine_model_type(args)
    
    print(f"Model type: {args.model_type}")
    print(f"Loading LLM model {args.model}")
    
    model = get_llm(args.model, args.cache_dir, access_token)
    model.eval()

    tokenizer_params = {'use_fast': False, 'token': access_token}
    if args.model in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf']:
        tokenizer_params['legacy'] = True

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_params)
    print("Model and tokenizer loaded")

    device = torch.device("cuda:0")
    print("Using device:", device)


    print("Loading calibration data")
    calib_data, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )


    print(f"Pruning method: {args.prune_method}")

    best_lambda_group, best_lambda_neuron = None, None
    if args.model_type == "opt":
        layers = model.model.decoder.layers
    elif args.model_type in ["phi", "llama", "mistral"]:
        layers = model.model.layers

    
    if args.top_up == 'neuroal': 
        sparsity_distribution, granularity, best_lambda_group,best_lambda_neuron = get_neuroal_sparsity_distribution(args, model, tokenizer, device, layers)
    elif args.top_up == 'owl':
        sparsity_distribution, granularity = get_owl_sparsity_distribution(args, model, tokenizer, device, calib_data)
    elif args.top_up == 'uniform':
        sparsity_distribution, granularity = get_uniform_sparsity_distribution(args, layers)


    with torch.no_grad():
        model.eval()
        args.nsamples = 128
        with torch.no_grad():
            if args.top_up != 'DSnoT':
                if args.model_type == "opt":
                    if args.prune_method == "wanda":
                        prune_wanda_opt(args, model, tokenizer , device, calib_data, dist=sparsity_distribution, granularity=granularity)
                    elif args.prune_method == "magnitude":
                        prune_magnitude_opt(args, model, tokenizer, device, dist=sparsity_distribution, granularity=granularity)
                    elif args.prune_method == "sparsegpt":
                        prune_sparsegpt_opt(args, model, tokenizer, device, calib_data, dist=sparsity_distribution)
                    elif args.prune_method == "multiflow":
                        prune_multiflow_opt(args, model, tokenizer, device, calib_data, dist=sparsity_distribution, granularity=granularity)
                elif args.model_type == "phi" or args.model_type == "llama" or args.model_type == "mistral":
                    if args.prune_method == "wanda":
                        prune_wanda(args, model, tokenizer, device, calib_data, dist=sparsity_distribution, granularity=granularity)
                    elif args.prune_method == "magnitude":
                        prune_magnitude(args, model, tokenizer, device, dist=sparsity_distribution,granularity=granularity)
                    elif args.prune_method == "sparsegpt":
                        prune_sparsegpt(args, model, tokenizer, device, calib_data, dist=sparsity_distribution)
                    elif args.prune_method == "multiflow":
                        prune_multiflow(args, model, tokenizer, device, calib_data, dist=sparsity_distribution, granularity=granularity)

            elif args.top_up == 'DSnoT':
                if args.skip_layer is not None:
                    if args.model_type == 'llama' or args.model_type == 'phi' or args.model_type == 'mistral':
                        args.skip_layer = 'mlp'
                    elif args.model_type == 'opt':
                        args.skip_layer = 'fc'
                
                if args.model_type == "opt":
                    prune_DSnoT_opt(args, model, tokenizer, device, calib_data)
                else:
                    prune_DSnoT(args, model, tokenizer, device, calib_data)

    

    print("*"*30)
    if args.model_type == "opt":
        sparsity_ratio = check_sparsity_opt(model)
    else:
        sparsity_ratio = check_sparsity(model)
    print(f"Overall Sparsity Ratio: {sparsity_ratio:.4f}")
    print("*"*30)
    model_name = args.model.split('/')[-1]

    if args.tasks == 'LM' or args.tasks == 'both':
        dataset = 'wikitext2'
        ppl_wikitext = eval_ppl(model, tokenizer, dataset, device)
        print(f"\nPerplexity on {dataset}: {ppl_wikitext}\n")

        dataset = 'c4'
        ppl_c4 = eval_ppl(model, tokenizer, dataset, device)
        print(f"\nPerplexity on {dataset}: {ppl_c4}\n")

        dataset = 'ptb'
        ppl_ptb = eval_ppl(model, tokenizer, dataset, device)
        print(f"\nPerplexity on {dataset}: {ppl_ptb}\n")

        filename_csv = 'perplexity_results.csv'

        if os.path.exists(filename_csv):
            append_write = 'a' 
        else:
            append_write = 'w'
        
        if args.top_up == None: args.top_up = 'none'
        with open(filename_csv, append_write) as f:
            f.write(f"{model_name},{args.sparsity_ratio},{args.prune_method},{args.top_up},{best_lambda_group},{best_lambda_neuron},{ppl_wikitext},{ppl_c4},{ppl_ptb},{sparsity_ratio}\n")

    elif args.tasks == 'zero-shot' or args.tasks == 'both':
        accelerate = False
        task_list = ["rte", "winogrande", "boolq","hellaswag","arc_easy","arc_challenge","openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate, access_token=access_token)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

        filename = f'zero_shot_results/{model_name}-{args.prune_method}-{args.top_up}-{args.sparsity_ratio}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

if __name__ == '__main__':
    main()


