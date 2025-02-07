# NeuroAL

**Zeroth-Order Adaptive Neuron Alignment Based Pruning without Retraining**

Preprint available on ArXiv 🔥.

[![arXiv](https://img.shields.io/badge/arXiv-2411.07066-b31b1b.svg)](https://arxiv.org/pdf/2411.07066?) 

[Elia Cunegatti](https://scholar.google.com/citations?hl=it&user=a2JJRjMAAAAJ), [Leonardo Lucio Custode](https://scholar.google.com/citations?user=3qvS-AwAAAAJ&hl=it), [Giovanni Iacca](https://sites.google.com/site/giovanniiacca/).

University of Trento, Italy


>**Abstract.**
*Network pruning focuses on computational techniques that aim to reduce a given model's computational cost by removing a subset of its parameters while having minimal impact on performance. Throughout the last decade, the most widely used pruning paradigm has been pruning and re-training, which nowadays is inconvenient due to the vast amount of pre-trained models, which are in any case too expensive to re-train. In this paper, we exploit functional information from dense pre-trained models, i.e., their activations, to obtain sparse models that maximize the activations' alignment w.r.t. their corresponding dense models. Hence, we propose **NeuroAL**, a *top-up* algorithm that can be used on top of any given pruning algorithm for LLMs, which modifies the block-wise and row-wise sparsity exploiting information from both the dense model and its sparse version to maximize the *neuron alignment* among activations. Differently from existing methods, our approach adaptively selects the best hyperparameters for the block-wise and row-wise sparsity ratios w.r.t. the model and the desired sparsity, and requires *no re-training*. We test our method over 276 cases combining four LLM families, three sparsity ratios, and ten language tasks (three language modeling and seven zero-shot datasets), showing how it consistently outperforms the latest state-of-the-art methods in terms of performance-runtime trade-off.*

```bibtex
@article{cunegatti2024zeroth,
  title={Zeroth-Order Adaptive Neuron Alignment Based Pruning without Re-Training},
  author={Cunegatti, Elia and Custode, Leonardo Lucio and Iacca, Giovanni},
  journal={arXiv preprint arXiv:2411.07066},
  year={2024}
}
```

## Setup

This project was developed using Python 3.9.18. You can find all dependencies in the `deps` folder, provided as a standard list of pip requirements and as a conda environment. To create a conda environment for this project, run `conda env create -f deps/environment.yaml`. To install dependencies using pip, run `pip install -r deps/requirements.txt`.

**Remark**: Some models require an access token from the ``transformers`` library. To enable this, create a ```config.py``` file with the line ```access_token = "xxx"```.

For the Zero-Shot tasks please install from ```lm-evaluation-harness``` folder, a modified version of EleutherAI LM Harness (```Wanda``` benchmark [here](https://github.com/locuslab/wanda)), adapted by us to work with models requiring access tokens, run the following commands:

```bash
cd lm-evaluation-harness
pip install -e .
```

To use the Customized Cuda Operator for efficient row-wise pruning (all credits to ```BESA``` authors 🙏, *we just re-used their Cuda implementation* available  [here](https://github.com/OpenGVLab/LLMPrune-BESA)) install as follows:

```bash
cd ops
python setup.py install
```

## Usage
In order To reproduce our method as well as the baselines, run:

```
python3 main.py --model microsoft/phi-2 --prune_method wanda --sparsity_ratio 0.7 --top_up neuroal --neuroal_step both
```

To run NeuroAL as presented in the paper, use `--neuroal_step both`. For block-only (faster) or row-only (slower) execution, set the flag to either `block` or `row`. To run the competitors, set `--top_up` to one of the three options available (`uniform`, `DSnoT`, `OWL`, or `alpha`). All additional details and options for the commands are listed in the help text below.



This is what you see if you run ```python main.py --help ```:
```
  -h, --help            show this help message and exit
  --model {facebook/opt-125m,facebook/opt-6.7B,microsoft/phi-2,baffo32/decapoda-research-llama-7B-hf,dfurman/LLaMA-13B,mistralai/Mistral-7B-v0.1,meta-llama/Llama-2-7b-hf,meta-llama/Llama-2-13b-hf}
                        LLM model to be pruned
  --seed SEED           Seed for sampling the calibration data
  --nsamples NSAMPLES   Number of calibration samples
  --sparsity_ratio SPARSITY_RATIO
                        Sparsity ratio
  --prune_method {wanda,sparsegpt,magnitude,dense,multiflow}
                        Pruning method
  --top_up {neuroal,owl,DSnoT,uniform}
                        Top-up method
  --cache_dir CACHE_DIR
                        Cache directory for model weights
  --neuroal_step {block,row,both}
                        NeuroAL step
  --M M                 M parameter for OWL
  --lam LAM             Lambda parameter for OWL
  --tasks {LM,zero-shot,both}
                        Evaluation tasks for the pruned model

```



## Ackdowlegments
Our codebase is developed on top of the following repositories:

- ```SparseGPT```, Frantar and Alistarh 2023, [https://github.com/IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt)
- ```Wanda``` , Sun et al. 2023, [https://github.com/locuslab/wanda](https://github.com/locuslab/wanda)
- ```DsNoT```, Zhang et al. 2024, [https://github.com/zyxxmu/DSnoT](https://github.com/zyxxmu/DSnoT)
- ```OWL```,  Yin et al. 2024, [https://github.com/luuyin/OWL](https://github.com/luuyin/OWL)



## Contact
The codebase underwent a major refactoring before publication. If you encounter any issues, please reach out to [elia.cunegatti@unitn.it](mailto:elia.cunegatti@unitn.it) or open a public issue 😀.
