## Adapting Transformers for NLP tasks
This repo contains utility scripts adapted from [AdapterHub.ml](https://adapterhub.ml/) and [Huggingface.co](https://huggingface.co/) 
transformer training scripts for masked language modeling and task adapter training.

This was an attempt to replicate domain-adaptive pretraining ("DAPT") and task-adaptive 
pretraining ("TAPT") as presented in ["Don't Stop Pretraining"](https://arxiv.org/abs/2004.10964) (Gururangan et al., 2020)
using lightweight adapters instead of fully fine-tuning model weights. 

There are Jupyter notebooks for masked language modeling (MLM) as well as task adapter 
training using randomized grid search to find optimal hyperparameter combinations. Thanks to 
[Jon Chen](https://github.com/jonchen1994) for contributing the random grid search code.

To reduce computation requirements for training, the code is currently configured
to use 5 of the 9 total [GLUE](https://gluebenchmark.com/) tasks, but can be expanded to use all of them.
Note that at the time of this writing, the stsb task doesn't work.
