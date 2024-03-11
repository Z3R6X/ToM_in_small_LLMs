# Boosting Theory of Mind in small LLMs
	
## Overview

The goal of this work is to investigate the Theory of Mind (ToM) capabilities of Large Language Models (LLMs) with a "small" number of parameters.
For that, the experiment of Moghaddam&Honey is reproduced for different different ~7B parameter LLMs: GPT4All-J, LLaMa2 and LLaMa2-Chat. 
To test the ToM capabilities the models are instructed to answer questions about 16 different ToM scenarios. 
To boost the models performance in answering these questions, different types of in-context prompting methods, two-shot chain-of-Thought reasoning and step-by-step thinking instructions, were used. 
Models trained without human preference finetuning, GPT4All-J v1.3 and LLaMa2, achieved only a poor performance, wheras LLaMa2, trained with human preference, could heavily improve its performance
with in-context prompting. This observation suggested that human preference finetuning might play a key-role for the development of ToM reasoning capabilities.
To test this hypothesis, one model, GPT4All-J v1.3, was finetuned for human preference with Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO).  

This repository contains...
- A report containing the results of the Theory of Mind experiment.
- Scripts to reproduce the Theory of Mind experiment for models of the GPT4All-J and the LLaMa2 family.
- Scripts to finetune GPT4All-J with PPO.
- Scripts to finetune GPT4All-J with DPO.

## Setup

- Clone the repository:
	```
	git clone https://github.com/Z3R6X/ToM_in_small_LLMs.git
	```
	
- Create and activate Conda environment:
	```
	conda create -n tom python=3.10
	conda activate tom
	```
	
- Install packages:
	```
	conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
	conda install -c conda-forge transformers
	pip install accelerate 
	pip install bitsandbytes
	pip install peft
	```
	
- For PPO finetuning TRL version 0.5.0 is required:
	```
	pip install trl==0.5.0
	```

	
- For DPO finetuning TRL version 0.7.1 is required:
	```
	pip install trl==0.7.1
	``` 
	
## Model Finetuning
### Proximal Policy Optimization 

First, finetune the model in a supervised fashion to avoid a distribution shift during reward modeling and ppo finetuning and merge the adapter weights with the base model:

	python PPO/ppo_supervised_finetuning.py
	python merge_peft_adapter.py --adapter_model_name <adapter checkpoint> --base_model_name nomic-ai/gpt4all-j --base_model_revision v1.3-groovy --output_name GPT4All-J_PPO_SFT

Second, finetune a reward model and merge the adapter weights with the previously finetuned model:

	python PPO/ppo_reward_modeling.py --model_name GPT4All-J_PPO_SFT
	python merge_peft_adapter.py --adapter_model_name <adapter checkpoint> --base_model_name GPT4All-J_PPO_SFT --output_name GPT4All-J_PPO_RM

Third, finetune the model with reinforcement learning by using the reward model and merge the adapter weights with the previously finetuned model:
	
	python PPO/ppo_rl_training.py --model_name GPT4All-J_PPO_SFT --reward_model_name GPT4All-J_PPO_RM
	python merge_peft_adapter.py --adapter_model_name <adapter checkpoint> --base_model_name GPT4All-J_PPO_SFT --output_name GPT4All-J_PPO

### Direct Preference Optimization 

First, finetune the model in a supervised fashion to avoid a distribution shift during preverence finetuning and merge the adapter weights with the base model:

	python DPO/dpo_supervised_finetuning.py
	python merge_peft_adapter.py --adapter_model_name <adapter checkpoint> --base_model_name nomic-ai/gpt4all-j --base_model_revision v1.3-groovy --output_name GPT4All-J_DPO_SFT

Second, finetune the model with preference pairs in a supervised fashion and merge the adapter weights with the previously finetuned model:

	python DPO/dpo_finetuning.py --model_name_or_path GPT4All-J_DPO_SFT
	python merge_peft_adapter.py --adapter_model_name <adapter checkpoint> --base_model_name GPT4All-J_DPO_SFT --output_name GPT4All-J_DPO

The finetuned GPT4All-J DPO model can be found on [Huggingface](https://huggingface.co/Z3R6X/gpt4all_dpo_instruct)

## Theory of Mind Experiment

The Theory of Mind experiment consists of 16 scenarios and corresponding questions that are fed to the model.
Each question is repeated 20 times. The answers to each question are stored in a json-file.
For example: 
	
> Answer the following question in context:
>
> Scenario: The weather was so warm today that all the tulips in Pam's backyard suddenly bloomed. The tulips next to Pam's office still have not yet flowered, though. Pam has been at work all day. Question: When Pam is driving home after work, does she assume her tulips have bloomed? Answer:


The ToM experiment can be run for different models with the following scripts:

	python ToM_Experiment/ToM_Eval_GPT4All.py
	python ToM_Experiment/ToM_Eval_GPT4All_Finetune.py --model GPT4All-J_FT
	python ToM_Experiment/Tom_Eval_LLaMa2.py
	python ToM_Experiment/Tom_Eval_LLaMa2-Chat.py

The type of prompting can be specified with the '--sbs' flag for Step-by-Step reasoning and with the '--cot' flag for Chain-of-Thought reasoning.
For sampling '--top_k', '--top_p', '--temp' and '--tokens' can also be specified.
The number of repetitions for every ToM question can be specified with '--reps'.
For example:
	
	python ToM_Experiment/ToM_Eval_GPT4All.py --sbs --cot --reps 20 --temp 0.4 --top_k 50 --top_p 0.95 --tokens 128 
	
## Results

| Model |  Base  | SbS | CoT | SbS+CoT |
|:------:|:------:|:------:|:------:|:------:|
| GPT4All-J v1.3 | **0.51** | 0.31 | 0.46 | 0.45 |
| GPT4All-J PPO | **0.62**  | 0.35 | 0.53 | 0.49 |
| GPT4All-J DPO | 0.27  | 0.35 | 0.49 | **0.6** |
| LLaMa2 |  0.34  | 0.43 | 0.56 | **0.58** |
| LLaMa2 Chat | 0.28 | 0.57 | 0.67 | **0.78** |

	
	
	
