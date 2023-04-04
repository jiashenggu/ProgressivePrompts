import torch
from torch import nn
import datasets
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse


from itertools import cycle
from copy import deepcopy
from transformers import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from sklearn.metrics import matthews_corrcoef, f1_score

from torch.nn.functional import kl_div
from torch import log_softmax, softmax

model_name = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
prefix_len = 64
max_length = 100
seq_len = 512
model.eval()


input_text = 'Now repeat the text:'

# # Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
input_embeddings = model.get_input_embeddings()(input_ids)
# Create a soft prompt embedding
soft_prompt = np.load('/mnt/beegfs/jiasheng/progressiveprompts_save/GPT_experiment/prompts.npy')
# convert to tensor
soft_prompt = torch.from_numpy(soft_prompt).to(device)
soft_prompt = soft_prompt.unsqueeze(0)
print(soft_prompt.shape)
# import ipdb; ipdb.set_trace()
# Get input embeddings and prepend the soft prompt


input_embeddings_with_soft_prompt = torch.cat([soft_prompt, input_embeddings], dim=1)

# Generate text from the model


# Decode the generated text
outputs = model.generate(inputs_embeds=input_embeddings_with_soft_prompt, max_length=200)
print("\ngenerate + inputs_embeds:", tokenizer.decode(outputs[0], skip_special_tokens=True))


# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

# text = "Hello world"
# input_ids = tokenizer.encode(text, return_tensors="pt")

# # Traditional way of generating text
# outputs = model.generate(input_ids)
# print("\ngenerate + input_ids:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# # From inputs_embeds -- exact same output if you also pass `input_ids`. If you don't
# # pass `input_ids`, you will get the same generated content but without the prompt
# prompt_input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
# inputs_embeds = model.transformer.wte(prompt_input_ids)
# outputs = model.generate(inputs_embeds=inputs_embeds, input_ids=input_ids)
# print("\ngenerate + inputs_embeds:", tokenizer.decode(outputs[0], skip_special_tokens=True))