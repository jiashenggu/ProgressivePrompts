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


# Define the prompt text
prompt_text = '''Frank and Cindy are bakers in the city of Paris, France. 
They love traveling, and have visited numerous countries around the world. 
They enjoy cruises, hiking, and visiting cities with history and flair. 
Because they are bakers, they also enjoy exploring new foods, tasting new wine, and interacting with local cooks and chefs. 
Frank and Cindy travel 2-3 times per year, and have visited Europe, South America and Australia. 
Now repeat the text:
'''

# Tokenize the prompt text
input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
# print(tokenizer.convert_ids_to_tokens(input_ids[0]))
# Generate text from the model
input_embeddings = model.get_input_embeddings()(input_ids)
output = model.generate(input_ids, max_length=200)
# import ipdb; ipdb.set_trace()
# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated text:")
print(generated_text)
