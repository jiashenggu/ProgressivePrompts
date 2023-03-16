import torch
from torch import nn
import datasets
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

import t5_dataset
from itertools import cycle
from copy import deepcopy
from transformers import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import matthews_corrcoef, f1_score

from torch.nn.functional import kl_div
from torch import log_softmax, softmax

model_name = 'google/flan-t5-base'
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)
prefix_len = 64
max_length = 100
seq_len = 512
model.eval()


# input_text = 'Repeat the following text: Frank and Cindy are bakers in the city of Paris, France. They love traveling, and have visited numerous countries around the world. They enjoy cruises, hiking, and visiting cities with history and flair. Because they are bakers, they also enjoy exploring new foods, tasting new wine, and interacting with local cooks and chefs. Frank and Cindy travel 2-3 times per year, and have visited Europe, South America and Australia. They have not visited Africa, but hope to someday. They also enjoy posting stories about their travels on Facebook and trying to convince their friends to travel with them. '
input_text = '''Frank and Cindy are bakers in the city of Paris, France. 
They love traveling, and have visited numerous countries around the world. 
They enjoy cruises, hiking, and visiting cities with history and flair. 
Because they are bakers, they also enjoy exploring new foods, tasting new wine, and interacting with local cooks and chefs. 
Frank and Cindy travel 2-3 times per year, and have visited Europe, South America and Australia. 
Now repeat the paragraph above:'''




# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Create a soft prompt embedding
soft_prompt = np.load('/mnt/beegfs/jiasheng/progressiveprompts_save/T5_experiment/prompts.npy')
# convert to tensor
soft_prompt = torch.from_numpy(soft_prompt).to(device)
soft_prompt = soft_prompt.unsqueeze(0)
# import ipdb; ipdb.set_trace()
# Get input embeddings and prepend the soft prompt
input_embeddings = model.get_input_embeddings()(input_ids)
# input_embeddings_with_soft_prompt = torch.cat([soft_prompt, input_embeddings], dim=1)

# Generate the output
with torch.no_grad():
    output = model.generate(inputs_embeds=input_embeddings, min_length=50, max_length=100)
    print(output[0].shape)

# Decode the generated output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(input_text)
print(decoded_output)

# from transformers import AutoTokenizer, T5ForConditionalGeneration

# tokenizer = AutoTokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# training
# input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
# labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
# outputs = model(input_ids=input_ids, labels=labels)
# loss = outputs.loss
# logits = outputs.logits

# # inference
# input_ids = tokenizer(
#     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
# ).input_ids  # Batch size 1
# input_embeddings = model.get_input_embeddings()(input_ids)
# outputs = model.generate(inputs_embeds=input_embeddings, min_length=40)
# print(outputs[0].shape)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# # studies have shown that owning a dog is good for you.