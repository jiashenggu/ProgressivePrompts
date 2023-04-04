import gpt_dataset
from transformers import AutoTokenizer
import datasets
import pandas as pd
model_name = 'gpt2'
task = 'example'
tokenizer = AutoTokenizer.from_pretrained(model_name)
ds = gpt_dataset.GPTDataset(tokenizer, task)
example_dataset = ds.get_final_ds(task, split='train', batch_size=2, k=2)

import ipdb; ipdb.set_trace()
for batch in example_dataset:
    print(batch)
    break
# split = 'train'
# df = pd.read_csv('../datasets/src/data/'+task+'/'+split+'.csv', header=None)
# df = df.rename(columns={0: "label", 1: "title", 2: "content"})
# df['label'] = df['label'] - 1
# dataset = datasets.Dataset.from_pandas(df)
# print(dataset)
# print(dataset[0])
