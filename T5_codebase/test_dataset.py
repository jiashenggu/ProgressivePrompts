import t5_dataset
from transformers import T5Tokenizer
import datasets
import pandas as pd
# model_name = 't5-small'
task = 'example'
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# ds = t5_dataset.T5Dataset(tokenizer, task)
# example_dataset = ds.get_final_ds(task, split='train', batch_size=2)

# for batch in example_dataset:
#     print(batch)
split = 'train'
df = pd.read_csv('../datasets/src/data/'+task+'/'+split+'.csv', header=None)
df = df.rename(columns={0: "label", 1: "title", 2: "content"})
df['label'] = df['label'] - 1
dataset = datasets.Dataset.from_pandas(df)
print(dataset)
print(dataset[0])
