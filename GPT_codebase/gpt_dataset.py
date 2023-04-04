import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import datasets


class GPTDataset:
    def __init__(self, tokenizer, task):
        """Dataset class for gpt model experiments.
        Args:
            task (str): Name of the downstream task.
            tokenizer (HuggingFace Tokenizer): gpt model tokenizer to use.
        """
        
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'sep_token': '<SEP>'})

        self.glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', \
                              'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        self.superglue_datasets = ['copa', 'boolq', 'wic', 'wsc', 'cb', 'record', 'multirc', 'rte_superglue', 'wsc_bool']
        
        # Column keys used in the dataset
        self.task_to_keys = {
            "amazon": ("content", None),
            "example": ("prompt", None),
        }
        
        # Label text for gpt tasks
        # (gpt has text-to-text format for text and labels)
        self.task_to_labels = {
            "amazon": ("terrible", "bad", "middle", "good", "wonderful"),
            "example": ("terrible", "bad", "middle", "good", "wonderful"),
        }

        self.task = task
        self.label_key = 'label'
        if 'yahoo_' in task: self.label_key = 'topic'
        if 'stsb' in task: self.label_key = 'similarity_score'
        if task=='record': self.label_key = 'answers'

    
    # Helper function to save idx of multirc questions (needed later for test metric computation)
    def save_multirc_questions_idx(self, val_ds):
        idx = []
        i = 0
        x_prev, y_prev= val_ds['paragraph'][0], val_ds['question'][0]

        for x,y in zip(val_ds['paragraph'], val_ds['question']):
            if x_prev!=x or y_prev!=y:
                i += 1
            x_prev = x
            y_prev = y
            idx.append(i)
        self.multirc_idx = np.array(idx)

    
    # Helper function to select a subset of k samples per class in a dataset
    def select_subset_ds(self, ds, k=2000, seed=0):
        if self.task in ['stsb', 'record', 'wsc']: # non-discrete labels
            idx_total = np.random.choice(np.arange(ds.shape[0]), min(k,ds.shape[0]), replace=False)

        else:
            label_key = self.label_key
            N = len(ds[label_key])
            idx_total = np.array([], dtype='int64')

            for l in set(ds[label_key]):
                idx = np.where(np.array(ds[label_key]) == l)[0]
                idx_total = np.concatenate([idx_total, # we cannot take more samples than there are available
                                            np.random.choice(idx, min(k, idx.shape[0]), replace=False)])

        np.random.seed(seed)
        np.random.shuffle(idx_total)
        return ds.select(idx_total)

    
    # WSC task function to preprocess raw input & label text into tokenized dictionary
    def process_wsc(self, wsc_row):
        text_proc = wsc_row['text'].split(' ')
        #text_proc[wsc_row['span1_index']] = '*' + text_proc[wsc_row['span1_index']] +'*'
        target = text_proc[wsc_row['span1_index']]
        text_proc[wsc_row['span2_index']] = '*' + text_proc[wsc_row['span2_index']] + '*'
        text_proc = (' ').join(text_proc)
        return text_proc, target

    
    # Function to preprocess raw input & label text into tokenized dictionary
    def preprocess_function(self, examples, task,
                            max_length=512, max_length_target=2,
                            prefix_list=[]):
        tokenizer = self.tokenizer
        keys = self.task_to_keys[task]
        label_key = self.label_key

        if keys[1]!=None:
            if task=='record':
                text = 'passage : ' + str(examples['passage']) + ' query: ' + str(examples['query']) + ' entities: ' + ('; ').join((examples['entities']))
            elif task=='wsc':
                text, target = self.process_wsc(examples)
            else:
                text = ''
                for key in keys:
                    text += key + ': ' + str(examples[key]) + ' '
        else:
            text = examples[keys[0]]

        if len(prefix_list)>0:
            text = (' ').join(prefix_list) + ' ' + text
        

        if task == 'example':
            target = examples['content']
        # import ipdb; ipdb.set_trace()
        full_text = tokenizer.bos_token + text + tokenizer.sep_token + target + tokenizer.eos_token
        encodings_dict = tokenizer(full_text,                                   
                                   truncation=True, 
                                   max_length=max_length, 
                                   padding="max_length")   
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        dict_final =    {'labels': torch.tensor(input_ids),
                        'input_ids': torch.tensor(input_ids), 
                        'attention_mask': torch.tensor(attention_mask)}
        return dict_final


    
    def get_final_ds(self, 
                     task, 
                     split,
                     batch_size,
                     k=-1,
                     seed=0,
                     return_test=False,
                     target_len=2,
                     max_length=512,
                     prefix_list=[]):
        """Function that returns final gpt dataloader.
        Args:
            task (str): Name of the downstream task.
            split (str): Which data split to use (train/validation/test).
            batch_size (int): Batch size to use in the dataloader.
            k (int, optional): Number of samples to use for each class. Defaults to -1, not sub-sample the data.
            seed (int, optional): Seed used for random shuffle. Defaults to 0.
            return_test (bool, optional): Whether to create a test split. 
                When True, two Dataloaders are returned. Defaults to False.
            target_len (int, optional): Length of the model output (in tokens). Defaults to 2.
            max_length (int, optional): Length of the model input (in tokens). Defaults to 512.
            prefix_list (List[str], optional): List of prompt virtual tokens to pre-pend to the input. 
                We do not encode soft prompt as extra virtual tokens in the latest implementation.
                Defaults to [], empty list.
            
        Returns:
            Dataloader: Torch Dataloader with preprocessed input text & label.
        """
        if task in ['example']:
            df = pd.read_csv('../datasets/src/data/'+'amazon'+'/'+split+'.csv', header=None)
            df = df.rename(columns={0: "label", 1: "title", 2: "content"})
            df['prompt'] = "Frank and Cindy are bakers in the city of Paris, France. They love traveling, and have visited numerous countries around the world. They enjoy cruises, hiking, and visiting cities with history and flair. Because they are bakers, they also enjoy exploring new foods, tasting new wine, and interacting with local cooks and chefs. Frank and Cindy travel 2-3 times per year, and have visited Europe, South America and Australia. They have not visited Africa, but hope to someday. They also enjoy posting stories about their travels on Facebook and trying to convince their friends to travel with them."
            df['label'] = df['label'] - 1
            dataset = datasets.Dataset.from_pandas(df)

        
        # Selecting k subset of the samples (if requested)
        if k!=-1:
            dataset = self.select_subset_ds(dataset, k=k)

        if k==-1 and split!='train' and self.task=='multirc':
            # we do not shuffle full validation set of multirc
            # but we save idx of the same questions
            # which are used for multirc test metric computation
            self.save_multirc_questions_idx(dataset)
        else:
            dataset = dataset.shuffle(seed=seed)
        
        # Returning the selected data split (train/val/test)
        if return_test==False:
            encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, task,
                                                                            max_length=max_length,
                                                                            max_length_target=target_len,
                                                                            prefix_list=prefix_list),
                                          batched=False)
            encoded_dataset.set_format(type='torch', columns=['labels','input_ids', 'attention_mask'])
            dataloader = DataLoader(encoded_dataset, batch_size=batch_size)

            return dataloader
        
        # Creating an extra test set from the selected data split
        else:
            N = len(dataset)
            dataset_val = dataset.select(np.arange(0, N//2))
            dataset_test = dataset.select(np.arange(N//2, N))

            dataloaders_val_test = []
            for dataset in [dataset_val, dataset_test]:
                encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, task,
                                                                                 max_length=max_length,
                                                                                 max_length_target=target_len,
                                                                                 prefix_list=prefix_list),
                                              batched=False)
                encoded_dataset.set_format(type='torch', columns=['labels','input_ids', 'attention_mask'])
                dataloader = DataLoader(encoded_dataset, batch_size=batch_size)
                dataloaders_val_test.append(dataloader)

            return dataloaders_val_test
