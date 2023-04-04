#!/bin/bash

#SBATCH -w c59
#SBATCH -J GPT_training              # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -t 10:00:00           # Run time (hh:mm:ss) - 1.5 hours

conda init bash
conda activate nlp

python train_gpt_cl.py \
--task_list example \
--select_k_per_class 10 \
--progressive 0 \
--lr 0.1 \
--num_epochs 30 \
--freeze_weights 1 \
--prefix_len 64 \
--model_name gpt2-large \
--early_stopping 1 \
--batch_size 4 \
--save_name GPT_experiment \
--save_dir /mnt/beegfs/jiasheng/progressiveprompts_save