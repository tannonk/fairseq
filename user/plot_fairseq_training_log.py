#!/usr/bin/env python
# coding: utf-8

import sys
import re
import pandas as pd
import matplotlib.pyplot as plt

"""
Parses Fairseq training log 
(e.g. output of `nohup fariseq-train...`)
and plots training and validation progress

Example call:
    python3 plot_fairseq_training_log.py log_file plot_file

with two log files:
    python3 plot_fairseq_training_log.py /srv/scratch2/kew/fairseq_materials/rrgen/de/ft100_rg/train.log /srv/scratch2/kew/fairseq_materials/rrgen/de/ft100_rg/train_cont.log /srv/scratch2/kew/fairseq_materials/rrgen/de/ft100_rg/train_plot_2.png


"""

log_files = sys.argv[1:-1] # takes one or more log files (e.g. for interrupted/continued training logs)
outfile = sys.argv[-1]

# for testing
# log_files = ['/home/user/kew/nohup.out']
# print(log_files)
# print(outfile)

# sys.exit()

# regex patterns for ID-ing/parsing lines
tr_inner = re.compile(r'\| INFO \| train_inner \|')
tr = re.compile(r'\| INFO \| train \|')
va = re.compile(r'\| INFO \| valid \|')
epc = re.compile(r'epoch (\d+):\s+(\d+) / (\d+) (.*)')


def parse_training_line(line):
    """
    Parse `train inner` log line for relevant values.
    These lines are written at each step.
    """
    m = re.search(epc, line)
    e, s, t, field_str = m.groups()

    fields = [i.strip().split('=') for i in field_str.split(',')
              if i.strip() and len(i.split('=')) == 2]
    fields = {k: float(v) for k, v in fields}

    return fields


def parse_train_val_line(line):
    """
    Parse `train / valid` lines from log file for relevant values.

    These lines are written at the end of each epoch.
    """
    fields = [i.strip().split() for i in line.split(
        '|') if i.strip() and len(i.split()) == 2]
    fields = {k: float(v) for k, v in fields}
    fields['epoch'] = int(fields['epoch'])
    return fields


train = []
valid = []
train_inner = []

for log_file in log_files:
    with open(log_file, 'r', encoding='utf8') as f:
        for line in f:
            # 2020-09-19 13:32:40 | INFO | train_inner | ...
            line = line[19:].strip()
            if re.search(tr, line):
                train.append(parse_train_val_line(line))
            elif re.search(va, line):
                valid.append(parse_train_val_line(line))
            elif re.search(tr_inner, line):
                train_inner.append(parse_training_line(line))

# epoch stats
train_progress = pd.DataFrame(train)
valid_progress = pd.DataFrame(valid)
# merge progress dfs into one for ease of plotting
epoch_df = train_progress.merge(valid_progress, how='left', on='epoch', suffixes=('_train', '_validation'))
# print(epoch_df)

# step stats
train_details = pd.DataFrame(train_inner)


# plot dataframes
fig, axes = plt.subplots(1, 3, figsize=(20, 4))

epoch_df.plot.line(x='epoch', y=['loss_train', 'loss_validation'], logy=False, ax=axes[0], xticks=epoch_df['epoch'])
axes[0].title.set_text('Loss / Epoch')

epoch_df.plot.line(x='epoch', y=['ppl_train', 'ppl_validation'], logy=False, ax=axes[1], xticks=epoch_df['epoch'])
axes[1].title.set_text('PPL / Epoch')

train_details.plot.line(
    x='num_updates', y=['loss', 'ppl'], logy=True, ax=axes[2])
axes[2].title.set_text('Training set / updates')
# if outfile:
plt.savefig(outfile)
print(f'Saved plots to {outfile}')
