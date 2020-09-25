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


"""

log_file = sys.argv[1]
outfile = sys.argv[2]

# for testing
# log_file = '/home/user/kew/nohup.out'

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

with open(log_file, 'r', encoding='utf8') as f:
    for line in f:
        line = line[19:].strip()
        if re.search(tr, line):
            train.append(parse_train_val_line(line))
        elif re.search(va, line):
            valid.append(parse_train_val_line(line))
        elif re.search(tr_inner, line):
            train_inner.append(parse_training_line(line))


train_progress = pd.DataFrame(train)
valid_progress = pd.DataFrame(valid)
train_details = pd.DataFrame(train_inner)

# plot dataframes
fig, axes = plt.subplots(1, 3, figsize=(20, 4))
train_progress.plot.line(x='epoch', y=[
                         'loss', 'ppl'], logy=True, ax=axes[0], xticks=train_progress['epoch'])
valid_progress.plot.line(x='epoch', y=[
                         'loss', 'ppl'], logy=True, ax=axes[1], xticks=valid_progress['epoch'])
train_details.plot.line(
    x='num_updates', y=['loss', 'ppl'], logy=True, ax=axes[2])

# if outfile:
plt.savefig(outfile)
print(f'Saved plots to {outfile}')
