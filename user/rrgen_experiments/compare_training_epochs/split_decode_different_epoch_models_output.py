#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re

infile = 'decode_different_epoch_models_output.txt'

with open(infile, 'r', encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if (m := re.match(r'--------------- BEGIN DECODE EPOCH (\d+) ---------------', line)):
            # print(m.group(1))
            outf = open(f'decode_epoch_{m.group(1)}.txt', 'w', encoding='utf8')
        elif (m := re.match(r'--------------- FINISH DECODE EPOCH (\d+) ---------------', line)):    
            outf.close()
        else:
            outf.write(f'{line}\n')
            
print('Finished.')