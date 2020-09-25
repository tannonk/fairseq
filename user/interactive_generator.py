#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pdb

import torch

from fairseq import checkpoint_utils, data, options, tasks, utils
from fairseq.data import encoders
from fairseq.models.rrgen_seq2seq_lstm import rrgen_lstm_arch

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='rrgen_translation')

input_args = [
    '/srv/scratch2/kew/fairseq_materials/translation_format/raw',
    '--path=/srv/scratch2/kew/fairseq_materials/translation_format/rrgen_lstm_emb_senti_cate_rate_100k/checkpoints/checkpoint_best.pt',
    '-s=src',
    '-t=tgt',
    '--task=rrgen_translation',
    '--dataset-impl=raw',
    '--nbest=5',
    '--sampling',
    '--sampling-topp=0.9',
    '--use-sentiment=senti',
    '--use-category=cate',
    '--use-rating=rate',
]

args = options.parse_args_and_arch(parser, input_args=input_args)

# args = options.parse_args_and_arch(parser)

# set device
use_cuda = torch.cuda.is_available() and not args.cpu

# Setup task
task = tasks.setup_task(args)

# Load model
print('| loading model from {}'.format(args.path))
models, _model_args = checkpoint_utils.load_model_ensemble(
    [args.path], task=task)
model = models[0]

if use_cuda:
    model.cuda()

# Load alignment dictionary for unknown word replacement
# (None if no unknown word replacement, empty if no path to align dictionary)
align_dict = utils.load_align_dict(args.replace_unk)

# initialise generator model
generator = task.build_generator(models, args)

# Handle tokenization and BPE
tokenizer = encoders.build_tokenizer(args)
bpe = encoders.build_bpe(args)


def decode_fn(x):
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


# assume no prefix tokens
prefix_tokens = None

# while True:

# sentence = input('\nReview: ')
sentence = 'pleasant stay ---SEP--- i thoroughly enjoyed my stay here . rooms were nice and clean . location was close to shopping area . customer service was excellent , especially bridget who always had a smile and greeted me warmly . would definitely stay here again .'
tokens = task.source_dictionary.encode_line(
    sentence, add_if_not_exist=False,
)

if args.use_sentiment:
    # ext_senti = input('\nSentiment: ')
    ext_senti = '4'
    try:
        ext_senti = task.ext_senti_dict[ext_senti]
    except:
        ext_senti = task.ext_senti_dict[int(ext_senti)]
if args.use_category:
    # ext_cate = input('\nCategory: ')
    ext_cate = 'Hotel'
    try:
        ext_cate = task.ext_cate_dict[ext_cate]
    except:
        ext_cate = task.ext_cate_dict[int(ext_cate)]
if args.use_rating:
    # ext_rate = input('\nRating: ')
    ext_rate = '5'
    try:
        ext_rate = task.ext_rate_dict[ext_rate]
    except:
        ext_rate = task.ext_rate_dict[int(ext_rate)]

# Build mini-batch to feed to the model
# batch = data.language_pair_dataset.collate(
#     samples=[{'id': -1, 'source': tokens}],  # bsz = 1
#     pad_idx=task.source_dictionary.pad(),
#     eos_idx=task.source_dictionary.eos(),
#     left_pad_source=False,
#     input_feeding=False,
# )

batch = data.rrgen_dataset.collate(
    samples=[{
        'id': 0,
        'source': tokens,
        'ext_senti': ext_senti,
        'ext_rate': ext_rate,
        'ext_cate': ext_cate
    }],
    pad_idx=task.source_dictionary.pad(),
    eos_idx=task.source_dictionary.eos(),
    left_pad_source=False,
    input_feeding=False)

# ensure correct dtype (int64)
batch['net_input']['src_tokens'] = batch['net_input']['src_tokens'].type(
    torch.LongTensor)

batch = utils.move_to_cuda(batch) if use_cuda else batch

# Remove padding --> not necessary since left_pad_source=False
# src_tokens = utils.strip_pad(batch['net_input']['src_tokens'][0, :], task.target_dictionary.pad())

# target_tokens = None
hypos = task.inference_step(generator, model, batch, prefix_tokens)

# pdb.set_trace()

# Feed batch to the model and get predictions
# preds = model(**batch['net_input'])

# (Pdb) preds = model(**batch['net_input'])
# *** TypeError: forward() missing 1 required positional argument: 'prev_output_tokens'

# Print top 3 predictions and their log-probabilities
# top_scores, top_labels = preds[0].topk(k=3)
# for score, label_idx in zip(top_scores, top_labels):
#     label_name = task.target_dictionary.string([label_idx])
#     print('({:.2f})\t{}'.format(score, label_name))

# Process top predictions
for j, hypo in enumerate(hypos[0][:args.nbest]):
    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=hypo['tokens'].int().cpu(),
        src_str=tokens,
        alignment=hypo['alignment'],
        align_dict=align_dict,
        tgt_dict=task.target_dictionary,
        remove_bpe=args.remove_bpe,
        # extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
    )
    detok_hypo_str = decode_fn(hypo_str)
    print(detok_hypo_str)
    print()
