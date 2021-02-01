#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Since decoding can be done on a single gpu qith only a
# small amount of memory consumption, it's best to copy
# these commands and execute them one after the other.

# CUDA_VISIBLE_DEVICES=0 \
# nohup \
# fairseq-generate \
# /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg6_up5/prep \
# --path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg6_up5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
# -s review -t response \
# --task rrgen_translation --truncate-source --truncate-target \
# --dataset-impl raw \
# --user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
# --batch-size 32 \
# --remove-bpe sentencepiece \
# --use-sentiment alpha_sentiment \
# --use-category domain \
# --use-rating rating \
# --use-length review_length \
# --gen-subset '../../FULL_TEST/bpe/test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/FULL_TEST/greedy/lg3_sg6_up5.lstm_srcl_bpemb200_hd200.txt &


# CUDA_VISIBLE_DEVICES=0 \
# nohup \
# fairseq-generate \
# /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg7_up5/prep \
# --path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg7_up5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
# -s review -t response \
# --task rrgen_translation --truncate-source --truncate-target \
# --dataset-impl raw \
# --user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
# --batch-size 32 \
# --remove-bpe sentencepiece \
# --use-sentiment alpha_sentiment \
# --use-category domain \
# --use-rating rating \
# --use-length review_length \
# --gen-subset '../../FULL_TEST/bpe/test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/FULL_TEST/greedy/lg3_sg7_up5.lstm_srcl_bpemb200_hd200.txt &


# CUDA_VISIBLE_DEVICES=0 \
# nohup \
# fairseq-generate \
# /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/prep \
# --path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
# -s review -t response \
# --task rrgen_translation --truncate-source --truncate-target \
# --dataset-impl raw \
# --user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
# --batch-size 32 \
# --remove-bpe sentencepiece \
# --use-sentiment alpha_sentiment \
# --use-category domain \
# --use-rating rating \
# --use-length review_length \
# --gen-subset '../../FULL_TEST/bpe/test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/FULL_TEST/greedy/SG0.7_SL3_LR1.8_UP5.lstm_srcl_bpemb200_hd200.txt &

# CUDA_VISIBLE_DEVICES=0 \
# nohup \
# fairseq-generate \
# /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/no_filter/prep \
# --path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/no_filter/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
# -s review -t response \
# --task rrgen_translation --truncate-source --truncate-target \
# --dataset-impl raw \
# --user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
# --batch-size 32 \
# --remove-bpe sentencepiece \
# --use-sentiment alpha_sentiment \
# --use-category domain \
# --use-rating rating \
# --use-length review_length \
# --gen-subset '../../FULL_TEST/bpe/test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/FULL_TEST/greedy/no_filter.lstm_srcl_bpemb200_hd200.txt &


# ------------------------

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg5_up5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg5_up5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/greedy/lg3_sg5_up5.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg6_up5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg6_up5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/greedy/lg3_sg6_up5.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg7_up5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg7_up5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/greedy/lg3_sg7_up5.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/greedy/SG0.7_SL3_LR1.8_UP5.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP3/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP3/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint20.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/greedy/SG0.7_SL3_LR1.8_UP3.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/no_filter/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/no_filter/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/greedy/no_filter.lstm_srcl_bpemb200_hd200.txt &

# lstm-base
CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg5_up5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg5_up5/lstm_bpemb200_hd200/checkpoints/checkpoint20.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/greedy/lg3_sg5_up5.lstm_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/lstm_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/greedy/SG0.7_SL3_LR1.8_UP5.lstm_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP3/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP3/lstm_bpemb200_hd200/checkpoints/checkpoint20.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/greedy/SG0.7_SL3_LR1.8_UP3.lstm_bpemb200_hd200.txt &





# ------------------------ sampling topp

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg5_up5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg5_up5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topp 0.9 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topp90/lg3_sg5_up5.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg6_up5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg6_up5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topp 0.9 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topp90/lg3_sg6_up5.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg7_up5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg7_up5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topp 0.9 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topp90/lg3_sg7_up5.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topp 0.9 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topp90/SG0.7_SL3_LR1.8_UP5.lstm_srcl_bpemb200_hd200.txt &


CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP3/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP3/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topp 0.9 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topp90/SG0.7_SL3_LR1.8_UP3.lstm_srcl_bpemb200_hd200.txt &


CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/no_filter/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/no_filter/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topp 0.9 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topp90/no_filter.lstm_srcl_bpemb200_hd200.txt &


# lstm-base
CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg5_up5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg5_up5/lstm_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--beam 10 --nbest 10 \
--sampling --sampling-topp 0.9 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topp90/lg3_sg5_up5.lstm_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/lstm_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--beam 10 --nbest 10 \
--sampling --sampling-topp 0.9 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topp90/SG0.7_SL3_LR1.8_UP5.lstm_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP3/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP3/lstm_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--beam 10 --nbest 10 \
--sampling --sampling-topp 0.9 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topp90/SG0.7_SL3_LR1.8_UP3.lstm_bpemb200_hd200.txt &


# -------------------------


CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg6_up5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg6_up5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topk 10 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topk10/lg3_sg6_up5.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg7_up5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/lg3_sg7_up5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topk 10 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topk10/lg3_sg7_up5.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP5/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topk 10 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topk10/SG0.7_SL3_LR1.8_UP5.lstm_srcl_bpemb200_hd200.txt &

CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/no_filter/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/no_filter/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topk 10 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topk10/no_filter.lstm_srcl_bpemb200_hd200.txt &


CUDA_VISIBLE_DEVICES=0 \
nohup \
fairseq-generate \
/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/no_filter/prep \
--path /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/no_filter/lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
-s review -t response \
--task rrgen_translation --truncate-source --truncate-target \
--dataset-impl raw \
--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
--batch-size 32 \
--remove-bpe sentencepiece \
--use-sentiment alpha_sentiment \
--use-category domain \
--use-rating rating \
--use-length review_length \
--beam 10 --nbest 10 \
--sampling --sampling-topk 10 \
--gen-subset '../../RE_TEST/bpe/re_test' >| /srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/topk10/no_filter.lstm_srcl_bpemb200_hd200.txt &

# -------------------------
