#!/usr/bin/env bash

set -e

exp_dir="/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/SG0.7_SL3_LR1.8_UP3"
model_type="lstm_srcl_bpemb200_hd200"
out_dir="/srv/scratch2/kew/fairseq_materials/rrgen_012021/de/RE_TEST/decoding/greedy/inspect_epoch_improvement"

for i in 4 8 12 16 20 24 28; do
    # check that decode output file does NOT already exist and model
    # checkpint DOES exist 
    # echo $i
    model="$exp_dir/$model_type/checkpoints/checkpoint$i.pt" 

    if [[ -f $model ]]
    then
        echo "--------------- BEGIN DECODE EPOCH $i ---------------"
        CUDA_VISIBLE_DEVICES=3 \
        fairseq-generate \
        "$exp_dir/prep" \
        --path "$model" \
        -s review -t response \
        --task rrgen_translation --truncate-source --truncate-target \
        --dataset-impl raw \
        --user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
        --batch-size 32 \
        --remove-bpe sentencepiece \
        --nbest 1 --beam 5 \
        --use-sentiment alpha_sentiment \
        --use-category domain \
        --use-rating rating \
        --use-length review_length \
        --gen-subset '../../RE_TEST/bpe/re_test' > "$out_dir/decode_epoch_$i.txt"
        echo "--------------- FINISH DECODE EPOCH $i ---------------"
    else
        echo "[!] Can not locate model checkpoint at $model"
    fi
done