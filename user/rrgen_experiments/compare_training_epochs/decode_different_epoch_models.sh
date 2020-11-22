#!/usr/bin/env bash

set -e

model_dir="/srv/scratch2/kew/fairseq_materials/rrgen/de/ft100_rg"
out_dir="$model_dir/inference"


# for i in 3 6 9 12 15 18; do
# for (( i=3; i<=52; i+=3 )); do
# -w ensure sero-padding
for i in `seq -w 3 3 52`; do

    # check that decode output file does NOT already exist and model
    # checkpint DOES exist 
    # echo $i
    if [[ ! -f "$out_dir/decode_epoch_$i.txt" ]] && [[ -f "$model_dir/checkpoints/checkpoint$i.pt" ]]
    then
        echo "--------------- BEGIN DECODE EPOCH $i ---------------"
        CUDA_VISIBLE_DEVICES=3 \
        fairseq-generate \
        /srv/scratch2/kew/fairseq_materials/rrgen/de/data_bin_rg/ \
        --path "$model_dir/checkpoints/checkpoint$i.pt" \
        -s review \
        -t response_rg \
        --task rrgen_translation \
        --dataset-impl mmap \
        --batch-size 8 \
        --data-buffer-size 4 \
        --max-source-positions 400 \
        --max-target-positions 400 \
        --skip-invalid-size-inputs-valid-test \
        --nbest 1 \
        --beam 5 \
        --num-workers 4 > "$out_dir/decode_epoch_$i.txt"
        echo "--------------- FINISH DECODE EPOCH $i ---------------"
    else
        echo "[!] Either $out_dir/decode_epoch_$i.txt already exists or no model checkpoint available. Skipping..."
    fi
done