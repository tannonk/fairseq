#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

GPU="3"
LANG="de"
ROOT_DIR="/srv/scratch2/kew/fairseq_materials/rrgen_012021/$LANG"
TARGET_DIR="$ROOT_DIR/RE_TEST"
model_type="lstm_srcl_bpemb200_hd200"

# string array containing trained models
declare -a experiments=("lg3_sg6_up5" "lg3_sg7_up5" "SG0.7_SL3_LR1.8_UP5" "no_filter"
)

if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p $TARGET_DIR
fi

# set for current shell
export CUDA_VISIBLE_DEVICES=$GPU

# Read the array values with space
for exp in "${experiments[@]}"; do
    prep="$ROOT_DIR/$exp/prep"
    model_path="$ROOT_DIR/$exp/$model_type/checkpoints/checkpoint_best.pt"
    output_file="$TARGET_DIR/$exp.$model_type.greedy.txt"
    if [ ! -f "$output_file" ]; then

        echo "Running decoding for $prep with $model_path ..."

        {
        fairseq-generate \
        $prep \
        --path $model_path \
        -s review -t response \
        --task rrgen_translation --truncate-source --truncate-target \
        --dataset-impl raw \
        --user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
        --batch-size 256 \
        --remove-bpe sentencepiece \
        --use-sentiment alpha_sentiment \
        --use-category domain \
        --use-rating rating \
        --use-length review_length \
        --gen-subset FULLTEST 
        } 2>&1 > $output_file

    fi
done