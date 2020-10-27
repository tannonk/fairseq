EXP_DIR := /srv/scratch2/kew/fairseq_materials/rrgen/en/
EMBEDDINGS := /srv/scratch2/kew/fasttext/FT_en.w2v.txt
# RAW_DATA_DIR := /srv/scratch2/kew/fairseq_materials/rrgen/en/data_raw/
# BIN_DATA_DIR := /srv/scratch2/kew/fairseq_materials/rrgen/en/data_bin_rg/
GPU := 5


# -------------
# PREPROCESSING without grettings
# -------------

binarize_data:
	fairseq-preprocess \
	--source-lang review \
	--target-lang response_rg \
	--trainpref $(EXP_DIR)/data_raw/train \
	--validpref $(EXP_DIR)/data_raw/valid \
	--testpref $(EXP_DIR)/data_raw/test \
	--dataset-impl mmap \
	--task rrgen_translation \
	--joined-dictionary \
	--destdir $(EXP_DIR)/data_bin_rg/


# --------
# TRAINING # NB. executed on s3it volta
# --------

train_ft100src_rg: ~/scratch/fairseq_materialss/rrgen/en/data_bin_rg
	fairseq-train ~/scratch/fairseq_materials/rrgen/en/data_bin_rg/ \
	--arch rrgen_lstm_arch \
	--task rrgen_translation \
	--dataset-impl mmap \
	--max-epoch 10 \
	--max-tokens 4000 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--lr 0.001 \
	--encoder-embed-path ~/scratch/embeddings/FT_en.w2v.txt \
	--share-all-embeddings \
	--encoder-embed-dim 100 \
	--decoder-embed-dim 100 \
	--decoder-out-embed-dim 100 \
	--encoder-hidden-size 200 \
	--decoder-hidden-size 200 \
	--save-dir ~/scratch/fairseq_materials/rrgen/en/ft100src_rg \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating \
	--skip-invalid-size-inputs-valid-test

# --------
# DECODING
# --------

decode_ft100src_rg_greedy: $(EXP_DIR)/ft100src_rg
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg/ \
	--path $@/checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 16 \
	--data-buffer-size 4 \
	--num-workers 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--model-overrides "{'encoder_embed_path': '/srv/scratch2/kew/fasttext/FT_en.w2v.txt', 'decoder_embed_path': '/srv/scratch2/kew/fasttext/FT_en.w2v.txt'}" \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $@/nbest5.txt &

decode_ft100src_rg_topk: $(EXP_DIR)/ft100src_rg
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg/ \
	--path $@/checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 16 \
	--data-buffer-size 4 \
	--num-workers 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--model-overrides "{'encoder_embed_path': '/srv/scratch2/kew/fasttext/FT_en.w2v.txt', 'decoder_embed_path': '/srv/scratch2/kew/fasttext/FT_en.w2v.txt'}" \
	--sampling \
	--sampling-topk 10 \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $@/nbest5_topk10.txt &



decode_ft100src_rg_topp: $(EXP_DIR)/ft100src_rg
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg/ \
	--path $@/checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 16 \
	--data-buffer-size 4 \
	--num-workers 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--model-overrides "{'encoder_embed_path': '/srv/scratch2/kew/fasttext/FT_en.w2v.txt', 'decoder_embed_path': '/srv/scratch2/kew/fasttext/FT_en.w2v.txt'}" \
	--sampling \
	--sampling-topp 0.9 \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $@/nbest5_topp90.txt &

