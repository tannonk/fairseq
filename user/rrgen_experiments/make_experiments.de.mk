EXP_DIR := /srv/scratch2/kew/fairseq_materials/rrgen/de
EMBEDDINGS := /srv/scratch2/kew/fasttext/FT_de.w2v.txt
GPU := 5


# -------------
# PREPROCESSING
# -------------

# NB. everywhere, rg = 'remove greetings'

binarize_data_remove_greetings:
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

# ---------
# TRAINING: baseline LSTM - no A-components
# ---------

train_ft100_rg: $(EXP_DIR)/ft100_rg
	CUDA_VISIBLE_DEVICES=3 \
	nohup \
	fairseq-train \
	$(EXP_DIR)/data_bin_rg/ \
	--arch rrgen_lstm_arch \
	--task rrgen_translation \
	--dataset-impl mmap \
	--max-epoch 52 \
	--max-tokens 1000 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--lr 0.001 \
	--encoder-embed-path $(EMBEDDINGS) \
	--share-all-embeddings \
	--encoder-embed-dim 100 \
	--decoder-embed-dim 100 \
	--decoder-out-embed-dim 100 \
	--encoder-hidden-size 200 \
	--decoder-hidden-size 200 \
	--skip-invalid-size-inputs-valid-test \
	--save-dir $</checkpoints > $</logs/train_cont_e52.log &

# ---------
# DECODING: baseline LSTM - no A-components
# ---------


decode_ft100_rg_greedy: $(EXP_DIR)/ft100_rg
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg/ \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 8 \
	--data-buffer-size 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--num-workers 4 > $</nbest5.txt &


decode_ft100_rg_topk: $(EXP_DIR)/ft100_rg
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 8 \
	--data-buffer-size 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--sampling \
	--sampling-topk 10 \
	--num-workers 4 > $</nbest5_topk10.txt &


decode_ft100_rg_topp: $(EXP_DIR)/ft100_rg
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg/ \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--sampling \
	--sampling-topp 0.9 \
	--num-workers 4 > $</nbest5_topp90.txt &



# ---------
# TRAINING: A-components without greetings
# ---------

train_ft100src_rg: $(EXP_DIR)/ft100src_rg/
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-train \
	$(EXP_DIR)/data_bin_rg/ \
	--arch rrgen_lstm_arch \
	--task rrgen_translation \
	--dataset-impl mmap \
	--max-epoch 10 \
	--max-tokens 1000 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--lr 0.001 \
	--encoder-embed-path $(EMBEDDINGS) \
	--share-all-embeddings \
	--encoder-embed-dim 100 \
	--decoder-embed-dim 100 \
	--decoder-out-embed-dim 100 \
	--encoder-hidden-size 200 \
	--decoder-hidden-size 200 \
	--skip-invalid-size-inputs-valid-test \
	--save-dir $</checkpoints \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $</train.log &

# --------
# DECODING - greedy
# --------

decode_ft100src_rg_greedy: $(EXP_DIR)/ft100src_rg/
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg/ \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 4 \
	--data-buffer-size 2 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $</nbest5.txt &

# --------
# DECODING - top-p sampling
# NB. speed up possible with increasing, e.g. --num-workers 4
# --------

decode_ft100src_rg_topp: $(EXP_DIR)/ft100src_rg/
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg/ \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 8 \
	--data-buffer-size 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--sampling \
	--sampling-topp 0.9 \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $</nbest5_topp95.txt &

# --------
# DECODING - top-k sampling
# --------

decode_ft100src_rg_topk: $(EXP_DIR)/ft100src_rg/
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg/ \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 8 \
	--data-buffer-size 4 \
	--num-workers 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--sampling \
	--sampling-topk 10 \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $</nbest5_topk10.txt &


# --------
# DECODING - top-p sampling # TODO
# --------

decode_ft100src_rg_topp: $(EXP_DIR)/ft100src_rg/
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg/ \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 8 \
	--data-buffer-size 4 \
	--num-workers 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--sampling \
	--sampling-topp 0.9 \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $</nbest5_topk10.txt &



# -------------
# PREPROCESSING with greetings in tact
# -------------

binarize_data_with_greetings:
	fairseq-preprocess \
	--source-lang review \
	--target-lang response_rg \
	--trainpref $(RAW_DATA_DIR)/train \
	--validpref $(RAW_DATA_DIR)/valid \
	--testpref $(RAW_DATA_DIR)/test \
	--dataset-impl mmap \
	--task rrgen_translation \
	--joined-dictionary \
	--destdir $(EXP_DIR)/data_bin

# ---------
# TRAINING: A-components with greetings
# ---------

train_ft100src: $(EXP_DIR)/ft100src/
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-train \
	$(EXP_DIR)/data_bin \
	--arch rrgen_lstm_arch \
	--task rrgen_translation \
	--dataset-impl mmap \
	--max-epoch 10 \
	--max-tokens 1000 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--lr 0.001 \
	--encoder-embed-path $(EMBEDDINGS) \
	--share-all-embeddings \
	--encoder-embed-dim 100 \
	--decoder-embed-dim 100 \
	--decoder-out-embed-dim 100 \
	--encoder-hidden-size 200 \
	--decoder-hidden-size 200 \
	--skip-invalid-size-inputs-valid-test \
	--save-dir $</checkpoints \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $</train.log &

# --------
# DECODING - greedy
# --------

decode_ft100src_greedy: $(EXP_DIR)/ft100src/
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin/ \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 8 \
	--data-buffer-size 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $</nbest5.txt &

# --------
# DECODING - top-p sampling #TODO
# --------


decode_ft100src_topp: $(EXP_DIR)/ft100src/
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin/ \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 8 \
	--data-buffer-size 4 \
	--num-workers 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--sampling \
	--sampling-topp 0.9 \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $</nbest5_topp95.txt &

# --------
# DECODING - top-k sampling #TODO
# --------

decode_ft100src_rg_topk: $(EXP_DIR)/ft100src/
	CUDA_VISIBLE_DEVICES=$(GPU) \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin/ \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task rrgen_translation \
	--dataset-impl mmap \
	--batch-size 8 \
	--data-buffer-size 4 \
	--num-workers 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 \
	--sampling \
	--sampling-topk 10 \
	--use-sentiment sentiment \
	--use-category domain \
	--use-rating rating > $</nbest5_topk10.txt &



##############################
# Alternative architectures
##############################

train_fconv_encoder_decoder: /srv/scratch2/kew/fairseq_materials/rrgen/de/fconv
	CUDA_VISIBLE_DEVICES=5 \
	nohup \
	fairseq-train \
	$(EXP_DIR)/data_bin_rg \
	--arch fconv \
	--task translation \
	--dropout 0.2 \
	--optimizer adam \
	--lr 0.001 \
	--clip-norm 0.1 \
	--dataset-impl mmap \
	--max-epoch 10 \
	--max-tokens 1000 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--encoder-embed-path $(EMBEDDINGS) \
	--share-input-output-embed \
	--encoder-embed-dim 100 \
	--decoder-embed-dim 100 \
	--decoder-out-embed-dim 100 \
	--encoder-layers "[(200, 3), (200, 3), (200, 3), (200, 3), (200, 3), (200, 3), (200, 3), (200, 3), (200, 3), (200, 3), (200, 3), (200, 3), (200, 3)]" \
	--decoder-layers "[(200, 5), (200, 5), (200, 5), (200, 5), (200, 5)]" \
	--decoder-attention "[True, True, True, True, True]" \
	--skip-invalid-size-inputs-valid-test \
	--save-dir $</checkpoints > $</train.log &

# --nenclayer 4
# --nlayer 3
# --timeavg
# --bptt 0

decode_fconv_greedy: /srv/scratch2/kew/fairseq_materials/rrgen/de/fconv/
	CUDA_VISIBLE_DEVICES=3,4 \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg \
	--dataset-impl mmap \
	--path $</checkpoints/checkpoint5.pt \
	-s review \
	-t response_rg \
	--task translation \
	--batch-size 16 \
	--data-buffer-size 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 > $</nbest5.txt &



train_transformer: /srv/scratch2/kew/fairseq_materials/rrgen/de/transformer
	CUDA_VISIBLE_DEVICES=6 \
	nohup \
	fairseq-train \
	$(EXP_DIR)/data_bin_rg \
	--arch transformer \
	--task translation \
	--dataset-impl mmap \
	--activation-fn relu \
	--lr 0.001 \
	--dropout 0.3 \
	--attention-dropout 0.3 \
	--activation-dropout 0.3 \
	--encoder-embed-path $(EMBEDDINGS) \
	--encoder-embed-dim 100 \
	--encoder-ffn-embed-dim 60 \
	--encoder-layers 6 \
	--encoder-attention-heads 4 \
	--decoder-embed-path $(EMBEDDINGS) \
	--decoder-embed-dim 100 \
	--decoder-ffn-embed-dim 60 \
	--decoder-layers 6 \
	--decoder-attention-heads 4 \
	--decoder-output-dim 100 \
	--share-all-embeddings \
	--max-epoch 10 \
	--max-tokens 1000 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--save-dir $</checkpoints > $</train.log &    
	
# --encoder-learned-pos
# --decoder-learned-pos
# --decoder-normalize-before
# --share-decoder-input-output-embed	
# --share-all-embeddings
# --no-token-positional-embeddings
# [--adaptive-softmax-cutoff EXPR] [--adaptive-softmax-dropout D]
# [--layernorm-embedding] [--no-scale-embedding] [--no-cross-attention]
# [--cross-self-attention] [--encoder-layerdrop D]
# [--decoder-layerdrop D]
# [--encoder-layers-to-keep ENCODER_LAYERS_TO_KEEP]
# [--decoder-layers-to-keep DECODER_LAYERS_TO_KEEP] [--quant-noise-pq D]
# [--quant-noise-pq-block-size D] [--quant-noise-scalar D]


decode_transformer_greedy: /srv/scratch2/kew/fairseq_materials/rrgen/de/transformer
	CUDA_VISIBLE_DEVICES=6 \
	nohup \
	fairseq-generate \
	$(EXP_DIR)/data_bin_rg \
	--dataset-impl mmap \
	--path $</checkpoints/checkpoint_best.pt \
	-s review \
	-t response_rg \
	--task translation \
	--batch-size 16 \
	--data-buffer-size 4 \
	--max-source-positions 400 \
	--max-target-positions 400 \
	--skip-invalid-size-inputs-valid-test \
	--nbest 5 \
	--beam 5 > $</nbest5.txt &
