GPU_NUM=1 # 2,4,8
MODEL_PATH="data/flux"
OUTPUT_DIR="data/rl_embeddings"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$GPU_NUM --master_port 100 \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/prompts.txt"
