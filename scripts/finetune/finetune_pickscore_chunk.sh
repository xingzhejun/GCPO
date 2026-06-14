export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export PYTHONPATH="./fastvideo:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port 19112 \
    ./fastvideo/train_grpo_flux_pickscore.py \
    --seed 42 \
    --pretrained_model_name_or_path ./data/flux \
    --vae_model_path ./data/flux \
    --data_json_path ./data/rl_embeddings/videos2caption.json \
    --train_batch_size 2 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 12 \
    --max_train_steps 151 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 30 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir ./data/outputs_pickscore/chunk/ \
    --h 720 \
    --w 720 \
    --sampling_steps 17 \
    --eta 0.7 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 0.01 \
    --weight_decay 0.0001 \
    --use_hpsv3 \
    --num_generations 12 \
    --shift 3 \
    --use_group \
    --timestep_fraction 0.5 \
    --gradient_checkpointing \
    --init_same_noise \
    --clip_range 5e-5 \
    --adv_clip_max 5.0 \
    --name pickscore_chunk \
    --right_clip_range 5e-5 \
    --grpo_step_mode flow \
    --use_chunk \
    --new_fix_chunk \
    --new_chunk_list '[2, 3, 4, 7]' \
    # --sample_weight \
    # --sample_weight_method normalized \
    # --load_from_before \
    # --load_path \
    # --fixed_chunk \
    # --chunk_idx '[2]' \
    # --use_global_std \
    # --fixed_chunk \
    # --chunk_idx '[0,1,2]' \
    # --chunk_size 8 \
    # --debug \
    # --use_reweight \
    # --kl_coeff 0 \
    # --use_half_half_adv \
    # --only_reward \
    
# idx 0 refers to high noise, while sampling_steps 0 is low noise