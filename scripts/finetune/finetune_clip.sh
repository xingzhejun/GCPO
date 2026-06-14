export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export PYTHONPATH="./fastvideo:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port 19112 \
    ./fastvideo/train_grpo_flux_clip.py \
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
    --output_dir ./data/outputs_clip/GRPO/ \
    --h 720 \
    --w 720 \
    --sampling_steps 17 \
    --eta 0.7 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 0.01 \
    --weight_decay 0.0001 \
    --use_clip \
    --num_generations 12 \
    --shift 3 \
    --use_group \
    --timestep_fraction 0.5 \
    --gradient_checkpointing \
    --init_same_noise \
    --clip_range 5e-5 \
    --adv_clip_max 5.0 \
    --name clip_GRPO \
    --right_clip_range 5e-5 \
    --grpo_step_mode flow \
    --use_step \
    # --load_from_before \
    # --load_path \
    # --use_global_std \
    # --use_sto_step \
    # --debug \
    # --use_reweight \
    # --kl_coeff 0 \
    # --use_half_half_adv \
    # --fixed_step \
    # --step_idx '[0,1,2,3]' \
    # --only_reward \