#!/bin/bash

export PYTHONPATH="./fastvideo:$PYTHONPATH"

ckpt_dir=''
mix_sampling_steps=30
output_dir=''

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 \
    ./test/sample_test.py \
    --model_path "${ckpt_dir}/diffusion_pytorch_model.safetensors" \
    --output_dir "${output_dir}_mix${mix_sampling_steps}" \
    --mix_sampling_steps $mix_sampling_steps \