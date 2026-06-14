# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.


import argparse
import json
import math
import os
from pathlib import Path
from typing import Union
from fastvideo.utils.parallel_states import destroy_sequence_parallel_group, get_sequence_parallel_state
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
import time
from torch.utils.data import DataLoader
import torch
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm.auto import tqdm
from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from fastvideo.utils.checkpoint import save_checkpoint_optimizer
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor
from collections import deque
import numpy as np
from torch.nn import functional as F
from typing import List
from PIL import Image
from diffusers import FluxTransformer2DModel, AutoencoderKL
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel, CLIPConfig
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# from diffusers.utils import check_min_version
# check_min_version("0.31.0")

from fastvideo.utils.logging_ import step_save, step_log, step_total_save
from fastvideo.utils.ema_utils import FSDP_EMA, save_ema_checkpoint
from fastvideo.utils.common_utils import (
    print_current_gpu_memory, assert_eq, init_everything, set_fsdp, 
    set_optimizer_and_lr_scheduler, repeat_tensor, sd3_time_shift, gather_tensor, 
    step_over, load_from_checkpoint, generate_perm, wandb_init, copy_dict
    )
from fastvideo.utils.reward_utils import initialize_hps_model, initialize_pic_model, calc_probs
from fastvideo.sample.flux_sample_utils import unpack_latents, pack_latents, prepare_latent_image_ids
from fastvideo.utils.chunk import set_chunk, get_chunk_list
from fastvideo.utils.loss_utils import loss_process
from safetensors.torch import load_file
from hpsv3 import HPSv3RewardInferencer
from fastvideo.utils.checkpoint import load_optim, resume_training
import random
from sklearn.cluster import KMeans
from typing import Optional, Union, List
from diffusers.utils.torch_utils import randn_tensor
import matplotlib.pyplot as plt
from open_clip import create_model_from_pretrained, get_tokenizer

def flow_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    grpo: bool = True,
    sde_solver: bool = True,
):
    device = model_output.device
    sigma = sigmas[index].to(device)
    sigma_prev = sigmas[index + 1].to(device)
    sigma_max = sigmas[1].item()
    dt = sigma_prev - sigma # neg dt

    pred_original_sample = latents - sigma * model_output

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta
    
    prev_sample_mean = latents*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape, 
            generator=generator, 
            device=device, 
            dtype=model_output.dtype
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise
    
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, pred_original_sample, log_prob

def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int, #i
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    
    prev_sample_mean = latents + dsigma * model_output
    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver: 
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None: 
        if sde_solver:
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        else:
            prev_sample = prev_sample_mean
        
    if grpo: 
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = (-((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))) - math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        image_ids, 
        grpo_sample,
    ):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:  
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            transformer.eval()
            with torch.autocast("cuda", torch.bfloat16):
                pred= transformer(
                    hidden_states=z,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps/1000,
                    guidance=torch.tensor([3.5], device=z.device, dtype=torch.bfloat16),
                    txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            if args.use_sto_step: # we didn't implement flow-sample in this case
                if i in args.step_idx: 
                    z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
                else:
                    z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=False)
            else:
                if i <= args.step_end_idx:
                    if args.grpo_step_mode == 'flow':
                        z, pred_original, log_prob = flow_grpo_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
                    else:
                        z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
                else:
                    z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=False)
            z.to(torch.bfloat16) # z is the next state, and pred_original is x0 
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        return z, latents, all_latents, all_log_probs
    else:
        raise ValueError("Invalid grpo_sample value. Must be True.")

def sample_reference_model(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
    step,
):
    w, h = args.w, args.h 
    sample_steps = args.sampling_steps 
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
    assert_eq(len(sigma_schedule), sample_steps + 1, "sigma_schedule must have length sample_steps + 1",)
    dir = os.path.join(args.output_dir, f"images/step_{step}/")
    main_print(f'save dir is: {dir}')
    os.makedirs(dir, exist_ok=True)
    rank = int(os.environ["RANK"])
    image_processor = VaeImageProcessor(16)
    vae.enable_tiling()

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE
    # the batch_size here is fixed to 1
    if args.use_chunk:
        batch_size = 1
        assert B % batch_size == 0
    else:
        batch_size = 1
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)
    if args.debug:
        main_print(f'B is: {B}')
        main_print(f'batch_indices is: {batch_indices}')
    all_latents = []
    all_log_probs = []
    all_rewards = []  
    all_image_ids = []
    caption_text = None
    if args.init_same_noise: # True
        input_latents = torch.randn(
                (batch_size, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )

    for index, batch_idx in enumerate(batch_indices):
        txt_filename = os.path.join(dir, f"{rank}_{index}.txt")
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_text_ids = text_ids[[index]]
        batch_caption = [caption[i] for i in batch_idx]
        if args.debug:
            main_print(f'batch_encoder_hidden_states shape is: {batch_encoder_hidden_states.shape}')
            main_print(f'text_ids shape is: {text_ids.shape}')
            main_print(f'text_ids is: {text_ids}') # all zero
            main_print(f'batch_text_ids shape is: {batch_text_ids.shape}')
            main_print(f'batch_caption is: {batch_caption}')
        if not args.init_same_noise: # False
            input_latents = torch.randn(
                    (len(batch_idx), IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                )
        input_latents_new = pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w) # 2*2 downsample
        image_ids = prepare_latent_image_ids(latent_h // 2, latent_w // 2, device, torch.bfloat16) # for position embedding
        if args.debug:
            main_print(f'image_ids shape is: {image_ids.shape}') 
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        
        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents_new,
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_pooled_prompt_embeds,
                batch_text_ids,
                image_ids,
                grpo_sample,
            )

        all_image_ids.append(image_ids)
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        # main_print(f'latents shape is: {latents.shape}') 
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(latents, h, w, 8)
                latents = (latents / 0.3611) + 0.1159
                image = vae.decode(latents, return_dict=False)[0]
                decoded_image = image_processor.postprocess(image)
        if args.debug:
            main_print(f'decoded_image length is: {len(decoded_image)}')
        
        with torch.no_grad():
            for k in range(batch_size):
                image_path = decoded_image[k]
                # image = preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                # text = tokenizer([batch_caption[k]]).to(device=device, non_blocking=True)

                if args.use_hpsv2:
                    image = preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                    text = tokenizer([batch_caption[k]]).to(device=device, non_blocking=True)
                    with torch.amp.autocast('cuda'):
                        outputs = reward_model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T
                        hps_score = torch.diagonal(logits_per_image)
                    all_rewards.append(hps_score)

                    hps_for_filename = hps_score[0].item()
                    if batch_caption[k] != caption_text:
                        caption_text = batch_caption[k]
                        with open(txt_filename, 'a', encoding='utf-8') as f:
                            f.write(caption_text + '\n')
        
                elif args.use_pickscore: 
                    score = calc_probs(tokenizer, reward_model, caption, image_path, device)
                    all_rewards.append(score)
                    pic_for_filename = score[0].item()

                elif args.use_clip:
                    # for apple
                    image_path = image_path.convert('RGB')
                    image_path = preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                    text = tokenizer([batch_caption[k]], context_length=reward_model.context_length).to(device=device, non_blocking=True)
                    with torch.amp.autocast('cuda'):
                        ## for laion
                        # clip_inputs = preprocess_val(text = [batch_caption[k]], images=[image_path], return_tensors="pt", padding=True, max_length=77)
                        # clip_inputs = clip_inputs.to(device)
                        # clip_outputs = reward_model(**clip_inputs)
                        # logits_per_image = clip_outputs.logits_per_image # this is the image-text similarity score
                        # logits = logits_per_image.squeeze(-1).float()
                        # hps_score = torch.tensor([logits]).to(device)
                        image_features = reward_model.encode_image(image_path)
                        text_features = reward_model.encode_text(text)
                        image_features = F.normalize(image_features, dim = -1)
                        text_features = F.normalize(text_features, dim = -1)
                        hps_score = image_features @ text_features.T 
                        hps_score = hps_score.squeeze(0)
                    all_rewards.append(hps_score)

                    hps_for_filename = hps_score.item()
                    if batch_caption[k] != caption_text:
                        caption_text = batch_caption[k]
                        with open(txt_filename, 'a', encoding='utf-8') as f:
                            f.write(caption_text + '\n')
                
                elif args.use_hpsv3:
                    with torch.amp.autocast('cuda'):
                        rewards = reward_model.reward([image_path], [batch_caption[k]])
                        hps_score = rewards[0][:1]
                    all_rewards.append(hps_score)

                    hps_for_filename = hps_score.item()
                    if batch_caption[k] != caption_text:
                        caption_text = batch_caption[k]
                        with open(txt_filename, 'a', encoding='utf-8') as f:
                            f.write(caption_text + '\n')

                if args.use_hpsv2 and args.use_pickscore:
                    pic_savepath = os.path.join(dir, f"{rank}_{index}_hps_{hps_for_filename:.4f}_pic_{pic_for_filename:.4f}.png")
                    decoded_image[0].save(os.path.join(dir, f"{rank}_{index}_hps_{hps_for_filename:.4f}_pic_{pic_for_filename:.4f}.png"))
                elif args.use_hpsv2 or args.use_hpsv3 or args.use_clip: # True
                    pic_savepath = os.path.join(dir, f"{rank}_{index}_{k}_hps_{hps_for_filename:.4f}.png")
                    decoded_image[k].save(pic_savepath)
                elif args.use_pickscore:
                    pic_savepath = os.path.join(dir, f"{rank}_{index}_pic_{pic_for_filename:.4f}.png")
                    decoded_image[0].save(os.path.join(dir, f"{rank}_{index}_pic_{pic_for_filename:.4f}.png"))
                else:
                    raise ValueError("Reward model is not defined")
                main_print(f'pic save to: {pic_savepath}')

    all_latents = torch.cat(all_latents, dim=0)
    if args.debug:
        main_print(f'all_log_probs is: {all_log_probs[0].shape}')
    all_log_probs = torch.cat(all_log_probs, dim=0)
    if args.debug:
        main_print(f'all_log_probs is: {all_log_probs.shape}')
    all_rewards = torch.cat(all_rewards, dim=0)
    all_image_ids = torch.stack(all_image_ids, dim=0)
    if all_image_ids.shape[0] != B:
        all_image_ids = all_image_ids.repeat(B // all_image_ids.shape[0], 1, 1)
    if args.debug:
        main_print(f'all_image_ids is: {all_image_ids.shape}') 
    
    return all_rewards, all_latents, all_log_probs, sigma_schedule, all_image_ids

def grpo_one_step(
            args,
            latents,
            pre_latents,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            image_ids,
            transformer,
            timesteps,
            i,
            sigma_schedule,
):
    chunk_size = latents.shape[0]
    if chunk_size == 1:
        mul_try = False
    else:
        mul_try = True # means chunk
    transformer.train()
    if args.debug and dist.get_rank()==0:
        print('mul_try is:', mul_try)
        print('encoder_hidden_states shape is:', encoder_hidden_states.shape) 
        print('pooled_prompt_embeds shape is:', pooled_prompt_embeds.shape) 
        print('img_ids shape is:', image_ids.shape)
        print('txt_ids shape is:', text_ids.shape) # torch.Size([1, 3])
        print('text ids shape is:', text_ids.repeat(encoder_hidden_states.shape[1],1).shape)
    
    if mul_try:
        with torch.autocast("cuda", torch.bfloat16):
            pred= transformer(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states.expand(chunk_size,-1,-1),
                timestep=timesteps/1000,
                guidance=torch.tensor(
                    [3.5],
                    device=latents.device,
                    dtype=torch.bfloat16
                ),
                txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
                pooled_projections=pooled_prompt_embeds.expand(chunk_size,-1),
                img_ids=image_ids.squeeze(0),
                joint_attention_kwargs=None,
                return_dict=False,
            ) # is a tuple
    
    else:
        with torch.autocast("cuda", torch.bfloat16):
            pred= transformer(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps/1000,
                guidance=torch.tensor(
                    [3.5],
                    device=latents.device,
                    dtype=torch.bfloat16
                ),
                txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
                pooled_projections=pooled_prompt_embeds,
                img_ids=image_ids.squeeze(0),
                joint_attention_kwargs=None,
                return_dict=False,
            ) # is a tuple of length 1
    # if args.debug and dist.get_rank()==0:
    #     print('pred length is:', len(pred)) # 1
    pred = pred[0]
    if mul_try: # i to i[0]
        log_probs_list = []
        for j in range(chunk_size):
            if args.grpo_step_mode == 'flow':
                _, _, log_prob = flow_grpo_step(pred[j].unsqueeze(0), latents[j].unsqueeze(0).to(torch.float32), args.eta, sigma_schedule, i[j], prev_sample=pre_latents[j].unsqueeze(0).to(torch.float32), grpo=True, sde_solver=True)
            else:
                _, _, log_prob = flux_step(pred[j].unsqueeze(0), latents[j].unsqueeze(0).to(torch.float32), args.eta, sigma_schedule, i[j], prev_sample=pre_latents[j].unsqueeze(0).to(torch.float32), grpo=True, sde_solver=True)
            
            log_probs_list.append(log_prob)
        log_prob = torch.stack(log_probs_list, dim=0).mean(dim=0) 
        if args.debug:
            float_list = [float(x.detach().cpu()) for x in log_probs_list] 
            main_print(f'log_probs_list is: {float_list}')
    else:
        # if args.debug:
        #     main_print(f'i is: {i}')
        if isinstance(i, list):
            i = i[0]
        if args.grpo_step_mode == 'flow':
            z, pred_original, log_prob = flow_grpo_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
        else:
            z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob

def train_one_step(
    args,
    device,
    transformer,
    vae,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    max_grad_norm,
    preprocess_val,
    ema_handler,
    step,
):
    chunk_loss = 0.0
    step_loss = 0.0
    total_loss = 0.0
    policy_chunk_loss = 0.0
    policy_step_loss = 0.0
    total_policy_loss = 0.0
    kl_chunk_loss = 0.0
    kl_step_loss = 0.0
    total_kl_loss = 0.0
    encoder_hidden_states, pooled_prompt_embeds, text_ids, caption = next(loader)
    if args.debug:
        main_print(f'encoder_hidden_states shape is: {encoder_hidden_states.shape}')

    if args.use_group: # True
        encoder_hidden_states = repeat_tensor(args, encoder_hidden_states)
        pooled_prompt_embeds = repeat_tensor(args, pooled_prompt_embeds)
        text_ids = repeat_tensor(args, text_ids)
        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    reward, all_latents, all_log_probs, sigma_schedule, all_image_ids = sample_reference_model(
            args,
            device, 
            transformer,
            vae,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            reward_model,
            tokenizer,
            caption,
            preprocess_val,
            step,
        )
    batch_size = all_latents.shape[0]
    if args.debug:
        main_print(f'batch_size is: {batch_size}')
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "rewards": reward.to(torch.float32),
        "image_ids": all_image_ids,
        "text_ids": text_ids,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }
    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank()==0:
        print("gathered_reward", gathered_reward)
        with open(os.path.join(args.output_dir, 'reward.txt'), 'a') as f: 
            f.write(f"{gathered_reward.mean().item()}\n")

    if args.use_group: # True
        n = len(samples["rewards"]) // (args.num_generations)
        advantages = torch.zeros_like(samples["rewards"])
        group_std_list = []
        if args.use_global_std:
            group_std = samples["rewards"].std() + 1e-8
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            if not args.use_global_std:
                group_std = group_rewards.std() + 1e-8
            group_std_list.append(group_std)
            adv = (group_rewards - group_mean) / group_std
            if args.use_half_half_adv:
                middle = torch.quantile(adv, 0.5)
                adv = adv - middle
            advantages[start_idx:end_idx] = adv
        samples["advantages"] = advantages
        if args.use_global_std:
            gathered_group_std = group_std
        else:
            group_std = torch.stack(group_std_list)
            gathered_group_std = gather_tensor(group_std)
        if args.debug:
            main_print(f'gathered_group_std is: {gathered_group_std}')
            main_print(f'advantages is: {advantages}')
    else:
        advantages = (samples["rewards"] - gathered_reward.mean())/(gathered_reward.std()+1e-8)
        samples["advantages"] = advantages

    if args.only_reward:
        return None, None, None, None, None, None, gathered_reward.mean().item(), gathered_group_std.mean().item()

    if args.debug and dist.get_rank()==0:
        # the case of num_gen = 12, gen_steps = 16
        print ('adv shape is:', samples["advantages"].shape) # torch.Size([24])
        print('reward shape is:', samples["rewards"].shape ) # torch.Size([24])
        print ('adv is:', samples["advantages"])
        print('reward is:', samples["rewards"])
        print('timesteps is:', samples["timesteps"].shape) # torch.Size([24, 15])
        print('latents is:', samples['latents'].shape) # torch.Size([24, 15, 2025, 64])
        print('log is:', samples['log_probs'].shape) # torch.Size([24, 15])
        print('log is:', samples['log_probs'])
    
    samples_chunk = copy_dict(samples)
    total_loss_list = []
    chunk_loss_list = []
    step_loss_list = []
    policy_chunk_loss_list = []
    policy_step_loss_list = []
    total_policy_loss_list = []
    kl_chunk_loss_list = []
    kl_step_loss_list = []
    total_kl_loss_list = []
    ratio_chunk_list = []
    ratio_step_list = []
    if args.use_chunk: # chunk-GRPO
        num_chunks, last_chunk_size, chunk_sizes = set_chunk(samples_chunk, args)
        main_print(f'chunk_sizes is: {chunk_sizes}')

        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples_chunk[key] = torch.split(samples_chunk[key], chunk_sizes, dim=1)
        samples_chunk["chunk_log_probs"] = torch.stack([chunk.mean(dim=1) for chunk in samples_chunk["log_probs"]], dim=1) # (batch_size, num_chunks)
        
        if args.debug and dist.get_rank()==0:
            # the case of num_gen = 12, gen_steps = 16
            main_print(f'chunk_sizes is: {chunk_sizes}')
            print ('adv shape is:', samples_chunk["advantages"].shape) # torch.Size([24])
            print('rew shape is:', samples_chunk["rewards"].shape ) # torch.Size([24])
            print('timesteps is:', samples_chunk["timesteps"]) # tuple
            # print('latents is:', samples['latents'].shape) # tuple
            print('log is:', samples_chunk['log_probs']) # tuple 
            print('chunk log is:', samples_chunk["chunk_log_probs"].shape)  # torch.Size([24, chunk_num])
            print('chunk log is:', samples_chunk["chunk_log_probs"])

        if args.sample_weight:
            l1_changes = (samples['next_latents'] - samples['latents']).abs().mean(dim=(-2, -1))
            prev = samples['latents'].abs().mean(dim=(-2, -1))
            l1_changes = l1_changes.detach()
            prev = prev.detach()
            l1_changes = l1_changes / (prev + 1e-8)
            l1_changes_in_chunks = list(torch.split(l1_changes, chunk_sizes, dim=1))

            chunk_weights = torch.cat([
                chunk.mean(dim=1, keepdim=True) for chunk in l1_changes_in_chunks
            ], dim=1)
            weights = chunk_weights + 1e-8

            if args.sample_weight_method == 'normalized':
                weights_normalized = F.normalize(weights, p=1, dim=1)
            else: # softmax otherwise
                weights_normalized = torch.softmax(weights, dim=1)

            perms = torch.multinomial(weights_normalized, num_samples=num_chunks - args.remove, replacement=False).to(device)
            
            # we fixed the half-number here. Please modify it when you change the sampling timesteps.
            first_half_mean = weights_normalized[:12, :].mean(dim=0)
            second_half_mean = weights_normalized[12:, :].mean(dim=0)
            final_result = torch.stack([first_half_mean, second_half_mean], dim=0)
            final_result = final_result.cpu().detach().numpy()
            numpy_path = os.path.join(args.output_dir, f"images/step_{step}/numpy.txt")
            if dist.get_rank()==0:
                with open(numpy_path, 'a') as f:
                    if os.path.getsize(numpy_path) > 0:
                        f.write("\n")
                    np.savetxt(f, final_result, fmt='%.8f', delimiter=',')

        else:
            perms = torch.stack(
                [torch.randperm(num_chunks - args.remove) for _ in range(batch_size)]
            ).to(device)
            if args.fixed_chunk:
                perms = generate_perm(batch_size, num_chunks, fixed_values = args.chunk_idx).to(device)

        main_print(f'perms is: {perms}')
        train_chunks = int(num_chunks * args.timestep_fraction)
        if args.fixed_chunk:
            train_chunks = len(args.chunk_idx)
        if args.debug and dist.get_rank()==0:
            # print('perm shape is:', perms.shape)
            print('perm is:', perms) 
            print('train_chunks is:', train_chunks)

        # reform to samples_batched_list
        samples_batched_list = get_chunk_list(samples_chunk, batch_size)
        grad_norm = torch.tensor(0.0, device=device) 
        indices_for_i = torch.randperm(len(samples_batched_list)).tolist()  
        count = 0
        for i in indices_for_i:
            sample = samples_batched_list[i]
            main_print(f'current i is: {i}')
            for chunk_perm_idx in range(train_chunks):
                chunk_idx = perms[i][chunk_perm_idx].item()
                if args.debug:
                    main_print(f'chunk_idx is:{chunk_idx}')
                current_chunk_size = chunk_sizes[chunk_idx]
                
                start_idx = sum(chunk_sizes[:chunk_idx])
                original_indices = [start_idx + k for k in range(chunk_sizes[chunk_idx])]
                if args.use_reweight:
                    middle_value = len(original_indices) // 2
                    mid = original_indices[middle_value]
                    reweight_factor = math.sqrt((args.sampling_steps - mid) / mid)
                else:
                    reweight_factor = 1.0
                if args.try_mul:
                    if args.debug:
                        main_print(f'current_chunk_size is: {current_chunk_size}')
                        main_print(f'latents shape is: {sample["latents"][chunk_idx].shape}')
                        main_print(f'next_latents shape is: {sample["next_latents"][chunk_idx].shape}')
                        main_print(f'encoder_hidden_states shape is: {sample["encoder_hidden_states"].unsqueeze(0).shape}')
                        main_print(f'pooled_prompt_embeds shape is: {sample["pooled_prompt_embeds"].unsqueeze(0).shape}')
                        main_print(f'text_ids shape is: {sample["text_ids"].unsqueeze(0).shape}')
                        main_print(f'image_ids shape is: {sample["image_ids"].unsqueeze(0).shape}')
                        main_print(f'timesteps shape is: {sample["timesteps"][chunk_idx].shape}')
                        main_print(f'original_indices is: {original_indices}')
                    
                    log_probs_in_chunk = grpo_one_step(
                        args,
                        sample["latents"][chunk_idx],          # (chunk_size, 2025, 64)
                        sample["next_latents"][chunk_idx],     
                        sample["encoder_hidden_states"].unsqueeze(0), 
                        sample["pooled_prompt_embeds"].unsqueeze(0),  
                        sample["text_ids"].unsqueeze(0),            
                        sample["image_ids"].unsqueeze(0),             
                        transformer,
                        sample["timesteps"][chunk_idx],        # # torch.Size([chunk_size])
                        original_indices,       # a list，like [4, 5, 6, 7]
                        sigma_schedule,
                    )
                    new_chunk_log_prob = log_probs_in_chunk
                    if args.only_on_policy:
                        old_chunk_log_prob = new_chunk_log_prob.detach()
                    else:
                        old_chunk_log_prob = sample["chunk_log_probs"][chunk_idx]
                    # old_chunk_log_prob = new_chunk_log_prob.detach()
                    if args.debug:
                        main_print(f'new_chunk_log_prob is: {new_chunk_log_prob}')
                        main_print(f'old_chunk_log_prob is: {old_chunk_log_prob}')
                    
                if args.debug:
                    main_print(f'new_chunk_log_prob shape is {new_chunk_log_prob.shape}') # torch.Size([1])
                advantages = torch.clamp(sample["advantages"], -args.adv_clip_max, args.adv_clip_max)
                ratio = torch.exp(new_chunk_log_prob - old_chunk_log_prob)
                if args.debug:
                    main_print(f'ratio is: {float(ratio.detach().cpu())}')
                ratio_chunk_list.append(float(ratio.detach().cpu()))
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.right_clip_range)
                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                policy_loss = policy_loss * reweight_factor
                if args.debug:
                    main_print(f'policy_loss is: {float(policy_loss.detach().cpu())}')
                if args.kl_coeff > 0:
                    anti_diff = old_chunk_log_prob - new_chunk_log_prob
                    kl_loss = torch.exp(anti_diff) - 1.0 - anti_diff
                    kl_loss = torch.mean(kl_loss)
                    if args.debug:
                        main_print(f'kl_loss is: {float(kl_loss.detach().cpu())}')
                    loss = (policy_loss + args.kl_coeff * kl_loss) / (args.gradient_accumulation_steps * train_chunks)
                else:
                    loss = policy_loss / (args.gradient_accumulation_steps * train_chunks)
                loss.backward()
                policy_chunk_loss, total_policy_loss, policy_chunk_loss_list, total_policy_loss_list = loss_process(
                    policy_loss, policy_chunk_loss, total_policy_loss, policy_chunk_loss_list, total_policy_loss_list
                )
                chunk_loss, total_loss, chunk_loss_list, total_loss_list = loss_process(
                    loss, chunk_loss, total_loss, chunk_loss_list, total_loss_list
                )
                if args.kl_coeff > 0:
                    kl_chunk_loss, total_kl_loss, kl_chunk_loss_list, total_kl_loss_list = loss_process(
                        kl_loss, kl_chunk_loss, total_kl_loss, kl_chunk_loss_list, total_kl_loss_list
                        )
            if (count+1)%args.gradient_accumulation_steps==0:
                kl_chunk_loss_list, policy_chunk_loss_list, chunk_loss_list, ratio_chunk_list = step_save(
                    args, count, new_chunk_log_prob, sample["chunk_log_probs"][chunk_idx], 
                    policy_chunk_loss_list, kl_chunk_loss_list,  chunk_loss_list, ratio_chunk_list, step
                    )
                if not args.use_step:
                    grad_norm = transformer.clip_grad_norm_(max_grad_norm)
                    step_over(args, total_loss, optimizer, step, count, lr_scheduler)
            if dist.get_rank()==0:
                step_log(sample["rewards"].item(), ratio, sample["advantages"].item(), loss.item(), total_loss)
            count = count + 1
            dist.barrier()
    if not args.use_chunk or args.use_step: # standard step_level GRPO
        perms = torch.stack(
            [
                torch.randperm(len(samples["timesteps"][0]))
                for _ in range(batch_size)
            ]
        ).to(device) 
        if args.fixed_step:
            perms = generate_perm(batch_size, len(samples["timesteps"][0]), fixed_values = args.step_idx).to(device)
        main_print(f'perm is: {perms}')
        if args.debug:
            main_print(f'perm is: {perms}')
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][
                torch.arange(batch_size).to(device) [:, None],perms
            ]
        samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
        # dict of lists -> list of dicts for easier iteration
        samples_batched_list = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]
        train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction) 
        if args.fixed_step:
            train_timesteps = len(args.step_idx)
        if args.debug:
            main_print(f'total i is: {len(list(enumerate(samples_batched_list)))}')
            main_print(f'train_timesteps is: {train_timesteps}')
        
        adv_clip_max = args.adv_clip_max
        count = 0
        for i,sample in list(enumerate(samples_batched_list)):
            main_print(f'current i is: {i}')
            for _ in range(train_timesteps):
                # reweight loss from tempflow
                if args.use_reweight:
                    reweight_factor = math.sqrt(sample["timesteps"][:,_].item() / (1000 - sample["timesteps"][:,_].item() + 1))
                    main_print(f'timestep is: {sample["timesteps"][:,_]}')
                    main_print(f'reweight factor is: {reweight_factor}')
                else:
                    reweight_factor = 1.0
                if args.debug and dist.get_rank()==0:
                    print('latent shape is:', sample["latents"][:,_].shape)  # torch.Size([1, 2025, 64])
                    print('timestep shape is:', sample["timesteps"][:,_].shape) # torch.Size([1])
                new_log_probs = grpo_one_step(
                    args,
                    sample["latents"][:,_],
                    sample["next_latents"][:,_],
                    sample["encoder_hidden_states"], # torch.Size([1, 512, 4096])
                    sample["pooled_prompt_embeds"], # torch.Size([1, 768])
                    sample["text_ids"], # torch.Size([1, 3])
                    sample["image_ids"], # # torch.Size([1, 2025, 3])
                    transformer,
                    sample["timesteps"][:,_],
                    perms[i][_],
                    sigma_schedule,
                )
                if args.debug and dist.get_rank()==0:
                    print('new_log_probs shape is:', new_log_probs.shape)
                    print('new_log_probs is:', new_log_probs)
                    print('old_log probs is:', sample["log_probs"][:,_])

                advantages = torch.clamp(
                    sample["advantages"],
                    -adv_clip_max,
                    adv_clip_max,
                )
                if args.only_on_policy:
                    old_log_probs = new_log_probs.detach()
                else: # we use this one for baseline, same from DanceGRPO
                    old_log_probs = sample["log_probs"][:,_]
                ratio = torch.exp(new_log_probs - old_log_probs)
                ratio_step_list.append(float(ratio.detach().cpu()))
                unclipped_loss = -advantages * ratio
                if args.debug and dist.get_rank()==0:
                    main_print(f'ratio is: {float(ratio.detach().cpu())}')
                clipped_loss = -advantages * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.right_clip_range)
                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                policy_loss = policy_loss * reweight_factor
                if args.kl_coeff > 0:
                    anti_diff = old_log_probs - new_log_probs
                    kl_loss = torch.exp(anti_diff) - 1.0 - anti_diff
                    kl_loss = torch.mean(kl_loss)
                    if args.debug:
                        main_print(f'kl_loss is: {float(kl_loss.detach().cpu())}')
                    loss = (policy_loss + args.kl_coeff * kl_loss) / (args.gradient_accumulation_steps * train_timesteps)
                else:
                    loss = policy_loss / (args.gradient_accumulation_steps * train_timesteps)
                loss.backward()
                policy_step_loss, total_policy_loss, policy_step_loss_list, total_policy_loss_list = loss_process(
                    policy_loss, policy_step_loss, total_policy_loss, policy_step_loss_list, total_policy_loss_list
                )
                step_loss, total_loss, step_loss_list, total_loss_list = loss_process(
                    loss, step_loss, total_loss, step_loss_list, total_loss_list
                )
                if args.kl_coeff > 0:
                    kl_step_loss, total_kl_loss, kl_step_loss_list, total_kl_loss_list = loss_process(
                        kl_loss, kl_step_loss, total_kl_loss, kl_step_loss_list, total_kl_loss_list
                        )
            if (count+1)%args.gradient_accumulation_steps==0:
                grad_norm = transformer.clip_grad_norm_(max_grad_norm)
                policy_step_loss_list, kl_step_loss_list, step_loss_list, ratio_step_list = step_save(
                            args, count, new_log_probs, sample["log_probs"][:, _], 
                            policy_step_loss_list, kl_step_loss_list, step_loss_list, ratio_step_list, step, False)
                if args.use_chunk:
                    total_policy_loss_list, total_kl_loss_list, total_loss_list = step_total_save(
                        args, count, advantages, total_policy_loss_list, total_kl_loss_list, total_loss_list, step
                    )
                step_over(args, total_loss, optimizer, step, count, lr_scheduler)
            if dist.get_rank()==0:
                step_log(sample["rewards"].item(), ratio, sample["advantages"].item(), loss.item(), total_loss)
            count = count + 1
            dist.barrier()
    return total_loss, total_kl_loss, total_policy_loss, chunk_loss, step_loss, grad_norm.item(), gathered_reward.mean().item(), gathered_group_std.mean().item()

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank, rank, world_size, device = init_everything(args)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    if args.use_hpsv2: # False
        reward_model, preprocess_val, processor = initialize_hps_model(args, device)
    elif args.use_pickscore: # False
        reward_model_pic, processor_pic = initialize_pic_model(args, device)
    elif args.use_clip: # True
        ## for laion
        # reward_model = CLIPModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K').to(device)
        # preprocess_val = CLIPProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        # reward_model.requires_grad_(False)
        # reward_model.eval()
        # processor = None
        # for apple
        processor = get_tokenizer('ViT-H-14')
        reward_model, preprocess_val = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
        reward_model.to(device).eval()
        reward_model.requires_grad_(False)
    else:
        reward_model = HPSv3RewardInferencer(device=f'cuda:{device}')
        preprocess_val = None
        processor = None
        
    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32
    )

    # if args.load_from_before:
    #     start_step = load_from_checkpoint(transformer, args, rank)
    #     main_print(f'resume training from step_{start_step}')
    # else:
    #     start_step = 1
    # fsdp setting
    transformer = set_fsdp(args, transformer)
    # emma setting
    ema_handler = None
    if args.use_ema:
        ema_handler = FSDP_EMA(transformer, args.ema_decay, rank)
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)
    #vae.enable_tiling()

    # Load everything and set model as trainable.
    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    main_print(f"--> model loaded")
    transformer.train()

    noise_scheduler = None
    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    optimizer, lr_scheduler, init_steps = set_optimizer_and_lr_scheduler(params_to_optimize, args)
    if args.load_from_before:
        transformer, optimizer, start_step = resume_training(transformer, optimizer, args)
        main_print(f'resume training from step_{start_step}')
    else:
        start_step = 1
    
    if args.debug:
        main_print(f"optimizer: {optimizer}")
        # args.num_generations = 4
        # args.gradient_accumulation_steps = 8
        # args.sampling_steps = 5
        args.name = 'debug'
        args.output_dir = './data/outputs_clip/debug/'

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,  # (stack: bsz*dim)
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    if rank <= 0:
        wandb_init(args)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B")
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps, # 0
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size, # how many parallel dataloader work
        args.train_sp_batch_size, # how many batch in each parallel dataloader
    )

    step_times = deque(maxlen=100)
    debug_path = os.path.join(args.output_dir, f'debug/')
    os.makedirs(debug_path, exist_ok = True)

    if args.only_reward:
        args.max_train_steps = 1000
    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler): # True
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch

        for step in range(init_steps+1, args.max_train_steps+1):
            main_print(f"Current step is: {step}")
            if step < start_step:
                _, _, _, _ = next(loader)
                if args.use_sto_step:
                    args.step_sto_idx = random.randrange(0, args.sampling_steps // 2 - 3)
                continue
            if args.debug:
                main_print(f'use chunk is: {args.use_chunk}')
                main_print(f'use step is: {args.use_step}')
            
            start_time = time.time()
            if not args.only_reward:
                if (step % args.checkpointing_steps == 0) and step != start_step:
                    save_checkpoint_optimizer(transformer, optimizer, rank, args.output_dir, step, epoch)
                    if args.use_ema: 
                        save_ema_checkpoint(ema_handler, rank, args.output_dir, step, epoch, dict(transformer.config))
                    dist.barrier()

            main_print(f'use_sto_step is:, {args.use_sto_step}')
            if args.use_sto_step: # like the way of mixgrpo and flow-grpo-s1. However we don't use it here.
                args.step_sto_idx = random.randrange(0, args.sampling_steps // 2 - 3)
                args.fixed_step = True
                args.step_idx = [args.step_sto_idx, args.step_sto_idx + 1, args.step_sto_idx + 2, args.step_sto_idx + 3]
                if args.debug or args.use_sto_step:
                    main_print(f'step_sto_idx is: {args.step_sto_idx}')
                    main_print(f'fixed step idx is: {args.step_idx}')

            loss, kl_loss, policy_loss, chunk_loss, step_loss, grad_norm, reward_mean, std_mean = train_one_step(
                args,
                device, 
                transformer,
                vae,
                reward_model,
                processor,
                optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                args.max_grad_norm,
                preprocess_val,
                ema_handler,
                step,
            )

            if args.use_ema and ema_handler: 
                ema_handler.update(transformer)
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            main_print(f"Current step using time is: {step_time:.2f}s")
            avg_step_time = sum(step_times) / len(step_times)
            main_print(f"Avg_step_time is: {avg_step_time:.2f}s")

            if args.only_reward:
                if rank <= 0:
                    wandb.log(
                        {
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "step_time": step_time,
                            'reward': reward_mean,
                            'std': std_mean
                        },
                        step=step,
                    )
            else:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "step_time": f"{step_time:.2f}s",
                        "grad_norm": grad_norm,
                    }
                )
                progress_bar.update(1)
                
                if rank <= 0:
                    wandb.log(
                        {
                            "train_loss": loss,
                            'kl_loss': kl_loss,
                            "policy_loss": policy_loss,
                            "chunk_loss": chunk_loss,
                            "step_loss": step_loss,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "step_time": step_time,
                            # "avg_step_time": avg_step_time,
                            "grad_norm": grad_norm,
                            'reward': reward_mean,
                            'std': std_mean
                        },
                        step=step,
                    )
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )   # 4
    parser.add_argument("--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )  # 2
    parser.add_argument("--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    ) # False

    # validation & logs
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.") # 42
    parser.add_argument("--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    ) # 30
    parser.add_argument("--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    ) 
    parser.add_argument("--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    ) 
    parser.add_argument("--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    ) # 1e-5
    parser.add_argument("--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    ) # 0
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.") 
    parser.add_argument("--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    ) # True
    parser.add_argument("--selective_checkpointing", type=float, default=1.0) # 1
    parser.add_argument("--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    ) # True
    parser.add_argument("--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    ) # False
    parser.add_argument("--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    ) # 1
    parser.add_argument("--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    ) # 1
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to apply.") 
    parser.add_argument("--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    ) 

    # parallel setting
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel") # 1
    parser.add_argument("--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    ) # 2 for 8gpu
    parser.add_argument("--fsdp_sharding_startegy", default="full") # full
    
    #GRPO training
    parser.add_argument("--h",
        type=int,
        default=None,   
        help="video height",
    ) # 720
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    ) # 720
    parser.add_argument("--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    ) 
    parser.add_argument("--eta",
        type=float,
        default=None,   
        help="noise eta",
    ) # 0.7
    parser.add_argument("--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    ) # 1223627
    parser.add_argument("--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    ) # True
    parser.add_argument("--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    ) # 12
    parser.add_argument("--use_hpsv2",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model",
    ) # False
    parser.add_argument("--use_pickscore",
        action="store_true",
        default=False,
        help="whether use pickscore as reward model",
    ) 
    parser.add_argument("--use_hpsv3",
        action="store_true",
        default=False,
        help="whether use hpsv3 as reward model",
    ) 
    parser.add_argument("--use_clip",
        action="store_true",
        default=False,
        help="whether use clip as reward model",
    ) # for clip as the reward model
    parser.add_argument("--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    ) # True
    parser.add_argument("--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    ) # 3
    parser.add_argument("--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    ) 
    parser.add_argument("--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    ) 
    parser.add_argument("--right_clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    ) 
    parser.add_argument("--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    ) 
    
    # wandb
    parser.add_argument("--name",
        type=str,
        default='trytry',
        help="The name of wandb",
    )

    # ema
    parser.add_argument("--use_ema", 
        action="store_true", 
        help="Enable Exponential Moving Average of model weights."
    ) 

    # fixed
    parser.add_argument("--hps_path",
        type=str,
        default='./hps_ckpt/open_clip_pytorch_model.bin',
        help="The path of hps model",
    )
    parser.add_argument("--hps_checkpoint_path",
        type=str,
        default='./hps_ckpt/HPS_v2.1_compressed.pt',
        help="The checkpoint path of hps model",
    )
    parser.add_argument(
        "--try_mul",
        action="store_true",
        default=True,
        help="whether",
    ) # True
    parser.add_argument(
        "--loss_clip",
        action="store_true",
        default=True,
        help="whether",
    ) # True, however it has no effect on training, just records some abnormal numerical issues.
    
    # chunk-setting or step-setting
    parser.add_argument("--chunk_size",
        type = int,
        default=4,
        help="size of chunk",
    ) 
    parser.add_argument("--use_chunk",
        action="store_true",
        default=False,
        help="whether using chunk",
    ) 
    parser.add_argument("--new_fix_chunk",
        action="store_true",
        default=False,
        help="set chunk segmention",
    ) # True

    parser.add_argument("--new_chunk_list",
        type=eval,
        default=[0],
        help="chunk segmention",
    ) 
    parser.add_argument("--use_step",
        action="store_true",
        default=False,
        help="whether using chunk",
    ) # for standard step level GRPO
    
    # debug
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="whether debug",
    ) 
    parser.add_argument(
        "--only_reward",
        action="store_true",
        default=False,
        help="whether only_reward instead of training",
    ) 

    # resume training, use this one!
    parser.add_argument(
        "--load_from_before",
        action="store_true",
        default=False,
        help="whether load from previous checkpoint",
    ) 
    parser.add_argument("--load_path",
        type=str,
        default=None,
        help="The previous checkpoint path",
    )

    # fixed chunk
    parser.add_argument(
        "--fixed_chunk",
        action="store_true",
        default=False,
        help="whether train with a/some specifc chunk/chunks",
    ) 
    parser.add_argument("--chunk_idx",
        type = eval,
        default=[0],
        help="identify which specific chunk",
    ) 
    parser.add_argument("--remove",
        type = int,
        default=0,
        help="remove which chunk",
    ) # we didn't use it

    # fixed step, only for the way like mixgrpo or flow-grpo-s1
    parser.add_argument("--step_end_idx",
        type = int,
        default=100,
        help="where ends stochastic sampling",
    ) # in our method you don't need this one. However for the way like mixgrpo or flow-grpo-s1, you need to adjust it,
    parser.add_argument("--step_sto_idx",
        type = int,
        default=0,
        help="which step for stochastic sampling",
    ) 
    parser.add_argument("--use_sto_step",
        action="store_true",
        default=False,
        help="whether use stochastic sampling",
    ) 
    parser.add_argument(
        "--fixed_step",
        action="store_true",
        default=False,
        help="whether train with a/some specifc step/steps",
    ) 
    parser.add_argument("--step_idx",
        type=eval,
        default=[0],
        help="identify which specific step",
    ) 

    # adv
    parser.add_argument("--use_global_std",
        action="store_true",
        default=False,
        help="whether using global std or group std",
    ) 
    parser.add_argument("--use_half_half_adv",
        action="store_true",
        default=False,
        help="whether making group adv half > 0 and half < 0",
    ) 

    # kl
    parser.add_argument("--kl_coeff",
        type=float,
        default=0,   
        help="kl control",
    ) # 0
    
    parser.add_argument("--use_reweight",
        action="store_true",
        default=False,
        help="whether using loss reweight",
    ) # for tempflow-grpo
    
    # some grpo tricks
    parser.add_argument(
        "--grpo_step_mode",
        type=str,
        default='flow',
        help="flow or dance",
    ) # we apply flow to both baseline and our approach, so please fix it.

    parser.add_argument("--only_on_policy",
        action="store_true",
        default=False,
        help="whether using chunk",
    ) # False for both baseline and our approach. However it's an interesting future work, and welcome to contact me if you have any progress or idea!

    parser.add_argument("--sample_weight",
        action="store_true",
        default=False,
        help="the weighted sampling strategy",
    ) # for the weighted strategy

    parser.add_argument(
        "--sample_weight_method",
        type=str,
        default='normalized',
        help="normalize or softmax",
    ) # normalized for our approach. You can change it to softmax.

    args = parser.parse_args()
    main(args)