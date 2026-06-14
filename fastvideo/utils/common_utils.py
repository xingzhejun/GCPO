import os
import copy
import torch
import random
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from safetensors.torch import load_file
import wandb


def print_current_gpu_memory():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
    print(f"[Current GPU {device}] Allocated memory: {allocated:.2f} MB")
    print(f"[Current GPU {device}] Reserved memory:  {reserved:.2f} MB")

def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

def init_everything(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size) # 1
    
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the dir creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    return local_rank, rank, world_size, device

def set_fsdp(args, transformer):
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    # fsdp_kwargs['use_orig_params'] = True
    
    transformer = FSDP(transformer, **fsdp_kwargs,)

    if args.gradient_checkpointing: # True
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )

    return transformer

def set_optimizer_and_lr_scheduler(params_to_optimize, args):
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    return optimizer, lr_scheduler, init_steps

def repeat_tensor(args, tensor):
    if tensor is None:
        return None
    return torch.repeat_interleave(tensor, args.num_generations, dim=0)

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

def gather_tensor(tensor):
    if not dist.is_initialized():
        print('not in dist training')
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def step_over(args, total_loss, optimizer, step, i, lr_scheduler, threshold = 0.00005):
    # you can ignor this threshold. 
    if total_loss >= threshold:
        # optimizer.zero_grad()
        abnormal_path = os.path.join(args.output_dir, f'abnormal.txt')
        if dist.get_rank()%8==0:
            with open(abnormal_path, 'a', encoding='utf-8') as f:
                f.write(f'abnormal:step_{step}_{i}' + '\n')
    # else:
    #     optimizer.step()
    #     lr_scheduler.step()
    #     optimizer.zero_grad()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

def load_from_checkpoint(transformer, args, rank):
    weight_path = os.path.join(args.load_path, "diffusion_pytorch_model.safetensors")
    state_dict = load_file(weight_path)
    transformer.load_state_dict(state_dict)
    print(f'{rank} load model state from {args.load_path}')
    folder_name = os.path.basename(os.path.normpath(args.load_path))
    step_str = folder_name.split("-")[1]
    start_step = int(step_str)
    return start_step

def load_optimizer(optimizer, device, args, rank):
    weight_path = os.path.join(args.load_path, "optimizer.pt")
    state_dict = torch.load(weight_path)
    optimizer.load_state_dict(state_dict)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    print(f'{rank} load optimizer state from {args.load_path}')
    return optimizer

def generate_perm(rows, cols, fixed_values):
    all_values = list(range(0, cols))
    perm_rows = []
    for _ in range(rows):
        remaining = [v for v in all_values if v not in fixed_values]
        random.shuffle(remaining)  
        row = fixed_values + remaining
        perm_rows.append(row)
    return torch.tensor(perm_rows)

def wandb_init(args):
    if args.debug:
        project = "flux_debug"
    elif args.grpo_step_mode == 'flow':
        if args.use_clip:
            project = 'flux_flow_clip'
        else:
            project = "flux_flow_hpsv3"
    elif args.use_hpsv3:
        project = "flux_hpsv3"
    else:
        project = "flux"
    wandb.init(project=project, name = args.name, config=args)

def copy_dict(samples):
    samples_new = {}
    for key, value in samples.items():
        if isinstance(value, torch.Tensor):
            samples_new[key] = value.clone()
        else:
            samples_new[key] = copy.deepcopy(value)
    return samples_new
