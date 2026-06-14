#This code file is from [https://github.com/hao-ai-lab/FastVideo], which is licensed under Apache License 2.0.

import os
import pdb
import sys
import json
import torch.distributed as dist

def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)


# ForkedPdb().set_trace()
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def step_save(args, i, new_log_prob, old_log_prob, 
                policy_loss_list, kl_loss_list, total_loss_list, ratio_list, step, is_chunk = True):
    if dist.get_rank()%8==0:
        if is_chunk:
            save_dict = {
                "new_chunk_log_prob": new_log_prob.detach().cpu().numpy().tolist(),
                "old_chunk_log_prob": old_log_prob.cpu().numpy().tolist(),
                'kl_loss_list': kl_loss_list,
                'policy_loss_list': policy_loss_list,
                'total_loss_list': total_loss_list,
                'ratio_list': ratio_list
            }
            save_path = os.path.join(args.output_dir, f'debug/debug_{step}_{i}_chunk.json')
        else:
            save_dict = {
                "new_step_log_prob": new_log_prob.detach().cpu().numpy().tolist(),
                "old_step_log_prob": old_log_prob.cpu().numpy().tolist(),
                'kl_loss_list': kl_loss_list,
                'policy_loss_list': policy_loss_list,
                'total_loss_list': total_loss_list,
                'ratio_list': ratio_list
            }
            save_path = os.path.join(args.output_dir, f'debug/debug_{step}_{i}_step.json')
        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4)
        if args.debug:
            print(f"Saved debug info to {save_path}")
    policy_loss_list = []
    kl_loss_list = []
    total_loss_list = []
    ratio_list = []
    return policy_loss_list, kl_loss_list, total_loss_list, ratio_list

def step_total_save(args, i, advantages, policy_loss_list, kl_loss_list, total_loss_list, step):
    if dist.get_rank()%8==0:
        save_dict = {
                "advantages": advantages.cpu().numpy().tolist(),
                'kl_loss_list': kl_loss_list,
                'policy_loss_list': policy_loss_list,
                'total_loss_list': total_loss_list
            }
        save_path = os.path.join(args.output_dir, f'debug/debug_{step}_{i}_total.json')
        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4)
        if args.debug:
            print(f"Saved debug info to {save_path}")
    policy_loss_list = []
    kl_loss_list = []
    total_loss_list = []
    return policy_loss_list, kl_loss_list, total_loss_list

def step_log(reward, ratio, advantage, loss, total_loss):
    print("reward", reward)
    print("ratio", float(ratio.detach().cpu()))
    print("advantage", advantage)
    print("loss", loss)
    print('total_loss', total_loss)

