import torch.distributed as dist


def loss_process(loss, loss_1, total_loss, loss_list_1, total_loss_list):
    avg_loss = loss.detach().clone()
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    loss_1 = loss_1 + avg_loss.item()
    total_loss = total_loss + avg_loss.item()
    loss_list_1.append(loss_1)
    total_loss_list.append(total_loss)
    return loss_1, total_loss, loss_list_1, total_loss_list

