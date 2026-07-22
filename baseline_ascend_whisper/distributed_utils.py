"""分布式训练工具函数。支持 TP（张量并行）和 DP（数据并行）。

全局 rank 编排: rank = dp_rank * tp + tp_rank
  - TP 组: 同一个 dp_rank 内的所有 tp_rank，共 dp 个组
  - DP 组: 同一个 tp_rank 内的所有 dp_rank，共 tp 个组
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed(rank: int = None, world_size: int = None,
                      backend: str = "hccl"):
    """初始化分布式进程组。

    可显式传入 rank/world_size，也可由环境变量自动推断（torchrun 模式）。
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if rank is not None and world_size is not None:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend=backend)


def create_tp_dp_groups(tp: int, dp: int):
    """创建 TP / DP 通信组，返回 (tp_group, dp_group, tp_rank, dp_rank)。"""
    rank = dist.get_rank()
    dp_rank = rank // tp
    tp_rank = rank % tp

    tp_group = None
    for d in range(dp):
        group_ranks = [d * tp + t for t in range(tp)]
        grp = dist.new_group(group_ranks)
        if rank in group_ranks:
            tp_group = grp

    dp_group = None
    for t in range(tp):
        group_ranks = [d * tp + t for d in range(dp)]
        grp = dist.new_group(group_ranks)
        if rank in group_ranks:
            dp_group = grp

    return tp_group, dp_group, tp_rank, dp_rank


def _get_tp_param_info(model):
    """扫描模型，返回 {state_dict_key: gather_dim} 映射。

    gather_dim 含义:
      >= 0 : 该参数在对应 dim 上 all_gather / 切分
      -1   : 该参数在 TP rank 间完全相同，无需 gather/shard
    """
    from model import ColumnParallelLinear, RowParallelLinear

    info = {}
    for name, module in model.named_modules():
        if isinstance(module, ColumnParallelLinear):
            prefix = name + '.'
            info[prefix + 'weight'] = 0
            if module.bias is not None:
                info[prefix + 'bias'] = 0
        elif isinstance(module, RowParallelLinear):
            prefix = name + '.'
            info[prefix + 'weight'] = 1
            if module.bias is not None:
                info[prefix + 'bias'] = -1
    return info


def gather_model_state_dict(model, tp_world_size: int, tp_group):
    """将 TP 分片的模型参数收集为完整 state_dict（用于保存检查点）。

    所有 TP rank 必须同时调用（内部有 all_gather 集合通信）。
    返回值在每个 rank 上都相同，但只有 rank 0 需要写文件。
    """
    state = model.state_dict()
    if tp_world_size <= 1 or tp_group is None:
        return state

    tp_param_info = _get_tp_param_info(model)

    gathered_state = {}
    for key, tensor in state.items():
        if key in tp_param_info:
            dim = tp_param_info[key]
            if dim >= 0:
                gathered = [torch.empty_like(tensor) for _ in range(tp_world_size)]
                dist.all_gather(gathered, tensor, group=tp_group)
                gathered_state[key] = torch.cat(gathered, dim=dim)
            else:
                gathered_state[key] = tensor
        else:
            gathered_state[key] = tensor

    return gathered_state


def shard_state_dict(state_dict: dict, model, tp_rank: int, tp_world_size: int) -> dict:
    """将完整 state_dict 切分为当前 TP rank 的分片（用于加载检查点）。"""
    if tp_world_size <= 1:
        return state_dict

    tp_param_info = _get_tp_param_info(model)

    sharded_state = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            sharded_state[key] = tensor
            continue
        if key in tp_param_info:
            dim = tp_param_info[key]
            if dim >= 0:
                chunk_size = tensor.size(dim) // tp_world_size
                slices = [slice(None)] * tensor.dim()
                slices[dim] = slice(tp_rank * chunk_size, (tp_rank + 1) * chunk_size)
                sharded_state[key] = tensor[tuple(slices)].clone()
            else:
                sharded_state[key] = tensor.clone()
        else:
            sharded_state[key] = tensor

    return sharded_state


def dp_allreduce_gradients(model, dp_group):
    """在 DP 组内对模型梯度做 all-reduce 平均。"""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, group=dp_group)
            world_size = dist.get_world_size(dp_group)
            param.grad.div_(world_size)


def cleanup_distributed():
    """清理分布式环境。"""
    if dist.is_initialized():
        dist.destroy_process_group()
