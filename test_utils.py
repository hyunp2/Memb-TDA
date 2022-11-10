import logging
import pathlib
from typing import List, Union
import os, sys
import pathlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import importlib
try:
    importlib.import_module("apex.optimizers")
    from apex.optimizers import FusedAdam, FusedLAMB
except Exception as e:
    pass
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, Logger
# from transformers import AdamW
import curtsies.fmtfuncs as cf
import torchmetrics

# https://fairscale.readthedocs.io/en/latest/
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper #anytime
from fairscale.experimental.nn.offload import OffloadModel #Single-GPU
from fairscale.optim.adascale import AdaScale #DDP
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP #Sharding
from fairscale.optim.oss import OSS #Sharding
import shutil
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
from torch.distributed.fsdp.wrap import (
					default_auto_wrap_policy,
					enable_wrap,
					wrap,
					)

from train_utils import load_state, single_val, single_test

def validate_and_test(model: nn.Module,
          get_loss_func: _Loss,
          val_dataloader: DataLoader,
	  test_dataloader: DataLoader,
          logger: Logger,
          args):
    """Includes evaluation and testing as well!"""

    #DDP options
    #Init distributed MUST be called in run() function, which calls this train function!

    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    tmetrics = torchmetrics.MeanAbsoluteError()
    
    #DDP Model
    if dist.is_initialized() and not args.shard:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model._set_static_graph()
        print(f"DDP is enabled {dist.is_initialized()} and this is local rank {local_rank} and sharding is {args.shard}!!")    
        model.train()
        if args.log: logger.start_watching(model) #watch a model!
    elif dist.is_initialized() and args.shard:
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=100
        )
        torch.cuda.set_device(local_rank)
        model = FSDP(model, fsdp_auto_wrap_policy=my_auto_wrap_policy)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    path_and_name = os.path.join(args.load_ckpt_path, "{}.pth".format(args.name))
    assert args.resume, "Validation and test must be under resumed keyword..."
    epoch_start, best_loss = load_state(model, None, None, path_and_name, use_artifacts=args.use_artifacts, logger=logger, name=args.name, model_only=True) 
    
    #DDP training: Total stats (But still across multi GPUs)
    init_start_event.record()
    
    model.eval()    	

    ###EVALUATION
    evaluate = single_val
    val_loss, loss_metrics, val_predictions = evaluate(args, model, val_dataloader, get_loss_func, None, None, logger, tmetrics, return_data=True) #change to single_val with DDP
    if dist.is_initialized():
        val_loss = torch.tensor(val_loss, dtype=torch.float, device=device)
        loss_metrics = torch.tensor(loss_metrics, dtype=torch.float, device=device)
        torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
        val_loss = (val_loss / world_size).item()
        torch.distributed.all_reduce(loss_metrics, op=torch.distributed.ReduceOp.SUM) #Sum to loss
        loss_metrics = (loss_metrics / world_size).item()
    if args.log: 
        logger.log_metrics({'ALL_REDUCED_val_loss': val_loss})
        logger.log_metrics({'ALL_REDUCED_val_MAE': loss_metrics}) #zero rank only
    #zero rank only
    #mae_reduced = tmetrics.compute() #Synced and reduced autometically!
    #logger.log_metrics({'ALL_REDUCED_val_mae_loss': mae_reduced.item()}, epoch_idx) #zero rank only
    tmetrics.reset()
    #scheduler_re.step(val_loss) #Not on individual stats but the total stats

    ###TESTING
    test_loss, loss_metrics, test_predictions = single_test(args, model, test_dataloader, get_loss_func, None, None, logger, tmetrics, return_data=True) #change to single_val with DDP
    if dist.is_initialized():
        test_loss = torch.tensor(test_loss, dtype=torch.float, device=device)
        loss_metrics = torch.tensor(loss_metrics, dtype=torch.float, device=device)
        torch.distributed.all_reduce(test_loss, op=torch.distributed.ReduceOp.SUM)
        test_loss = (test_loss / world_size).item()
        torch.distributed.all_reduce(loss_metrics, op=torch.distributed.ReduceOp.SUM) #Sum to loss
        loss_metrics = (loss_metrics / world_size).item()
    if args.log: 
        logger.log_metrics({'ALL_REDUCED_test_loss': test_loss}) #zero rank only
    logger.log_metrics({'ALL_REDUCED_test_MAE': loss_metrics}) #zero rank only
    #mae_reduced = tmetrics.compute() #Synced and reduced autometically!
    #logger.log_metrics({'ALL_REDUCED_test_mae_loss': mae_reduced.item()}, epoch_idx) #zero rank only
    tmetrics.reset()

    init_end_event.record()

    if local_rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
	
    print(cf.on_green("Saving returned validation and test_predictions!"))
    val_gts = torch.cat([batch["temp"] for batch in val_dataloader], dim=0) #B,1
    test_gts = torch.cat([batch["temp"] for batch in test_dataloader], dim=0) #B,1

    np.savez("PH_val_test.npz", validation_gt=val_gts.detach().cpu().numpy(), test_gt=test_gts.detach().cpu().numpy(), validation_pred=val_predictions.detach().cpu().numpy(), test_pred=test_predictions.detach().cpu().numpy())
    
    print(cf.on_yellow("Validation and test are OVER..."))
    dist.destroy_process_group()


