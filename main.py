from __future__ import print_function, division
import abc, sys
import collections
import torch_geometric
from torch_geometric.data import Data, Dataset
import pathlib
import persim
import ripser
import MDAnalysis as mda
import argparse
from typing import *
import functools
import itertools 
import functools
import numpy as np
import time
import ray
import os
import pickle
import collections
import warnings
import curtsies.fmtfuncs as cf
import tqdm
import pymatgen as pg
from pymatgen.core import Structure
import dataclasses
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader #Can this handle DDP? yeah!
import torch.distributed as dist 
from dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity
from torch.utils.data import DistributedSampler
from typing import *
from topologylayer.nn import RipsLayer, AlphaLayer
import gc
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.align import AlignTraj
from MDAnalysis import transformations
import data_utils
import data_utils_mem 
from dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, WandbLogger
from train_utils import train as train_function
from kd_train_utils import train as distill_train_function
from model import MPNN, Vision
from gpu_utils import *
from loss_utils import * #TEMP_RANGES
from test_utils import validate_and_test, InferenceDataset

def get_args():
    parser = argparse.ArgumentParser()
    
    #Directories
    parser.add_argument('--data_dir', type=str, default="/Scr/hyunpark/Monster/vaegan_md_gitlab/data", help="DEPRECATED!") 
    parser.add_argument('--save_dir', type=str, default="/Scr/hyunpark-new/Protein-TDA/pickled_indiv/")  
    parser.add_argument('--load_ckpt_path', type=str, default="/Scr/hyunpark-new/Protein-TDA/saved")
    parser.add_argument('--filename', type=str, default="dppc.pickle")  
    parser.add_argument('--pdb_database', type=str, default="/Scr/arango/Sobolev-Hyun/2-MembTempredict/testing/") 
    parser.add_argument('--search_temp', default=123, help="e.g. keyword for T.123 directory") 

    #MDAnalysis utils
    parser.add_argument('--psf', type=str, default=None)  
    parser.add_argument('--pdb', type=str, default=None)  
    parser.add_argument('--last', type=int, default=200) 
    parser.add_argument('--trajs', default=None, nargs="*") 
    parser.add_argument('--atom_selection', type=str, default="backbone")  

    #PH utils
    parser.add_argument('--maxdim', type=int, default=1)  
    parser.add_argument('--multiprocessing', action="store_true")  
    parser.add_argument('--multiprocessing_backend', type=str, default="ray", choices=["multiprocessing", "dask", "joblib", "ray"])  
    parser.add_argument('--tensor', action="store_true", help="DEPRECATED!")  

    #Dataloader utils
    parser.add_argument('--train_frac', type=float, default=0.8)  
    parser.add_argument('--pin_memory', type=bool, default=True)  
    parser.add_argument('--num_workers', type=int, default=0)  
    parser.add_argument('--batch_size', type=int, default=32)  
    parser.add_argument('--preprocessing_only', action="store_true", help="to get RIPSER based PH!")  
    parser.add_argument('--ignore_topologicallayer', action="store_true", help="forcefully use RIPSER for subscription!")  
    parser.add_argument('--truncated', action="store_true", help="Use only 10% of data. It MUST have original dataset, however.")  

    #Training utils
    parser.add_argument('--epoches', type=int, default=2)
    parser.add_argument('--label_smoothing', '-ls', type=float, default=0.)
    parser.add_argument('--learning_rate','-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--warm_up_split', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0)
#     parser.add_argument('--distributed',  action="store_true")
    parser.add_argument('--low_memory',  action="store_true")
    parser.add_argument('--amp', action="store_true", help="floating 16 when turned on.")
    parser.add_argument('--optimizer', type=str, default='adam', choices=["adam","lamb","sgd","torch_adam","torch_adamw","torch_sparse_adam"])
    parser.add_argument('--scheduler', type=str, default='reduce', choices=["linear","reduce"])
    parser.add_argument('--gradient_clip', type=float, default=None) 
    parser.add_argument('--accumulate_grad_batches', type=int, default=1) 
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_artifacts', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--shard', action='store_true')
    parser.add_argument('--loss', choices=["mse", "mae", "smooth", "hybrid"], default="hybrid")
    parser.add_argument('--ce_weights', nargs="*", default=[1]*TEMP_RANGES[2], type=float, help="CE weights for class")
    parser.add_argument('--ce_re_ratio', nargs=2, default=[1., 1.], type=float, help="CE and Reg loss weights")

    #Model utils
    parser.add_argument('--backbone', type=str, default='vit', choices=["mpnn", "vit", "swin", "swinv2", "convnext", "restv2", "clip_resnet"])
    
    #Callback utils
    parser.add_argument('--log', action="store_true", help="to log for W&B")  
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--name', type=str, default="mpnn", help="saved torch model name...")
    parser.add_argument('--teacher_name', type=str, default="mpnn", help="saved torch model name...")

    #Mode utils
    parser.add_argument('--which_mode', type=str, choices=["preprocessing", "train", "infer", "infer_custom"], default="preprocessing")  

    args = parser.parse_args()
    return args

def preprocessing(args):
    ds = dutils.PH_Featurizer_Dataset(args)
    print(ds[0], ds[5])
#         dl = PH_Featurizer_DataLoader(opt=args)
#         testset = dl.test_dataloader()
#         print(iter(testset).next())

def job_submit(args):
    #Initalize DDP
    is_distributed = init_distributed() #normal python vs torchrun!
    local_rank = get_local_rank()

    #WARNING: Call dataloader & logger after initializing DDP
    dl = dutils.PH_Featurizer_DataLoader(opt=args)
    train_loader, val_loader, test_loader = [getattr(dl, key)() for key in ["train_dataloader", "val_dataloader", "test_dataloader"]]
    print(cf.on_blue("STEP 1 of training: Loading data is done!"))
    
    if args.backbone == "mpnn":
        net = MPNN()
    elif args.backbone in ["vit", "swin", "swinv2", "convnext", "restv2", "clip_resnet"]:
        net = Vision(args)
        
    if args.gpu:
        net = net.to(torch.cuda.current_device())
        
    if args.loss == "mse":
        loss_func = torch.nn.MSELoss()
    elif args.loss == "mae":
        loss_func = torch.nn.L1Loss()
    elif args.loss == "smooth":
        loss_func = torch.nn.SmoothL1Loss()
    elif args.loss == "hybrid":
#         print(args.ce_re_ratio)
        ce_re_ratio = torch.tensor(args.ce_re_ratio).to(torch.cuda.current_device()).float()
        loss_func = lambda pred, targ: ce_re_ratio[0] * ce_loss(args, targ, pred) + ce_re_ratio[1] * reg_loss(args, targ, pred)
    elif args.loss == "distill":
#         print(args.ce_re_ratio)
        ce_re_ratio = torch.tensor(args.ce_re_ratio).to(torch.cuda.current_device()).float()
        loss_func = lambda pred, targ, teacher_pred, T, alpha: ce_re_ratio[0] * distillation_loss(args, targ, pred, teacher_pred, T, alpha) + ce_re_ratio[1] * reg_loss(args, targ, pred)
        
    if args.log:
#         https://docs.wandb.ai/guides/artifacts/storage
        logger = WandbLogger(name=args.name, project="Memb-TDA", entity="hyunp2")
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")
        os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), ".cache/wandb")
        os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), ".config/wandb")
    else:
        logger = None
    
    #Dist training
    if is_distributed:         
        nproc_per_node = torch.cuda.device_count()
        affinity = set_affinity(local_rank, nproc_per_node)
    increase_l2_fetch_granularity()
    
    print(cf.on_yellow("STEP 2 of training: Initalizing training..."))
    train_function(net, loss_func, train_loader, val_loader, test_loader, logger, args)
    #python -m main --which_mode train --ignore_topologicallayer

def job_submit_distill(args):
    #Initalize DDP
    is_distributed = init_distributed() #normal python vs torchrun!
    local_rank = get_local_rank()

    #WARNING: Call dataloader & logger after initializing DDP
    dl = dutils.PH_Featurizer_DataLoader(opt=args)
    train_loader, val_loader, test_loader = [getattr(dl, key)() for key in ["train_dataloader", "val_dataloader", "test_dataloader"]]
    print(cf.on_blue("STEP 1 of training: Loading data is done!"))
    
    if args.backbone == "mpnn":
        net = MPNN()
    elif args.backbone in ["vit", "swin", "swinv2", "convnext", "restv2", "clip_resnet"]:
        student = net = Vision(args)
        args.backbone = "convnext" #Force it...
        teacher = Vision(args)
        
    if args.gpu:
        net = net.to(torch.cuda.current_device())
        teacher = teacher.to(torch.cuda.current_device())

    if args.loss == "mse":
        loss_func = torch.nn.MSELoss()
    elif args.loss == "mae":
        loss_func = torch.nn.L1Loss()
    elif args.loss == "smooth":
        loss_func = torch.nn.SmoothL1Loss()
    elif args.loss == "hybrid":
#         print(args.ce_re_ratio)
        ce_re_ratio = torch.tensor(args.ce_re_ratio).to(torch.cuda.current_device()).float()
        loss_func = lambda pred, targ: ce_re_ratio[0] * ce_loss(args, targ, pred) + ce_re_ratio[1] * reg_loss(args, targ, pred)
    elif args.loss == "distill":
#         print(args.ce_re_ratio)
        ce_re_ratio = torch.tensor(args.ce_re_ratio).to(torch.cuda.current_device()).float()
        loss_func = lambda pred, targ, teacher_pred, T, alpha: ce_re_ratio[0] * distillation_loss(args, targ, pred, teacher_pred, T, alpha) + ce_re_ratio[1] * reg_loss(args, targ, pred)
    
    assert args.loss == "distill", "For distillatin, distill loss must be chosen!"
    
    if args.log:
#         https://docs.wandb.ai/guides/artifacts/storage
        logger = WandbLogger(name=args.name, project="Memb-TDA", entity="hyunp2")
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")
        os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), ".cache/wandb")
        os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), ".config/wandb")
    else:
        logger = None
    
    #Dist training
    if is_distributed:         
        nproc_per_node = torch.cuda.device_count()
        affinity = set_affinity(local_rank, nproc_per_node)
    increase_l2_fetch_granularity()
    
    print(cf.on_yellow("STEP 2 of training: Initalizing training..."))
    distill_train_function(net, teacher, loss_func, train_loader, val_loader, test_loader, logger, args)
    #python -m main --which_mode train --ignore_topologicallayer
    
def infer_submit(args):
    #Initalize DDP
    is_distributed = init_distributed() #normal python vs torchrun!
    local_rank = get_local_rank()

    #WARNING: Call dataloader & logger after initializing DDP
    dl = dutils.PH_Featurizer_DataLoader(opt=args)
    
    train_loader, val_loader, test_loader = [getattr(dl, key)() for key in ["train_dataloader", "val_dataloader", "test_dataloader"]]
    print(cf.on_blue("STEP 1 of validation and testing: Loading data is done!"))
    
    if args.backbone == "mpnn":
        net = MPNN()
    elif args.backbone in ["vit", "swin", "swinv2", "convnext", "restv2", "clip_resnet"]:
        net = Vision(args)
        
    if args.gpu:
        net = net.to(torch.cuda.current_device())
        
    if args.loss == "mse":
        loss_func = torch.nn.MSELoss()
    elif args.loss == "mae":
        loss_func = torch.nn.L1Loss()
    elif args.loss == "smooth":
        loss_func = torch.nn.SmoothL1Loss()
    elif args.loss == "hybrid":
#         print(args.ce_re_ratio)
        ce_re_ratio = torch.tensor(args.ce_re_ratio).to(torch.cuda.current_device()).float()
        loss_func = lambda pred, targ: ce_re_ratio[0] * ce_loss(args, targ, pred) + ce_re_ratio[1] * reg_loss(args, targ, pred)
    elif args.loss == "distill":
#         print(args.ce_re_ratio)
        ce_re_ratio = torch.tensor(args.ce_re_ratio).to(torch.cuda.current_device()).float()
        loss_func = lambda pred, targ, teacher_pred, T, alpha: ce_re_ratio[0] * distillation_loss(args, targ, pred, teacher_pred, T, alpha) + ce_re_ratio[1] * reg_loss(args, targ, pred)
        
#     print(cf.red("Forcefully changeing loss function for evaluation to MAE..."))
#     args.loss = "mae" 
#     loss_func = torch.nn.L1Loss()
    
    if args.log:
#         https://docs.wandb.ai/guides/artifacts/storage
        logger = WandbLogger(name=args.name, project="Memb-TDA", entity="hyunp2")
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")
        os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), ".cache/wandb")
        os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), ".config/wandb")
    else:
        logger = None
    
    #Dist training
    if is_distributed:         
        nproc_per_node = torch.cuda.device_count()
        affinity = set_affinity(local_rank, nproc_per_node)
    increase_l2_fetch_granularity()
    
    print(cf.on_yellow("STEP 2 of validation and testing: Initalizing validation and testing..."))
    validate_and_test(net, loss_func, train_loader, val_loader, test_loader, logger, args)
    
def infer_for_customdata(args):
    #Initalize DDP
    is_distributed = init_distributed() #normal python vs torchrun!
    local_rank = get_local_rank()
    
    if args.backbone == "mpnn":
        net = MPNN()
    elif args.backbone in ["vit", "swin", "swinv2", "convnext", "restv2", "clip_resnet"]:
        net = Vision(args)
        
    if args.gpu:
        net = net.to(torch.cuda.current_device())
        
    if args.log:
#         https://docs.wandb.ai/guides/artifacts/storage
        logger = WandbLogger(name=args.name, project="Memb-TDA", entity="hyunp2")
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")
        os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), ".cache/wandb")
        os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), ".config/wandb")
    else:
        logger = None
    
    #Dist training
    if is_distributed:         
        nproc_per_node = torch.cuda.device_count()
        affinity = set_affinity(local_rank, nproc_per_node)
    increase_l2_fetch_granularity()
    
    print(cf.on_yellow("Inferring on given custom dataset!"))
    
    infds = InferenceDataset(args, net)
    infds()
    
    if args.log and (not dist.is_initialized() or dist.get_rank() == 0):
        logger.experiment.finish()
    
    
if __name__ == "__main__":
    args = get_args()
    print(args.__dict__)
#     dutils = data_utils if args.which_mode == "preprocessing" else data_utils_mem
    dutils = data_utils
    
    if args.which_mode == "preprocessing":
        preprocessing(args)
    elif args.which_mode == "train":
        job_submit(args)
    elif args.which_mode == "infer":
        infer_submit(args)
    elif args.which_mode == "infer_custom":
        infer_for_customdata(args)
        
    #python -m main --which_mode train --name vit_model --filename vit.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio "[1, 0.1]"
