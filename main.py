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
    using_tensor_cores, increase_l2_fetch_granularity
from train_utils import train as train_function
from model import MPNN

def get_args():
    parser = argparse.ArgumentParser()
    
    #Directories
    parser.add_argument('--data_dir', type=str, default="/Scr/hyunpark/Monster/vaegan_md_gitlab/data") 
    parser.add_argument('--save_dir', type=str, default="/Scr/arango/Sobolev-Hyun/2-MembTempredict/analysis/")  
    parser.add_argument('--load_ckpt_path', type=str, default="/Scr/hyunpark/Protein-TDA/saved")
    parser.add_argument('--filename', type=str, default="default.pickle")  
    
    #MDAnalysis utils
    parser.add_argument('--psf', type=str, default=None)  
    parser.add_argument('--pdb', type=str, default=None)  
    parser.add_argument('--last', type=int, default=200) 
    parser.add_argument('--trajs', default=None, nargs="*") 
    parser.add_argument('--atom_selection', type=str, default="backbone")  

    #PH utils
    parser.add_argument('--maxdim', type=int, default=1)  
    parser.add_argument('--multiprocessing', action="store_true")  
    parser.add_argument('--tensor', action="store_true", help="DEPRECATED!")  

    #Dataloader utils
    parser.add_argument('--train_frac', type=float, default=0.8)  
    parser.add_argument('--pin_memory', type=bool, default=True)  
    parser.add_argument('--num_workers', type=int, default=0)  
    parser.add_argument('--batch_size', type=int, default=32)  
    parser.add_argument('--preprocessing_only', action="store_true", help="to get RIPSER based PH!")  
    parser.add_argument('--ignore_topologicallayer', action="store_true", help="forcefully use RIPSER for subscription!")  

    #Training utils
    parser.add_argument('--epoches', type=int, default=2)
    parser.add_argument('--learning_rate','-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--warm_up_split', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0)
#     parser.add_argument('--distributed',  action="store_true")
    parser.add_argument('--low_memory',  action="store_true")
    parser.add_argument('--amp', action="store_true", help="floating 16 when turned on.")
    parser.add_argument('--optimizer', type=str, default='adam', choices=["adam","lamb","sgd","torch_adam","torch_adamw","torch_sparse_adam"])
    parser.add_argument('--gradient_clip', type=float, default=None) 
    parser.add_argument('--accumulate_grad_batches', type=int, default=1) 
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--shard', action='store_true')
    
    #Model utils
    parser.add_argument('--backbone', type=str, default='mpnn', choices=["mpnn"])
    
    #Callback utils
    parser.add_argument('--log', action="store_true", help="to log for W&B")  
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--name', type=str, default="mpnn", help="saved torch model name...")
    
    #Mode utils
    parser.add_argument('--which_mode', type=str, choices=["preprocessing", "train", "infer"], default="preprocessing")  

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args.__dict__)
    dutils = data_utils if args.which_mode == "preprocessing" else data_utils_mem
    
    if args.which_mode == "preprocessing":
        ds = dutils.PH_Featurizer_Dataset(args)
        print(ds[0], ds[5])
#         dl = PH_Featurizer_DataLoader(opt=args)
#         testset = dl.test_dataloader()
#         print(iter(testset).next())
    elif args.which_mode == "train":
        dl = dutils.PH_Featurizer_DataLoader(opt=args)
        train_loader, val_loader, test_loader = [getattr(dl, key)() for key in ["train_dataloader", "val_dataloader", "test_dataloader"]]
        net = MPNN()
        loss_func = torch.nn.MSELoss()
        logger = None
        
        #Initalize DDP
        is_distributed = init_distributed() #normal python vs torchrun!
        local_rank = get_local_rank()
        if args.gpu:
            net = net.to(torch.cuda.current_device())
        #Dist training
        if is_distributed:         
            nproc_per_node = torch.cuda.device_count()
            affinity = set_affinity(local_rank, nproc_per_node)
        increase_l2_fetch_granularity()
        
        print("Initalizing training...")
        train_function(net, loss_func, train_loader, val_loader, test_loader, logger, args)
        #python -m main --which_mode train --ignore_topologicallayer
