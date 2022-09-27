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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/grand/ACO2RDS/xiaoliyan/hMOF/cif") 
    parser.add_argument('--save_dir', type=str, default=os.getcwd())  
    parser.add_argument('--filename', type=str, default="default.pickle")  
    parser.add_argument('--maxdim', type=int, default=1)  
    parser.add_argument('--multiprocessing', action="store_true")  
    parser.add_argument('--tensor', action="store_true", help="DEPRECATED!")  
    parser.add_argument('--train_frac', type=float, default=0.8)  
    parser.add_argument('--pin_memory', type=bool, default=True)  
    parser.add_argument('--num_workers', type=int, default=0)  
    parser.add_argument('--batch_size', type=int, default=32)  
    parser.add_argument('--psf', type=str, default=None)  
    parser.add_argument('--pdb', type=str, default=None)  
    parser.add_argument('--last', type=int, default=200) 
    parser.add_argument('--trajs', default=None, nargs="*")  
    parser.add_argument('--atom_selection', type=str, default="backbone")  

    args = parser.parse_args()
    return args
