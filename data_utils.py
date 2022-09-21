from __future__ import print_function, division
import abc, sys
import collections
import torch_geometric
from torch_geometric.data import Data, Dataset
import pathlib
# roots = pathlib.Path(__file__).parent.parent
# sys.path.append(roots) #append top directory
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
    using_tensor_cores, increase_l2_fetch_granularity, WandbLogger
from torch.utils.data import DistributedSampler
from typing import *
from topologylayer.nn import RipsLayer

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

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

    args = parser.parse_args()
    return args

def get_coordinates(filenames: List[str]):
    """Logic to return coordinates of files"""
    structure: callable = lambda cif: Structure.from_file(cif).cart_coords
    coords_list: List[np.ndarray] = list(map(lambda inp: structure(inp), filenames ))
    return coords_list

@ray.remote
def get_coordinates_mp(filename: str):
    """Logic to return coordinates of files"""
    structure: callable = lambda cif: Structure.from_file(cif).cart_coords
    coords = structure(filename)
    return coords

def persistent_diagram(graph_input_list: List[np.ndarray], maxdim: int):
    assert isinstance(graph_input_list, list), f"graph_input_list must be a type list..."
    Rs_total = list(map(lambda info: ripser.ripser(info, maxdim=maxdim)["dgms"], graph_input_list ))
    return Rs_total

@ray.remote
def persistent_diagram_mp(graph_input: np.ndarray, maxdim: int, tensor: bool=False):
    assert isinstance(graph_input, np.ndarray), f"graph_input must be a type array..."
    #Definition of information has changed from List[np.ndarray] to np.ndarray
    #Multiprocessing changes return value from "List of R" to "one R"
    if not tensor:
        R_total = ripser.ripser(graph_input, maxdim=maxdim)["dgms"]
    else:
        graph_input = torch.from_numpy(graph_input).to("cuda").type(torch.float)
        layer = RipsLayer(graph_input.size(0), maxdim=maxdim)
        R_total = layer(graph_input)
    return R_total

# @dataclasses.dataclass
class PH_Featurizer_Dataset(Dataset):
    def __init__(self, args: argparse.ArgumentParser):
        super().__init__()
        [setattr(self, key, val) for key, val in args.__dict__.items()]
        self.files_to_pg = list(map(lambda inp: os.path.join(self.data_dir, inp), os.listdir(self.data_dir)))
        self.files_to_pg = list(filter(lambda inp: os.path.splitext(inp)[-1] == ".cif", self.files_to_pg ))
        # print(self.files_to_pg)
        self.graph_input_list, self.Rs_total = self.get_values()

    def get_persistent_diagrams(self, ):
        if not self.multiprocessing:
            print("Single CPU Persistent Diagram...")
            if not (os.path.exists(os.path.join(self.save_dir, "PH_" + self.filename)) and os.path.exists(os.path.join(self.save_dir, "coords_" + self.filename))):
                s=time.time()
                graph_input_list = get_coordinates(self.files_to_pg)
                print(cf.on_yellow("Coordinate extraction done!"))
                Rs_total = persistent_diagram(graph_input_list, maxdim)
                print(cf.on_yellow("Persistent diagram extraction done!"))
                e=time.time()
                print(f"{e-s} seconds taken...")
                f = open(os.path.join(self.save_dir, "coords_" + self.filename), "wb")
                pickle.dump(graph_input_list, f)   
                f = open(os.path.join(self.save_dir, "PH_" + self.filename), "wb")
                pickle.dump(Rs_total, f)  
            else:
                f = open(os.path.join(self.save_dir, "coords_" + self.filename), "rb")
                graph_input_list = pickle.load(f) #List of structures: each structure has maxdim PHs
                f = open(os.path.join(self.save_dir, "PH_" + self.filename), "rb")
                Rs_total = pickle.load(f) #List of structures: each structure has maxdim PHs
        else:
            print(f"Multiple CPU Persistent Diagram... with {os.cpu_count()} CPUs")
            if not (os.path.exists(os.path.join(self.save_dir, "PH_" + self.filename)) and os.path.exists(os.path.join(self.save_dir, "coords_" + self.filename))):
                s=time.time()
                futures = [get_coordinates_mp.remote(i) for i in self.files_to_pg] 
                graph_input_list = ray.get(futures) #List of structures: each structure has maxdim PHs
                print(cf.on_yellow("Coordinate extraction done!"))
                maxdims = [self.maxdim] * len(graph_input_list)
                tensor_flags = [self.tensor] * len(graph_input_list)
                futures = [persistent_diagram_mp.remote(i, maxdim, tensor_flag) for i, maxdim, tensor_flag in zip(graph_input_list, maxdims, tensor_flags)] 
                Rs_total = ray.get(futures) #List of structures: each structure has maxdim PHs
                print(cf.on_yellow("Persistent diagram extraction done!"))
                e=time.time()
                print(f"{e-s} seconds taken...")
                f = open(os.path.join(self.save_dir, "coords_" + self.filename), "wb")
                pickle.dump(graph_input_list, f)   
                f = open(os.path.join(self.save_dir, "PH_" + self.filename), "wb")
                pickle.dump(Rs_total, f)   
            else:
                f = open(os.path.join(self.save_dir, "coords_" + self.filename), "rb")
                graph_input_list = pickle.load(f) #List of structures: each structure has maxdim PHs
                f = open(os.path.join(self.save_dir, "PH_" + self.filename), "rb")
                Rs_total = pickle.load(f) #List of structures: each structure has maxdim PHs

        return graph_input_list, Rs_total #List of structures: each structure has maxdim PHs

    def get_values(self, ):
        graph_input_list, Rs_total = self.get_persistent_diagrams()
        return graph_input_list, Rs_total

    def len(self, ):
        return len(self.files_to_pg)

    def get(self, idx):
        graph_input = torch.from_numpy(self.graph_input_list[idx]).type(torch.float)
        Rs = self.Rs_total[idx]
        Rs_dict = dict()
        for i in range(self.maxdim+1):
            Rs_dict[f"ph{i}"] = torch.from_numpy(Rs[i]).type(torch.float)
        return Data(x=graph_input, y=torch.tensor([0.]) ,**Rs_dict)

class PH_Featurizer_DataLoader(abc.ABC):
    """ Abstract DataModule. Children must define self.ds_{train | val | test}. """

    def __init__(self, **dataloader_kwargs):
        super().__init__()
        self.opt = opt = dataloader_kwargs.pop("opt")
        self.dataloader_kwargs = {'pin_memory': opt.pin_memory, 'persistent_workers': opt.num_workers > 0,
                                        'batch_size': opt.batch_size}

        if get_local_rank() == 0:
            self.prepare_data()
            print(f"{get_local_rank()}-th core is parsed!")
#             self.prepare_data(opt=self.opt, data=self.data, mode=self.mode) #torch.utils.data.Dataset; useful when DOWNLOADING!

        # Wait until rank zero has prepared the data (download, preprocessing, ...)
        if dist.is_initialized():
            dist.barrier(device_ids=[get_local_rank()]) #WAITNG for 0-th core is done!
                    
        full_dataset = PH_Featurizer_Dataset(self.opt)
        self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(full_dataset, _get_split_sizes(self.opt.train_frac, full_dataset),
                                                                generator=torch.Generator().manual_seed(42))
    
    def prepare_data(self, ):
        """ Method called only once per node. Put here any downloading or preprocessing """
        full_dataset = PH_Featurizer_Dataset(self.opt)
        full_dataset.get_values()
        print("Preparation is done!")
            
    def train_dataloader(self) -> DataLoader:
        return get_dataloader(self.ds_train, shuffle=True, collate_fn=None, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return get_dataloader(self.ds_val, shuffle=False, collate_fn=None, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return get_dataloader(self.ds_test, shuffle=False, collate_fn=None, **self.dataloader_kwargs)    

if __name__ == "__main__":
    args = get_args()
    ph = PH_Featurizer_Dataset(args)
    print(ph[5])
    dataloader = PH_Featurizer_DataLoader(opt=args)
    print(iter(dataloader.test_dataloader()).next())

    # graph_input_list, Rs_total = ph
    # print(graph_input_list[0], Rs_total[0])

    # python -m tda.ph --save_dir ../temp_save_cif --data_dir ../cif --multiprocessing --tensor --filename temp2.pickle
