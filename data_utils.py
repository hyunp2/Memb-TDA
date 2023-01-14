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
import glob
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
from torch.utils.data import DistributedSampler, Subset
from torch._utils import _accumulate
from typing import *
from topologylayer.nn import RipsLayer, AlphaLayer
import gc
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.align import AlignTraj
from MDAnalysis import transformations
from math_utils import wasserstein
import mdtraj

#https://colab.research.google.com/github/shizuo-kaji/TutorialTopologicalDataAnalysis/blob/master/TopologicalDataAnalysisWithPython.ipynb#scrollTo=Y6fj2UqWHPbs
##ABOVE: cubicle-Ripser
__all__ = ["PH_Featurizer_Dataset", "PH_Featurizer_DataLoader"]

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import torch
import numpy as np

def remove_filler(dgm, val=np.inf):
    """
    remove filler rows from diagram
    """
    inds = (dgm[:,0] != val)
    return dgm[inds,:]

def remove_zero_bars(dgm):
    """
    remove zero bars from diagram
    """
    inds = dgm[:,0] != dgm[:,1]
    return dgm[inds,:]

def remove_infinite_bars(dgm, issub):
    """
    remove infinite bars from diagram
    """
    if issub:
        inds = dgm[:, 1] != np.inf
        return dgm[inds,:]
    else:
        inds = dgm[:, 1] != -np.inf
        return dgm[inds,:]

def order_dgm(dgm):
    dgm = remove_zero_bars(dgm)
    dgm = remove_infinite_bars(dgm, True)
    order_data = np.abs(dgm[:,1] - dgm[:,0]) # abs(death - birth)
    args = np.argsort(order_data, axis=0) #Largest to smallest length
    dgm = dgm[args]
    return dgm
    
def _get_split_sizes(train_frac: float, full_dataset: Dataset) -> Tuple[int, int, int]:
    """DONE: Need to change split schemes!"""
    len_full = len(full_dataset)
    len_train = int(len_full * train_frac)
    len_test = int(0.1 * len_full)
    len_val = len_full - len_train - len_test
    return len_train, len_val, len_test  
  
def get_dataloader(dataset: Dataset, shuffle: bool, collate_fn: callable=None, **kwargs):
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
    loader = DataLoader(dataset, shuffle=(shuffle and sampler is None), sampler=sampler, collate_fn=collate_fn, **kwargs)
    return loader

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
    assert isinstance(graph_input, (torch.Tensor, np.ndarray)), f"graph_input must be a type array..."
    #Definition of information has changed from List[np.ndarray] to np.ndarray
    #Multiprocessing changes return value from "List of R" to "one R"
    graph_input = graph_input.detach().cpu().numpy() if isinstance(graph_input, torch.Tensor) else np.array(graph_input)
    if not tensor:
        R_total = ripser.ripser(graph_input, maxdim=maxdim)["dgms"]
    else:
#         graph_input = torch.from_numpy(graph_input).to("cuda").type(torch.float)
#         layer = RipsLayer(graph_input.size(0), maxdim=maxdim)
        layer = AlphaLayer(maxdim=maxdim)
        layer.cuda()
        R_total = layer(graph_input)
    return R_total

# @ray.remote
def persistent_diagram_tensor(graph_input: torch.Tensor, maxdim: int):
    assert isinstance(graph_input, torch.Tensor), f"graph_input must be a type array..."
    #Definition of information has changed from List[np.ndarray] to np.ndarray
    #Multiprocessing changes return value from "List of R" to "one R"
#     layer = RipsLayer(graph_input.size(0), maxdim=maxdim)
    layer = AlphaLayer(maxdim=maxdim)
    layer #.to(torch.cuda.current_device())
    R_total = layer(graph_input)
    return R_total

def traj_preprocessing(prot_traj, prot_ref, align_selection):
    if (prot_traj.trajectory.ts.dimensions is not None): 
        box_dim = prot_traj.trajectory.ts.dimensions
    else:
        box_dim = np.array([1,1,1,90,90,90])
#         print(box_dim, prot_traj.atoms.positions, prot_ref.atoms.positions, align_selection)
    transform = transformations.boxdimensions.set_dimensions(box_dim)
    prot_traj.trajectory.add_transformations(transform)
    #AlignTraj(prot_traj, prot_ref, select=align_selection, in_memory=True).run()
    return prot_traj

@ray.remote
def mdtraj_loading(root: str, topology: str):
    return mdtraj.load(os.path.join(root, topology)).xyz[0]

# @ray.remote
def images_processing(Images_total: dict, make_more_channels=False, index=None):
    img0 = Images_total[0][index] #(H,W)
    img1 = Images_total[1][index] #(H,W)
#     img2 = Images_total[2][index] #(H,W)
    
    from PIL import Image
    from torchvision.transforms import ToTensor

    newsize = (128, 128)
    # imgs0 = list(map(lambda inp: inp, ToTensor(Image.fromarray(inp).resize(newsize)), img0 )) #List[Tensor(PIL.Image)]; ONLY can parse ONE image at a time!
    # imgs1 = list(map(lambda inp: inp, ToTensor(Image.fromarray(inp).resize(newsize)), img1 )) #List[Tensor(PIL.Image)]
    # imgs2 = list(map(lambda inp: inp, ToTensor(Image.fromarray(inp).resize(newsize)), img2 )) #List[Tensor(PIL.Image)]
    imgs0, imgs1 = list(map(lambda inp: ToTensor()(Image.fromarray(inp).resize(newsize, resample=Image.LANCZOS)), (img0, img1) )) #(1HW ea.)
    imgs2 = (imgs0 + imgs1) / 2 #(1,H,W)
    # imgs0, imgs1, imgs2 = torch.tensor(imgs0), torch.tensor(imgs1), torch.tensor(img2) #(batch, H, W)

    imgs = torch.cat([imgs0, imgs1, imgs2], dim=0) #(3HW)

    if make_more_channels:
        img01 = (imgs0 - imgs1).abs()
        img02 = (imgs0 - imgs2).abs()
        img12 = (imgs1 - imgs2).abs()
    else:
        return imgs
    
def sanity_check_mdtraj(directory: str, pdbs: List[str]) -> List[int]:
    valid_pdbs = []
    valid_pdbs = list(filter(lambda inp: os.stat(os.path.join(directory, inp)).st_size != 0, pdbs ))
    print(cf.on_red(f"{len(pdbs) - len(valid_pdbs)} PDBs have been removed!"))
    return valid_pdbs
    
# @dataclasses.dataclass
class PH_Featurizer_Dataset(Dataset):
    def __init__(self, args: argparse.ArgumentParser, directories: str=None, image_stats: collections.namedtuple=None):
        super().__init__()
        [setattr(self, key, val) for key, val in args.__dict__.items()]
#         self.files_to_pg = list(map(lambda inp: os.path.join(self.data_dir, inp), os.listdir(self.data_dir)))
#         self.files_to_pg = list(filter(lambda inp: os.path.splitext(inp)[-1] == ".cif", self.files_to_pg ))

        if self.trajs is not None:
            #args.trajs is now DEPRECATED!
            #Do not use args.data_dir either!
            
            assert self.trajs is None, "DCD trajectories is NOT passed and MDAnalysis is NOT used!"
            self.reference, self.prot_traj = self.load_traj(data_dir=self.data_dir, pdb=self.pdb, psf=self.psf, trajs=self.trajs, selection=self.atom_selection)
            self.coords_ref, self.coords_traj = self.get_coordinates_for_md(self.reference), self.get_coordinates_for_md(self.prot_traj)
        else:
            #pdb_database: /Scr/arango/Sobolev-Hyun/2-MembTempredict/testing/
            self.coords_ref = []
            self.coords_traj = []
            self.temperatures = []
            
            directories = sorted(glob.glob(os.path.join(self.pdb_database, "T.*"))) if directories is None else directories #For a specific directory during inference!
            self.image_stats = image_stats
            if not (os.path.exists(os.path.join(self.save_dir, "PH_" + self.filename)) 
                    and os.path.exists(os.path.join(self.save_dir, "coords_" + self.filename)) 
                    and os.path.exists(os.path.join(self.save_dir, "Im_" + self.filename)) 
                    and os.path.exists(os.path.join(self.save_dir, "temperature_" + self.filename)) ):
                for direct in directories:
                    pdbs = os.listdir(direct) #all PDBs inside a directory
    #                 print(os.path.join(direct,pdbs[0]))
    #                 univ_pdbs = [mda.Universe(os.path.join(direct,top)) for top in pdbs] #List PDB universes
    #                 self.coords_traj += [self.get_coordinates_for_md(univ_pdb)[0] for univ_pdb in univ_pdbs]
                    valid_pdbs = sanity_check_mdtraj(direct, pdbs) #List[str]
                    self.coords_traj += [ mdtraj.load(os.path.join(direct,top)).xyz[0] for top in valid_pdbs ] if not self.multiprocessing else ray.get([mdtraj_loading.remote(root, top) for root, top in zip([direct]*len(pdbs), valid_pdbs)])
                    self.temperatures += [int(os.path.split(direct)[1].split(".")[1])] * len(valid_pdbs)
                f = open(os.path.join(self.save_dir, "temperature_" + self.filename), "wb")
                pickle.dump(self.temperatures, f)   
                assert len(self.coords_traj) == len(self.temperatures), "coords traj and temperatures must have the same data length..."
                print(cf.on_blue("STEP 0: Saved temperature!"))
            else:
                f = open(os.path.join(self.save_dir, "temperature_" + self.filename), "rb")
                self.temperatures = pickle.load(f)
                print(cf.on_blue("STEP0: Loaded temperature!"))
                
#         self.graph_input_list, self.Rs_total, self.Rs_list_tensor = self.get_values()
        self.graph_input_list, self.Rs_total, self.Images_total = self.get_values()

        del self.coords_ref
        del self.coords_traj
        gc.collect()
        
    def get_persistent_diagrams(self, ):
        if not self.multiprocessing:
            print("Single CPU Persistent Diagram...")
            if not (os.path.exists(os.path.join(self.save_dir, "PH_" + self.filename)) and os.path.exists(os.path.join(self.save_dir, "coords_" + self.filename))):
                s=time.time()
#                 graph_input_list = get_coordinates(self.files_to_pg)
                graph_input_list = self.coords_ref + self.coords_traj
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
            if not (os.path.exists(os.path.join(self.save_dir, "PH_" + self.filename)) 
                    and os.path.exists(os.path.join(self.save_dir, "coords_" + self.filename)) 
                    and os.path.exists(os.path.join(self.save_dir, "Im_" + self.filename)) 
                    and os.path.exists(os.path.join(self.save_dir, "temperature_" + self.filename)) ):
                s=time.time()
#                 futures = [get_coordinates_mp.remote(i) for i in self.files_to_pg] 
#                 graph_input_list = ray.get(futures) #List of structures: each structure has maxdim PHs
                graph_input_list = self.coords_ref + self.coords_traj
                graph_input_list = list(map(lambda inp: torch.tensor(inp), graph_input_list )) #List of (L,3) Arrays
                f = open(os.path.join(self.save_dir, "coords_" + self.filename), "wb")
                pickle.dump(graph_input_list, f) 
                print(cf.on_yellow("STEP 1: Coordinate extraction done!"))
                
                maxdims = [self.maxdim] * len(graph_input_list)
                tensor_flags = [self.tensor] * len(graph_input_list)
                futures = [persistent_diagram_mp.remote(i, maxdim, tensor_flag) for i, maxdim, tensor_flag in zip(graph_input_list, maxdims, tensor_flags)] 
                Rs_total = ray.get(futures) #List of structures: each structure has maxdim PHs
                f = open(os.path.join(self.save_dir, "PH_" + self.filename), "wb")
                pickle.dump(Rs_total, f)   
                print(cf.on_yellow("STEP 2: Persistent diagram extraction done!"))

                images_total = list(zip(*Rs_total))
                print(len(images_total) , (self.maxdim + 1))
                assert len(images_total) == (self.maxdim + 1), "images_total must be the same as maxdim!"
                pers = persim.PersistenceImager(pixel_size=0.01) #100 by 100 image
                pers_images_total = collections.defaultdict(list)
                
                PREPROCESS_FLAG = self.which_mode == "preprocessing"
                
                for i, img in enumerate(images_total):
#                     img = list(map(lambda inp: torch.from_numpy(inp), img))
                    img = list(map(order_dgm, img)) #list of Hi 
#                     img = list(map(lambda inp: inp.detach().cpu().numpy(), img))
                    pers.fit(img)
                    if self.image_stats is None:
                        bmax, pmax = pers.birth_range[1], pers.pers_range[1]  
                    elif self.image_stats is not None and i == 0:
                        bmax, pmax = self.image_stats.bmax0, self.image_stats.pmax0
                    elif self.image_stats is not None and i == 1:
                        bmax, pmax = self.image_stats.bmax1, self.image_stats.pmax1
                    if PREPROCESS_FLAG:
                        print("Preprocessing BD: ", i, bmax, pmax)
                    pers.birth_range = (0, bmax+0.5)
                    pers.pers_range = (0, pmax+0.5)
                    img_list = pers.transform(img, n_jobs=-1)
                    temp = np.stack(img_list, axis=0)
                    if self.image_stats is None:
                        mins, maxs = temp.min(), temp.max()
                    elif self.image_stats is not None and i == 0:
                        mins, maxs = self.image_stats.mins0, self.image_stats.maxs0
                    elif self.image_stats is not None and i == 1:
                        mins, maxs = self.image_stats.mins1, self.image_stats.maxs1
                    if PREPROCESS_FLAG:
                        print("Preprocessing mimmax: ", i, mins, maxs)
                    img_list = list(map(lambda inp: (inp - mins) / (maxs - mins), img_list )) #range [0,1]
                    pers_images_total[i] += img_list
                Images_total = pers_images_total
                f = open(os.path.join(self.save_dir, "Im_" + self.filename), "wb")
                pickle.dump(pers_images_total, f)   
                print(cf.on_yellow("STEP 3: Persistent image extraction done!"))
                
                pbar = tqdm.tqdm(range(len(Images_total[0])))
                imgs = [images_processing(Images_total, index=ind) for ind in pbar]
                Processed_images_total = imgs
                f = open(os.path.join(self.save_dir, "ProcessedIm_" + self.filename), "wb")
                pickle.dump(imgs, f)
                e=time.time()
                print(cf.on_yellow("STEP 4: PIL resized images done!"))

                print(f"{e-s} seconds taken...")
                
                print(cf.on_yellow("STEP 5: Coords and PH and Images and Resized files saved!"))
  
#                 if not self.preprocessing_only: Rs_list_tensor = list(map(alphalayer_computer_coords, graph_input_list, maxdims ))
            else:
                f = open(os.path.join(self.save_dir, "coords_" + self.filename), "rb")
                graph_input_list = pickle.load(f) #List of structures: each structure has maxdim PHs
                graph_input_list = list(map(lambda inp: torch.tensor(inp), graph_input_list )) #List of (L,3) Arrays
                f = open(os.path.join(self.save_dir, "PH_" + self.filename), "rb")
                Rs_total = pickle.load(f) #List of structures: each structure has maxdim PHs
                maxdims = [self.maxdim] * len(graph_input_list)
#                 if not self.preprocessing_only: Rs_list_tensor = list(map(alphalayer_computer_coords, graph_input_list, maxdims ))
#                 f = open(os.path.join(self.save_dir, "Im_" + self.filename), "rb")
#                 Images_total = pickle.load(f) #List of structures: each structure has maxdim PHs #######IGNORE!
                f = open(os.path.join(self.save_dir, "ProcessedIm_" + self.filename), "rb")
                Processed_images_total = pickle.load(f) #List of structures: each structure has maxdim PHs
                
        if self.preprocessing_only or self.ignore_topologicallayer:
            return graph_input_list, Rs_total, Processed_images_total #None #List of structures: each structure has maxdim PHs
        else:
            return graph_input_list, Rs_total, Processed_images_total #Rs_list_tensor #List of structures: each structure has maxdim PHs

    def get_values(self, ):
        graph_input_list, Rs_total, Images_total = self.get_persistent_diagrams()
        return graph_input_list, Rs_total, Images_total

    def len(self, ):
        return len(self.graph_input_list)

    def get(self, idx):
        if self.preprocessing_only:
            raise NotImplementedError("Get item method is not available with preprocessing_only option!")
#         graph_input = torch.from_numpy(self.graph_input_list[idx]).type(torch.float)
#         graph_input = self.graph_input_list[idx].type(torch.float)
        
#         if self.ignore_topologicallayer:
#             Rs = self.Rs_total[idx]
#             Rs_dict = dict()
#             for i in range(self.maxdim+1):
#                 Rs_dict[f"ph{i}"] = torch.from_numpy(Rs[i]).type(torch.float)
#         else:
#             Rs = list(self.Rs_list_tensor[idx])
#             Rs_dict = dict()
#     #         Rs_list_tensor = list(persistent_diagram_tensor(graph_input, maxdim=self.maxdim))
#             del Rs[0] #Remove H0
#             for i in range(1, self.maxdim+1):
#                 Rs_dict[f"ph{i}"] = order_dgm(Rs[i-1]) #ordered!
#         img_temp = []
#         img = np.stack([self.Images_total[i][idx] for i in range(len(self.Images_total))], axis=0) #(2,H,W)
#         img = np.concatenate((img, 0.5*img[:1, ...] + 0.5*img[1:2, ...]), axis=0) #->(3,H,W)
        img = self.Images_total[idx] #(3,H,W)
        img = torch.from_numpy(img).type(torch.float) if isinstance(img, np.ndarray) else img.type(torch.float) #pin_memory for CPU tensors!
        temps = torch.tensor(self.temperatures).view(-1,1).to(img)[idx]
#         return {"Coords": Data(x=graph_input, y=torch.tensor([0.])), "PH": Data(x=Rs_dict["ph1"], **Rs_dict)}
        return {"PH": img, "temp": temps}

    def load_traj(self, data_dir: str, pdb: str, psf: str, trajs: List[str], selection: str):
        assert (pdb is not None) or (psf is not None), "At least either PDB of PSF should be provided..."
        assert trajs is not None, "DCD(s) must be provided"
        top = pdb if (pdb is not None) else psf
        top = os.path.join(data_dir, top)
        trajs = list(map(lambda inp: os.path.join(data_dir, inp), trajs ))
        universe = mda.Universe(top, *trajs)
        reference = mda.Universe(top)
        print("MDA Universe is created")
    #         print(top, universe,reference)
        #prot_traj = traj_preprocessing(universe, reference, selection)
        prot_traj = universe
        print("Aligned MDA Universe is RETURNED!")

        return reference, prot_traj #universes

    def get_coordinates_for_md(self, mda_universes_or_atomgroups: mda.AtomGroup):
        ags = mda_universes_or_atomgroups #List of AtomGroups 
        assert isinstance(ags, (mda.AtomGroup, mda.Universe)), "mda_universes_or_atomgroups must be AtomGroup or Universe!"

        prot_traj = ags.universe if hasattr(ags, "universe") else ags #back to universe
        coords = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                               prot_traj.atoms.select_atoms(self.atom_selection)).run().results['timeseries'] #B,L,3
        information = torch.from_numpy(coords).unbind(dim=0) #List of (L,3) Tensors
#         information = list(map(lambda inp: inp.detach().cpu().numpy(), information )) #List of (L,3) Arrays

        return information
    
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
        if opt.which_mode in ["preprocessing", "train"]:
            self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(full_dataset, _get_split_sizes(self.opt.train_frac, full_dataset),
                                                                generator=torch.Generator().manual_seed(42))
        elif opt.which_mode in ["infer"]:
            """Deterministic Sequential sampling"""
            len_train, len_val, len_test = _get_split_sizes(self.opt.train_frac, full_dataset)
            lengths = [len_train, len_val, len_test]
            indices = torch.arange(sum(lengths)).tolist()
            if sum(lengths) != len(full_dataset):    # type: ignore[arg-type]
                raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
            self.ds_train, self.ds_val, self.ds_test = [Subset(full_dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

    def prepare_data(self, ):
        """ Method called only once per node. Put here any downloading or preprocessing """
        full_dataset = PH_Featurizer_Dataset(self.opt)
        full_dataset.get_values()
        print(cf.on_blue("Preparation is done!"))
            
    def train_dataloader(self) -> DataLoader:
        return get_dataloader(self.ds_train, shuffle=True, collate_fn=None, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return get_dataloader(self.ds_val, shuffle=False, collate_fn=None, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return get_dataloader(self.ds_test, shuffle=False, collate_fn=None, **self.dataloader_kwargs)    

def alphalayer_computer(batches: Data, maxdim: int):
    batches = batches.to(torch.cuda.current_device())
    poses = batches.x
    batch = batches.batch
    phs = []
    pos_list = []
    for b in batch.unique():
        sel = (b == batch)
        pos = poses[sel]
        pos_list.append(pos)
        ph, _ = persistent_diagram_tensor(pos, maxdim=maxdim)
        phs.append(ph)
    return phs #List[List[torch.Tensor]]   

def alphalayer_computer_coords(coords: torch.Tensor, maxdim: int):
#     coords = coords.to(torch.cuda.current_device())
    ph, _ = persistent_diagram_tensor(coords, maxdim=maxdim)
    return ph #List[List[torch.Tensor]]  

if __name__ == "__main__":
    from main import get_args
    args = get_args()
#     ph = PH_Featurizer_Dataset(args)
#     print(ph[5])
#     dataloader = PH_Featurizer_DataLoader(opt=args)
#     print(iter(dataloader.test_dataloader()).next())
#     for i, batches in enumerate(dataloader.train_dataloader()):
# #     batches = iter(dataloader.test_dataloader()).next() #num_nodes, 3
#         phs = alphalayer_computer(batches, ph.maxdim)
#         print(phs)
#         print(f"{i} is done!")
#     maxdims = [ph.maxdim] * batch.unique().size(0)
#     tensor_flags = [ph.tensor] * batch.unique().size(0)
#     futures = [persistent_diagram_tensor.remote(i, maxdim, tensor_flag) for i, maxdim, tensor_flag in zip(pos_list, maxdims, tensor_flags)] 
#     Rs_total = ray.get(futures) #List of structures: each structure has maxdim PHs
#         print(phs)
#     graph_input_list, Rs_total = ph
#     print(graph_input_list[0], Rs_total[0])

    # python -m data_utils --psf reference_autopsf.psf --pdb reference_autopsf.pdb --trajs adk.dcd --save_dir . --data_dir /Scr/hyunpark/Monster/vaegan_md_gitlab/data --multiprocessing --filename temp2.pickle
    
    f = open(os.path.join(args.save_dir, "coords_" + args.filename), "rb")
    graph_input_list = pickle.load(f) #List of structures: each structure has maxdim PHs
#     graph_input_list = list(map(lambda inp: torch.tensor(inp), graph_input_list )) #List of (L,3) Arrays
    maxdims = [args.maxdim] * len(graph_input_list)
    tensor_flags = [args.tensor] * len(graph_input_list)
    futures = [persistent_diagram_mp.remote(i, maxdim, tensor_flag) for i, maxdim, tensor_flag in zip(graph_input_list, maxdims, tensor_flags)] 
    Rs_total = ray.get(futures) #List of structures: each structure has maxdim PHs
    f = open(os.path.join(args.save_dir, "PH_" + args.filename), "wb")
    pickle.dump(Rs_total, f)   
    print(cf.on_yellow("STEP 2: Persistent diagram extraction done!"))
    
#     with open("./pickled/PH_vit.pickle", "rb") as f:
#         Rs_total = pickle.load(f)
#     maxdim = 1
#     images_total = list(zip(*Rs_total))
#     assert len(images_total) == (maxdim + 1), "images_total must be the same as maxdim!"
#     pers = persim.PersistenceImager(pixel_size=0.01) #100 by 100 image
#     pers_images_total = collections.defaultdict(list)
#     for i, img in enumerate(images_total):
# #         img = list(map(lambda inp: torch.from_numpy(inp), img))
#         img = list(map(order_dgm, img)) #list of Hi 
# #         img = list(map(lambda inp: inp.detach().cpu().numpy(), img))
#         pers.fit(img)
#         bmax, pmax = pers.birth_range[1], pers.pers_range[1]
#         pers.birth_range = (0, bmax+0.5)
#         pers.pers_range = (0, pmax+0.5)
#         img_list = pers.transform(img, n_jobs=-1)
#         temp = np.stack(img_list, axis=0)
#         mins, maxs = temp.min(), temp.max()
#         img_list = list(map(lambda inp: (inp - mins) / (maxs - mins), img_list )) #range [0,1]
#         pers_images_total[i] += img_list
#         print(f"br: {bmax} vs pr: {pmax}")
#         print(f"min max {mins}-{maxs}")
#     Images_total = pers_images_total
#     print(Images_total)
#     with open("./pickled/Im_vit.pickle", "wb") as f:
#         pickle.dump(Images_total, f)
    
#     import tqdm
#     with open("./pickled/Im_vit.pickle", "rb") as f:
#         Im_dict = pickle.load(f)
# #     Im_dict_put = ray.put(Im_dict)
#     pbar = tqdm.tqdm(range(len(Im_dict[0])))
#     imgs = [images_processing(Im_dict, index=ind) for ind in pbar]
# #     imgs = ray.get(futures) #List[np.ndarray] of each shape (3,H,W)
#     f = open("./pickled/ProcessedIm_vit.pickle", "wb")
#     pickle.dump(imgs, f)
