import persim
import ripser
import MDAnalysis as mda
import argparse
from typing import *
import functools
import itertools 
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.align import AlignTraj
from MDAnalysis import transformations
import functools
import numpy as np
import torch
import multiprocessing as mp
import time
import ray
import os

parser = argparse.ArgumentParser()
parser.add_argument('--pdbs', nargs="*", type=str, default="3CLN")
parser.add_argument('--data_dir', type=str, default="/Scr/hyunpark/Monster/vaegan_md_gitlab/data")
parser.add_argument('--pdb', type=str, default=None)
parser.add_argument('--psf', type=str, default=None)
parser.add_argument('--trajs', nargs="*", type=str, default=None) #List of dcds
parser.add_argument('--filename', type=str, default="default.npy") #List of dcds
parser.add_argument('--selections', nargs="*", type=str, default="backbone and segid A")
parser.add_argument('--get_cartesian', type=bool, default=True, help="MDA data extraction")
parser.add_argument('--multip', action="store_true", help="enable multiprocessing?")

def persistent_diagram(information: Union[np.ndarray, List[np.ndarray]]):
    Rs = list(map(lambda info: ripser.ripser(info)["dgms"][1], information ))
    return Rs

@ray.remote
def persistent_diagram_mp(information: Union[np.ndarray, List[np.ndarray]]):
    #Definition of information has changed from List[np.ndarray] to np.ndarray
    #Multiprocessing changes return value from "List of R" to "one R"
    R = ripser.ripser(information)["dgms"][1]
    return R

class PersistentHomology(object):
    def __init__(self, args: argparse.ArgumentParser):
#         pdbs = args.pdbs
#         selections = args.selections
#         get_cartesian = args.get_cartesian
        
        [setattr(self, key, val) for key, val in args.__dict__.items()]
        
#         self.pdbs = pdbs
#         self.selections = selections
#         self.get_cartesian = get_cartesian

    @staticmethod
    def load_mmtf(pdbs: List[str]):
        us = list(map(lambda pdb: mda.fetch_mmtf(pdb), pdbs)) #List of universes
        return us
    
    @staticmethod
    def traj_preprocessing(prot_traj, prot_ref, align_selection):
        box_dim = prot_traj.trajectory.ts.dimensions 
#         print(box_dim, prot_traj.atoms.positions, prot_ref.atoms.positions, align_selection)
        transform = transformations.boxdimensions.set_dimensions(box_dim)
        prot_traj.trajectory.add_transformations(transform)
        AlignTraj(prot_traj, prot_ref, select=align_selection, in_memory=True).run()
        return prot_traj
    
    @staticmethod
    def load_traj(data_dir: str, pdb: str, psf: str, trajs: List[str], selections: List[str]):
        assert (pdb is not None) or (psf is not None), "At least either PDB of PSF should be provided..."
        assert trajs is not None, "DCD(s) must be provided"
        top = pdb if (pdb is not None) else psf
        top = os.path.join(data_dir, top)
        trajs = list(map(lambda inp: os.path.join(data_dir, inp), trajs ))
        universe = mda.Universe(top, *trajs)
        reference = mda.Universe(top)
        print("MDA Universe is created")
#         print(top, universe,reference)
        selections = selections[0]
        prot_traj = PersistentHomology.traj_preprocessing(universe, reference, selections)
        print("Aligned MDA Universe is RETURNED!")

        return reference, prot_traj #universes
    
    @staticmethod
    def get_atomgroups(mda_universes: List[mda.Universe], selections: List[str] = "backbone and segid A"):
        if isinstance(mda_universes, list):
            if len(selections) == 1: 
                print("there is one atom selection criteria...; Applying the same selection for all molecules!")
                selections = selections * len(mda_universes) #proliferate selection of elements same as pdb lists 
            else: 
                print(selections, mda_universes)
                assert len(selections) == len(mda_universes), "number of Universes and selections should match!"

            ags = list(map(lambda u, sel: u.select_atoms(sel), mda_universes, selections ))
            return ags
        else:
            ag = [mda_universes.select_atoms(selections[0])] #Make it into a List of AtomGroup
            return ag #List[one_AtomGroup]
    
    @staticmethod
    def birth_and_death(mda_universes_or_atomgroups: Union[List[mda.Universe], List[mda.AtomGroup]], get_cartesian: bool = True, 
                        selections: List[str] = "backbone and segid A", traj_flag: bool=False, multip: bool=False):
        if isinstance(mda_universes_or_atomgroups[0], mda.Universe):
            ags = PersistentHomology.get_atomgroups(mda_universes_or_atomgroups, selections)
        else:
            ags = mda_universes_or_atomgroups #List of AtomGroups 

        if get_cartesian and len(ags) >= 2 and not traj_flag:
            information = list(map(lambda ag: ag.atoms.positions, ags )) #List of atomgroup positions
        elif get_cartesian and len(ags) == 1 and traj_flag:
            prot_traj = ags[0].universe #back to universe
            coords = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                   prot_traj.atoms.select_atoms(selections[0])).run().results['timeseries'] #B,L,3
            information = torch.from_numpy(coords).unbind(dim=0) #List of (L,3) Tensors
            information = list(map(lambda inp: inp.detach().cpu().numpy(), information )) #List of (L,3) Arrays
#             information = np.split(coords, indices_or_sections=coords.shape[0], axis=0) #[(L,3)] * B
        else:
            raise NotImplementedError("Not implemented for non-positional information!")
        
#         print(information)
        print("Ripser for DGMS...")
        if not multip:
            print("Normal Ripser...")
            Rs = persistent_diagram(information)
#         else:
#             return information
        else:
            print("Multiprocessing Ripser...")
#             with mp.Pool() as pool:
#                 Rs = pool.map(persistent_diagram_mp, information)
            futures = [persistent_diagram_mp.remote(i) for i in information]
            Rs = ray.get(futures)
        return Rs


    @staticmethod
    def get_wassersteins(ripser_objects: List[ripser.ripser], traj_flag: bool=False):
        if not traj_flag:
            assert len(ripser_objects) >= 2, "for Wasserstein, it must have more than two Ripser objects!"
            ripser_pair = list(itertools.combinations(ripser_objects, 2))
            wdists = list(map(lambda pair: persim.wasserstein(*pair), ripser_pair ))
            return wdists
        else:
            wdists = list(map(lambda pair: functools.partial(persim.wasserstein, dgm1=ripser_objects[0])(dgm2 = pair), ripser_objects[slice(1, None)] ))
            return wdists

    @staticmethod
    def get_wassersteins_pairwise(ripser_objects: List[ripser.ripser]):
        firsts = ripser_objects[slice(0,-1)]
        seconds = ripser_objects[slice(1,None)]
        wdists = list(map(lambda first, second: persim.wasserstein(dgm1=first, dgm2=second), firsts, seconds ))
        return wdists
        
    @property
    def calculate_wdists_pdbs(self, ):
        s = time.time()
        print(self.__dict__)
        
        us = self.load_mmtf(self.pdbs)
        ags = self.get_atomgroups(us, self.selections)
        Rs = self.birth_and_death(ags, self.get_cartesian, self.selections)
        wdists = self.get_wassersteins(Rs)
        
        e = time.time()
        print(f"Took {e-s} seconds...")
        print("Done!")
        return us, ags, Rs, wdists
    
    @property
    def calculate_wdists_trajs(self, ):
        s = time.time()
        print(self.__dict__)
        
        reference, prot_traj = self.load_traj(self.data_dir, self.pdb, self.psf, self.trajs, self.selections)
        ags_ref = self.get_atomgroups(reference, self.selections)
        ags_trajs = self.get_atomgroups(prot_traj, self.selections)
        traj_flag = (self.trajs is not None)
        
        if os.path.exists(os.path.join(self.data_dir, filename)):
            Rs = np.load(os.path.join(self.data_dir, filename))
            Rs_ = torch.from_numpy(Rs).unbind(dim=0)
            Rs = list(map(lambda inp: inp.detach().cpu().numpy(), Rs_))
            print(f"Loading saved diagrams from {filename}...")
        else:
            Rs_ref = self.birth_and_death(ags_ref, self.get_cartesian, self.selections, traj_flag)
            print("Rs for Ref done...")
            Rs_trajs = self.birth_and_death(ags_trajs, self.get_cartesian, self.selections, traj_flag, self.multip)
            print("Rs for Trajs done...")
            Rs = Rs_ref + Rs_trajs 
            np.save(os.path.join(self.data_dir, filename), Rs)
            
        wdists = self.get_wassersteins(Rs, traj_flag)
        wdist_pairs = self.get_wassersteins_pairwise(Rs)
        
        e = time.time()
        print(f"Took {e-s} seconds...")
        print("Done!")
        return [reference, prot_traj], [ags_ref, ags_trajs], Rs, (wdists, wdist_pairs)
    
if __name__ == "__main__":
    args = parser.parse_args()
    ph = PersistentHomology(args)
    _, _, Rs, wdists = ph.calculate_wdists_trajs
    print(wdists[0], wdists[1])

    """
    s = time.time()
    print(ph.__dict__)
    reference, prot_traj = ph.load_traj(ph.pdb, ph.psf, ph.trajs, ph.selections)
    ags_ref = ph.get_atomgroups(reference, ph.selections)
    ags_trajs = ph.get_atomgroups(prot_traj, ph.selections)
    traj_flag = (ph.trajs is not None)
    Rs_ref = ph.birth_and_death(ags_ref, ph.get_cartesian, ph.selections, traj_flag)
    print("Rs for Ref done...")
    
    if not ph.multip:
        Rs_trajs = ph.birth_and_death(ags_trajs, ph.get_cartesian, ph.selections, traj_flag, ph.multip)
        print("Rs for Trajs done...")
    else:
        information = ph.birth_and_death(ags_trajs, ph.get_cartesian, ph.selections, traj_flag, ph.multip)
#         print(information)
        print("Multiprocessing Ripser...")
#         time.sleep(10)
#         with mp.Pool() as pool:
#             Rs_trajs = pool.map(persistent_diagram_mp, information)
        futures = [persistent_diagram_mp.remote(i) for i in information]
        Rs_trajs = ray.get(futures)
        print("Rs for Trajs done...")
        
    Rs = Rs_ref + Rs_trajs 
    wdists = ph.get_wassersteins(Rs, traj_flag)
    
    print(wdists)
    e = time.time()
    print(f"Took {e-s} seconds...")
    print("Done!")
    """
