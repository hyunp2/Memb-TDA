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

parser = argparse.ArgumentParser()
parser.add_argument('--pdbs', nargs="*", type=str, default="3CLN")
parser.add_argument('--pdb', type=str, default=None)
parser.add_argument('--psf', type=str, default=None)
parser.add_argument('--trajs', nargs="*", type=str, default=None) #List of dcds
parser.add_argument('--selections', nargs="*", type=str, default="backbone and segid A")
parser.add_argument('--get_cartesian', type=bool, default=True, help="MDA data extraction")

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
        print(box_dim, prot_traj, prot_ref, align_selection)
        transform = transformations.boxdimensions.set_dimensions(box_dim)
        prot_traj.trajectory.add_transformations(transform)
        AlignTraj(prot_traj, prot_ref, select=align_selection, in_memory=True).run()
        return prot_traj
    
    @staticmethod
    def load_traj(pdb: str, psf: str, trajs: List[str], selections: List[str]):
        assert (pdb is not None) or (psf is not None), "At least either PDB of PSF should be provided..."
        assert trajs is not None, "DCD(s) must be provided"
        top = pdb if (pdb is not None) else psf
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
    def birth_and_death(mda_universes_or_atomgroups: Union[List[mda.Universe], List[mda.AtomGroup]], get_cartesian: bool = True, selections: List[str] = "backbone and segid A", traj_flag: bool=False):
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
            information = np.split(coords, indices_or_sections=coords.shape[0], axis=0) #[(L,3)] * B
        else:
            raise NotImplementedError("Not implemented for non-positional information!")

        Rs = list(map(lambda info: ripser.ripser(info)["dgms"][1], information ))
        return Rs

    @staticmethod
    def get_wassersteins(ripser_objects: List[ripser.ripser], traj_flag: bool=False):
        if not traj_flag:
            assert len(ripser_objects) >= 2, "for Wasserstein, it must have more than two Ripser objects!"
            ripser_pair = list(itertools.combinations(ripser_objects, 2))
            wdists = list(map(lambda pair: persim.wasserstein(*pair), ripser_pair ))
            return wdists
        else:
            wdists = list(map(lambda pair: functols.partial(persim.wasserstein, dgm1=ripser_objects[0])(dgm2 = pair), ripser_objects[slice(1, None)] ))
            return wdists
            
    @property
    def calculate_wdists_pdbs(self, ):
        us = self.load_mmtf(self.pdbs)
        ags = self.get_atomgroups(us, self.selections)
        Rs = self.birth_and_death(ags, self.get_cartesian, self.selections)
        wdists = self.get_wassersteins(Rs)
        return us, ags, Rs, wdists
    
    @property
    def calculate_wdists_trajs(self, ):
        reference, prot_traj = self.load_traj(self.pdb, self.psf, self.trajs, self.selections)
        ags_ref = self.get_atomgroups(reference, self.selections)
        ags_trajs = self.get_atomgroups(prot_traj, self.selections)
        traj_flag = (self.trajs is not None)
        Rs_ref = self.birth_and_death(ags_ref, self.get_cartesian, self.selections, traj_flag)
        Rs_trajs = self.birth_and_death(ags_trajs, self.get_cartesian, self.selections, traj_flag)
        Rs = Rs_ref + Rs_trajs 
        wdists = self.get_wassersteins(Rs, traj_flag)
        return [reference, prot_traj], [ags_ref, ags_trajs], Rs, wdists
    
if __name__ == "__main__":
    args = parser.parse_args()
#     us = load_mmtf(args.pdbs)
#     ags = get_atomgroups(us, args.selections)
#     Rs = birth_and_death(ags, args.get_cartesian, args.selections)
#     wdists = get_wassersteins(Rs)
    ph = PersistentHomology(args)
    us, ags, Rs, wdists = ph.calculate_wdists_trajs
    print(Rs, wdists)

    
    
# u_open = mda.fetch_mmtf('3CLN')
# u_inter = mda.fetch_mmtf('1CFD')
# u_closed = mda.fetch_mmtf('1L7Z')
# Pa1 = u_open.select_atoms("backbone and segid A")
# Pa2 = u_inter.select_atoms("backbone and segid A")
# Pa3 = u_closed.select_atoms("backbone and segid A")

# R1 = ripser.ripser(Pa1.atoms.positions)['dgms'][1]
# R2 = ripser.ripser(Pa2.atoms.positions)['dgms'][1]
# R3 = ripser.ripser(Pa3.atoms.positions)['dgms'][1]

# print(persim.wasserstein(R1, R2))
# print(persim.wasserstein(R2, R3))
# print(persim.wasserstein(R1, R3))
