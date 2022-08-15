import persim
import ripser
import MDAnalysis as mda
import argparse
from typing import *
import functools
import itertools 

parser = argparse.ArgumentParser()
parser.add_argument('--pdbs', nargs="*", type=str, default="3CLN")
parser.add_argument('--selections', nargs="*", type=str, default="backbone and segid A")
parser.add_argument('--get_cartesian', type=bool, default=True, help="MDA data extraction")

def load_mmtf(pdbs: List[str]):
    us = list(map(lambda pdb: mda.fetch_mmtf(pdb), pdbs)) #List of universes
    return us
  
def get_atomgroups(mda_universes: List[mda.Universe], selections: List[str] = "backbone and segid A"):
    if len(selection) == 1: 
        print("there is one atom selection criteria...")
        selections = selections * len(mda_universes) #proliferate selection of elements same as pdb lists 
    else: 
        assert len(selections) == len(mda_universes), "number of Universes and selections should match!"
        
    ags = list(map(lambda u, sel: u.select_atoms(sel), mda_universes, selections ))
    return ags

def birth_and_death(mda_universes_or_atomgroups: Union[List[mda.Universe], List[mda.AtomGroup]], get_cartesian: bool = True, selections: List[str] = "backbone and segid A"):
    if isinstance(mda_universes_or_atomgroups[0], mda.Universe):
        ags = get_atomgroups(mda_universes_or_atomgroups, selections)
    else:
        ags = mda_universes_or_atomgroups
        
    if get_cartesian:
        information = list(map(lambda ag: ag.atoms.positions, ags )) #List of atomgroup positions
    else:
        raise NotImplementedError("Not implemented for non-positional information!")
    
    Rs = list(map(lambda info: ripser.ripser(info)["dgms"][1], information ))
    return Rs

def get_wassersteins(ripser_objects: List[ripser.ripser]):
    assert len(ripser_objects) >= 2, "for Wasserstein, it must have more than two Ripser objects!"
    ripser_pair = list(itertools.combinations(ripser_objects, 2))
    wdists = list(map(lambda pair: persim.wasserstein(*pair), ripser_pair ))
    return wdists
  
if __name__ == "__main__":
    args = parser.parse_args()
    us = load_mmtf(args.pdbs)
    ags = get_atomgroups(us, args.selections)
    Rs = birth_and_death(ags, args.get_cartesian, args.selections)
    wdists = get_wassersteins(Rs)
    

    
    
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
