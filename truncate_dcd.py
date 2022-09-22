import argparse
import MDAnalysis
import os
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
    
args = get_args()
print(args)
psf = os.path.join(args.data_dir, args.psf)
trajs = list(map(lambda inp: os.path.join(args.data_dir, inp), args.trajs))
print(psf , trajs)
u = MDAnalysis.Universe(psf, *trajs)
protein = u.select_atoms("all")
with MDAnalysis.Writer("test.dcd", u.atoms.n_atoms) as W:
    for ts in u.trajectory[-200:]:
        W.write(protein)
        
        
