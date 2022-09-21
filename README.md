# Protein-TDA

[For multiple PDBs] To get Wasserstein distances, enter below code in bash shell... </br>
<code>
python -m test_persistent_homology --pdbs 3CLN 1CFD 1L7Z --selections "backbone and segid A" 
</code>

[For a reference PDB and trajectories with single-CPU] To get Wasserstein distances, enter below code in bash shell... </br>
<code>
python -m test_persistent_homology --data_dir /Scr/hyunpark/Monster/vaegan_md_gitlab/data --psf reference_autopsf.psf --pdb reference_autopsf.pdb --trajs adk.dcd --selection "backbone"
</code>

[For a reference PDB and trajectories with multi-CPUs] To get Wasserstein distances, enter below code in bash shell... </br>
<code>
python -m test_persistent_homology --data_dir /Scr/hyunpark/Monster/vaegan_md_gitlab/data --psf reference_autopsf.psf --pdb reference_autopsf.pdb --trajs adk.dcd --selection "backbone" --multip --filename TEST.pickle
</code>

#### For now, use below conda environment
<code> /Scr/hyunpark/anaconda3/envs/deeplearning/bin/python </code>

#### (single CPU) 228.51532316207886 seconds... versus (multi CPU) 14.778266191482544 seconds... <br>

[To get PyTorch dataLoader of graph format using PyTorchGeometric] This extracts XYZ coordinates and multiprocesses PH; subscriptable by index for Dataset and loadable for DataLoader </br>
<code>
python -m data_utils --save_dir /Scr/hyunpark/ArgonneGNN/temp_save_cif --data_dir /Scr/hyunpark/ArgonneGNN/cif --multiprocessing --filename temp.pickle
</code>
