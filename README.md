# Protein-TDA

[For multiple PDBs] To get Wasserstein distances, enter below code in bash shell... </br>
<code>
python -m test_persistent_homology --pdbs 3CLN 1CFD 1L7Z --selections "backbone and segid A" 
</code>

[For a reference PDB and trajectories with single-CPU] To get Wasserstein distances, enter below code in bash shell... </br>
<code>
python -m test_persistent_homology --psf /Scr/hyunpark/Monster/vaegan_md_gitlab/data/reference_autopsf.psf --pdb /Scr/hyunpark/Monster/vaegan_md_gitlab/data/reference_autopsf.pdb --trajs /Scr/hyunpark/Monster/vaegan_md_gitlab/data/adk.dcd --selection "backbone"
</code>

[For a reference PDB and trajectories with multi-CPUs] To get Wasserstein distances, enter below code in bash shell... </br>
<code>
python -m test_persistent_homology --psf /Scr/hyunpark/Monster/vaegan_md_gitlab/data/reference_autopsf.psf --pdb /Scr/hyunpark/Monster/vaegan_md_gitlab/data/reference_autopsf.pdb --trajs /Scr/hyunpark/Monster/vaegan_md_gitlab/data/adk.dcd --selection "backbone" --multip
</code>

#### (single CPU) 228.51532316207886 seconds... versus (multi CPU) 14.778266191482544 seconds... <br>
