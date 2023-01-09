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

[To get PyTorch dataLoader of graph format using <s>PyTorchGeometric</s> PyTorch] This extracts XYZ coordinates and multiprocesses PH; subscriptable by index for Dataset and loadable for DataLoader </br>
<code>
<s>python -m data_utils --psf reference_autopsf.psf --pdb reference_autopsf.pdb --trajs adk.dcd --save_dir . --data_dir /Scr/hyunpark/Monster/vaegan_md_gitlab/data --multiprocessing --filename temp2.pickle</s>
</code>
<code>
python -m main  --which_mode preprocessing --pdb_database /Scr/arango/Sobolev-Hyun/2-MembTempredict/testing/ --save_dir /Scr/hyunpark/Protein-TDA/pickled --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1 
</code>

[To train from saved pickle/dat files] Assuming that pickle/dat files for coordinates, PH and temperature are saved, we can start training neural network model...</br>

<code>
<s>python -m main --ignore_topologicallayer --optimizer torch_adamw --which_mode train --gpu --log --batch_size 8 --epoches 100</s>
</code> 
</br>
<code>
  python -m main --which_mode train --load_ckpt_path /Scr/hyunpark/Protein-TDA/saved --name vit_model --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1
</code>

<br><br> For distributed data parallelization <br>

<code>
<s>python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module main --gpu --log --ignore_topologicallayer --optimizer torch_adam --which_mode train  --batch_size 8 --epoches 100</s>
</code>
</br>
<code>
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module main --which_mode train --name vit_model --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1
</code>
  
<br><br> For DGX-3 submission, assuming submit_local contains proper job scheduling...<br>
<code>
  <s>submit_local main.py main main dgx-test</s>
</code>
<br><br> To continue training...<br>
<code>
python -m main --which_mode train --name vit_model --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1 --resume
</code>

<br><br> To infer on all data...<br>
<code>
python -m main --which_mode infer --name vit_model --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1 --resume
</code>

<br><br> To infer PDB patches' temperatures inside e.g. **inference_save/T.123** directory, and to save inside **inference_save** directory as pickles<br>
<code>
python -m main --which_mode infer_custom --name convnext_model --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 512 --ce_re_ratio 1 0.1 --backbone convnext --resume --pdb_database inference_folder --save_dir inference_save --search_temp 123
</code>
