# Memb-TDA

Biological membranes play a key role in cellular compartmentalization, structure, and its signaling pathways. At varying temperatures, individual membrane lipids sample from different configurations, frequently leading to higher order phase phenomena. Order parameters of lipid tails are often used as a metric for quantifying phase behavior, however typically only representing bulk ensembles. Here we present a persistent homology-based method for quantifying the structural features of individual and bulk lipids, providing local and contextual information on lipid tail organization. Our method leverages the mathematical machinery of algebraic topology and machine learning to infer temperature dependent structural information of lipids from static coordinates. 
To train our model, we generated multiple molecular dynamic trajectories of DPPC membranes at varying temperatures. A fingerprint was then constructed for each set of lipid coordinates by a persistent homology filtration, in which spheres were grown around the lipid atoms while tracking their intersections. 
The sphere filtration formed a simplicial complex that captures enduring key topological features of the configuration landscape. Following fingerprint extraction for physiologically relevant temperatures, the persistence data were used to train an attention-based neural network for assignment of temperature values to selected membrane regions. Attention is a mathematical algorithm widely used in deep learning, traditionally used for natural language processing (NLP), having shown great promise in biological problems as demonstrated by Alphafold2. The attention mechanism uses global context of the data to predict a desired property; in this case pair-wise interactions between atoms, the global context, and lipid coordinates are used to predict a structural score, indicative of temperature. Our persistence homology-based method captures the local structural effects of lipids adjacent to other membrane constituents, e.g., sterols and proteins, quantifying structural features that facilitate entropically driven membrane organization. 
This topological learning approach predicts membrane temperature values from static coordinates across multiple spatial resolutions.




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

[To get PyTorch dataLoader of graph format using PyTorch] This extracts XYZ coordinates and multiprocesses PH; subscriptable by index for Dataset and loadable for DataLoader </br>

<code>
python -m main  --which_mode preprocessing --pdb_database /Scr/arango/Sobolev-Hyun/2-MembTempredict/testing/ --save_dir /Scr/hyunpark/Protein-TDA/pickled --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1 
</code>

[To train from saved pickle/dat files] Assuming that pickle/dat files for coordinates, PH and temperature are saved, we can start training neural network model...</br>


</br>
<code>
  python -m main --which_mode train --load_ckpt_path /Scr/hyunpark/Protein-TDA/saved --name vit_model --backbone vit --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1
</code>

<br><br> For distributed data parallelization <br>


</br>
<code>
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module main --which_mode train --name vit_model --backbone vit --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1
</code>
  
<br><br> For DGX-3 submission, assuming submit_local contains proper job scheduling...<br>

<br><br> To continue training...<br>
<code>
python -m main --which_mode train --name vit_model --backbone vit --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1 --resume
</code>

<br><br> To infer on all data...<br>
<code>
python -m main --which_mode infer --name vit_model --backbone vit --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1 --resume
</code>

<br><br> To infer PDB patches' temperatures inside e.g. **inference_save/T.123** directory, and to save inside **inference_save** directory as pickles<br>
<code>
python -m main --which_mode infer_custom --name convnext_model --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 512 --ce_re_ratio 1 0.1 --backbone convnext --resume --pdb_database inference_folder --save_dir inference_save --search_temp 123
</code>

# Train/Inference databases
<br> ***Patch Lipids for training***:
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/2-MembTempredict/testing/
<br> ***Patch Lipids for inference***:
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/5-Analysis/DPPC-CHL/inference_folder/
<br> ***Individual Lipids for training***:
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/2-MembTempredict/indiv_lips_H/
<br> ***Individual Lipids for inference***: 
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/5-Analysis/AQP5-PC
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/5-Analysis/LAINDY-PE
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/5-Analysis/B2GP1-PC
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/5-Analysis/B2GP1-PS
