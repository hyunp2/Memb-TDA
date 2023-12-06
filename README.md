# Memb-TDA (*abbr* of Membrane Topological Data Analysis)
Welcome to **Memb-TDA**! This repo contains topological data analysis based machine learning algorithm to predict effective temperature of biological membranes. It predicts configuration based temperature from static coordinates of lipid configuration.

## A. Preprocessing
[Preprocessing] This extracts XYZ coordinates and multiprocesses PH; subscriptable by index for Dataset and loadable for DataLoader </br>
<code>python -m main --which_mode preprocessing --pdb_database /Scr/arango/Sobolev-Hyun/2-MembTempredict/indiv_lips_H/ --save_dir /Scr/hyunpark-new/Memb-TDA/pickled_indiv --filename dppc.pickle --multiprocessing --log --gpu --load_ckpt_path /Scr/hyunpark-new/Memb-TDA/saved
</code>

## B. Training
[Training after preprocessing] Assuming that pickle/dat files for coordinates, PH and temperature are saved, we can start training neural network model...
<br><br> For distributed data parallelization <br>
<code>python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module main --which_mode train --load_ckpt_path /Scr/hyunpark-new/Memb-TDA/saved --name convnext_model_indiv --backbone convnext --save_dir /Scr/hyunpark-new/Memb-TDA/pickled_indiv --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1
</code>
  
<br>To continue training...<br>
<code>python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module main --which_mode train --load_ckpt_path /Scr/hyunpark-new/Memb-TDA/saved --name convnext_model_indiv --backbone convnext --save_dir /Scr/hyunpark-new/Memb-TDA/pickled_indiv --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1 --resume
</code>

## C. Inference
[Inference after training] Assuming that we have a pretrained neural network model, we can pretict temperature distributions...
<br><br>To infer on all data...<br>
<code>python -m main --which_mode infer --load_ckpt_path /Scr/hyunpark-new/Memb-TDA/saved --name convnext_model_indiv --backbone convnext --save_dir /Scr/hyunpark-new/Memb-TDA/pickled_indiv --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1 --resume
</code>

### **Below is the MOST important code snippet! Inference on OOD data!**
To infer PDB patches' temperatures inside e.g. **inference_save/T.307** directory, and to save inside **inference_save** directory as pickles<br> 
<code>python -m main --which_mode infer_custom --load_ckpt_path /Scr/hyunpark-new/Memb-TDA/saved --name convnext_model_indiv --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 512 --ce_re_ratio 1 0.1 --backbone convnext --resume --pdb_database inference_folder --save_dir inference_save --search_temp 307
</code>

## D. Train/Inference directories
  ***Patch Lipids for training***:
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/2-MembTempredict/testing/
<br> ***Patch Lipids for inference***:
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/5-Analysis/DPPC-CHL/inference_folder/
<br> ***Individual Lipids for training***:
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/2-MembTempredict/indiv_lips_H/
<br> ***Individual Lipids for inference***: 
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/5-Analysis/AQP5-PC/inference_folder/
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/5-Analysis/LAINDY-PE/inference_folder/
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/5-Analysis/B2GP1-PC/inference_folder/
<br>    &ensp;&ensp; /Scr/arango/Sobolev-Hyun/5-Analysis/B2GP1-PS/inference_folder/

## E. Input/Output format (based on MOST imporant code snippet)
1. Input: directory argument *pdb_database* takes ***inference_folder***. ***inference_folder*** contains multiple directories and we infer on argument *search_temp* input 307 (K).
2. Output: directory argumnet *inference_save* saves 6 output prefixed files: **coords, PH, IM, ProcessedIm, temperature, Predicted**. (e.g. coords_convnext_307.pickle where convnext is argument *backbone* and 307 (K) is *search_temp*).
3. Model: directory argument *load_ckpt_path* is where model checkpoint argument *name* (e.g., convnext_model_indiv without ".pth") is placed.

PS. Download model checkpoints, input and output from [Zenodo](https://doi.org/10.5281/zenodo.10258742). Our paper is in [BioArxiv](https://www.biorxiv.org/content/10.1101/2023.11.28.569053v1.abstract).

## F. Interesting arguments to try
* --pdb_database inference_folder where there are folders starting with T. (e.g., T.300) with --search_temp 300 to infer all PDBs inside.
* --backbone can be "convnext", "clip_resnet" or "swinv2".
* --ce_re_ratio 1 0.1 to balance loss weight between CE and regression loss.

## G. Explainable AI (XAI)
To get highlights of important persistence image features...<br>
<code>python -m main --which_mode xai --name convnext_model_indiv --backbone convnext --save_dir /Scr/hyunpark-new/Memb-TDA/pickled_indiv --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --which_xai gradcam --resume
</code>

## H. Plot distributions
To get our paper's temperature distirubtion plot<br>
<code> python -m main --which_mode train --name vit_model --save_dir inference_save --multiprocessing --log --gpu </code>
