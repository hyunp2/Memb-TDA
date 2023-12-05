# Memb-TDA

[To get PyTorch dataLoader of graph format using <s>PyTorchGeometric</s> PyTorch] This extracts XYZ coordinates and multiprocesses PH; subscriptable by index for Dataset and loadable for DataLoader </br>
<code>
python -m main --which_mode preprocessing --pdb_database /Scr/arango/Sobolev-Hyun/2-MembTempredict/testing/ --save_dir /Scr/hyunpark/Protein-TDA/pickled --filename dppc.pickle --multiprocessing --log --gpu
</code>

[To train from saved pickle/dat files] Assuming that pickle/dat files for coordinates, PH and temperature are saved, we can start training neural network model...</br>
<br><br> For distributed data parallelization <br>

</br>
<code>
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module main --which_mode train --name vit_model --backbone vit --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1
</code>
  

<br><br> To continue training...<br>
<code>
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module main --which_mode train --name vit_model --backbone vit --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1 --resume
</code>

<br><br> To infer on all data...<br>
<code>
python -m main --which_mode infer --name vit_model --backbone vit --filename dppc.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 16 --ce_re_ratio 1 0.1 --resume
</code>

## **Below is MOST important!**
<br><br> To infer PDB patches' temperatures inside e.g. **inference_save/T.123** directory, and to save inside **inference_save** directory as pickles<br> 
<code>
python -m main --which_mode infer_custom --name convnext_model_indiv --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 512 --ce_re_ratio 1 0.1 --backbone convnext --resume --pdb_database inference_folder --save_dir inference_save --search_temp 307
</code>

# Train/Inference databases
<br> ***Patch Lipids for training***:
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
