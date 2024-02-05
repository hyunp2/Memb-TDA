import logging
import collections
import pathlib
from typing import List, Union
import matplotlib.pyplot as plt
import os, sys
import pathlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import importlib
import pickle
import argparse
try:
    importlib.import_module("apex.optimizers")
    from apex.optimizers import FusedAdam, FusedLAMB
except Exception as e:
    pass
from sklearn.metrics import confusion_matrix
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, Logger
# from transformers import AdamW
import curtsies.fmtfuncs as cf
import torchmetrics

# https://fairscale.readthedocs.io/en/latest/
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper #anytime
from fairscale.experimental.nn.offload import OffloadModel #Single-GPU
from fairscale.optim.adascale import AdaScale #DDP
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP #Sharding
from fairscale.optim.oss import OSS #Sharding
import shutil
if torch.__version__.startswith('1.11'):
    from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
    from torch.distributed.fsdp.wrap import (
                        default_auto_wrap_policy,
                        enable_wrap,
                        wrap,
                        )
elif torch.__version__.startswith('1.13'):
    from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload, BackwardPrefetch
    from torch.distributed.fsdp.wrap import (
                        size_based_auto_wrap_policy,
                        enable_wrap,
                        wrap,
                        )

from data_utils import get_dataloader, PH_Featurizer_Dataset, mdtraj_loading, sanity_check_mdtraj
from train_utils import load_state, single_val, single_test
from loss_utils import * #TEMP_RANGES variable
from log_utils import * #Confusion matrix func
from model import Vision
from transformers import ViTFeatureExtractor, ConvNextFeatureExtractor, ViTModel, SwinModel, Swinv2Model, ConvNextModel, ViTConfig, SwinConfig, Swinv2Config, ConvNextConfig


#https://github.com/taki0112/denoising-diffusion-gan-Tensorflow/blob/571a99022ccc07a31b6c3672f7b5b30cd46a7eb6/src/utils.py#L156:~:text=def%20merge(,return%20img
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h * j:h * (j + 1), w * i:w * (i + 1), :] = image

    return img

class InferenceDataset(PH_Featurizer_Dataset):
    #python -m main --which_mode infer_custom --name convnext_model --filename vit.pickle --multiprocessing --optimizer torch_adam --log --gpu --epoches 1000 --batch_size 512 --ce_re_ratio 1 0.1 --backbone convnext --resume --pdb_database inference_folder --save_dir inference_save --search_temp 307
    def __init__(self, args: argparse.ArgumentParser, model: torch.nn.Module):
        import pathlib
#         assert os.path.basename(args.pdb_database).startswith("inference_") or os.path.basename(args.pdb_database).startswith("individual_"), "pdb_database directory MUST start with a prefix inference_ to differentiate from training directory!"
        assert os.path.basename(args.save_dir).startswith("inference_"), "saving directory MUST start with a prefix inference_ to differentiate from training directory!"
        assert args.search_temp != None, "this argument only exists for InferenceDataset!"
        args.filename = args.backbone + "_" + args.search_temp + ".pickle" #To save pickle files
	
        self.device = device = torch.cuda.current_device()
        model.to(device=device)
        local_rank = get_local_rank()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.world_size = world_size
        tmetrics = torchmetrics.MeanAbsoluteError()
    
        #DDP Model
        if dist.is_initialized() and not args.shard:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            model._set_static_graph()
            print(f"DDP is enabled {dist.is_initialized()} and this is local rank {local_rank} and sharding is {args.shard}!!")    
            model.train()
            if args.log: logger.start_watching(model) #watch a model!
        elif dist.is_initialized() and args.shard:
            my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=100
            )
            torch.cuda.set_device(local_rank)
            model = FSDP(model, fsdp_auto_wrap_policy=my_auto_wrap_policy)

#         init_start_event = torch.cuda.Event(enable_timing=True)
#         init_end_event = torch.cuda.Event(enable_timing=True)

        path_and_name = os.path.join(args.load_ckpt_path, "{}.pth".format(args.name))
        assert args.resume, "Validation and test must be under resumed keyword..."
        epoch_start, best_loss = load_state(model, None, None, path_and_name, use_artifacts=args.use_artifacts, logger=None, name=args.name, model_only=True) 
        self.model = model #To call a pretrained model!
        self.model.eval()
        self.search_system = args.search_system
        self.directories = [os.path.join(args.pdb_database, f"T.{args.search_temp}")] #e.g. List of str dir ... ["inference_pdbdatabase/T.123"]
        #index_for_searchTemp is NO LONGER NECESSARY!
	
						############################################
						############################################
						################KEY STATISTICS##############
						############################################
						############################################
        
        assert ("patch" in args.name) or ("indiv" in args.name), "model name MUST have patch or indiv in its string..."
        if "patch" in args.name:
            image_stats = collections.namedtuple("image_stats", ["bmax0","pmax0","mins0","maxs0","bmax1","pmax1","mins1","maxs1"])
            [setattr(image_stats, key, val) for key, val in zip(["bmax0","pmax0","mins0","maxs0","bmax1","pmax1","mins1","maxs1"], 
							    [0.0, 5.63134450674057, 3.708039700764246e-12,0.004038097944668425,
							     5.176542000174523, 1.3557068485021593, 2.3814288004517375e-13, 0.0004667655730385265] )]
						# BELOW Patches
						#         br: 0.0 vs pr: 5.63134450674057
						#         min max 3.708039700764246e-12-0.004038097944668425
						#         br: 5.176542000174523 vs pr: 1.3557068485021593
						#         min max 2.3814288004517375e-13-0.0004667655730385265
						#         image_stats = None #Placeholder for now!
        elif "indiv" in args.name:
            image_stats = collections.namedtuple("image_stats", ["bmax0","pmax0","mins0","maxs0","bmax1","pmax1","mins1","maxs1"])
            [setattr(image_stats, key, val) for key, val in zip(["bmax0","pmax0","mins0","maxs0","bmax1","pmax1","mins1","maxs1"], 
							    [0.0, 0.17649484410881996, 0.00019561807973112353, 0.0002599278183368139,
							     1.639571475684643, 0.768688747882843, 4.088960698780252e-08, 2.4840175665985345e-05] )]

						# BELOW Individiuals
						# Preprocessing BD:  0 0.0 0.17649484410881996
						# Preprocessing mimmax:  0 0.00019561807973112353 0.0002599278183368139
						# Preprocessing BD:  1 1.639571475684643 0.768688747882843
						# Preprocessing mimmax:  1 4.088960698780252e-08 2.4840175665985345e-05
						############################################
						############################################
						############################################
						############################################
						############################################
	
        super().__init__(args=args, directories=self.directories, image_stats=image_stats, search_system=self.search_system) #Get all the values from inheritance!
        print(cf.on_red(f"Argument args.search_temp {self.search_temp} is an integer keyword to find the correct directory e.g. inference_pdbdatabase/T.128/*.pdb"))
        self.index_for_searchTemp = np.where(np.array(self.temperatures) == int(self.search_temp))[0] #Index to get only the correponding temperature-related data!
#         self.graph_input_list, self.Rs_total, self.Images_total, self.temperature = self.graph_input_list[self.index_for_searchTemp], self.Rs_total[self.index_for_searchTemp], self.Images_total[self.index_for_searchTemp], self.temperature[self.index_for_searchTemp]
        self.feature_extractor = ViTFeatureExtractor(do_resize=False, size=Vision.IMAGE_SIZE, do_normalize=True, image_mean=Vision.IMAGE_MEAN, image_std=Vision.IMAGE_STD, do_rescale=False) if self.backbone in ["vit", "swin", "swinv2"] else ConvNextFeatureExtractor(do_resize=False, size=Vision.IMAGE_SIZE, do_normalize=True, image_mean=Vision.IMAGE_MEAN, image_std=Vision.IMAGE_STD, do_rescale=False)

    @property
    def infer_all_temperatures(self, ):
        how_many_patches = len(self) #number of temperature patches (i.e. PDBs) inside e.g. T.123 directory 
        direct = self.directories[0] #self.directories is a list of str dir!
        all_pdbs = os.listdir(direct)
        valid_pdbs = sanity_check_mdtraj(direct, all_pdbs)
        pdbs_ = np.array(list(map(lambda inp: inp.split(".") ,  valid_pdbs)) ) #(vallid_num_temps, 3)
	
        orders = np.lexsort((pdbs_[:,1].astype(int), pdbs_[:,0].astype(int))) #keyword-wise order --> lexsort((a,b)) is to sort by b and then a
        pdbs = pdbs_[orders] #e.g. ([0,1,"pdb"], [0,2,"pdb"] ... [199, 24,"pdb"], [199, 25,"pdb"])
        print(how_many_patches, pdbs.shape)
        assert how_many_patches == pdbs.shape[0] and pdbs.ndim == 2, "something is wrong! such as dimension or number of temperature patches! this maybe due to previous files were not deleted... start from scratch!"
	
        print(cf.on_red(f"Truncating data for specific temperature!"))
        self.Images_total, self.temperature = torch.stack(self.Images_total, dim=0)[self.index_for_searchTemp][orders], np.array(self.temperatures)[self.index_for_searchTemp][orders] #For a given temperature identifier (i.e. search_temp);; ORDERED!
        self.pdb2str = list(map(lambda inp: ".".join(inp), pdbs.tolist() )) #e.g. "0.1.pdb,... 199.25.pdb";; ORDERED!
#         quotient, remainder = divmod(how_many_patches, self.batch_size)

        img : torch.FloatTensor = self.Images_total.detach().cpu().unbind(dim=0)
        img : List[np.ndarray] = list(map(lambda inp: inp.numpy(),  img))
        img: Dict[str, torch.FloatTensor] = self.feature_extractor(img, return_tensors="pt") #range [-1, 1]
        img = img["pixel_values"] #BCHW tensor! range: [-1,1]
        self.Images_total = img
	
        dataset = torch.utils.data.TensorDataset(self.Images_total, torch.from_numpy(self.temperature)) #(how_many_patches,3,H,W)
        kwargs = {'pin_memory': True, 'persistent_workers': False}
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, **kwargs)
        dataloader = get_dataloader(dataset, shuffle=False, collate_fn=None, batch_size=self.batch_size, **kwargs)

        # (confmat, acc_global, acc, iu), (temps_all, predictions_all) = self.get_statistics(dataloader)
        _, (temps_all, predictions_all) = self.get_statistics_sklearn(dataloader)

        print("HERE")
        # print(confmat.mat, acc_global.nonzero(), acc, iu, temps_all, predictions_all)
	      
        temps_all = torch.cat(temps_all, dim=0) #(how_many_patches, 48)
        predictions_all = torch.cat(predictions_all, dim=0) #(how_many_patches, 48)
        h = confmat = confusion_matrix(temps_all.detach().cpu().numpy(), predictions_all.argmax(1).detach().cpu().numpy(), labels=np.arange(TEMP_RANGES[2]))	
        acc_global = np.diag(h).sum() / h.sum()
        acc = np.diag(h) / h.sum(1)
        iu = np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h))
        print(confmat, acc_global, acc, iu)	    
	    
        if dist.is_initialized():
            temps_all = torch.tensor(temps_all, dtype=torch.float, device=self.device) #(B,num_classes)
            temps_all_list = [temps_all.new_zeros(temps_all.size()).to(temps_all) for _ in range(self.world_size)] #list of (B,num_classes)
            torch.distributed.all_gather(temps_all_list, temps_all) #Gather to empty list!
            temps_all = torch.cat([temps for temps in temps_all_list], dim=0) #(Bs, num_classes)
            predictions_all = torch.tensor(predictions_all, dtype=torch.float, device=self.device) #(B,num_classes)
            predictions_all_list = [predictions_all.new_zeros(predictions_all.size()).to(predictions_all) for _ in range(self.world_size)] #list of (B,num_classes)
            torch.distributed.all_gather(predictions_all_list, predictions_all) #Gather to empty list!
            predictions_all = torch.cat([pred for pred in predictions_all_list], dim=0) #(Bs, num_classes)
	
        ranges = torch.arange(TEMP_RANGES[0], TEMP_RANGES[1] + 1).to(predictions_all).float() #temperatures
#         ranges = torch.arange(283, 331).to(predictions_all).float() #temperatures

        predictions_all_probs = F.softmax(predictions_all, dim=-1) #-->(Batch, numclass)
        print(predictions_all_probs[0])
        assert predictions_all_probs.size(-1) == ranges.size(0), "Num class must match!"
        predictions_all_probs_T = predictions_all_probs * ranges[None, :]  #-->(Batch, numclass)
        print(predictions_all_probs_T[0])
        predictions_all_probs_T = predictions_all_probs_T.sum(dim=-1) #-->(Batch,)
        predictions_all_probs_T_std = ((ranges[None, :] - predictions_all_probs_T.view(-1,)[:, None]).pow(2) * predictions_all_probs).sum(dim=-1).sqrt().view(-1, ) #(Batch, )
	
        if get_local_rank() == 0:
            f = open(os.path.join(self.save_dir, f"{self.search_system}_" + "Predicted_" + self.filename), "wb")
            save_as = collections.defaultdict(list)
            for key, val in zip(["predictions", "predictions_std", "images", "pdbnames"], [predictions_all_probs_T, predictions_all_probs_T_std, self.Images_total, self.pdb2str]):
                save_as[key] = val
            # save_as["METADATA"] = (confmat.mat, *confmat.compute())
            pickle.dump(save_as, f)   
        # confmat.reset()

        if dist.is_initialized():
    	    dist.destroy_process_group()
	
    def __call__(self):
        self.infer_all_temperatures

    def get_statistics(self, dataloader: torch.utils.data.DataLoader):
        """Compute confusion matrix"""
        temps_all = []
        predictions_all = []
        confmat = ConfusionMatrix(num_classes=TEMP_RANGES[2])
	    
        with torch.inference_mode():
            for batch in dataloader:
                imgs = batch[0].to(self.device)
                predictions = self.model(imgs)
                temps = batch[1].to(self.device) - TEMP_RANGES[0] #confmat must have a range from 0-47
#                 temps = batch[1].to(self.device) - 283 #confmat must have a range from 0-47
#                 print(batch.size(), predictions.size())
                confmat.update(temps.flatten().long(), predictions.argmax(1).flatten())
                temps_all.append(temps)
                predictions_all.append(predictions)
	    
        confmat.reduce_from_all_processes()
        acc_global, acc, iu = confmat.compute()
        return (confmat, acc_global, acc, iu), (temps_all, predictions_all)

    def get_statistics_sklearn(self, dataloader: torch.utils.data.DataLoader):
        """Compute confusion matrix"""
        temps_all = []
        predictions_all = []
	    
        with torch.inference_mode():
            for batch in dataloader:
                imgs = batch[0].to(self.device)
                predictions = self.model(imgs)
                temps = batch[1].to(self.device) - TEMP_RANGES[0] #confmat must have a range from 0-47
#                 temps = batch[1].to(self.device) - 283 #confmat must have a range from 0-47
#                 print(batch.size(), predictions.size())
                temps_all.append(temps)
                predictions_all.append(predictions)
	    
        return None, (temps_all, predictions_all)

def validate_and_test(model: nn.Module,
          get_loss_func: _Loss,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
	  test_dataloader: DataLoader,
          logger: Logger,
          args):
    """Includes evaluation and testing as well!"""

    #DDP options
    #Init distributed MUST be called in run() function, which calls this train function!

    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    tmetrics = torchmetrics.MeanAbsoluteError()
    
    #DDP Model
    if dist.is_initialized() and not args.shard:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model._set_static_graph()
        print(f"DDP is enabled {dist.is_initialized()} and this is local rank {local_rank} and sharding is {args.shard}!!")    
        model.train()
        if args.log: logger.start_watching(model) #watch a model!
    elif dist.is_initialized() and args.shard:
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=100
        )
        torch.cuda.set_device(local_rank)
        model = FSDP(model, fsdp_auto_wrap_policy=my_auto_wrap_policy)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    path_and_name = os.path.join(args.load_ckpt_path, "{}.pth".format(args.name))
    assert args.resume, "Validation and test must be under resumed keyword..."
    epoch_start, best_loss = load_state(model, None, None, path_and_name, use_artifacts=args.use_artifacts, logger=logger, name=args.name, model_only=True) 
    
    #DDP training: Total stats (But still across multi GPUs)
    init_start_event.record()
    
    model.eval()    	
	
    ds_train, ds_val, ds_test = train_dataloader.dataset, val_dataloader.dataset, test_dataloader.dataset
    dataloader_kwargs = {'pin_memory': args.pin_memory, 'persistent_workers': args.num_workers > 0,
                                        'batch_size': args.batch_size}
    ds = torch.utils.data.ConcatDataset([ds_train, ds_val, ds_test])
    val_dataloader = get_dataloader(ds, shuffle=False, collate_fn=None, **dataloader_kwargs)
    gts = torch.cat([batch["temp"] for batch in val_dataloader], dim=0).reshape(-1,) #B
    print(cf.yellow("All the data are concatenated into one! It is still named val_dataloader!"))

    ###EVALUATION
    evaluate = single_val
    val_loss, loss_metrics, val_predictions = evaluate(args, model, val_dataloader, get_loss_func, None, None, logger, tmetrics, return_data=True) #change to single_val with DDP

    if dist.is_initialized():
        val_loss = torch.tensor(val_loss, dtype=torch.float, device=device)
        loss_metrics = torch.tensor(loss_metrics, dtype=torch.float, device=device)
        torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
        val_loss = (val_loss / world_size).item()
        torch.distributed.all_reduce(loss_metrics, op=torch.distributed.ReduceOp.SUM) #Sum to loss
        loss_metrics = (loss_metrics / world_size).item() if (hasattr(loss_metrics, "item") and loss_metrics.numel() == 1) else loss_metrics
	
        gts = torch.tensor(gts, dtype=torch.float, device=device) #B
        gts_list = [gts.new_zeros(gts.size()).to(gts) for _ in range(world_size)] #list of (B)
        torch.distributed.all_gather(gts_list, gts) #Gather to empty list!
        gts = torch.cat([gt for gt in gts_list], dim=0) #Bs
	
        val_predictions = torch.tensor(val_predictions, dtype=torch.float, device=device) #B
        val_predictions_list = [val_predictions.new_zeros(val_predictions.size()).to(val_predictions) for _ in range(world_size)] #list of (B)
        torch.distributed.all_gather(val_predictions_list, val_predictions) #Gather to empty list!
        val_predictions = torch.cat([pred for pred in val_predictions_list], dim=0) #(Bs, 2)

    if args.log: 
        logger.log_metrics({'ALL_REDUCED_val_loss': val_loss})
        logger.log_metrics({'ALL_REDUCED_val_MAE': loss_metrics}) #zero rank only
    #zero rank only
    #mae_reduced = tmetrics.compute() #Synced and reduced autometically!
    #logger.log_metrics({'ALL_REDUCED_val_mae_loss': mae_reduced.item()}, epoch_idx) #zero rank only
    tmetrics.reset()
    #scheduler_re.step(val_loss) #Not on individual stats but the total stats

    init_end_event.record()

    if local_rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
	
    print(cf.on_green("Saving returned validation and test_predictions!"))
    gts = gts.detach().cpu().numpy()
    val_predictions = val_predictions.detach().cpu().numpy().reshape(-1, 2) #(B,2)
    if local_rank == 0:
        np.savez(os.path.join(args.save_dir, f"{args.backbone}_all_temps.npz"), gt=gts, pred=val_predictions[:,0], pred_std=val_predictions[:,1]) #(B,) for pred and pred_std
        shutil.copyfile(os.path.join(args.save_dir, f"{args.backbone}_all_temps.npz"), os.path.join(pathlib.Path(args.save_dir).parent, "inference_save", f"{args.backbone}_all_temps.npz"))
#     val_gts = torch.cat([batch["temp"] for batch in val_dataloader], dim=0) #B,1
#     test_gts = torch.cat([batch["temp"] for batch in test_dataloader], dim=0) #B,1
#     np.savez("PH_val_test.npz", validation_gt=val_gts.detach().cpu().numpy(), test_gt=test_gts.detach().cpu().numpy(), validation_pred=val_predictions.detach().cpu().numpy(), test_pred=test_predictions.detach().cpu().numpy())
    
    print(cf.on_yellow("Validation and test are OVER..."))
    if dist.is_initialized():
    	dist.destroy_process_group()

def compute_confusion(gt, pred) -> None:
    temps_all = gt
    predictions_all = pred
    h = confmat = confusion_matrix(temps_all, predictions_all, labels=np.arange(TEMP_RANGES[2]))	
    acc_global = np.diag(h).sum() / h.sum()
    acc = np.diag(h) / h.sum(1)
    iu = np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h))
    print(confmat, acc_global, acc, iu)	    
	
if __name__ == "__main__":
    # plot_analysis("PH_all_test.npz")
#     infer_all_temperatures()
    from main import get_args
    args = get_args()
    data = np.load(os.path.join(pathlib.Path(args.save_dir).parent, "inference_save", f"{args.backbone}_all_temps.npz"))
    gt, pred = data["gt"], data["pred"]
    gt = gt - TEMP_RANGES[0]
    ranges = np.arange(TEMP_RANGES[0], TEMP_RANGES[1] + 1).astype(float) #temperatures
    pred = np.searchsorted(ranges, pred) #Method 1; (Method 2) try on classification!
    compute_confusion(gt, pred)
    ## git pull && python -m test_utils --save_dir pickled_indiv --backbone convnext
