import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from torch_geometric.data import Data, Batch
import argparse

TEMP_RANGES = (280, 330, 51)

__all__ = ["ce_loss", "reg_loss", "distillation_loss", "TEMP_RANGES"]

def ce_loss(args: argparse.ArgumentParser, y_true: Union[torch.LongTensor, torch.FloatTensor], y_pred: torch.FloatTensor, label_smoothing: float=0.):
    """Get temperature class prediction"""
#     onehots = F.one_hot(torch.arange(0, 48), num_classes=48) #48 temp bins
#     onehots.index_select(dim=0, index = y_true.view(-1,).long() - 283) # -->(Batch, numclass)
    assert y_true.size(0) == y_pred.size(0), "Batch size must match!"
    ranges = torch.arange(0, TEMP_RANGES[2]).to(y_pred).long() #48 temp bins
    y_true = ranges.index_select(dim=0, index = y_true.to(y_pred).view(-1,).long() - TEMP_RANGES[0]) # --> (Batch, ) of LongTensor;; y_pred is (Batch, numclasses)
    weights = torch.tensor(args.ce_weights).to(y_pred)
    ce = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
#     print("ce_loss inside loss_utils", y_pred.shape, y_true.shape)
    loss = ce(y_pred, y_true)
    return loss
  
def reg_loss(args: argparse.ArgumentParser, y_true: Union[torch.LongTensor, torch.FloatTensor], y_pred: torch.FloatTensor):
    """Get expected temperature and regress on True Temp"""
    assert y_true.size(0) == y_pred.size(0), "Batch size must match!"
    ranges = torch.arange(TEMP_RANGES[0], TEMP_RANGES[1] + 1).to(y_pred).float() #temperatures
    y_pred_probs = F.softmax(y_pred, dim=-1) #-->(Batch, numclass)
    assert y_pred_probs.size(-1) == ranges.size(0), "Num class must match!"
    y_pred_expected_T = y_pred_probs * ranges[None, :]  #-->(Batch, numclass)
    y_pred_expected_T = y_pred_expected_T.sum(dim=-1) #-->(Batch,)
#     mse = torch.nn.MSELoss()
    mse = torch.nn.SmoothL1Loss()
#     print(y_true.to(y_pred).view(-1,).shape, y_pred_expected_T.view(-1,).shape)
    loss_mean = mse(y_true.to(y_pred).view(-1,), y_pred_expected_T.view(-1,) )
    loss_std = ((ranges[None, :] - y_pred_expected_T.view(-1,)[:, None]).pow(2) * y_pred_probs).sum(dim=-1).sqrt().mean()
    loss = loss_mean + loss_std
    return loss

def distillation_loss(args: argparse.ArgumentParser, y_true, y_pred, teacher_scores: torch.FloatTensor, T: float, alpha: float):
    print(y_pred.size(), teacher_scores.size(), y_true.size())
    return nn.KLDivLoss()(F.log_softmax(y_pred/T), F.softmax(teacher_scores/T)) * (T*T * 2.0 * alpha) + torch.nn.CrossEntropyLoss(weight=torch.tensor(args.ce_weights).to(y_pred))(y_pred, y_true.view(-1,)) * (1. - alpha)
  
def contrastive_loss(y_true: Union[torch.LongTensor, torch.FloatTensor], y_pred_tensor: torch.FloatTensor):
    """WIP: Extract tensor from forward hook and do contrastive learning
    SEE: ProtoNet: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/12-meta-learning.html#:~:text=class%20ProtoNet(pl.LightningModule)%3A
    """
    data_list = []
    for one_tensor, one_temp in zip(y_pred_tensor.unbind(dim=0), y_true.unbind(dim=0)):
        data = Data(x=one_tensor.view(1,-1), y=y_true.view(1,-1))
        data_list.append(data)
    batch = Batch.from_data_list(data_list)
    ranges = torch.arange(TEMP_RANGES[0], TEMP_RANGES[1] + 1).to(y_true).long() #temperatures
    mean_list = []
    random_list = []
    for t in ranges:
        tensor_at_t = batch.x[batch.y.long().view(-1,) == t]
        mean = tensor_at_t.mean(dim=0) #(dim,)
        if mean.size(0) == 0:
            continue
        else:
            mean_list.append(mean)
            random_list.append()
    mean_list = torch.stack(mean_list, dim=0) #maybe maximum (num_temp_range, dim)
    #https://github.com/lucidrains/contrastive-learner/blob/master/contrastive_learner/contrastive_learner.py
    logits = queries @ keys.t()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits /= temperature
    return F.cross_entropy(logits, torch.arange(b, device=device))
        
def roost_RobustL2Loss(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    # NOTE can we scale log_std by something sensible to improve the OOD behaviour?
    loss = 0.5 * torch.pow(output - target, 2.0) * torch.exp(-2.0 * log_std) + log_std
    return torch.mean(loss)
    
def roost_sampled_softmax(pre_logits, log_std, samples=10):
    """
    Draw samples from gaussian distributed pre-logits and use these to estimate
    a mean and aleatoric uncertainty.
    """
    # NOTE here as we do not risk dividing by zero should we really be
    # predicting log_std or is there another way to deal with negative numbers?
    # This choice may have an unknown effect on the calibration of the uncertainties
    sam_std = torch.exp(log_std).repeat_interleave(samples, dim=0)

    # TODO here we are normally distributing the samples even if the loss
    # uses a different prior?
    epsilon = torch.randn_like(sam_std)
    pre_logits = pre_logits.repeat_interleave(samples, dim=0) + torch.mul(
        epsilon, sam_std
    )
    logits = F.softmax(pre_logits, dim=1).view(len(log_std), samples, -1) #B,s,dim
    logits = logits.mean(dim=1) #B,dim
    return logits.exp()
    
