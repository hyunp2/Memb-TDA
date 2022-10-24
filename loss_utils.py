import torch
import torch.nn.functional as F
from typing import *
from torch_geometric.data import Data, Batch

def ce_loss(y_true: Union[torch.LongTensor, torch.FloatTensor], y_pred: torch.FloatTensor):
    """Get temperature class prediction"""
#     onehots = F.one_hot(torch.arange(0, 48), num_classes=48) #48 temp bins
#     onehots.index_select(dim=0, index = y_true.view(-1,).long() - 283) # -->(Batch, numclass)
    assert y_true.size(0) == y_pred.size(0), "Batch size must match!"
    ranges = torch.arange(0, 48).to(y_pred).long() #48 temp bins
    y_true = ranges.index_select(dim=0, index = y_true.view(-1,).long() - 283) # --> (Batch, ) of LongTensor;; y_pred is (Batch, numclasses)
    ce = torch.nn.CrossEntropyLoss()
    loss = ce(y_pred, y_true)
    return loss
  
def reg_loss(y_true: Union[torch.LongTensor, torch.FloatTensor], y_pred: torch.FloatTensor):
    """Get expected temperature and regress on True Temp"""
    assert y_true.size(0) == y_pred.size(0), "Batch size must match!"
    ranges = torch.arange(283, 283+48).to(y_pred).float() #temperatures
    y_pred_probs = F.softmax(y_pred, dim=-1) #-->(Batch, numclass)
    assert y_pred_probs.size(-1) == ranges.size(0), "Num class must match!"
    y_pred_expected_T = y_pred_probs * ranges[None, :]  #-->(Batch, numclass)
    y_pred_expected_T = y_pred_expected_T.sum(dim=-1) #-->(Batch,)
    mse = torch.nn.MSELoss()
    loss = mse(y_true.view(-1,), y_pred_expected_T.view(-1,) )
    return loss
  
def contrastive_loss(y_true: Union[torch.LongTensor, torch.FloatTensor], y_pred_tensor: torch.FloatTensor):
    """WIP: Extract tensor from forward hook and do contrastive learning"""
    data_list = []
    for one_tensor, one_temp in zip(y_pred_tensor.unbind(dim=0), y_true.unbind(dim=0)):
        data = Data(x=one_tensor.view(1,-1), y=y_true.view(1,-1))
        data_list.append(data)
    batch = Batch.from_data_list(data_list)
    ranges = torch.arange(283, 283+48).to(y_true).long() #temperatures
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
        
        
    
