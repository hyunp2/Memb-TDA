import torch
import torch.nn.functional as F
from typing import *

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
    y_pred_probs = F.softmax(y_pred_probs, dim=-1) #-->(Batch, numclass)
    assert y_pred_probs.size(-1) == ranges.size(0), "Num class must match!"
    y_pred_expected_T = y_pred_probs * ranges[None, :]  #-->(Batch, numclass)
    y_pred_expected_T = y_pred_expected_T.sum(dim=-1) #-->(Batch,)
    mse = torch.nn.MSELoss()
    loss = mse(y_true.view(-1,), y_pred_expected_T.view(-1,) )
    return loss
  
