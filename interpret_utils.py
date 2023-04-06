
"""
Functions taken and modified from https://github.com/hila-chefer/Transformer-MM-Explainability

Credit:
@InProceedings{Chefer_2021_ICCV,
   author    = {Chefer, Hila and Gur, Shir and Wolf, Lior},
   title     = {Generic Attention-Model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers},
   booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
   month     = {October},
   year      = {2021},
   pages     = {397-406}
} 

"""

from typing import Dict, Optional

from topologylayer.nn import RipsLayer, AlphaLayer

import cv2
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# from .misc import patchify, unpatchify
import os
from captum.attr import Saliency, Lime, LayerGradCam, LayerAttribution
from captum.attr._core.lime import get_exp_kernel_similarity_function
from model import Vision
from transformers import ViTFeatureExtractor, ConvNextFeatureExtractor, ViTModel, SwinModel, Swinv2Model, ConvNextModel, ViTConfig, SwinConfig, Swinv2Config, ConvNextConfig
from loss_utils import TEMP_RANGES
from train_utils import load_state, single_val, single_test

def xai(args, images: torch.Tensor, gts: torch.LongTensor, model: torch.nn.Module, method="saliency", title="lows"):
    feature_extractor = ViTFeatureExtractor(do_resize=False, size=Vision.IMAGE_SIZE, do_normalize=True, image_mean=Vision.IMAGE_MEAN, image_std=IVision.MAGE_STD, do_rescale=False) if args.backbone in ["vit", "swin", "swinv2"] else ConvNextFeatureExtractor(do_resize=False, size=Vision.IMAGE_SIZE, do_normalize=True, image_mean=Vision.IMAGE_MEAN, image_std=Vision.IMAGE_STD, do_rescale=False)

    img : torch.FloatTensor = images.detach().cpu().unbind(dim=0)
    img : List[np.ndarray] = list(map(lambda inp: inp.numpy(),  img))
    img: Dict[str, torch.FloatTensor] = feature_extractor(img, return_tensors="pt") #range [-1, 1]
    img = img["pixel_values"] #BCHW tensor! range: [-1,1]
    images = img
      
    assert method in ["saliency", "gradcam", "lime"]
   
    class Layer4Gradcam(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            assert args.backbone == "convnext"
            self.model = model
            
            def fhook(m, i, o):
                self.layer_forward_output = o[0] #BCHW
                print(f"Forward {m.__class__.__name__} is registered...")
               
            def bhook(m, i, o):
                self.layer_backward_output = o[0] #BCHW
                print(f"Backward {m.__class__.__name__} is registered...")
               
            self.model.pretrained.encoder.register_forward_hook(fhook)    
            self.model.pretrained.encoder.register_full_backward_hook(bhook)   
            
        def attribute(self, inputs: torch.Tensor, target: torch.LongTensor):
            inputs = inputs.detach().requires_grad_(True) #make it leaf and differentiable!
            
            preds = self.model(inputs)
#             preds = torch.gather(input=preds, dim=1, index=target.view(-1, 1)) # -> (B,1)
#             torch.autograd.grad(preds, inputs, grad_outputs=torch.ones_like(preds))[0]
            preds = preds.amax(dim=-1)
#             print(preds.size())
            preds.backward(gradient=torch.ones_like(preds))
   
            module_output = self.layer_forward_output
            module_upstream_gradient = self.layer_backward_output
         
            grads_power_2 = module_upstream_gradient**2 #Bcddd
            grads_power_3 = grads_power_2 * module_upstream_gradient
            sum_activations = module_output.sum(dim=(2,3,4), keepdim=True) #Bc11
            eps = 0.000001
            aij = grads_power_2 / (2 * grads_power_2 +
                                 sum_activations * grads_power_3 + eps) #Bcdd
            aij = torch.where(module_upstream_gradient != module_upstream_gradient.new_tensor(0.), aij, module_upstream_gradient.new_tensor(0.)) #Non-zeros #Bcddd
            weights = torch.maximum(module_upstream_gradient, module_upstream_gradient.new_tensor(0.)) * aij #Only positive #Bcddd
            weights = weights.sum(dim=(2,3,4), keepdim=True) #Bc11
            gradcampp = (module_output * weights).sum(dim=1, keepdim=True) #Bcdd --> Bcdd
            gradcampp = torch.maximum(gradcampp, torch.tensor(0.)) #Only positives

            return gradcampp #B1HW
        
    def forward_func(images):
        preds: torch.Tensor = model(images) #-> (B,C)
        return preds
    
    def perturb_func(original_input: torch.Tensor,
                     **kwargs)->torch.Tensor:
        return original_input + original_input.new_tensor(torch.randn_like(original_input))

    similarity_func = get_exp_kernel_similarity_function(distance_mode="euclidean")
    
    if method == "saliency":
        attribute_method = Saliency
        attrs = attribute_method(forward_func=forward_func)
        attr_output = attrs.attribute(images, target=gts.view(-1)) #->(B,C,N,N)
    elif method == "gradcam_defunct":
        attribute_method = LayerGradCam
        attrs = attribute_method(forward_func=forward_func, layer=layer)
        attr_output = attrs.attribute(images, target=gts.view(-1)) #->(B,C,N,N)
        attr_output = LayerAttribution.interpolate(attr_output, (Vision.IMAGE_SIZE, Vision.IMAGE_SIZE))
    elif method == "gradcam":
        attribute_method = Layer4Gradcam
        attrs = attribute_method(model)
        attr_output = attrs.attribute(images, target=gts.view(-1)) #->(B,1,N,N)
        attr_output = torch.nn.functional.interpolate(attr_output, images.size())
    elif method == "lime":
        attribute_method = Lime
        attrs = attribute_method(forward_func=forward_func, similarity_func=similarity_func, perturb_func=perturb_func)
        attr_output = attrs.attribute(images, target=gts.view(-1)) #->(B,C,N,N)
         
    fig, ax = plt.subplots(4,4,figsize=(8,8))
    mins, maxs = attr_output.min().data, attr_output.max().data
    attr_output.data = (attr_output.data - mins) / (maxs - mins)
    for idx in range(images.size(0)):
        ax.flatten()[idx].imshow(attr_output[idx].permute(1,2,0).detach().cpu().numpy(), cmap=plt.cm.get_cmap("cool"), vmin=0.5, vmax=1)
    fig.savefig(os.path.join(args.save_dir, title))
    plt.close()
    return attr_output

# rule 5 from paper
def avg_heads(cam: torch.Tensor, grad: torch.Tensor):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss: torch.Tensor, cam_ss: torch.Tensor):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_relevance(model: nn.Module, inputs: Dict[str, torch.Tensor], index: Optional[int] = None):
    output = model(**inputs, register_hook=True)["logits"]

    if index is None:
        index = torch.argmax(output, dim=-1)

    one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
    one_hot[0, index] = 1
    one_hot = one_hot.requires_grad_(True)
    one_hot = torch.sum(one_hot * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    num_tokens = model.vit.encoder.layer[0].attention.attention.get_attention_map().shape[-1]

    R = torch.eye(num_tokens, num_tokens)
    for layer in model.vit.encoder.layer:
        grad = layer.attention.attention.get_attention_gradients()
        cam = layer.attention.attention.get_attention_map()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R, cam)

    return R[0, 1:]


# create heatmap from mask on image
def show_cam_on_image(img: torch.Tensor, mask: torch.Tensor):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = torch.einsum("chw->hwc", unpatchify(patchify(torch.einsum("hwc->chw", img))))
    img = np.float32(img)
    cam = heatmap + img
    cam = cam / np.max(cam)
    return cam


def generate_visualization(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    image_hw: int,
    patch_size: int = 16,
    class_index: Optional[int] = None,
):

    transformer_attribution = generate_relevance(model, inputs, index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, image_hw // patch_size, image_hw // patch_size)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=16, mode="bilinear"
    )
    transformer_attribution = transformer_attribution.reshape(image_hw, image_hw)
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
        transformer_attribution.max() - transformer_attribution.min()
    )

    original_image = inputs["pixel_values"].squeeze()
    image_transformer_attribution = original_image.permute(1, 2, 0)
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
        image_transformer_attribution.max() - image_transformer_attribution.min()
    )
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

if __name__ == "__main__":
    from main import get_args
    args = get_args()
   
    path_and_name = os.path.join(args.load_ckpt_path, "{}.pth".format(args.name))
    assert args.resume, "Validation and test must be under resumed keyword..."
    model = Vision(args)
    epoch_start, best_loss = load_state(model, None, None, path_and_name, use_artifacts=args.use_artifacts, logger=None, name=args.name, model_only=True) 
    model.eval()
   
    images = torch.rand(16, 3, 128, 128)
    y_true = torch.LongTensor(16).random_(TEMP_RANGES[0], TEMP_RANGES[1])
    ranges = torch.arange(0, TEMP_RANGES[2]).to(images).long() #48 temp bins
    gts = ranges.index_select(dim=0, index = y_true.to(images).view(-1,).long() - TEMP_RANGES[0]) 
   
    xai(args, images, gts, model, method=args.which_xai)   
