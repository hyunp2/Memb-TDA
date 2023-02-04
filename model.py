import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, GRU
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    NNConv,
    TransformerConv
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, radius_graph
from transformers import ViTFeatureExtractor, ViTModel
import os
import curtsies.fmtfuncs as cf 
from transformers import ViTFeatureExtractor, ConvNextFeatureExtractor, ViTModel, SwinModel, Swinv2Model, ConvNextModel, ViTConfig, SwinConfig, Swinv2Config, ConvNextConfig
from resTv2 import ResTV2 as ResTV2Model
from clip_resnet import ResNetForCLIP as ResNetForCLIPModel
from loss_utils import * #TEMP_RANGES
 

__all__ = ["MPNN", "Vit", "feature_extractor"]

##############################
############GNN###############
##############################

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

# CGCNN
class MPNN(torch.nn.Module):
    def __init__(
        self,
        num_features = 64,
        num_edge_features = 64,
        dim1=64,
        dim2=64,
        dim3=64,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        cutoff = 10.,
        max_num_neighbors=32,
        num_gaussians=64,
        nlp="transformer",
        heads=8,
        **kwargs
    ):
        super(MPNN, self).__init__()

        self.embedding = torch.nn.Embedding(10000, num_features)
        self.rbf = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.explain = kwargs.get("explain", False)
        if self.explain:
            def hook(module, inputs, grad):
                self.embedded_grad = grad
            self.embedding.register_backward_hook(hook)

        self.nlp = nlp
        self.cutoff = cutoff
        self.max_num_neighbors=max_num_neighbors
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"        
        if pre_fc_count == 0:
            gc_dim = num_features
        else:
            gc_dim = dim1
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            post_fc_dim = num_features
        else:
            post_fc_dim = dim1
        ##Determine output dimension length
        output_dim = 1

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(num_features, dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        ##Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.gru_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            nn = Sequential(
                Linear(num_edge_features, dim3), ReLU(), Linear(dim3, gc_dim * gc_dim)
            )
            conv = NNConv(
                gc_dim, gc_dim, nn, aggr="mean"
            )            
            self.conv_list.append(conv)
            if nlp == "gru":
                nlp_ = GRU(gc_dim, gc_dim)  
            elif nlp == "transformer":
                nlp_ = TransformerConv(gc_dim, gc_dim//heads, heads=heads, edge_dim=gc_dim)
            elif nlp == "point_transformer":
                raise NotImplementedError
            self.gru_list.append(nlp_)

            ##Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm == "True":
                bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        ##Set up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim*2, output_dim)
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)   

        ##Set up set2set pooling (if used)
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)

    def forward(self, pos, batch: torch.LongTensor=None, metadata: dict=None):
        #pos #(nodes, 3)
        #z #(nodes,)
        z = torch.cat([torch.arange((b==batch).sum()) for b in batch.unique()], dim=0).to(pos).long() #positional encoding
#         print(z.unique())
        pos.requires_grad_()

        h = self.embedding(z) #(nodes, dim)
        data = Data()
        data.x = h
        
        edge_index = metadata["edge_index"] if metadata != None and metadata["edge_index"] else radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1) #distance (nodes,)
        edge_attr = self.rbf(edge_weight) #(edges, dfilter)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.batch = batch
        
        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        ##GNN layers
        if len(self.pre_lin_list) == 0:
            h = data.x.unsqueeze(0)    
        else:
            h = out.unsqueeze(0)                
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    m = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                    m = self.bn_list[i](m)
                else:
                    m = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
            else:
                if self.batch_norm == "True":
                    m = self.conv_list[i](out, data.edge_index, data.edge_attr)
                    m = self.bn_list[i](m)
                else:
                    m = self.conv_list[i](out, data.edge_index, data.edge_attr)            
            m = getattr(F, self.act)(m)          
            m = F.dropout(m, p=self.dropout_rate, training=self.training)

            if self.nlp == "gru":
                out, h = self.gru_list[i](m.unsqueeze(0), h) #NOT GOOD! Sequence parsing is not aware of different molecules !! --> m: (1,natoms,dim);; h: (1,natoms.dim)?????
                out = out.squeeze(0)                
            elif self.nlp == "transformer":
                out = self.gru_list[i](m, data.edge_index, data.edge_attr)

        if self.explain:
            self.final_conv_acts = out
            def hook(grad):
                self.final_conv_grads = grad
            self.final_conv_acts.register_hook(hook) #only when backpropped!	

        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)

        elif self.pool_order == "late":
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                
        return out

##############################
############VIT###############
##############################
# Vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=os.path.join(os.getcwd(), "huggingface_cache"))
# Swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224", cache_dir=os.path.join(os.getcwd(), "huggingface_cache"))
# Swinv2 = Swinv2Model.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k", cache_dir=os.path.join(os.getcwd(), "huggingface_cache"))
# Convnext = ConvNextModel.from_pretrained("facebook/convnext-xlarge-384-22k-1k", cache_dir=os.path.join(os.getcwd(), "huggingface_cache"))

class Vision(torch.nn.Module):
    IMAGE_SIZE = 128
    PATCH_SIZE = 8 #change to 8
    NUM_CHANNELS = 3
    IMAGE_MEAN = [0.5] * NUM_CHANNELS
    IMAGE_STD = [0.5] * NUM_CHANNELS
    NUM_CLASSES = TEMP_RANGES[2]
    def __init__(self, args, **configs):
        super().__init__()
        IMAGE_SIZE = Vision.IMAGE_SIZE
        PATCH_SIZE = Vision.PATCH_SIZE
        NUM_CHANNELS = Vision.NUM_CHANNELS
        NUM_CLASSES = Vision.NUM_CLASSES
        IMAGE_MEAN = Vision.IMAGE_MEAN
        IMAGE_STD = Vision.IMAGE_STD
        config_vit = ViTConfig(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_channels=NUM_CHANNELS)
        config_swin = SwinConfig(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_channels=NUM_CHANNELS)
        config_swinv2 = Swinv2Config(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_channels=NUM_CHANNELS)
        config_convnext = ConvNextConfig(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_channels=NUM_CHANNELS)
        config_restv2 = dict(in_chans=NUM_CHANNELS, num_classes=NUM_CLASSES, embed_dims=[96, 192, 384, 768],num_heads=[1, 2, 4, 8],
                             drop_path_rate=0., depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
        config_resnetclip= dict(layers = (3, 4, 6, 3), output_dim = 512, heads = 1024, input_resolution = IMAGE_SIZE, width = 64, use_clip_init = True,)
        
        Vit = ViTModel(config_vit)
        Swin = SwinModel(config_swin)
        Swinv2 = Swinv2Model(config_swinv2)
        Convnext = ConvNextModel(config_convnext)
        ResTV2 = ResTV2Model(**config_restv2)
        ResNetForCLIP = ResNetForCLIPModel(**config_resnetclip)
        
        if args.backbone == "vit":
            self.pretrained = Vit
            self.feature_extractor = ViTFeatureExtractor(do_resize=False, size=IMAGE_SIZE, do_normalize=True, image_mean=IMAGE_MEAN, image_std=IMAGE_STD)
#             hidden_from_ = self.pretrained.pooler.dense.out_features
        elif args.backbone == "swin":
            self.pretrained = Swin
            self.feature_extractor = ViTFeatureExtractor(do_resize=False, size=IMAGE_SIZE, do_normalize=True, image_mean=IMAGE_MEAN, image_std=IMAGE_STD)
#             hidden_from_ = self.pretrained.layernorm.weight.size()[0]
        elif args.backbone == "swinv2":
            self.pretrained = Swinv2
            self.feature_extractor = ViTFeatureExtractor(do_resize=False, size=IMAGE_SIZE, do_normalize=True, image_mean=IMAGE_MEAN, image_std=IMAGE_STD)
#             hidden_from_ = self.pretrained.layernorm.weight.size()[0]
        elif args.backbone == "convnext":
            self.pretrained = Convnext
            self.feature_extractor = ConvNextFeatureExtractor(do_resize=False, size=IMAGE_SIZE, do_normalize=True, image_mean=IMAGE_MEAN, image_std=IMAGE_STD)
        elif args.backbone == "restv2":
            self.pretrained = ResTV2
            self.feature_extractor = ConvNextFeatureExtractor(do_resize=False, size=IMAGE_SIZE, do_normalize=True, image_mean=IMAGE_MEAN, image_std=IMAGE_STD)
        elif args.backbone == "clip_resnet":
            self.pretrained = ResNetForCLIP
            self.feature_extractor = ConvNextFeatureExtractor(do_resize=False, size=IMAGE_SIZE, do_normalize=True, image_mean=IMAGE_MEAN, image_std=IMAGE_STD)
        
        if args.backbone in ["vit", "swin", "swinv2", "convnext"]:
            hidden_from_ = self.pretrained.layernorm.weight.size()[0]  
        elif args.backbone == "restv2":
            hidden_from_ = self.pretrained.embed_dims[3]
        elif args.backbone == "clip_resnet":
            hidden_from_ = self.pretrained.output_dim
            
        self.add_module("last_layer_together", torch.nn.Sequential(torch.nn.Linear(hidden_from_, 512), torch.nn.SiLU(True), 
                                                            torch.nn.Linear(512,256), torch.nn.SiLU(True), 
                                                                torch.nn.Linear(256,64), torch.nn.SiLU(True), 
                                                                torch.nn.Linear(64, NUM_CLASSES), )) #48 temperature classes

    def forward(self, img_ph: torch.FloatTensor):
        img_ph : List[torch.FloatTensor] = img_ph.detach().cpu().unbind(dim=0)
        img_ph : List[np.ndarray] = list(map(lambda inp: inp.numpy(), img_ph ))
        img_inputs: Dict[str, torch.FloatTensor] = self.feature_extractor(img_ph, return_tensors="pt") #range [-1, 1]
        img_inputs = dict(pixel_values=img_inputs["pixel_values"].to(torch.cuda.current_device()))
        out_ph = self.pretrained(**img_inputs).pooler_output #batch, dim
        out = self.last_layer_together(out_ph)
        
        return out

if __name__ == "__main__":
    from tda.ph import get_args, PH_Featurizer_DataLoader
    args = get_args()
    print(cf.green(f"Arguments: {args.__dict__}"))
    dataloader = PH_Featurizer_DataLoader(opt=args)
    ds = iter(dataloader.test_dataloader()).next()
    images = ds["PH"] #range [0,1]

    inputs = feature_extractor(images.unbind(dim=0), return_tensors="pt") #range [-1, 1]
    outs = model(**inputs)
    print(outs)

    
    
    
    





    
# if __name__ == "__main__":
#     model = MPNN()
#     from data_utils import *
#     args = get_args()
#     dataloader = PH_Featurizer_DataLoader(opt=args)
#     testset = iter(dataloader.test_dataloader()).next()["PH"]
#     testset_ph = testset.x
#     testset_batch = testset.batch
#     print(model(testset_ph, batch=testset_batch))
    
    #python -m model --psf reference_autopsf.psf --pdb reference_autopsf.pdb --trajs adk.dcd --save_dir . --data_dir /Scr/hyunpark/Monster/vaegan_md_gitlab/data --multiprocessing --filename temp2.pickle
