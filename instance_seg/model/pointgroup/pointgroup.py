
'''
PointGroup
Written by Li Jiang
'''

import torch
import time
import torch.nn as nn
import spconv
import numpy as np
import copy
import random
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys
sys.path.append('../../')
sys.path.append('lib/spconv/build/lib.linux-x86_64-3.7/spconv')
sys.path.append('lib/spconv')
from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
import torch.nn.functional as F
import math




class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
       
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
               
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
   
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        #print ('xshape', x.shape, seq_len)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return x

def attention(q, k, v, d_k, mask=None, dropout=None):
   
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
   
    #print ('mask score ', mask.shape, scores.shape)
    #print (scores)
   
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    #print (scores.shape)
    scores = F.softmax(scores, dim=-1)
   
    if dropout is not None:
        scores = dropout(scores)
    #print ('output', scores.shape, v.shape)       
    output = torch.matmul(scores, v)
    #print ('output', output.shape)
    return output
   
   
   
       
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
       
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
       
        self.q_linear = nn.Linear(d_model, d_model, bias=True)
        self.v_linear = nn.Linear(d_model, d_model, bias=True)
        self.k_linear = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model, bias=True)
   
    def forward(self, q, k, v, mask=None):
       
        bs = q.size(0)
       
        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
       
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        #print ('kqv',k.shape, q.shape, v.shape)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        #print ('scores.shape',scores.shape)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        #print ('concat', concat.shape)
       
        output = self.out(concat)
        #print ('output', output.shape)
   
        return output
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
       
       
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
       
       
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-5):
        super().__init__()
   
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
       

        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
       
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()
       
    def forward(self, x, e_outputs, src_mask=None):
        #print ('1',self.norm_2.bias)
        #print ('input',x.shape, e_outputs.shape)
        x = self.norm_2(x)
        #print (x.shape)
        #print ('2',torch.unique(x))
        x = x+self.dropout_2(self.attn_2(x, e_outputs, e_outputs, src_mask))
        #print ('att2.shape',  self.attn_2(x, e_outputs, e_outputs, src_mask).shape)
        #print ('3',torch.unique(x))
        x = self.norm_3(x)
        #print ('4',torch.unique(x))
        x = x+self.dropout_3(self.ff(x))
        #print (x.shape)
        #print ('5',torch.unique(x))
        return x
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])




class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            if nPlanes[0]==32:
              self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
              )
            else:
              self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
              )

            blocks_tail = {}

            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output


class PointGroup(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel
        m = cfg.m
        self.m=m
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        self.input_conv_mom = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        for param_q, param_k in zip(self.input_conv.parameters(), self.input_conv_mom.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False 


        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, indice_key_id=1)
        self.unet_mom = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, indice_key_id=1)


        for param_q, param_k in zip(self.unet.parameters(), self.unet_mom.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False 


        self.output_layer = spconv.SparseSequential(
            norm_fn(m) #,
            #nn.ReLU()
        )
        self.output_layer_mom = spconv.SparseSequential(
            norm_fn(m) #,
            #nn.ReLU()
        )

        for param_q, param_k in zip(self.output_layer.parameters(), self.output_layer_mom.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False 



        self.linear_1 = nn.Linear(m, m) 
        self.linear_2 = nn.Linear(m, m) 
        
        self.N=4
        self.layers=get_clones(DecoderLayer(m,1), self.N)
        


        #### semantic segmentation
        self.linear = nn.Linear(m, classes) # bias(default): True

        self.feat=torch.Tensor(1,20,m).cuda() #.detach()
        self.start=[]
        for i in range(20):
          self.start.append(1)


        self.linear_sv = nn.Linear(m*2, classes)
        
        self.mom=0.999

        #### offset
        self.offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        '''#### score branch
        self.score_unet = UBlock([m, 2*m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )
        self.score_linear = nn.Linear(m, 1)

        self.apply(self.set_bn_init)

        #### fix parameter
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'linear': self.linear, 'offset': self.offset, 'offset_linear': self.offset_linear,
                      'score_unet': self.score_unet, 'score_outputlayer': self.score_outputlayer, 'score_linear': self.score_linear}

        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False'''

        #### load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.input_conv.parameters(), self.input_conv_mom.parameters()):
            param_k.data = param_k.data * self.mom + param_q.data * (1. - self.mom)
        for param_q, param_k in zip(self.unet.parameters(), self.unet_mom.parameters()):
            param_k.data = param_k.data * self.mom + param_q.data * (1. - self.mom)
        for param_q, param_k in zip(self.output_layer.parameters(), self.output_layer_mom.parameters()):
            param_k.data = param_k.data * self.mom + param_q.data * (1. - self.mom)


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map


    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, epoch):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        unary_feats = output.features[input_map.long()]

        #### semantic segmentation
        semantic_scores = self.linear(unary_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]    # (N), long

        ret['semantic_scores'] = semantic_scores

        '''output_feats = self.linear_1(unary_feats.detach())
        output_feats = F.relu(output_feats)        
        output_feats = self.linear_2(output_feats) '''
        return ret,  unary_feats






def calc_bound(points):
  x1=torch.min(points[:,0])#[0]
  y1=torch.min(points[:,1])#[0]
  z1=torch.min(points[:,2])#[0]
  x2=torch.max(points[:,0])#[0]
  y2=torch.max(points[:,1])#[0]
  z2=torch.max(points[:,2])#[0]  
  return (x1,y1,z1,x2,y2,z2)
  
  
def intersect(anno_bound, bound):
  #print (anno_bound,bound)
  the=0.1
  if bound[0]-anno_bound[3]>the or bound[3]-anno_bound[0]<-the or bound[1]-anno_bound[4]>the or bound[4]-anno_bound[1]<-the or bound[2]-anno_bound[5]>the or bound[5]-anno_bound[2]<-the:
    return 0
  return 1
    

def merge(anno_bound, bound):
  x1=min(anno_bound[0], bound[0])
  y1=min(anno_bound[1], bound[1])
  z1=min(anno_bound[2], bound[2])
  x2=max(anno_bound[3], bound[3])
  y2=max(anno_bound[4], bound[4])
  z2=max(anno_bound[5], bound[5])
  return (x1,y1,z1,x2,y2,z2)



def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch, mode):
        #print ('test model fn')
        
        feats=torch.zeros((20,16))
        
        coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()              # (N, C), float32, cuda

        batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']
        labels = batch['labels']  
        
        group2points=batch['g2p_map']
        group_labels=batch['group_labels']
        
        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda
        
        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret, unary_feat_pred = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        #ret_pair, pair_feat_pred = model_mom(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        output = model.input_conv_mom(input_)
        output = model.unet_mom(output)
        output = model.output_layer_mom(output)
        pair_feat_pred = output.features[p2v_map.long()]
        
        pair_feat_pred=pair_feat_pred.detach()


        with torch.no_grad():  # no gradient to keys
            model._momentum_update_key_encoder()


        

        semantic_scores_pred = ret['semantic_scores']        

        voxel_xyzs=torch.zeros((0,3)).cuda()     

        
        
        labels_sv=torch.zeros((0,1)) #.cuda()

        label_to_append = torch.zeros((1,1)).cuda()
          
        pair_feats=torch.zeros((0,32)).cuda()
        pair_preds=torch.zeros((0,20)).cuda()
        unary_feats=torch.zeros((0,32)).cuda()
        unary_preds=torch.zeros((0,20)).cuda()
        tmp_feat=torch.zeros((20, model.m)).cuda().detach()
        tmp_feat[:]=0 
        tmp_feat_cnt=torch.zeros((20, )).cuda().detach()
        tmp_feat_cnt[:]=0





        pair_feat_pred=pair_feat_pred.detach()


        voxel_batch_idxs=torch.zeros((0,)).cuda()
        voxel_batch_offsets=torch.zeros((batch_offsets.shape[0],)).cuda()



        batch_idxs_sv=torch.zeros((0,)).cuda()
        


        for b in range(batch_offsets.shape[0]-1):



          #####################cuda
          voxel_xyz=pointgroup_ops.voxelization(coords_float[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          unary_feat=pointgroup_ops.voxelization(unary_feat_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          unary_pred=pointgroup_ops.voxelization(semantic_scores_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          pair_feat=pointgroup_ops.voxelization(pair_feat_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b]) 

          unary_feats=torch.cat((unary_feats, unary_feat),0)      
          unary_preds=torch.cat((unary_preds, unary_pred),0)
          #pair_feats=torch.cat((pair_feats, unary_feat),0)  
           
          labels_sv=torch.cat((labels_sv, group_labels[b]),0) 
          ####################end cuda
          voxel_xyzs=torch.cat((voxel_xyzs, voxel_xyz),0)

          batch_idx_sv=torch.zeros((unary_feat.shape[0])).cuda()
          batch_idx_sv[:]=b
          batch_idxs_sv=torch.cat((batch_idxs_sv, batch_idx_sv),0)
          '''for i in range(group_labels[b].shape[0]):
            l=int(group_labels[b][i].item())

            if l!=-100:
              if model.start[l]==1:
                model.start[l]==0
                model.feat[0,l,:]=pair_feats[i,:] #.detach().cpu().numpy()[0,:]  #torch.mean(pair_feat.detach(),1)
              else:
                #print (model.feat.shape, pair_feat.shape)
                model.feat[0,l,:]=0.9*model.feat[0,l,:]+0.1*pair_feat[i,:] #.detach().cpu().numpy()[0,:]'''

        labels_sv=labels_sv.long().cuda()

        
        unary_feats_expand=torch.unsqueeze(unary_feats,0)
        #pair_feats_expand=torch.unsqueeze(pair_feats,0)
        

        unary_feats_transformer_only=model.layers[0](unary_feats_expand,model.feat)
        unary_feats_transformer_only=model.layers[1](unary_feats_transformer_only,model.feat)
        unary_feats_transformer_only=model.layers[2](unary_feats_transformer_only,model.feat)     
        unary_feats_transformer_only=model.layers[3](unary_feats_transformer_only,model.feat)    

        
        
        
        unary_feats_transformer_only=unary_feats_transformer_only[0]
        unary_feats_transformer=torch.cat((unary_feats, unary_feats_transformer_only), -1)
        
          
        
        
        unary_preds_transformer=model.linear_sv(unary_feats_transformer)



        




        '''feat_point=torch.zeros((coords_float.shape[0],32)).cuda()
        feat_point[:]=-100
        for b in range(batch_offsets.shape[0]-1):
          sum=0
          for k in batch['group2point_full'][b].keys():
              #print ( batch_offsets[b].item(), batch['group2point_full'][b][k])
              idxs=batch['group2point_full'][b][k]
              idxs=np.asarray(idxs)+batch_offsets[b].item()
              feat_point[idxs]=unary_feats[k]
              sum+=len(idxs)'''
        
        feat_point=unary_feat_pred
        #print (feat_point, torch.where(feat_point==-100), sum, feat_point.shape)
        
        #### offset
        pt_offsets_feats = model.offset(feat_point) #output_feats)
        pt_offsets = model.offset_linear(pt_offsets_feats)   # (N, 3), float32


        voxel_xyz_offsets=torch.zeros((0,3)).cuda() 
        for b in range(batch_offsets.shape[0]-1):
          #####################cuda
          voxel_xyz_offset=pointgroup_ops.voxelization(pt_offsets[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          voxel_xyz_offsets=torch.cat((voxel_xyz_offsets, voxel_xyz_offset),0)
        voxel_xyz=voxel_xyz +voxel_xyz_offsets



        '''gt=torch.load('../PointGroup/train_official/scene0000_01_inst_nostuff.pth')
        seg_gt=gt[2]
        
        seg_gt_sv=torch.zeros((unary_preds_transformer.shape[0],)).cuda()
        seg_gt_sv[:]=-100
        for i in range(group2points[0].shape[0]):
          lb=seg_gt[group2points[0][i]][1]
          #print ('label',group2points[0][i],lb)
          seg_gt_sv[i]=lb'''


        
        
        unary_preds_transformer=torch.nn.functional.softmax(unary_preds_transformer,1)
        '''conf=torch.max(unary_preds_transformer,1)[0]
        print (conf.shape)
        for i in range(conf.shape[0]):
          if conf[i]<0.5:
            unary_preds_transformer[i,:]=0'''
        
        if mode=='train':
          for b in range(batch_offsets.shape[0]-1):
            #print ('batch',b)
            pred_label=torch.argmax(unary_preds_transformer[batch['offsets'][b]:batch['offsets'][b+1]],-1)    #seg_gt_sv #
            pred_xyz=voxel_xyz[batch['offsets'][b]:batch['offsets'][b+1]]
            annotated_idx= torch.nonzero(labels_sv[batch['offsets'][b]:batch['offsets'][b+1]][:,0] > 1).view(-1)
            annotated_xyz=voxel_xyz[batch['offsets'][b]:batch['offsets'][b+1]][annotated_idx]
            annotated_label=labels_sv[batch['offsets'][b]:batch['offsets'][b+1]][annotated_idx]
            anno_idxs= torch.nonzero(annotated_label[:,0]>1).view(-1)          
            #print ('annotated label',annotated_label)
                      
            #print ('annoated_label', annotated_label.shape)
            
            
            from kms import KMeans
            
  
            instance_pred=np.zeros((labels_sv.shape[0]))#.cuda()
            instance_pred[:]=-100
            #print ('labels_sv', labels_sv, labels_sv.shape)
            
            
            
            for cate in range(2,20):
              #print ('cate', cate)
              #print ('pred_label',pred_label)
              if cate not in pred_label:
                continue
              data_cate_idxs= torch.nonzero(pred_label[:]==cate).view(-1)
              anno_cate_idxs= torch.nonzero(annotated_label[:,0]==cate).view(-1)
              
              #print (anno_cate_idxs)
  
              kmeans = KMeans(n_clusters=anno_cate_idxs.shape[0], mode='euclidean', verbose=0)
              #print ('anno cate idxs', anno_cate_idxs)
              if anno_cate_idxs.shape[0]==0:
                continue
              x = pred_xyz[data_cate_idxs] #torch.randn(100000, 64, device='cuda')
              
              #print ('x.shape, ann.shape', x.shape, annotated_xyz[anno_cate_idxs].shape)
              #print (annotated_xyz[anno_cate_idxs].shape)
              #print (x.shape, annotated_xyz[anno_cate_idxs])
              labels = kmeans.fit_predict(x, centroids=annotated_xyz[anno_cate_idxs])
              #print ('lbs',labels.shape, labels)
              #print ('cate idxs',data_cate_idxs)
              #for ll in range(labels.shape[0]):
              #print (labels[ll].item(),ll,'label')
              instance_pred[data_cate_idxs.detach().cpu().numpy()]=anno_cate_idxs.detach().cpu().numpy()[labels.detach().cpu().numpy()] #.detach().cpu().numpy() #[ll] #.item()
  
            #print ('anno idx1',instance_pred)
            #print ('annoidx',annotated_idx)
            instance_pred[annotated_idx.cpu().numpy()]=anno_idxs.cpu().numpy() #[annotated_idx]
            #print ('anno idx2',instance_pred)
            
            
  
            uni=np.unique(instance_pred)
            #print ('uni', uni)
            for i in range(uni.shape[0]):
              #print ('uni',i)
              ins=uni[i]
              if ins==-1:
                continue
              ins_idxs=np.where(instance_pred==ins)[0]
              anno_idx=-1
  
              for idx in ins_idxs:
                if labels_sv[idx,0]!=-100:
                  anno_idx=idx
                  break
              
  
              anno_points=coords_float[np.asarray(batch['group2point_full'][anno_idx])]
              
              anno_bound=calc_bound(anno_points)
              
              assert anno_idx!=-1
              clusters=ins_idxs.shape[0]
              merged_idxs=[anno_idx]
              
              init_clusters=clusters
              
              while 1:
                origin_clusters=clusters
                for idx in ins_idxs:
                  if idx in merged_idxs:
                    continue
                  points=coords_float[np.asarray(batch['group2point_full'][idx])]
                  bound=calc_bound(points)
                  #if (intersect(anno_bound, bound)):
                  #  print ('1111111111111111111111111111111111111111111111111111111111111111111')
                  
                  if intersect(anno_bound, bound):
                    #print ('1111',anno_bound, bound)
                    anno_bound=merge(anno_bound, bound)
                    merged_idxs.append(idx)
                    #print ('2222',anno_bound)
                    clusters-=1
                if origin_clusters==clusters or clusters==1:
                  #print ('clusters',init_clusters,origin_clusters, clusters)
                  break
              
              if clusters>1:
                for idx in ins_idxs:
                  if idx not in merged_idxs:
                    instance_pred[idx]=-100
                
                    
                  
                
  
  
  
  
            instance_pred_point=np.zeros((semantic_scores_pred.shape[0],))
            instance_pred_point[:]=-100
            colors=np.zeros((semantic_scores_pred.shape[0],3))
            
            
            
            ###############filter###############
            #for i in range(labels_sv[batch['offsets'][0]:batch['offsets'][1]].shape[0]):
            #  if 
                
              
            
            
            
            
            
            
    
            #for i in range(instance_pred.shape[0]):
            for i in np.unique(instance_pred):
                if i==-100:
                  continue
                r0=random.uniform(0.2, 1)
                r1=random.uniform(0.2, 1)
                r2=random.uniform(0.2, 1)
                svs=np.where(instance_pred==i)[0]
                
                for sv in svs:
                  
                  idxs=batch['group2point_full'][sv]
                  instance_pred_point[idxs]=instance_pred[sv]#.cpu().numpy()
                  
                  colors[idxs,0]=r0
                  colors[idxs,1]=r1
                  colors[idxs,2]=r2
                
            vertices = coords_float[batch['offsets'][0]:batch['offsets'][1]].cpu().numpy()
  
  
  
  
          semantic_scores_pred_transformer=torch.zeros((semantic_scores_pred.shape[0],20)).cuda()
          semantic_scores_pred_transformer[:]=-100
          
          if 'group2point_full' in batch.keys():
            for k in batch['group2point_full'].keys():
              idxs=batch['group2point_full'][k]
              
              #print ('idxs',idxs, unary_preds[k])
              semantic_scores_pred_transformer[idxs]=unary_preds_transformer[k]

              
          #print ('11111111111111')
          #unary_preds_transformer=torch.nn.functional.softmax(unary_preds_transformer,1)
          with torch.no_grad():
              preds = {}
              #preds['semantic'] = result
              #preds['feat'] = output_feats
              #preds['semantic_crf'] = result_crf
              preds['semantic']=semantic_scores_pred
              preds['semantic_transformer']=semantic_scores_pred_transformer
              preds['Q']=unary_preds_transformer
              preds['conf']=torch.max(unary_preds_transformer,1)[0]
              preds['vertices']=vertices
              preds['colors']= colors
              preds['inst_sv']=instance_pred
              preds['inst_point']=instance_pred_point
            
          return preds
  
        elif mode=='test':

          feat_point=unary_feat_pred
          #print (feat_point, torch.where(feat_point==-100), sum, feat_point.shape)
          
          #### offset
          pt_offsets_feats = model.offset(feat_point) #output_feats)
          pt_offsets = model.offset_linear(pt_offsets_feats) 

          semantic_scores_pred_transformer=torch.zeros((semantic_scores_pred.shape[0],20)).cuda()
          semantic_scores_pred_transformer[:]=-100
          
          if 'group2point_full' in batch.keys():
            for k in batch['group2point_full'].keys():
              idxs=batch['group2point_full'][k]
              #print (k, idxs)
              semantic_scores_pred_transformer[idxs]=unary_preds_transformer[k]
          semantic_preds=torch.max(semantic_scores_pred_transformer,1)[1]
          #### get prooposal clusters
          object_idxs = torch.nonzero(semantic_preds > 1).view(-1)
          #print ('object idxs', object_idxs.shape, semantic_preds.shape)
          
          batch_idxs=torch.zeros((semantic_scores_pred_transformer.shape[0])).cuda().int()

          batch_idxs_ = batch_idxs[object_idxs]
          #print ('batch_idxs',batch_idxs,batch_idxs.shape,batch_idxs_.shape)
          batch_offsets_ = utils.get_batch_offsets(batch_idxs_, 1)
          coords_ = coords_float[object_idxs]
          pt_offsets_ = pt_offsets[object_idxs]

          semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()

          idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, model.cluster_radius, model.cluster_shift_meanActive)
          #print ('coords', coords_.shape, 'meanactive', self.cluster_shift_meanActive)
          #print ('ball query', idx_shift, idx_shift.shape, start_len_shift, start_len_shift.shape, torch.max(start_len_shift[:,0]), torch.max(start_len_shift[:,1]))

          proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), model.cluster_npoint_thre)
          #print ('proposal idx shift before',proposals_idx_shift.shape, proposals_offset_shift.shape)
          proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
          #print ('proposal idx shift after',proposals_idx_shift.shape, proposals_offset_shift.shape)
          # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
          # proposals_offset_shift: (nProposal + 1), int

          '''idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
          
          
          
          proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
          proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
          # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
          # proposals_offset: (nProposal + 1), int
          #print (proposals_idx.shape, proposals_idx, proposals_offset.shape)
          
          
          proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
          proposals_offset_shift += proposals_offset[-1]
          proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
          proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))'''
          proposals_idx=proposals_idx_shift
          proposals_offset=proposals_offset_shift
          #print ('shape', proposals_idx.shape, proposals_offset.shape)

          
          scores=torch.zeros((proposals_offset.shape[0]-1,)).cuda()
          
          #print (semantic_scores_pred_transformer,'semantic')
          for i in range (proposals_offset.shape[0]-1):
            idxs=torch.where(proposals_idx[:,0]==i)
            idxs2=proposals_idx[:,1][idxs].long()
            
            #print (idxs, idxs2)
            score=torch.mean(torch.max(semantic_scores_pred_transformer,1)[0][idxs2]) #[0]
            #print (score, semantic_scores_pred_transformer[idxs2].shape)
            #print (semantic_preds[sv_idxs] ,score)
            scores[i]=score
          #print (scores,'scores')
          #ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)



          ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)
          with torch.no_grad():
              preds = {}
              #preds['semantic'] = result
              #preds['feat'] = output_feats
              #preds['semantic_crf'] = result_crf
              preds['semantic']=semantic_scores_pred
              preds['semantic_transformer']=semantic_scores_pred_transformer
              
              preds['Q']=unary_preds_transformer
              preds['conf']=torch.max(unary_preds_transformer,1)[0]
              preds['coords_offset']=coords_float + pt_offsets
              preds['coords']=coords_float
              preds['coords_instance']=coords_
              #preds['colors']= colors
              #preds['inst_sv']=instance_pred
              #preds['inst_point']=instance_pred_point
              preds['proposal_scores']=ret['proposal_scores']
              preds['pt_offsets'] = pt_offsets
              preds['score'] = scores
              preds['proposals'] = (proposals_idx, proposals_offset)
              preds['group2point_full']= batch['group2point_full']
          return preds


    def model_fn(batch, model, epoch):
        #print ('model fn')
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        labels = batch['labels'].cuda()                        # (N), long, cuda

        groups=batch['groups']
        group2points=batch['g2p_map']
        group_labels=batch['group_labels']



        instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100
        instance_info = batch['instance_info'][:,:3].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        '''classes=[]
        poss=[]
        negs=[]
        for i in range(20):
            classes.append([])

        for g in range(len(groups)):
            group=groups[g]
            for i in range(20):
                for s in range(len(group[i])):
                    classes[i].append((i,g,group[i][s]))  
         
        ignore=[]
        mini=10 #min(min(map(len, classes)),30)
        for i in range(20):
            random.shuffle(classes[i])
            if len(classes[i])==0:
              ignore.append(i)
              continue
            if len(classes[i])>=mini:
              classes[i]=classes[i][:mini]
            else:
              while(len(classes[i])<10):
                classes[i].append(random.choice(classes[i]))'''



        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda
        
        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret, unary_feat_pred = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        #ret_pair, pair_feat_pred = model_mom(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        output = model.input_conv_mom(input_)
        output = model.unet_mom(output)
        output = model.output_layer_mom(output)
        pair_feat_pred = output.features[p2v_map.long()]
        
        pair_feat_pred=pair_feat_pred.detach()


        with torch.no_grad():  # no gradient to keys
            model._momentum_update_key_encoder()

        semantic_scores_pred = ret['semantic_scores']
        
        labels_sv=torch.zeros((0,1)) #.cuda()

        label_to_append = torch.zeros((1,1)).cuda()
          
        pair_feats=torch.zeros((0,32)).cuda()
        pair_preds=torch.zeros((0,20)).cuda()
        unary_feats=torch.zeros((0,32)).cuda()
        unary_preds=torch.zeros((0,20)).cuda()
        tmp_feat=torch.zeros((20, model.m)).cuda().detach()
        tmp_feat[:]=0 
        tmp_feat_cnt=torch.zeros((20, )).cuda().detach()
        tmp_feat_cnt[:]=0



        voxel_xyzs=torch.zeros((0,3)).cuda()  

        pair_feat_pred=pair_feat_pred.detach()
        
        batch_idxs_sv=torch.zeros((0,)).cuda()
        
        
        #instance_infos=torch.zeros((0,3)).cuda()  
        #instance_labels=torch.zeros((0,1)).cuda()
        #instance_labels[:]=-100
                 
        
        #instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100
        #instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        #instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda
        #print ('inst',instance_labels.shape, instance_info.shape, instance_pointnum.shape)
        
        
        
        #instance_labels_point=torch.unsqueeze(instance_labels_point,1).float()
        for b in range(batch_offsets.shape[0]-1):



          #instance_label=pointgroup_ops.voxelization(instance_labels_point[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          #print ('bat',b, instance_label)
          #instance_info=pointgroup_ops.voxelization(instance_info_point[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          
          
          
          
          
          #####################cuda
          voxel_xyz=pointgroup_ops.voxelization(coords_float[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
        
          unary_feat=pointgroup_ops.voxelization(unary_feat_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          unary_pred=pointgroup_ops.voxelization(semantic_scores_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          pair_feat=pointgroup_ops.voxelization(pair_feat_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b]) 

          unary_feats=torch.cat((unary_feats, unary_feat),0)      
          unary_preds=torch.cat((unary_preds, unary_pred),0)
          #pair_feats=torch.cat((pair_feats, unary_feat),0)  
           
          labels_sv=torch.cat((labels_sv, group_labels[b]),0)
          
          batch_idx_sv=torch.zeros((unary_feat.shape[0])).cuda()
          batch_idx_sv[:]=b
          batch_idxs_sv=torch.cat((batch_idxs_sv, batch_idx_sv),0)
          voxel_xyzs=torch.cat((voxel_xyzs, voxel_xyz),0)          



          #instance_labels=torch.cat((instance_labels, instance_label),0)
          #instance_infos=torch.cat((instance_infos, instance_info),0)          
          ####################end cuda




          for i in range(group_labels[b].shape[0]):
            l=int(group_labels[b][i].item())


            if l!=-100:
              if model.start[l]==1:
                model.start[l]==0
                model.feat[0,l,:]=pair_feats[i,:] #.detach().cpu().numpy()[0,:]  #torch.mean(pair_feat.detach(),1)
              else:
                #print (model.feat.shape, pair_feat.shape)
                model.feat[0,l,:]=0.9*model.feat[0,l,:]+0.1*pair_feat[i,:] #.detach().cpu().numpy()[0,:]





        labels_sv=labels_sv.long().cuda()
        '''instance_labels=torch.round(instance_labels)
        #print (torch.unique(instance_labels))
        #print (torch.unique(instance_labels_point),'point')    
        instance_num = int(instance_labels_point.max()) + 1

        instance_pointnum=torch.zeros((instance_num)).cuda()
        for i_ in range(instance_num):
            inst_idx_i = torch.where(instance_labels == i_)
            #print ('inst idx i', inst_idx_i, inst_idx_i[0], inst_idx_i[0].shape[0])

            ### instance_pointnum

            instance_pointnum[i_]=inst_idx_i[0].shape[0]'''
  
        
        
        


        unary_feats_expand=torch.unsqueeze(unary_feats,0)
        #pair_feats_expand=torch.unsqueeze(pair_feats,0)
        

        unary_feats_transformer_only=model.layers[0](unary_feats_expand,model.feat)
        unary_feats_transformer_only=model.layers[1](unary_feats_transformer_only,model.feat)
        unary_feats_transformer_only=model.layers[2](unary_feats_transformer_only,model.feat)     
        unary_feats_transformer_only=model.layers[3](unary_feats_transformer_only,model.feat)    

        
        
        
        unary_feats_transformer_only=unary_feats_transformer_only[0]
        unary_feats_transformer=torch.cat((unary_feats, unary_feats_transformer_only), -1)
        
          
        
        
        unary_preds_transformer=model.linear_sv(unary_feats_transformer)



        




        '''feat_point=torch.zeros((coords_float.shape[0],32)).cuda()
        feat_point[:]=-100
        for b in range(batch_offsets.shape[0]-1):
          sum=0
          for k in batch['group2point_full'][b].keys():
              #print ( batch_offsets[b].item(), batch['group2point_full'][b][k])
              idxs=batch['group2point_full'][b][k]
              idxs=np.asarray(idxs)+batch_offsets[b].item()
              feat_point[idxs]=unary_feats[k]
              sum+=len(idxs)'''
        
        feat_point=unary_feat_pred
        #print (feat_point, torch.where(feat_point==-100), sum, feat_point.shape)
        
        #### offset
        pt_offsets_feats = model.offset(feat_point) #output_feats)
        pt_offsets = model.offset_linear(pt_offsets_feats)   # (N, 3), float32
        #print ('1111',unary_feats, feat_point.shape, feat_point, pt_offsets, pt_offsets.shape)

        '''ret['pt_offsets'] = pt_offsets

        if(epoch > 0): #self.prepare_epochs):
            #### get prooposal clusters
            object_idxs = torch.nonzero(unary_preds_transformer > 1).view(-1)
            print ('object idxs', object_idxs.shape, unary_preds_transformer.shape)

            batch_idxs_ = batch_idxs_sv[object_idxs]
            print ('batch_idxs',batch_idxs,batch_idxs.shape,batch_idxs_.shape)
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
            coords_ = voxel_xyzs[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]

            semantic_preds_cpu = unary_preds_transformer[object_idxs].int().cpu()

            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
            print ('coords', coords_.shape, 'meanactive', self.cluster_shift_meanActive)
            print ('ball query', idx_shift, idx_shift.shape, start_len_shift, start_len_shift.shape, torch.max(start_len_shift[:,0]), torch.max(start_len_shift[:,1]))

            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
            print ('proposal idx shift before',proposals_idx_shift.shape, proposals_offset_shift.shape)
            proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
            print ('proposal idx shift after',proposals_idx_shift.shape, proposals_offset_shift.shape)
            # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset_shift: (nProposal + 1), int

            idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
            
            
            
            proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
            # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int
            print (proposals_idx.shape, proposals_idx, proposals_offset.shape)
            
            
            proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
            proposals_offset_shift += proposals_offset[-1]
            proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
            proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))
            
            max_len=0
            for i in range(0,proposals_offset.shape[0]-1):
              length=proposals_offset[i+1]-proposals_offset[i]
              if length>max_len:
                max_len=length
            print (max_len, proposals_offset.shape[0])
            
            max_length=100
            
            cluster2point=torch.zeros((cluster_size,100)).cuda()
            for i in range(0,proposals_offset.shape[0]-1):
              length=min(99,proposals_offset[i+1]-proposals_offset[i])
              cluster2point[i,0]=length
              cluster2point[i,1:(length+1)]=proposals_idx[proposals_offset[i]:proposals_offset[i+1]]
              
            
        
            cluster_scores=torch.zeros((0,)).cuda()
            for b in range(batch_offsets.shape[0]-1):
    
    
    
              #####################cuda

              voxel_xyz=pointgroup_ops.voxelization(proposals_idx, group2points[b])
              

            scores = self.score_linear(score_feats)  # (nProposal, 1)
            

            ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

        return ret'''
        
        


        semantic_scores_pred = ret['semantic_scores'] # (N, nClass) float32, cudas
        loss_inp = {}

        loss_inp['semantic_scores_pred'] = (semantic_scores_pred, labels)
        loss_inp['unary_transformer'] = (unary_preds_transformer, labels_sv)
        #loss_inp['pair'] = (pair_preds, labels_sv)
        loss_inp['unary_init'] = (unary_preds, labels_sv)
        #print (voxel_xyzs.shape, instance_info.shape)
        loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)

        loss, semantic_loss_pred, semantic_loss_transformer,  semantic_loss_pair, semantic_loss_unary ,  offset_norm_loss, offset_dir_loss, loss_out, infos = loss_fn(loss_inp, epoch)


        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():

            preds = {}
            #preds['semantic'] = semantic_scores

            
            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, semantic_loss_pred, semantic_loss_transformer, semantic_loss_pair, semantic_loss_unary ,  offset_norm_loss, offset_dir_loss, visual_dict, meter_dict


    def loss_fn(loss_inp, epoch):
        #print ('loss fn')

        loss_out = {}
        infos = {}

        '''semantic loss'''
        
        
        #loss_inp['semantic_scores_pred'] = (semantic_scores_pred, labels)
        #loss_inp['unary_transformer'] = (unary_preds, labels_sv)
        #loss_inp['pair'] = (pair_preds, labels_sv)
        #loss_inp['unary_init'] = (unary_preds_init, labels_sv)
        
        
        

        semantic_scores_pred, semantic_labels_pred = loss_inp['semantic_scores_pred']
        unary_preds_transformer, labels_sv = loss_inp['unary_transformer']
        #pair_preds, labels_sv = loss_inp['pair']
        unary_preds, labels_sv = loss_inp['unary_init']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss_pred = semantic_criterion(semantic_scores_pred, semantic_labels_pred)
        #print (unary_preds.shape, labels_sv.shape, semantic_labels_pred.shape)
        semantic_loss_transformer = semantic_criterion(unary_preds_transformer, labels_sv[:,0])

        #semantic_loss_pair = semantic_criterion(pair_preds, labels_sv[:,0])
        semantic_loss_unary = semantic_criterion(unary_preds, labels_sv[:,0])

        loss_out['semantic_pred'] = (semantic_loss_pred, semantic_scores_pred.shape[0])
        loss_out['semantic_transformer'] = (semantic_loss_transformer, unary_preds.shape[0])

        #loss_out['voxel_pair'] = (semantic_loss_pair, pair_preds.shape[0])
        loss_out['voxel_unary'] = (semantic_loss_unary, unary_preds.shape[0])















        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        #instance_labels=torch.squeeze(instance_labels,1)
        #print (pt_offsets, coords, instance_info, instance_labels)
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        #print (gt_offsets)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        #print (pt_diff.shape, gt_offsets.shape, coords.shape, instance_info, instance_labels.shape)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        #print (pt_dist)
        valid = (instance_labels != cfg.ignore_label).float()
        #print ('instance labels', instance_labels.shape, torch.unique(instance_labels))

        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
        
        #print (pt_dist.shape, valid.shape,  torch.sum(pt_dist * valid),torch.sum(valid), torch.unique(valid))

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6) 
        #print (gt_offsets_ , pt_offsets_, pt_offsets)
        #print (torch.unique(direction_diff), torch.sum(direction_diff * valid), torch.sum(valid))

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())





        '''total loss'''
        loss = cfg.loss_weight[0] * (semantic_loss_pred +semantic_loss_transformer+semantic_loss_unary) + cfg.loss_weight[1] * offset_norm_loss + cfg.loss_weight[2] * offset_dir_loss 
        #if(epoch > cfg.prepare_epochs):
        #    loss += (cfg.loss_weight[3] * score_loss)

        return loss, semantic_loss_pred, semantic_loss_transformer,  semantic_loss_unary, semantic_loss_unary , offset_norm_loss, offset_dir_loss, loss_out, infos


    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        '''
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores


    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn
