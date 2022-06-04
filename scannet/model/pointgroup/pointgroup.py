
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
   
    #print ('attmask score ',  scores.shape)
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
    #print ('attoutput', output.shape)
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
        #print (x.shape, e_outputs.shape)
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

        '''#### offset
        self.offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        #### score branch
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
                param.requires_grad = False

        #### load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))'''

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

def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):
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





        for b in range(batch_offsets.shape[0]-1):



          #####################cuda

          unary_feat=pointgroup_ops.voxelization(unary_feat_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          unary_pred=pointgroup_ops.voxelization(semantic_scores_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          pair_feat=pointgroup_ops.voxelization(pair_feat_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b]) 
          
          #print (unary_feat.shape, pair_feat.shape, unary_feat_pred.shape, pred_feat_pred.shape)

          unary_feats=torch.cat((unary_feats, unary_feat),0)      
          unary_preds=torch.cat((unary_preds, unary_pred),0)
          pair_feats=torch.cat((pair_feats, unary_feat),0)  
           
          labels_sv=torch.cat((labels_sv, group_labels[b]),0) 
          ####################end cuda




        labels_sv=labels_sv.long().cuda()
    
              
        

        unary_feats_expand=torch.unsqueeze(unary_feats,0)
        #pair_feats_expand=torch.unsqueeze(pair_feats,0)

        
        #for i in range(model.N):
        #unary_feats_transformer_only=model.layers[i](unary_feats_expand,model.feat)
        unary_feats_transformer_only=model.layers[0](unary_feats_expand,model.feat)
        unary_feats_transformer_only=model.layers[1](unary_feats_transformer_only,model.feat)
        unary_feats_transformer_only=model.layers[2](unary_feats_transformer_only,model.feat)     
        unary_feats_transformer_only=model.layers[3](unary_feats_transformer_only,model.feat)       
        #print (torch.unique(unary_feats), torch.unique(unary_feats_transformer_only))
        unary_feats_transformer_only=unary_feats_transformer_only[0]
        unary_feats_transformer=torch.cat((unary_feats, unary_feats_transformer_only), -1)
        
          
        
        
        unary_preds_transformer=model.linear_sv(unary_feats_transformer)


        

        
        
        
        '''unary_feats_expand=torch.unsqueeze(unary_feats,0)
        pair_feats_expand=torch.unsqueeze(pair_feat,0)
        for i in range(model.N):
          unary_feats_transformer=model.layers[i](unary_feats_expand,pair_feats_expand)
        unary_feats_transformer=unary_feats_transformer[0]
          
        
        
        unary_preds_transformer=model.linear_sv(unary_feats_transformer)'''




        semantic_scores_pred_transformer=torch.zeros((semantic_scores_pred.shape[0],20)).cuda()
        semantic_scores_pred_transformer[:]=-100
        
        if 'group2point_full' in batch.keys():
          for k in batch['group2point_full'].keys():
            idxs=batch['group2point_full'][k]
            
            #print ('idxs',idxs, unary_preds[k])
            semantic_scores_pred_transformer[idxs]=unary_preds_transformer[k]
            
        
        unary_preds_transformer=torch.nn.functional.softmax(unary_preds_transformer,1)
        with torch.no_grad():
            preds = {}
            #preds['semantic'] = result
            #preds['feat'] = output_feats
            #preds['semantic_crf'] = result_crf
            preds['semantic']=semantic_scores_pred
            preds['semantic_transformer']=semantic_scores_pred_transformer
            preds['Q']=unary_preds_transformer
            preds['conf']=torch.max(unary_preds_transformer,1)[0]
            #print (preds['conf'])
            '''preds['pt_offsets'] = pt_offsets
            if (epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)'''

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





        pair_feat_pred=pair_feat_pred.detach()
        
        
        for b in range(batch_offsets.shape[0]-1):



          #####################cuda

          unary_feat=pointgroup_ops.voxelization(unary_feat_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          unary_pred=pointgroup_ops.voxelization(semantic_scores_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b])
          pair_feat=pointgroup_ops.voxelization(pair_feat_pred[batch['offsets'][b]:batch['offsets'][b+1]], group2points[b]) 

          unary_feats=torch.cat((unary_feats, unary_feat),0)      
          unary_preds=torch.cat((unary_preds, unary_pred),0)
          pair_feats=torch.cat((pair_feats, pair_feat),0)  
           
          labels_sv=torch.cat((labels_sv, group_labels[b]),0) 
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
    
        

        
        
        
        

        unary_feats_expand=torch.unsqueeze(unary_feats,0)
        #pair_feats_expand=torch.unsqueeze(pair_feats,0)
        

        #for i in range(model.N):
        #  unary_feats_transformer_only=model.layers[i](unary_feats_expand,model.feat)

        unary_feats_transformer_only=model.layers[0](unary_feats_expand,model.feat)
        unary_feats_transformer_only=model.layers[1](unary_feats_transformer_only,model.feat)
        unary_feats_transformer_only=model.layers[2](unary_feats_transformer_only,model.feat)     
        unary_feats_transformer_only=model.layers[3](unary_feats_transformer_only,model.feat)
        
        
        unary_feats_transformer_only=unary_feats_transformer_only[0]
        unary_feats_transformer=torch.cat((unary_feats, unary_feats_transformer_only), -1)
        
          
        
        
        unary_preds_transformer=model.linear_sv(unary_feats_transformer)


        semantic_scores_pred = ret['semantic_scores'] # (N, nClass) float32, cudas
        loss_inp = {}

        loss_inp['semantic_scores_pred'] = (semantic_scores_pred, labels)
        loss_inp['unary_transformer'] = (unary_preds_transformer, labels_sv)
        #loss_inp['pair'] = (pair_preds, labels_sv)
        loss_inp['unary_init'] = (unary_preds, labels_sv)


        loss, semantic_loss_pred, semantic_loss_transformer,  semantic_loss_pair, semantic_loss_unary ,  loss_out, infos = loss_fn(loss_inp, epoch)


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

        return loss, semantic_loss_pred, semantic_loss_transformer, semantic_loss_pair, semantic_loss_unary ,  visual_dict, meter_dict


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



        '''total loss'''
        loss = cfg.loss_weight[0] * (semantic_loss_pred +semantic_loss_transformer+semantic_loss_unary) 
        #if(epoch > cfg.prepare_epochs):
        #    loss += (cfg.loss_weight[3] * score_loss)

        return loss, semantic_loss_pred, semantic_loss_transformer,  semantic_loss_unary, semantic_loss_unary , loss_out, infos


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
