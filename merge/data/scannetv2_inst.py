'''
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
'''

import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch,json
from torch.utils.data import DataLoader
#from util_iou import *
sys.path.append('../')

from util.config import cfg
from util.log import logger
from lib.pointgroup_ops.functions import pointgroup_ops

class Dataset:
    def __init__(self, test=False):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode

        if test:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1


    def trainLoader(self):
        #do not use this 
        train_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, '*' + self.filename_suffix)))

        self.train_files = []
        for i in train_file_names:
          print (i)
          self.train_files.append(torch.load(i))


        logger.info('Training samples: {}'.format(len(self.train_files)))

        train_set = list(range(len(self.train_files)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers,
                                            shuffle=True, sampler=None, drop_last=True, pin_memory=True)


    def valLoader(self):
        val_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'val_fully', '*' + self.filename_suffix)))
        self.val_file_names=val_file_names
        self.val_files = [torch.load(i) for i in val_file_names]

        logger.info('Validation samples: {}'.format(len(self.val_files)))

        val_set = list(range(len(self.val_files)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers,
                                          shuffle=False, drop_last=False, pin_memory=True)


    def testLoader(self):
        self.test_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, '*' + self.filename_suffix)))
        
        self.test_file_names.sort()
        self.test_file_names=self.test_file_names[0:]
        
        self.test_files=[]
        cnt=0
        for i in self.test_file_names:
          print (i)
          cnt+=1        
          data=torch.load(i)
          name=i.split('/')[-1]
          '''fn3 = self.data_root+'/scans/'+name[:12]+'/'+name[:12]+'_vh_clean_2.0.010000.segs.json'

          with open(fn3) as jsondata:
            d = json.load(jsondata)
            seg = d['segIndices']'''
          seg=data[-1]
            
          

          
          full_group=np.unique(seg)
          
          full_group=full_group.tolist()
          
          
          full_group2point=seg
          data=list(data)
          data.append(full_group)
          data.append(full_group2point)
          data.append(i)
          data=tuple(data)
          
          
          
          self.test_files.append(data)
        logger.info('Testing samples ({}): {}'.format(self.test_split, len(self.test_files)))

        test_set = list(np.arange(len(self.test_files)))
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge, num_workers=self.test_workers,
                                           shuffle=True, drop_last=False, pin_memory=True)

    #Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag


    def getInstanceInfo(self, xyz, instance_label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # (nInst), int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}


    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)


    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        #print (valid_idxs,len(xyz),len(valid_idxs))
        onezero=valid_idxs.copy()
        onezero[np.where(valid_idxs==True)]=1
        #print (onezero.sum())
        return xyz_offset, valid_idxs


    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label


    def trainMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []




        groups=[]
        group_to_points=[]
        group_fulls=[]
        group_to_point_fulls=[]

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]
        names=[]        

        total_inst_num = 0
        for i, idx in enumerate(id):
            xyz_origin, rgb, label, group, point2seg, group_full, point2seg_full, name = self.train_files[idx]
            names.append(name)            



            
            xyz_origin=xyz_origin.astype('float32')
            rgb=rgb.astype('float32')
            


            ### jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### elastic
            xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

            ### offset
            xyz -= xyz.min(0)

            ### crop
            xyz, valid_idxs = self.crop(xyz)




            point2seg_ori=np.asarray(point2seg)
            point2seg=point2seg_ori[valid_idxs]
            point2seg=point2seg.tolist()
            group_to_point = {}
            for j in range(len(point2seg)):
              if point2seg[j] not in group_to_point:
                group_to_point[point2seg[j]] = []
              group_to_point[point2seg[j]].append(j)
            for j in range(20):
              group[j]=list(set(group[j]) & set(group_to_point.keys()))
              
              
            point2seg_ori_full=np.asarray(point2seg_full)
            point2seg_full=point2seg_ori_full[valid_idxs]
            point2seg_full=point2seg_full.tolist()
            group_to_point_full = {}
            for j in range(len(point2seg_full)):
              if point2seg_full[j] not in group_to_point_full:
                group_to_point_full[point2seg_full[j]] = []
              group_to_point_full[point2seg_full[j]].append(j)
            
            group_full=list(set(group_full) & set(group_to_point_full.keys()))       
                  
            

            #print ('keys',group_to_point.keys())
            #print ('groups',group)

            '''np1=np.asarray(group_to_point.keys())
            np2=np.asarray(group)

            #print ('111111111111111111',group_to_point.keys())
            #print('222222222222222222',group)

            g=[]
            for j in range(20):
               g.extend(group[j])
            #for j in group_to_point:
            #   print (j)
            #   assert (j in g)

            for j in g:

               if(j not in group_to_point):
                 print ('j',j,'group',group_to_point.keys(),self.train_file_names[idx],(j in point2seg),(272071 in point2seg),(j in point2seg_ori))
                 print ('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                 exit()
            print (len(g),len(group_to_point))'''

            #print ('bef',xyz_middle.shape)
            xyz_middle = xyz_middle[valid_idxs]

            #print ('aft',xyz_middle.shape)
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            #instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            ### get instance information
            '''inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]   # (nInst), list

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num'''

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            groups.append(group)
            group_fulls.append(group_full)
            group_to_points.append(group_to_point)
            group_to_point_fulls.append(group_to_point_full)
            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)
            labels.append(torch.from_numpy(label))
            '''instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)'''

        ### merge all the scenes in the batchd
        

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)                              # float (N, C)
        labels = torch.cat(labels, 0).long()                     # long (N)
        '''instance_labels = torch.cat(instance_labels, 0).long()   # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)'''

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, #'instance_labels': instance_labels,
                #'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'groups': groups, 'group2points': group_to_points, 'group_fulls': group_fulls, 'group2point_fulls': group_to_point_fulls, 'names': name}


    def valMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):
            print ('iiiiiii',i)
            xyz_origin, rgb, label= self.val_files[idx]
            xyz_origin=xyz_origin.astype('float32')
            rgb=rgb.astype('float32')
            ### flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            ### crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            '''instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num'''

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))
            labels.append(torch.from_numpy(label))
            '''instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)'''

        ### merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)    # float (N, 3)
        feats = torch.cat(feats, 0)                                # float (N, C)
        labels = torch.cat(labels, 0).long()                       # long (N)
        '''instance_labels = torch.cat(instance_labels, 0).long()     # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)               # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)          # int (total_nInst)'''

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, #'instance_labels': instance_labels,
                #'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}


    def testMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        batch_offsets = [0]
        groups=[]
        group_to_points=[]
        group_fulls=[]
        group_to_point_fulls=[]


        for i, idx in enumerate(id):
            if 1:
                xyz_origin, rgb, label, group, point2seg, group_full, point2seg_full,name = self.test_files[idx]



                unary=torch.from_numpy(np.load(self.data_root+'/unary_pred/'+name.split('/')[-1][:12]+'.npy'))
                unary_feat=torch.from_numpy(np.load(self.data_root+'/unary_feat/'+name.split('/')[-1][:12]+'.npy'))
                pairwise=torch.from_numpy(np.load(self.data_root+'/rel_feat/'+name.split('/')[-1][:12]+'.npy'))
                prod=torch.from_numpy(np.load(self.data_root+'/rel_pred/'+name.split('/')[-1][:12]+'.npy'))
                


            xyz_origin=xyz_origin.astype('float32')
            rgb=rgb.astype('float32')
            ### flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)
            
            
            point2seg_ori=np.asarray(point2seg)
            point2seg=point2seg_ori #[valid_idxs]

            point2seg=point2seg.tolist()


            group_to_point = {}
            for j in range(len(point2seg)):
              if point2seg[j] not in group_to_point:
                group_to_point[point2seg[j]] = []
              group_to_point[point2seg[j]].append(j)


            for j in range(20):
              group[j]=list(set(group[j]) & set(group_to_point.keys()))
              
              
            point2seg_ori_full=np.asarray(point2seg_full)
            point2seg_full=point2seg_ori_full  #[valid_idxs]
            point2seg_full=point2seg_full.tolist()
            group_to_point_full = {}
            for j in range(len(point2seg_full)):
              if point2seg_full[j] not in group_to_point_full:
                group_to_point_full[point2seg_full[j]] = []
              group_to_point_full[point2seg_full[j]].append(j)
            
            group_full=list(set(group_full) & set(group_to_point_full.keys()))       
              
            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))
            labels.append(torch.from_numpy(label))
            groups.append(group)
            group_fulls.append(group_full)
            group_to_points.append(group_to_point)
            group_to_point_fulls.append(group_to_point_full)
        ### merge all the scenes in the batch
        

        
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                         # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)           # float (N, 3)
        
        feats = torch.cat(feats, 0)                                       # float (N, C)
        labels = torch.cat(labels, 0).long()   
        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)
        #print ('feats',feats,feats.shape)
        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'groups': groups, 'group2points': group_to_points, 'group_fulls': group_fulls, 'group2point_fulls': group_to_point_fulls, 'unary':unary, 'pairwise':pairwise, 'prod':prod, 'unary_feat':unary_feat}
