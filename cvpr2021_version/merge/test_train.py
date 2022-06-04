'''
PointGroup test.py
Written by Li Jiang
'''


import torch
import time
import numpy as np
import random
import os
from util.config import cfg
cfg.task = 'test'
from util.log import logger
import util.utils as utils
import util.eval as eval
from util_iou import *

import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, argparse

import scannet_util,os,csv

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

mode='train'
conf_thres=0.9

dic={}
dic['wall']=0
dic['floor']=1
dic['cabinet']=2
dic['bed']=3
dic['chair']=4
dic['sofa']=5
dic['table']=6
dic['door']=7
dic['window']=8
dic['bookshelf']=9
dic['picture']=10
dic['counter']=11
dic['desk']=12
dic['curtain']=13
dic['refridgerator']=14
dic['shower curtain']=15
dic['toilet']=16
dic['sink']=17
dic['bathtub']=18
dic['otherfurniture']=19





def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, data_name, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')


    from data.scannetv2_inst import Dataset
    dataset = Dataset(test=True)
    dataset.testLoader()

    dataloader = dataset.test_data_loader

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    try:
      os.mkdir(cfg.data_root+'/train'+str(conf_thres))
    except:
      pass

    with torch.no_grad():
        model = model.eval()
        start = time.time()

        matches = {}
        for i, batch in enumerate(dataloader):
            N = batch['feats'].shape[0]
            test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1][:12]
            


            start1 = time.time()
            preds = model_fn(batch, model, epoch)
            unary=np.amax(preds['unary'].cpu().numpy(),1)

            
            end1 = time.time() - start1
            Q=torch.argmax(preds['Q'],1).cpu().numpy()
            
            crf=preds['semantic_crf'].cpu().numpy()


            confs=preds['conf'].cpu().numpy()
            

            
            group_all=preds['group_full']
            group_to_point=preds['group2point_full']

        
        
            sem_labels = np.zeros((unary.shape[0],)) #remapper[np.array(f2.elements[0]['label'])]
            sem_labels[:]=-100


            groups=[]
            for i in range(20):
                groups.append([])
        
            cnt_group=0
            cnt=len(group_all)

            for i in range(cnt):
              conf=confs[i]
              c=Q[i]
              if conf<conf_thres:#
                continue
              cnt_group+=1
              groups[c].append(group_all[i])
              idxs=group_to_point[group_all[i]]
              sem_labels[idxs]=c
        

            data=torch.load(cfg.data_root+'/train_weakly/'+test_scene_name+'_vh_clean_2.ply_inst_nostuff.pth')

            coords=data[0]
            colors=data[1]
            prev_sem=data[2]
            prev_groups=data[3]
            full_seg=data[4]
            
            
            prev_group_cnt=0
            for g in range(len(prev_groups)):
              g2=prev_groups[g]
              prev_group_cnt+=len(g2)
              for i in g2:
                if (i not in groups[g]) and (i in group_to_point.keys()):
                   groups[g].append(i)
                   idxs=group_to_point[i]
                   sem_labels[idxs]=g
            
            

            sem_labels[np.where(prev_sem!=-100)]=prev_sem[np.where(prev_sem!=-100)]



            torch.save((coords, colors, sem_labels, groups, full_seg), cfg.data_root+'/train'+str(conf_thres)+'/'+test_scene_name+'_inst_nostuff.pth')


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == '__main__':
    init()
    cfg.dataset='train_weakly'
    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    if model_name == 'pointgroup':
        from model.pointgroup.pointgroup import PointGroup as Network
        from model.pointgroup.pointgroup import model_fn_decorator
    else:
        print("Error: no model version " + model_name)
        exit(0)
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(test=True)

    ##### load model
    
    print (model)
    
    #utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, model_fn, data_name, cfg.test_epoch)
