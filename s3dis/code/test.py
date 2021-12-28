'''
PointGroup test.py
Written by Li Jiang
Revised by Zhengzhe Liu
'''
colors=[[0, 255, 0],
[174, 199, 232],
[152, 223, 138],
[31, 119, 180 ],
[255, 187, 120],
[188, 189, 34],
[140, 86, 75 ],
[255, 152, 150],
[214, 39, 40],
[197, 176, 213],
[148, 103, 189],
[196, 156, 148],
[23, 190, 207],
[247, 182, 210],
[66, 188, 102],
[219, 219, 141],
[140, 57, 197],
[202, 185, 52],
[51, 176, 203],
[200, 54, 131],
[92, 193, 61],
[78,71, 183],
[172, 114, 82],
[255, 127, 14],
[91, 163, 138],
[153, 98, 156],
[140, 153, 101],
[158, 218, 229],
[100, 125, 154],
[178, 127, 135],
[146, 111, 194],
[44, 160, 44],
[112, 128, 144],
[96, 207, 209],
[227, 119, 194],
[213, 92, 176],
[94, 106, 211],
[82, 84, 163],
[100, 85, 144],
[0, 0, 255],
[0, 0, 0]]

names=['wall',
'floor',
'cabinet',
'bed',
'chair',
'sofa',
'table',
'door',
'window',
'bookshelf',
'picture',
'counter',
'desk',
'curtain',
'refridgerator',
'shower curtain',
'toilet',
'sink',
'bathtub',
'otherfurniture']


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

fx_rgb = 5.1885790117450188e+02
fy_rgb = 5.1946961112127485e+02
cx_rgb = 3.2558244941119034e+02
cy_rgb = 2.5373616633400465e+02


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
    #logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    '''try:
      os.mkdir(cfg.data_root+cfg.dataset+'/unary_pred')
    except:
      pass
    try:
      os.mkdir(cfg.data_root+cfg.dataset+'/unary_feat')
    except:
      pass'''
    try:
      os.mkdir('result')
    except:
      pass
    try:
      os.mkdir('result/pred')
    except:
      pass

    from data.scannetv2_inst import Dataset
    dataset = Dataset(test=True)
    dataset.testLoader('../data/val_cuda/')

    dataloader = dataset.test_data_loader
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    with torch.no_grad():
        model = model.eval()
        start = time.time()

        matches = {}
        for i, batch in enumerate(dataloader):
            N = batch['feats'].shape[0]
            test_scene_name = dataset.test_file_names[int(batch['id'][0]/3)].split('/')[-1].split('.')[0] #[:12]
            print (test_scene_name)


            start1 = time.time()
            preds = model_fn(batch, model, epoch)
            end1 = time.time() - start1


            semantic_scores = preds['semantic_transformer'] 

            if i%3==0:
              semantic_acc=semantic_scores*0

            semantic_acc+=semantic_scores

            if i%3==0:
              print (i)
              semantic_pred = semantic_acc.max(1)[1]  # (N) long, cuda
              semantic_pred=semantic_pred.detach().cpu().numpy()   
              labels=batch['labels'].detach().cpu().numpy()  #[:int(N/3)]

              #print (cfg.data_root+cfg.dataset+'/unary_pred/'+test_scene_name+'.npy')
              #np.save(cfg.data_root+cfg.dataset+'/unary_pred/'+test_scene_name+'.npy',semantic_scores.detach().cpu().numpy())
              #np.save(cfg.data_root+cfg.dataset+'/unary_feat/'+test_scene_name+'.npy',preds['feats'].detach().cpu().numpy())
              #if 'lobby' in test_scene_name:
              #  continue
              
              f1=open('result/pred/'+test_scene_name.replace('_nostuff','')+'.txt','w')
              for j in range(labels.shape[0]):
                f1.write(str(semantic_pred[j])+'\n')
              f1.close()

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
    #init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    #logger.info('=> creating model ...')
    ##logger.info('Classes: {}'.format(cfg.classes))

    cfg.dataset='val_weakly'

    if model_name == 'pointgroup':
        from model.pointgroup.pointgroup import PointGroup as Network
        from model.pointgroup.pointgroup import model_fn_decorator
    else:
        print("Error: no model version " + model_name)
        exit(0)
    model = Network(cfg)
    use_cuda = torch.cuda.is_available()
    #logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    #model = model.c()

    # logger.info(model)
    #logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(test=True)

    ##### load model
    utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, model_fn, data_name, cfg.test_epoch)
