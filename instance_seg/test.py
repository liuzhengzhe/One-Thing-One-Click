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
import glob

def write_ply_point_normal(name, vertices, colors):
  fout = open(name, 'w')
  fout.write("ply\n")
  fout.write("format ascii 1.0\n")
  fout.write("element vertex "+str(len(vertices))+"\n")
  fout.write("property float x\n")
  fout.write("property float y\n")
  fout.write("property float z\n")
  fout.write("property uchar red\n")
  fout.write("property uchar green\n")
  fout.write("property uchar blue\n")
  fout.write("end_header\n")
  for ii in range(len(vertices)):
    fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(min(255,int(255*colors[ii,2])))+" "+str(min(255,int(255*colors[ii,1])))+" "+str(min(255,int(255*colors[ii,0])))+"\n")

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

    try:
      os.mkdir('instance_pred')
    except:
      pass
    try:
      os.mkdir('instance_result')
    except:
      pass

    from data.scannetv2_inst import Dataset
    dataset = Dataset(test=True)
    dataset.testLoader(cfg.data_root+'val_cuda/')

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
            test_scene_name = dataset.test_file_names[int(batch['id'][0]/3)].split('/')[-1][:12]
            print (test_scene_name)


            start1 = time.time()
            preds = model_fn(batch, model, epoch, 'test')
            end1 = time.time() - start1


            semantic_scores = preds['semantic_transformer'] 
            
            
            
            
            if i%3==0:
              semantic_acc=semantic_scores*0
            semantic_acc+=semantic_scores
            #print (semantic_acc)
            if i%3!=2:
              continue
            #semantic_acc=semantic_scores
            #print (torch.unique(semantic_acc))            
            semantic_pred = semantic_acc.max(1)[1]  # (N) long, cuda
            semantic_pred=semantic_pred.detach().cpu().numpy()  #.cpu().numpy()   
            semantic_acc=semantic_acc.cpu().numpy()/3.0

            #if i%3==2:

            labels=batch['labels'].detach().cpu().numpy()  #[:int(N/3)]

            #print (cfg.data_root+cfg.dataset+'/unary_pred/'+test_scene_name+'.npy')
            #np.save(cfg.data_root+cfg.dataset+'/unary_pred/'+test_scene_name+'.npy',semantic_scores.detach().cpu().numpy())
            #np.save(cfg.data_root+cfg.dataset+'/unary_feat/'+test_scene_name+'.npy',preds['feats'].detach().cpu().numpy())

            f1=open('result/pred/'+test_scene_name+'.txt','w')
            #print (labels.shape, semantic_pred.shape)
            for j in range(labels.shape[0]):
              f1.write(str(semantic_pred[j])+'\n')
            f1.close()
              
            

            #print (preds.keys())
            scores = preds['score']   # (nProposal, 1) float, cuda
            scores_pred = scores.view(-1) #torch.sigmoid(scores.view(-1))
            #print (torch.unique(scores_pred), scores_pred.shape)

            proposals_idx, proposals_offset = preds['proposals']
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device) # (nProposal, N), int, cuda
            proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

            semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long
            #print ('scores_pred',scores_pred)
            ##### score threshold
            score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
            scores_pred = scores_pred[score_mask]

            #print (proposals_offset.shape, scores_pred.shape, proposals_pred.shape, score_mask.shape)
            proposals_pred = proposals_pred[score_mask]
            semantic_id = semantic_id[score_mask]


            ##### npoint threshold
            proposals_pointnum = proposals_pred.sum(1)
            npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
            scores_pred = scores_pred[npoint_mask]
            proposals_pred = proposals_pred[npoint_mask]
            semantic_id = semantic_id[npoint_mask]
            #print (proposals_pred.shape)
            ##### nms
            if semantic_id.shape[0] == 0:
                pick_idxs = np.empty(0)
            else:
                proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(), cfg.TEST_NMS_THRESH)  # int, (nCluster, N)
            clusters = proposals_pred[pick_idxs]
            cluster_scores = scores_pred[pick_idxs]
            cluster_semantic_id = semantic_id[pick_idxs]
            #print (proposals_pred.shape, clusters.shape)
            nclusters = clusters.shape[0]

            ##### prepare for evaluation
            if cfg.eval:
                pred_info = {}
                pred_info['conf'] = cluster_scores.cpu().numpy()
                pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                pred_info['mask'] = clusters.cpu().numpy()
                gt_file = os.path.join('./', cfg.split + '_gt', test_scene_name + '.txt')
                gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)
                matches[test_scene_name] = {}
                matches[test_scene_name]['gt'] = gt2pred
                matches[test_scene_name]['pred'] = pred2gt

            ##### save files
            start3 = time.time()
            if cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)
    
            if cfg.save_pt_offsets:
                os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                pt_offsets_np = pt_offsets.cpu().numpy()
                coords_np = batch['locs_float'].numpy()
                coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)   # (N, 6)
                np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)
    
    
    
    
            vertices_coords = preds['coords'].cpu().numpy()
            vertices_coords_offset = preds['coords_offset'].cpu().numpy()  # coords_float[batch['offsets'][0]:batch['offsets'][1]].cpu().numpy()
    
            #print (vertices_coords.shape, semantic_pred.shape)
            #semantic_pred_sv_l2=semantic_pred_sv[np.where(semantic_pred_sv>1)]
            #print (vertices_coords.shape, semantic_pred.shape)     
            #semantic_pred=torch.load(glob.glob('../data/val_cuda/scene0011_00*')[0])[2]            
            #print (semantic_pred.shape, np.unique(semantic_pred))             
            colors2=np.zeros((semantic_pred.shape[0],3))
            for i in range(20):
                r0=random.uniform(0.2, 1)
                r1=random.uniform(0.2, 1)
                r2=random.uniform(0.2, 1)
    
                idxs=np.where(semantic_pred==i)[0]
                  
                colors2[idxs,0]=r0
                colors2[idxs,1]=r1
                colors2[idxs,2]=r2  
            
    
            write_ply_point_normal('instance_pred/'+test_scene_name+'_origin_vertices_origin.ply', vertices_coords, colors2)
            write_ply_point_normal('instance_pred/'+test_scene_name+'_origin_vertices_offset.ply', vertices_coords_offset, colors2)
    
            #print ('cluster', clusters.shape, N)

            colors=np.zeros((N,3))
            for i in range(clusters.shape[0]):
                xx=random.uniform(0.2, 1)
                r0=random.uniform(0.2, 1)
                r1=random.uniform(0.2, 1)
                r2=random.uniform(0.2, 1)
                idxs=np.where(clusters[i,:].cpu().numpy()==1)[0]

                  
                colors[idxs,0]=r0
                colors[idxs,1]=r1
                colors[idxs,2]=r2
            #print (preds['coords_instance'].shape, colors.shape)
            write_ply_point_normal('instance_pred/'+test_scene_name+'_inst.ply', vertices_coords, colors)
    
    
    
            if(epoch > cfg.prepare_epochs and cfg.save_instance):
                f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    score = cluster_scores[proposal_id]
                    f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                    if proposal_id < nclusters - 1:
                        f.write('\n')
                    np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                f.close()
            end3 = time.time() - start3
            end = time.time() - start
            start = time.time()
    
            ##### print
            logger.info("instance iter: {}/{} point_num: {} ncluster: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(batch['id'][0] + 1, len(dataset.test_files), N, nclusters, end, end1, end3))
    
        ##### evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            avgs = eval.compute_averages(ap_scores)
            eval.print_results(avgs)
    
              
              
                  
                  
              
              

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
    model = model.cuda()

    # logger.info(model)
    #logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(test=True)

    ##### load model
    utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, model_fn, data_name, cfg.test_epoch)
