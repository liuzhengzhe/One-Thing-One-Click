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


import numpy as np
import random,torch




data=torch.load('gt_train/Area_1_WC_1_inst_nostuff.pth')
#data=torch.load('../data/train_cuda_s3dis/Area_6_office_1_inst_nostuff.pth')
vertices_coords=data[0]
semantic_pred=data[4]
colors2=np.zeros((semantic_pred.shape[0],3))
for i in np.unique(semantic_pred):
    r0=random.uniform(0.2, 1)
    r1=random.uniform(0.2, 1)
    r2=random.uniform(0.2, 1)

    idxs=np.where(semantic_pred==i)[0]
     
    colors2[idxs,0]=r0
    colors2[idxs,1]=r1
    colors2[idxs,2]=r2  


write_ply_point_normal('vis_sv.ply', vertices_coords, colors2)

#data=torch.load('../data/train_cuda_s3dis/Area_6_office_1_inst_nostuff.pth')
vertices_coords=data[0]
semantic_pred=data[2]
colors2=np.zeros((semantic_pred.shape[0],3))
for i in range(13):
    r0=random.uniform(0.2, 1)
    r1=random.uniform(0.2, 1)
    r2=random.uniform(0.2, 1)

    idxs=np.where(semantic_pred==i)[0]
     
    colors2[idxs,0]=r0
    colors2[idxs,1]=r1
    colors2[idxs,2]=r2  


write_ply_point_normal('vis_seg.ply', vertices_coords, colors2)
