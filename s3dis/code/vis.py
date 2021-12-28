import torch,glob
import numpy as np
paths=glob.glob('train_cuda/Area_1_conferenceRoom_2_inst_nostuff.pth')
for path in paths:
  data=torch.load(path)
  seg=data[-1]
  print (len(seg), np.unique(seg))
  #for i in range(seg.shape[])