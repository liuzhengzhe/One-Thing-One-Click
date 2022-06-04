import torch
import glob
import numpy as np
paths=glob.glob('train_cuda/*.pth')

nums=0
for path in paths:
  name=path.split('/')[-1]
  data=torch.load(path)
  #print (data[1].shape)
  
  data2=torch.load('train_cuda_512/'+name)
  #print (data[1].shape, )
  num=np.where(data2[2]!=-100)[0].shape[0]
  nums+=num

print (nums/len(paths))