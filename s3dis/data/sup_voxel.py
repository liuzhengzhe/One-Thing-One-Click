import glob,json,os,h5py,torch,random
import numpy as np
from multiprocessing import Pool
semantic_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter']
semantic_name2id = {}
for i, names in enumerate(semantic_names):
    semantic_name2id[names] = i



#data=torch.load('/home/lzz/sdc1/pg/s3dis/train_weakly2/Area_1_office_3_inst_nostuff.pth')
#print (data[0].shape,data[1].shape,data[2].shape,data[4].shape,data[3],np.unique(data[4]))
#exit()

reg=str(0.01)

paths=glob.glob('features'+reg+'/Area_*/*.h5')
paths.sort()
#for path in paths:
def process(path):
  
  #def process(path):
  area=path.split('/')[-2]
  room=path.split('/')[-1].split('.')[0]
  print (path)
  
  
  if 'Area_2_auditorium_2' in path:  #this data will cause out of memory for 11G GPU. You can keep it if your GPU has larger memory.
    return
  #if area!='Area_5': # and room=='office_21':
  #  return
  '''if area=='Area_4' and room=='office_21':
    return
  if area=='Area_4' and room=='office_16':
    return
  if area=='Area_4' and room=='office_6':
    return
  if area=='Area_4' and room=='office_15':
    return
  if area=='Area_4' and room=='office_20':
    return
  if area=='Area_4' and room=='office_19':
    return
  if area=='Area_4' and room=='office_17':
    return
  if area=='Area_5' and room=='lobby_1':
    return
  if area=='Area_2' and room=='office_4':
    return
  if area=='Area_2' and room=='office_5':
    return
  if area=='Area_4' and room=='office_9':
    return'''
  print (path)
  #if os.path.exists('/home/lzz/sdc1/pg/s3dis/train_weakly/'+area+'_'+room+'_inst_nostuff.pth'):
  #  continue
  
  f = h5py.File('features'+reg+'/'+area+'/'+room+'.h5', 'r')
  #print (f['point2group'].value, np.unique(f['point2group'].value))
  point2voxel=f['point2group'].value.astype('int') #-1
  
  
  f = h5py.File('superpoint_graphs'+reg+'/'+area+'/'+room+'.h5', 'r')
  voxel2group=f['in_component'].value
  
  
  
  
  point2group=voxel2group[point2voxel]
  
  
  
  #print (np.unique(f['point2group'].value),np.unique(f['point2group'].value).shape)
  
  
  
  groups=[]
  for g in range(13):
    groups.append([])
  
  
  group2point = {}
  for i in range(len(point2group)):
      if point2group[i] not in group2point:
          group2point[point2group[i]] = []
      group2point[point2group[i]].append(i)
  
  
  
  group_num=point2group.shape[0]
  
  origintxt=np.loadtxt('rawdata/'+area+'/'+room+'/'+room+'.txt')
  total_num=origintxt.shape[0]
  coords = np.zeros((total_num,3))
  colors = np.zeros((total_num,3))
  masks = np.zeros((total_num,))
  sem_labels = np.zeros((total_num,))
  inst = np.zeros((total_num,))
  sem_labels[:]=-100
  inst[:]=-100
  instance_files = glob.glob(os.path.join('rawdata/',area,room, 'Annotations', '*.txt'))
  instance_files.sort()
  #### 1
  print ('total num',total_num)
  fracs=0
  cnt=0
  sums=0
  
  offset=0
  for i, f in enumerate(instance_files):
      print (f)
      #try:
      #print (i,f)
      #print (offset)
      class_name = f.split('/')[-1].split('.')[0].split('_')[0]
      # assert class_name in semantic_names, f
      if class_name in semantic_names:
          class_id = semantic_name2id[class_name]
      else:
          class_id = -100
          
      # print(f)
      v = np.loadtxt(f)   # (Ni, 6)
      count=v.shape[0]

      
      
      #print (v[0][0],origintxt[offset][0])
      #assert (v[0][0]==origintxt[offset][0])
      #if v[0][0]==origintxt[offset][0]:
      #  pass
      find=0
      
      #if offset>=origintxt.shape[0] or v[0][0]!=origintxt[offset][0]:
      '''for j in range(total_num):
        #print (v.shape, origintxt[j:j+count].shape)
        if np.array_equal(v[:10], origintxt[j:j+10]):
          #if v[0][0]==origintxt[j][0] and v[0][1]==origintxt[j][1] and v[0][2]==origintxt[j][2] and v[2][0]==origintxt[j+2][0] and v[2][1]==origintxt[j+2][1] and v[2][2]==origintxt[j+2][2] and v[count-1][0]==origintxt[j+count-1][0] and v[count-1][1]==origintxt[j+count-1][1] and v[count-1][2]==origintxt[j+count-1][2] and v[3][0]==origintxt[j+3][0] and v[3][1]==origintxt[j+3][1] and v[3][2]==origintxt[j+3][2] :
          offset=j
          #print (offset)
          find=1
          break'''
      #else:
      #  find=1
      #print (sums, v[:10], origintxt[:10])
      j=sums
      assert v[0][0]==origintxt[j][0] and v[0][1]==origintxt[j][1] and v[0][2]==origintxt[j][2] and v[2][0]==origintxt[j+2][0] and v[2][1]==origintxt[j+2][1] and v[2][2]==origintxt[j+2][2] and v[count-1][0]==origintxt[j+count-1][0] and v[count-1][1]==origintxt[j+count-1][1] and v[count-1][2]==origintxt[j+count-1][2] and v[3][0]==origintxt[j+3][0] and v[3][1]==origintxt[j+3][1] and v[3][2]==origintxt[j+3][2]
      offset=j
      #assert find==1
      #assert (v[0][0]==origintxt[offset][0])
      #semantic_labels.append(np.ones(v.shape[0]) * class_id)
      #assert np.sum(masks[offset:offset+count])==0
      coords[offset:offset+count]=v[:, :3]
      colors[offset:offset+count]=v[:, 3:6]
      #assert (1 not in masks[offset:offset+count])
      masks[offset:offset+count]=1
      inst[offset:offset+count]=i
      point2group_tmp=point2group[offset:offset+count]
  
      (values,counts) = np.unique(point2group_tmp, return_counts=True)
      ind=random.choice(point2group_tmp)
      

      idxs=np.where(point2group==ind)[0]
      idxs_this=np.where(point2group[offset:offset+count]==ind)[0]
      #print (idxs_this.shape,idxs.shape)
      frac=idxs_this.shape[0]/idxs.shape[0]
      fracs+=frac
      cnt+=1
      if area!="Area_5":
        sem_labels[idxs]=class_id
      else:
        sem_labels[offset:offset+count]=class_id
      #print ('classid',class_id)
      if class_id!=-100:
        groups[class_id].append(ind)    
      offset+=count
      sums+=count
      #print (f,count,offset,total_num)
      #except:
      #print (path,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
  print (fracs/cnt)
  print ('sums',sums, np.sum(masks), masks.shape)
  #assert (0 not in masks)
  #assert offset==total_num
  #print (groups)
  print ('sem',np.where(sem_labels!=-100)[0].shape,total_num)
  coords = np.ascontiguousarray(coords - coords.mean(0))
  colors = np.ascontiguousarray(colors)/127.5-1
  
  
  #print (np.unique(coords.mean(1)),'mean', np.where(np.unique(coords.mean(1))==0))
  #for i in np.unique(coords.mean(1)):
  #  print (i)
  

  if area!='Area_5':
    #f=open('gt_train/'+area+'_'+room+'_inst.txt','w')
    #for i in sem_labels:
    #  f.write(str(int(i))+'\n')
    #f.close()
    torch.save((coords, colors, sem_labels, groups, point2group), 'gt_train/'+area+'_'+room+'_inst_nostuff.pth')
  else:
    f=open('gt/'+area+'_'+room+'_inst.txt','w')
    for i in sem_labels:
      f.write(str(int(i))+'\n')
    f.close()
    torch.save((coords, colors, sem_labels, groups, point2group), 'gt_val/'+area+'_'+room+'_inst_nostuff.pth')

if __name__ == "__main__":
  pool=Pool()
  pool.map(process,paths)
  print ('finished')










'''coords = []
colors = []
semantic_labels = []
instance_files = glob.glob(os.path.join('/home/lzz/sdc1/pg/s3dis/data/',area,room, 'Annotations', '*.txt'))
instance_files.sort()
#### 1
offset=0
for i, f in enumerate(instance_files):
    #print (i,f)
    class_name = f.split('/')[-1].split('.')[0].split('_')[0]
    # assert class_name in semantic_names, f
    if class_name in semantic_names:
        class_id = semantic_name2id[class_name]
    else:
        class_id = -100
    # print(f)
    v = np.loadtxt(f)   # (Ni, 6)
    count=v.shape[0]
    #print (v.shape)
    coords.append(v[:, :3])
    colors.append(v[:, 3:6])
    semantic_labels.append(np.ones(v.shape[0]) * class_id)
    
    #assert v[0][0]=
    
    
    
    


coords = np.concatenate(coords).astype(np.float32)
colors = np.concatenate(colors).astype(np.float32)
semantic_labels = np.concatenate(semantic_labels).astype(np.int)

semantic_labels2=semantic_labels.copy()

coords = np.ascontiguousarray(coords - coords.mean(0))
colors = np.ascontiguousarray(colors)/127.5-1


for g in segid_to_pointid.keys():
  idxs=np.asarray(segid_to_pointid[g])
  print (idxs)
  print (semantic_labels.shape)
  sems=semantic_labels[idxs]
  print (sems.shape)
  counts = np.bincount(sems)
  (values,counts) = np.unique(sems,return_counts=True)
  ind=np.argmax(counts)
  target=values[ind]
  print (target)
  semantic_labels2[idxs]=target
  
  
f=open('1/1.txt','w')
for i in semantic_labels2:
  f.write(str(int(i))+'\n')
f.close()
f=open('2/1.txt','w')
for i in semantic_labels:
  f.write(str(int(i))+'\n')
f.close()'''



    

  