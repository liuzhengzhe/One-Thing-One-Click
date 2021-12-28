import glob
import numpy as np
from multiprocessing import Pool
paths=glob.glob('Area_*/*')
#for path in paths:
def process(path):
  print (path)
  area=path.split('/')[-2]
  room=path.split('/')[-1]
  things=glob.glob(path+'/Annotations/*.txt')
  things.sort()
  datas=np.zeros((0,6))
  for thing in things:
    data=np.loadtxt(thing)
    datas=np.concatenate((datas,data),0)
  np.savetxt(area+'/'+room+'/'+room+'.txt',datas)
  #print (area,room)

if __name__ == "__main__":
  pool=Pool()
  pool.map(process,paths)