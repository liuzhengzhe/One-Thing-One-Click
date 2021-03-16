'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, argparse

import scannet_util,os,csv

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))
files = sorted(glob.glob('/home/lzz/sdc1/scannetv2/scannetv2/scans/scene*_*/*_vh_clean_2.ply'))
print (len(files))

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



name2label={}
with open("scannetv2-labels.combined.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    start=1
    for row in rd:
        if start==1:
           start=0
           continue
        key=row[1]
        value=row[7]
        if value not in dic.keys():
            continue
        name2label[key]=value
        #print (key,value,dic[value])


def f(fn):
    name=fn.split('/')[-1]
    #if os.path.exists('train/'+name+'_inst_nostuff.pth'):
    #    return

    fn2 = fn[:-3] + 'labels.ply'
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    f2 = plyfile.PlyData().read(fn2)
    sem_labels = remapper[np.array(f2.elements[0]['label'])]

    with open(fn3) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    instance_segids = []
    labels = []
    with open(fn4) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
                instance_segids.append([x['segments'][0]])  #We have asked the author of ScanNet and got the reply that "there is no particular order of 'segments'". Here we choose the first segment by default (equlize to choose a random point in it). 
                labels.append(x['label'])
                assert(x['label'] in scannet_util.g_raw2scannetv2.keys())

    check = []
    for i in range(len(instance_segids)): check += instance_segids[i]

    instance_labels = np.ones(sem_labels.shape[0]) * -100

    groups=[]

    for i in range(20):
        groups.append([])

    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        label=labels[i]
        if label not in name2label.keys():
            continue
        
        labelnum=dic[name2label[label]]
        for segid in segids:
            groups[labelnum].append(segid) #[segid]

    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        assert(len(np.unique(sem_labels[pointids])) == 1)
    mask=np.where(instance_labels==-100)
    sem_labels[mask]=-100
    torch.save((coords, colors, sem_labels, groups, seg), 'train_weakly/'+name+'_inst_nostuff.pth')
    print('Saving to ' + fn[:-15]+'_inst_nostuff.pth')

# for fn in files:
#     f(fn)

p = mp.Pool(processes=mp.cpu_count())
if opt.data_split == 'test':
    p.map(f_test, files)
else:
    p.map(f, files)
p.close()
p.join()
