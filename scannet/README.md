Semantic Segmentation Extension for the paper One Thing One Click: A Self-Training Approach for Weakly Supervised 3D Semantic Segmentation.

Authors: Zhengzhe Liu, Xiaojuan Qi, Chi-Wing Fu

## Update on 2022.06.03

* In this implementation, we achieve 69.97% on ScanNet-v2 validation set using "One Thing One Click" with 0.02% annotation. 

* The implementation of Relation-Net is modified. We use a momentum-updated Relation-Net in this version. 

* Efficient super-voxel average pooling using CUDA. 

## Data Preparation

Please follow "Data Preparation" in the "cvpr2021_version" folder. Copy training data (*.pth) into "train_cuda", and copy validation data (*.pth) into "val_cuda"


## Inference and Evaluation

download the weights and features from [here]() and [here]()


```
cd scannet/
python test.py --config config/pointgroup_run1_scannet.yaml --pretrain pointgroup_run1_scannet-000000976_weight.pth
```

The results is:

```
i 0 wall (0.8489608272159432, 10338403, 12177715)
i 1 floor (0.9720041516304324, 8856514, 9111601)
i 2 cabinet (0.6135148966046426, 1034626, 1686391)
i 3 bed (0.7949386965609758, 764421, 961610)
i 4 chair (0.8911075302609167, 3025507, 3395221)
i 5 sofa (0.8222355150810922, 717523, 872649)
i 6 table (0.7118906009059712, 1243255, 1746413)
i 7 door (0.6310385200857385, 1206752, 1912327)
i 8 window (0.624381709044705, 1068801, 1711775)
i 9 bookshelf (0.7083803225260141, 907664, 1281323)
i 10 picture (0.37114424262150814, 123149, 331809)
i 11 counter (0.5899744216287278, 165148, 279924)
i 12 desk (0.6179378805922905, 476330, 770838)
i 13 curtain (0.6873812228533244, 514333, 748250)
i 14 refrigerator (0.49910139718244534, 120526, 241486)
i 15 shower curtain (0.6411151273422244, 108017, 168483)
i 16 toilet (0.9385854583046175, 108951, 116080)
i 17 sink (0.6523031083258439, 80564, 123507)
i 18 bathtub (0.8415696713960711, 89150, 105933)
i 19 otherfurniture (0.5369330131755925, 1059473, 1973194)
avg iou:  0.6997249156669538
```








## Train


We use GPU 0 to train, and GPU 1 for pseudo label updating during training. Please keep GPU 1 free during training. You can choose other GPU to be free in config/pointgroup_run1_scannet.yaml line 6: update_gpu

```
mkdir train_cuda
cp ../data/gt_train/* train_cuda/
python train.py --config config/pointgroup_run1_scannet.yaml
```
