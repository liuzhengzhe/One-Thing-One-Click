# One Thing One Click
## One Thing One Click: A Self-Training Approach for Weakly Supervised 3D Semantic Segmentation (CVPR2021)

Code for the paper **One Thing One Click: A Self-Training Approach for Weakly Supervised 3D Semantic Segmentation**, CVPR 2021.

This code is based on PointGroup https://github.com/llijiang/PointGroup 

**Authors**: Zhengzhe Liu, Xiaojuan Qi, Chi-Wing Fu

## Installation

### Requirements
* Python 3.7.0
* Pytorch 1.3.0
* CUDA 10.1

### Virtual Environment
```
conda create -n pointgroup python==3.7
source activate pointgroup
```

### Install `PointGroup`

(1) Clone the PointGroup repository.
```
git clone https://github.com/liuzhengzhe/One-Thing-One-Click --recursive 
cd One-Thing-One-Click
```

(2) Install the dependent libraries.
```
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

(3) For the SparseConv, we apply the implementation of [spconv](https://github.com/traveller59/spconv). The repository is recursively downloaded at step (1). We use the version 1.0 of spconv. 

**Note:** The author of PointGroup further modified `spconv\spconv\functional.py` to make `grad_output` contiguous. Make sure you use our modified `spconv`.

* To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```
Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Compile the `spconv` library.
```
cd lib/spconv
python setup.py bdist_wheel
```

* Run `cd dist` and use pip to install the generated `.whl` file.



(4) Compile the `pointgroup_ops` library.
```
cd lib/pointgroup_ops
python setup.py develop
```
If any header files could not be found, run the following commands. 
```
python setup.py build_ext --include-dirs=$INCLUDE_PATH$
python setup.py develop
```
`$INCLUDE_PATH$` is the path to the folder containing the header files that could not be found.


## Data Preparation

* Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

* Put the data in the corresponding folders. 

* Put the file `scannetv2-labels.combined.tsv` in the `data/` folder.

* Change the path in prepare_data_otoc.py Line 20. 

```
cd data/
python prepare_data_otoc.py 
```

* Split the generated files into the `data/train_weakly` and `data/val_weakly` folders according to the ScanNet v2 train/val [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 


## Pretrained Model
We provide a pretrained model trained on ScanNet v2 dataset. Download it [here](https://drive.google.com/drive/folders/1zqd-V-w1eQ5tpy3Gz6fxbYMaCPDx5nx1?usp=sharing). Its performance on ScanNet v2 validation set is 71.94 mIoU.



## Inference and Evaluation

(1) 3D U-Net Evaluation 

set the data_root in config/pointgroup_run1_scannet.yaml

```
cd 3D-U-Net
python test.py --config config/pointgroup_run1_scannet.yaml --pretrain pointgroup_run1_scannet-000001250.pth
```
Its performance on ScanNet v2 validation set is 68.96 mIoU.

(2) Relation Net Evaluation 

```
cd relation
python test.py --config config/pointgroup_run1_scannet.yaml --pretrain pointgroup_run1_scannet-000002891_weight.pth
```

(3) Overall Evaluation

```
cd merge
python test.py --config config/pointgroup_run1_scannet.yaml
```


## Self Training

(1) Train 3D U-Net

set the data_root/dataset in config/pointgroup_run1_scannet.yaml

```
cd 3D-U-Net
CUDA_VISIBLE_DEVICES=0 python train.py --config config/pointgroup_run1_scannet.yaml 
```

(2) Generate features and predictions of 3D U-Net

```
CUDA_VISIBLE_DEVICES=0 python test_train.py --config config/pointgroup_run1_scannet.yaml --pretrain $PATH_TO_THE_MODEL$.pth
```

(3) Train Relation Net

set the data_root/dataset in config/pointgroup_run1_scannet.yaml

```
cd relation
CUDA_VISIBLE_DEVICES=0 python train.py --config config/pointgroup_run1_scannet.yaml 
```

(4) Generate features and predictions of Relation Net

```
CUDA_VISIBLE_DEVICES=0 python test_train.py --config config/pointgroup_run1_scannet.yaml --pretrain $PATH_TO_THE_MODEL$_weight.pth
```

(5) Merge the Results via Graph Propagation

```
cd merge
CUDA_VISIBLE_DEVICES=0 python test_train.py --config config/pointgroup_run1_scannet.yaml
```

(6) Repeat from (1) to (5) for self-training for 3 to 5 times


## Acknowledgement
This repo is built upon several repos, e.g., [PointGrouop](https://github.com/Jia-Research-Lab/PointGroup), [SparseConvNet](https://github.com/facebookresearch/SparseConvNet), [spconv](https://github.com/traveller59/spconv) and [ScanNet](https://github.com/ScanNet/ScanNet). 

## Contact
If you have any questions or suggestions about this repo, please feel free to contact me (liuzhengzhelzz@gmail.com).

