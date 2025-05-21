<div align="center">

# LaSSM: Efficient Semantic-Spatial Query Decoding via Local Aggregation and State Space Models for 3D Instance Segmentation
[Lei Yao](https://rayyoh.github.io/), [Yi Wang](https://wangyintu.github.io/), [Yawen Cui](https://scholar.google.com/citations?hl=zh-CN&user=Er0gOskAAAAJ&view_op=list_works&sortby=pubdate), [Moyun Liu](https://lmomoy.github.io/), [Lap-Pui Chau](https://www.eie.polyu.edu.hk/~lpchau/)

[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Static Badge](https://img.shields.io/badge/Weights-grey?style=plastic&logo=huggingface&logoColor=yellow)](https://huggingface.co/RayYoh/LaSSM)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)


</div>

## :memo: To-Do List
- [x] Release trained weights and experiment record.
- [x] Installation instructions.
- [x] Processing datasets.
- [ ] Release training configs.
- [x] Release training code.



## :floppy_disk: Trained Results
| Model | Benchmark | Num GPUs | mAP | AP50 | AP25 | Config | Tensorboard | Exp Record | Model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LaSSM | ScanNet++ V2 Val | 4 | 29.1 | 43.5 | 51.6 | [Link](https://github.com/RayYoh/LaSSM/blob/main/configs/scannetpp/insseg-lassm-spunet-v2-3.py) | [Link](https://huggingface.co/RayYoh/LaSSM/tensorboard) | [Link](https://huggingface.co/RayYoh/LaSSM/raw/main/scannetpp-lassm-spunet-v2-3/train.log) | [Link](https://huggingface.co/RayYoh/LaSSM/blob/main/scannetpp-lassm-spunet-v2-3/model/model_best.pth) |
| LaSSM | ScanNet Val | 4 | 58.4 | 78.1 | 86.1 | [Link](https://github.com/RayYoh/LaSSM/blob/main/configs/scannet/insseg-lassm-spunet-v2-3.py) | - | [Link](https://huggingface.co/RayYoh/LaSSM/raw/main/scannet-lassm-spunet-v2-3/train.log) | [Link](https://huggingface.co/RayYoh/LaSSM/blob/main/scannet-lassm-spunet-v2-3/model/model_best.pth) |
| LaSSM | ScanNet200 Val | 4 | 29.3 | 39.2 | 44.5 | [Link](https://github.com/RayYoh/LaSSM/blob/main/configs/scannet200/insseg-lassm-minkunet-3.py) | - | [Link](https://huggingface.co/RayYoh/LaSSM/raw/main/scannet200-lassm-minkunet-3/train.log) | [Link](https://huggingface.co/RayYoh/LaSSM/blob/main/scannet200-lassm-minkunet-3/model/model_best.pth) |


## :hammer: Installation
Our model is built on Pointcept toolkit, you can follow its [official instructions](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#installation) to install the packages:

```bash
conda create -n 3dseg python=3.8 -y
conda activate 3dseg

pip install ninja==1.11.1.1

# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install h5py pyyaml
pip install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm

# Note: (recommended) if you encounter problem in this step, try to download packages on official web and install locally
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu117

pip install transformers==4.44.0 mamba-ssm==2.0.4 causal-conv1d==1.2.0.post2
```
Note that they also provide scripts to build correponding docker image: [build_image.sh](https://github.com/Pointcept/Pointcept/blob/main/scripts/build_image.sh)


## :mag: Data Preprocessing 
**ScanNet V2 & ScanNet200**
- Download the [ScanNet V2](http://www.scan-net.org/) dataset.
- Run preprocessing code for raw ScanNet as follows:

```bash
# RAW_SCANNET_DIR: the directory of downloaded ScanNet raw dataset.
# PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset (output dir).
# Comparing with Pointcept, we just add superpoint extraction
python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```
- Link processed dataset to codebase:
```bash
# PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset.
mkdir data
ln -s ${PROCESSED_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
```

**ScanNet++ V2**
- Download the [ScanNet++ V2](https://kaldir.vc.in.tum.de/scannetpp/) dataset.
- Run preprocessing code for raw ScanNet++ as follows:
```bash
# RAW_SCANNETPP_DIR: the directory of downloaded ScanNet++ raw dataset.
# PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet++ dataset (output dir).
# NUM_WORKERS: the number of workers for parallel preprocessing.
python pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp.py --dataset_root ${RAW_SCANNETPP_DIR} --output_root ${PROCESSED_SCANNETPP_DIR} --num_workers ${NUM_WORKERS}
```
- Sampling and chunking large point cloud data in train/val split as follows (only used for training):
```bash
# PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet++ dataset (output dir).
# NUM_WORKERS: the number of workers for parallel preprocessing.
python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root ${PROCESSED_SCANNETPP_DIR} --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers ${NUM_WORKERS}
python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root ${PROCESSED_SCANNETPP_DIR} --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split val --num_workers ${NUM_WORKERS}
```
- Link processed dataset to codebase:
```bash
# PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet dataset.
mkdir data
ln -s ${PROCESSED_SCANNETPP_DIR} ${CODEBASE_DIR}/data/scannetpp
```

## 🚀 Training
Same to [Pointcept](https://github.com/Pointcept/Pointcept), the training process is based on configs in `configs` folder. The training scripts will create an experiment folder in `exp` and backup essential code in the experiment folder. Training config, log file, tensorboard, and checkpoints will also be saved during the training process.

**Attention:** Note that a cricital difference from Pointcept is that most of data augmentation operations are conducted on GPU in this [file](/pointcept/custom/transform_tensor.py). Make sure `ToTensor` is before the augmentation operations.

**ScanNet V2, LaSSM**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannet -c insseg-lassm-spunet-v2-3 -n insseg-lassm-spunet-v2-3
```

**ScanNet200, LaSSM**
First download the pre-trained backbone from [Mask3D](https://github.com/JonasSchult/Mask3D) [Weight](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/mask3d_scannet200.pth), you can also use our provided weight [mask3d_scannet200](https://huggingface.co/RayYoh/SGIFormer/blob/main/mask3d_scannet200.pth).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannet200 -c insseg-lassm-minkunet-3 -n insseg-lassm-minkunet-3
```

**ScanNet++ V2, LaSSM**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannetpp -c insseg-lassm-spunet-v2-3 -n insseg-lassm-spunet-v2-3
```
**Note**: we load the model pre-trained on ScanNet V2, you need to train ScanNet V2 first or use our provided weight [scannet-lassm-spunet-v2-3](https://huggingface.co/RayYoh/LaSSM/blob/main/scannet-lassm-spunet-v2-3/model/model_best.pth).


## :books: License

This repository is released under the [MIT license](LICENSE).

## :clap: Acknowledgement

Our code is primarily built upon [Pointcept](https://github.com/Pointcept/Pointcept), [OneFormer3D](https://github.com/oneformer3d/oneformer3d), [SGIFormer](https://github.com/RayYoh/SGIFormer). We also thank [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), [Point Cloud Matters](https://github.com/HaoyiZhu/PointCloudMatters?tab=readme-ov-file), and [Mask3D](https://github.com/JonasSchult/Mask3D) for their excellent templates.

## :pencil: Citation

```bib
@article{yao2025lassm,
  title={LaSSM: Efficient Semantic-Spatial Query Decoding via Local Aggregation and State Space Models for 3D Instance Segmentation},
  author={Yao, Lei and Wang, Yi and Yawen, Cui and Liu, Moyun and Chau, Lap-Pui},
  journal={xxx},
  year={2025},
  publisher={xxx}
}
```