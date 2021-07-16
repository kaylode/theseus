# Pytorch Custom Template for Image Classification

## Notebook:
Demo using this template:  [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zk87VMlG4DyguhYsXo_NYVvqUypUaKj4?usp=sharing)

## To-do list:
- [ ] Improved Progressive Learning ([EfficientNetV2](https://arxiv.org/abs/2104.00298v1))
- [ ] Test time augmentation
- [ ] Semi-supervised integrations ([UDA](https://arxiv.org/abs/1904.12848), [MetaPseudoLabel](https://arxiv.org/abs/2003.10580))
- [ ] Gradient checkpointing
- [ ] Distributed data parallel
- [ ] Sync BatchNorm
- [x] K-Fold splitting
- [x] Multi-GPU support (nn.DataParallel)
- [x] GradCAM vizualization
- [x] Gradient Accumulation
- [x] Mixed precision

## Dataset Structure:
```
this repo
│   detect.py
│   train.py
│
└───configs
│      configs.yaml
│
└───data  
│   └───<dataset's name>
│       └───images
│           └───train
│               └───<class 0's name>
│               │     00000.jpg
│               │     00001.jpg
│               │     ...
│               └───<class 1's name>
│               └─── ...
│           └───val
│               └───<class 0's name>
│               │     00000.jpg
│               │     00001.jpg
│               │     ...
│               └───<class 1's name>
│               └─── ...
```

## Configuration for custom dataset:
Open file configs/configs.yaml
```
settings:
  project_name: <dataset's name> (name of the folder of the dataset that under ./data folder)
  train_imgs: train
  val_imgs: val
  test_imgs: test
```

## Training:
```
python train.py 
```
See [train.py](https://github.com/kaylode/custom-template/blob/classification/train.py) for more extra parameters

## Inference:
```
python detect.py --weight=<weight path>
```
See [detect.py](https://github.com/kaylode/custom-template/blob/classification/detect.py) for more extra parameters

## Reference:
- timm models from https://github.com/rwightman/pytorch-image-models
