# Pytorch Custom Template for Object Detection

## To-do list:
- [ ] Gradient checkpointing
- [ ] Distributed data parallel
- [ ] Sync BatchNorm
- [x] Multi-GPU support (nn.DataParallel)
- [x] Cutmix, Mixup, strong augmentations
- [x] Test time augmentation
- [x] Gradient Accumulation
- [x] Mixed precision

## Reference:
- Efficientdet from https://github.com/rwightman/efficientdet-pytorch
- FasterRCNN from torchvision
- Box fusion ensemble from https://github.com/ZFTurbo/Weighted-Boxes-Fusion
