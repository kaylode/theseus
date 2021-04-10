# Pytorch Custom Template for Object Detection

## To-do list:
- [ ] Multi-scale training
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
- Scaled YOLOv4 from https://github.com/WongKinYiu/ScaledYOLOv4
- YOLOv5 from https://github.com/ultralytics/yolov5
- Box fusion ensemble from https://github.com/ZFTurbo/Weighted-Boxes-Fusion
