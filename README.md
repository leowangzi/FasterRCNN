# FasterRCNN

### Renferences

  - https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0
  - https://github.com/ruotianluo/pytorch-faster-rcnn

### Prerequisites

- Python 2.7 or 3.6
- Pytorch 1.1 or higher
- CUDA 9.0 or higher
- tensorboardX

First of all, clone the code
```
git clone https://github.com/leowangzi/FasterRCNN.git
cd FasterRCNN
```
Then, create a folder:
```
mkdir data
```
or
```
ln -s [source_data] data
```

### What we are doing and going to do

- [x] Support pytorch-1.1 (master).
- [x] Support torchvision-0.3 (master).

### Benchmarking

We benchmark our code thoroughly on pascal voc datasets, using resnet101 network architecture. Below are the results:

1). PASCAL VOC 2007 (Train/Test: 07trainval/07test, scale=600, ROI Align)

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
[Res-101]   | 1 | 1 | 1e-3 | 5   | 10   |  0.88 hr | 3200 MB  | 75.06

2). PASCAL VOC 2007&2012 (Train/Test: 07+12trainval/07test, scale=600, ROI Align)

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
[Res-101]   | 1 | 1 | 1e-3 | 5   | 10   |  0.88 hr | 3200 MB  | 79.80
