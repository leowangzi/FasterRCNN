#!/bin/bash
set -e


CUDA_VISIBLE_DEVICES=2  python trainval_net.py --dataset pascal_voc --net res101 --bs 1 --nw 8 --lr 1e-3 --lr_decay_step 5 --epochs 20 --cuda
