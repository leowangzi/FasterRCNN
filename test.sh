#!/bin/bash
set -e


CUDA_VISIBLE_DEVICES=7  python test_net.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 16 --checkpoint 10021 --cuda
