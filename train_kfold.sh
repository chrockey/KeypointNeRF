#!/bin/bash

for i in 0 1 2 3 4
do
    python train.py --config configs/zju.json --data_root data/zju_mocap --fold $i --expname "zju-fold${i}"
done