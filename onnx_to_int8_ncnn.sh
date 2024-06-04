#!/bin/bash

model=$1

n=$(nproc)

python3 -m onnxsim $model.onnx $model-sim.onnx
onnx2ncnn $model-sim.onnx $model-sim.param $model-sim.bin
ncnnoptimize $model-sim.param $model-sim.bin $model-sim-opt.param $model-sim-opt.bin 0
find dataset/RLB_dataset/train -type f > imagelist.txt
ncnn2table $model-sim-opt.param $model-sim-opt.bin imagelist.txt $model.table mean=[0,0,0] norm=[0.003922,0.003922,0.003922] shape=[352,352,3] pixel=BGR thread=$n method=kl
rm imagelist.txt
ncnn2int8 $model-sim-opt.param $model-sim-opt.bin $model-sim-opt-int8.param $model-sim-opt-int8.bin $model.table