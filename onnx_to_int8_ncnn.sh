#!/bin/bash

checkpoint=$1

n=$(nproc)

python3 -m onnxsim $checkpoint/model.onnx $checkpoint/model-sim.onnx
onnx2ncnn $checkpoint/model-sim.onnx $checkpoint/model-sim.param $checkpoint/model-sim.bin
ncnnoptimize $checkpoint/model-sim.param $checkpoint/model-sim.bin $checkpoint/model-sim-opt.param $checkpoint/model-sim-opt.bin 0
find dataset/RLB_dataset/train -type f > imagelist.txt
ncnn2table $checkpoint/model-sim-opt.param $checkpoint/model-sim-opt.bin imagelist.txt $checkpoint/model.table mean=[0,0,0] norm=[0.003922,0.003922,0.003922] shape=[352,352,3] pixel=BGR thread=$n method=kl
rm imagelist.txt
ncnn2int8 $checkpoint/model-sim-opt.param $checkpoint/model-sim-opt.bin $checkpoint/model-sim-opt-int8.param $checkpoint/model-sim-opt-int8.bin $checkpoint/model.table
ncnnoptimize  $checkpoint/model-sim-opt.param $checkpoint/model-sim-opt.bin $checkpoint/model-sim-opt-fp16.param $checkpoint/model-sim-opt-fp16.bin 65536