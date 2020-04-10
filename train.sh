#!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${1}


batch_size=48
learning_rate=${2}
train_database='/mnt/other-home/mmajursk/argus/train-yolov3-20200409.lmdb'
test_database='/mnt/other-home/mmajursk/argus/test-yolov3-20200409.lmdb'
#train_database='/mnt/other-home/mmajursk/argus/train-mini-yolov3-20200410.lmdb'
#test_database='/mnt/other-home/mmajursk/argus/test-mini-yolov3-20200410.lmdb'
str=${1//,/-}
output_dir="/mnt/other-home/mmajursk/argus/model-$str"

python train.py --batch_size=${batch_size} --learning_rate=${learning_rate} --train_database=${train_database} --test_database=${test_database} --output_dir=${output_dir}