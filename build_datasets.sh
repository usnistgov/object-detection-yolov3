#!/bin/bash


ofp='/mnt/other-home/mmajursk/argus/'

img_fp='/mnt/isgnas/restricted/argus/cnn-data/img'
csv_fp='/mnt/isgnas/restricted/argus/cnn-data/csv'
name="mini-yolov3-$(date +%Y%m%d)"

python build_lmdb_argus.py  --image_folder=${img_fp} --csv_folder=${csv_fp} --output_folder=${ofp} --dataset_name=${name} --image_format=jpg

