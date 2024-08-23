#!/bin/bash

# download validation.zip (from DeepFashion2 Source)
# https://drive.google.com/file/d/1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK/view?usp=drive_link
gdown 1O45YqhREBOoLudjA06HcTehcEebR0o9y

# download train.zip (from DeepFashion2 Source)
# https://drive.google.com/file/d/1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK/view?usp=drive_link
gdown 1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK

# unzip validation.zip 
echo "Password Required. To obtain password, please check https://github.com/switchablenorms/DeepFashion2#download-the-data"
unzip validation.zip

# unzip train.zip 
echo "Password Required. To obtain password, please check https://github.com/switchablenorms/DeepFashion2#download-the-data"
unzip train.zip

mkdir deepfashion2
mv validation deepfashion2/
mv train deepfashion2/
rm -rf *.zip

# filter images from the raw Deepfashion2 dataset
echo "filtering training data.."
export SPLIT='train'
python street_tryon_benchmark/process_deepfashion2_images.py \
--source_img_dir deepfashion2/${SPLIT}/image/ \
--bbox_dir street_tryon/${SPLIT}/raw_bbox \
--target_img_dir street_tryon/${SPLIT}/image \
--image_list_path street_tryon/annotations/${SPLIT}_image_list.txt 

echo "filtering validation data.."
export SPLIT='validation'
python street_tryon_benchmark/process_deepfashion2_images.py \
--source_img_dir deepfashion2/${SPLIT}/image/ \
--bbox_dir street_tryon/${SPLIT}/raw_bbox \
--target_img_dir street_tryon/${SPLIT}/image \
--image_list_path street_tryon/annotations/${SPLIT}_image_list.txt 

# remove deepfashion2
rm -rf deepfashion2