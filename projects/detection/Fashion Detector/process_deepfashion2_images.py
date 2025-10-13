import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

def filter_and_crop(image_list, source_img_dir, bbox_dir, target_img_dir):
    os.makedirs(target_img_dir, exist_ok=True)
    for fn in tqdm(image_list):
        curr_img = np.array(Image.open(f'{source_img_dir}/{fn}').convert('RGB'))
        with open(f'{bbox_dir}/{fn}.json') as f:
            anno = json.load(f)
        x1,x2,y1,y2 = anno['human_bbox']
        cropped_image = curr_img[x1:x2, y1:y2]
        Image.fromarray(cropped_image).save(f'{target_img_dir}/{fn}')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_dir', type=str, default='deepfashion2/train/image/')
    parser.add_argument('--bbox_dir', type=str, default='street_tryon_release/train/raw_bbox')
    parser.add_argument('--target_img_dir', type=str, default='street_tryon_release/train/image')
    parser.add_argument('--image_list_path', type=str, default='street_tryon_release/annotations/train_image_list.txt')
    args = parser.parse_args()

    with open(args.image_list_path) as f:
        image_list = [a[:-1] for a in f.readlines()]
    
    filter_and_crop(image_list, args.source_img_dir, args.bbox_dir, args.target_img_dir)