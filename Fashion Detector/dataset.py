"""
We require data to have preprocessed (pimg, pseg, piuv, gimg, gseg, giuv)
"""

import os
import cv2
import collections
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms


to_tensor = transforms.ToTensor()

def load_kpt(fn, size=(256,176)):
  """
  Utility function: load image from file
  """
  img = Image.open(fn).convert("RGB")
  img = img.resize((size[1],size[0]))
  img = to_tensor(img)
  return img

def load_img(fn, size=(256,176)):
  """
  Utility function: load image from file
  """
  img = Image.open(fn).convert("RGB")
  img = to_tensor(img)
  img = F.interpolate(img[None], size,mode='bilinear')[0]
  return img*2-1 # img in range [0,1]

def load_iuv(fn, size=(256,176)):
  """
  Utility function: load densepose. 3xHxW for three channels: I,U,V.
    - I channel: segmentation mask for 25 classes, range [0,1,...,24]
    - U channel: normlized U coordinate. range (0,1)
    - V channel: normlized V coordinate. range (0,1)
  """
  iuv = cv2.imread(fn).transpose([2,0,1])
  iuv = torch.from_numpy(iuv).float() 
  iuv[1:] = iuv[1:] / 255.0 # normalize UV coord from [0,255] to [0,1]
  iuv = F.interpolate(iuv[None], size=size, mode='nearest')[0]
  return iuv

def load_parse(fn, size=[256,176]):
  img = np.array(Image.open(fn))
  img = cv2.resize(img,(size[1],size[0]),interpolation=cv2.INTER_NEAREST)
  #if len(img.shape) == 3:
    # import pdb; pdb.set_trace()
  #  img = (img[:,:,-1] != 0).astype(np.float32)
  #  img[img == 1] = 5.0
  img = torch.from_numpy(img).unsqueeze(0)
  
  if img.max() == 255:
     img[img > 127.5] = 5
  return img

class GeneralTryOnDataset(Dataset):
    def __init__(self, dataroot, config, split):
        super().__init__()
        self.config = config
        self.size = config['size']
        self.split = split
        self.dataroot = dataroot
        self.parse_data_config(dataroot, split, config)
       
    def parse_data_config(self, dataroot, split, configs):
    
        self.pairs = []

        self.all_dicts = collections.defaultdict(dict)
        for dataset_name in configs['datasets']:
            config = configs['datasets'][dataset_name]
            curr_dicts = self._parse_datapath(dataroot, split, config)
            for key in curr_dicts:
                self.all_dicts[key] = {**self.all_dicts[key], **curr_dicts[key]}

            if config['setting'] != 'paired':
                curr_dicts = copy.deepcopy(curr_dicts)
                for key in curr_dicts:
                    self.all_dicts[f'garment_{key}'] = {**self.all_dicts[f'garment_{key}'], **curr_dicts[key]}
            else:
                curr_dicts = self._parse_datapath(dataroot, split, config, is_gimg=True)
                for key in curr_dicts:
                    self.all_dicts[f'garment_{key}'] = {**self.all_dicts[f'garment_{key}'], **curr_dicts[key]}

            curr_annos = self.parse_pairs(dataroot, config)
            self.pairs += curr_annos
            
        
    
    def _parse_datapath(self, dataroot, split, config, is_gimg=False):
        if not config['pair_annotation_path'].startswith('configs/'):
            pair_path = os.path.join(dataroot, config['pair_annotation_path'])
        else:
            pair_path = config['pair_annotation_path']
        with open(pair_path) as f:
            all_fns = f.readlines()
            if config['pair_annotation_path'].endswith('.csv'):
                all_fns = [fn[:-1].split(",") for fn in all_fns[1:]]
                all_fns = [a for _,a,b in all_fns] + [b for _,a,b in all_fns]
            elif config['pair_annotation_path'].endswith('.txt'):
                all_fns = [fn[:-1].split(" ") for fn in all_fns[1:]]
                all_fns = [a for a,b in all_fns] + [b for a,b in all_fns]
        
        all_dicts = collections.defaultdict(dict)
        info = {}
        img_dir = config['garment_image_dir'] if is_gimg else config['image_dir']
        img_postfix = config['garment_image_postfix'] if is_gimg else config['image_postfix']
        image_dir = os.path.join(dataroot, img_dir)
        for anno_name in ['image', 'densepose', 'segm', 'keypoint', 'image_undress']:
            curr_dir_key = f'garment_{anno_name}_dir' if is_gimg else f'{anno_name}_dir'
            if not curr_dir_key in config:
                continue
            curr_dir = os.path.join(dataroot, config[curr_dir_key])
            if not os.path.exists(curr_dir):
                continue
            curr_postfix = config[f'garment_{anno_name}_postfix'] if is_gimg else config[f'{anno_name}_postfix']
            
            for to_fn in all_fns:
                img_fn = "{}/{}".format(image_dir, to_fn)
                curr_fn = img_fn = "{}/{}".format(curr_dir, to_fn.replace(img_postfix, curr_postfix))
                all_dicts[anno_name][to_fn] = curr_fn
        return all_dicts
    
   
            
    def parse_pairs(self, dataroot, config):
        
        if not config['pair_annotation_path'].startswith('configs/'):
            pair_path = os.path.join(dataroot, config['pair_annotation_path'])
        else:
            pair_path = config['pair_annotation_path']

        with open(pair_path) as f:
            annos = f.readlines()
            if config['pair_annotation_path'].endswith('.csv'):
                annos = [anno[:-1].split(',') for anno in annos[1:]]
                annos = [anno[1:] for anno in annos]
            elif config['pair_annotation_path'].endswith('.txt'):
                annos = [anno[:-1].split(' ') for anno in annos[1:]]
            
        return annos

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        
        from_fn, to_fn = self.pairs[idx] # from_fn is source garment; to_fn is source person
        
        pimg = load_img(self.all_dicts['image'][to_fn],size=self.size)
        piuv = load_iuv(self.all_dicts['densepose'][to_fn],size=self.size)
        pseg = load_parse(self.all_dicts['segm'][to_fn],size=self.size)
        

        gimg = load_img(self.all_dicts['garment_image'][from_fn],size=self.size)
        giuv = load_iuv(self.all_dicts['garment_densepose'][from_fn],size=self.size)
        gseg = load_parse(self.all_dicts['garment_segm'][from_fn],size=self.size)

        return dict(
                    pimg=pimg,pseg=pseg,piuv=piuv,
                    gimg=gimg,gseg=gseg,giuv=giuv,
                    garment_fn=from_fn.replace('/','__'),
                    person_fn=to_fn.replace('/','__')
            )
        



