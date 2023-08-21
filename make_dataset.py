import torch
from torch.utils.data import Dataset
import os
import json
import cv2
import numpy as np
from transforms import *

name2num = {
    'road': 1,
    'full_line': 2,
    'dotted_line': 3,
    'road_mark': 4,
    'crosswalk': 5,
    'speed_dump': 6,
    'curb': 7,
    'static': 8,
    'sidewalk': 9,
    'parking_place': 10,
    'vehicle': 11,
    'motorcycle': 12,
    'bicycle': 13,
    'pedestrian': 14,
    'rider': 15,
    'dynamic': 16,
    'traffic_sign': 17,
    'traffic_light': 18,
    'pole': 19,
    'building': 20,
    'guardrail': 21,
    'sky': 22,
    'water': 23,
    'mountain': 24,
    'vegetation': 25,
    'bridge': 26,
    'undefined_object/area' : 100
}
for key in name2num:
    name2num[key] -= 1 #index 0부터 시작 위함
NUM_CLASS = 26
HEIGHT=1080
WIDTH=1920

class Segmentation2D(Dataset):
    def __init__(self, json_path, image_path, json_files, image_files, transform=None, is_test=False):
        self.json_path = json_path
        self.image_path = image_path
        self.json_files = json_files
        self.image_files = image_files
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):

        image_dir = os.path.join( self.image_path, self.image_files[idx] )
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = change_json2label(self.json_path,self.json_files,idx) #3차원 label받아옴

        if self.is_test:
            image = test_transform(image=image)['image']
            return image, label

        if self.transform:
            temp = self.transform(image=image,label=label)
            image = temp['image']
            label = temp['label']

        return image, label

import random

def change_json2label(json_path,json_files, idx):
  """
    json 파일 읽어서 3차원 label로 변환

    Args:
        json_path (_type_): _description_
        json_files (_type_): _description_
        idx (_type_): _description_

    Returns:
        _type_: _description_
  """
  dir = os.path.join(json_path,json_files[idx])

  with open(dir, 'r') as json_file:
    temp = json.load(json_file)

  assert temp["Source_Image_Info"]["Resolution"] == [1920,1080]

  final_array = np.zeros((NUM_CLASS, HEIGHT, WIDTH), dtype=int) # 0/1 mapping된 array

  for cur_info in temp['Annotation']:
    assert cur_info['Type']=='polygon'

    cur_class = name2num[cur_info['Label']]
    if(cur_class==99): #skip unknown
      continue

    try: #오류 있으면 다음꺼로 스킵
        cur_polygon = cur_info['Coordinate']
        cur_polygon = np.array(cur_polygon)
        cur_polygon = cur_polygon.reshape(-1,2)

        temp_array = cv2.fillPoly(np.zeros((HEIGHT,WIDTH)), pts=[np.int32(cur_polygon)], color=255)

        final_array[cur_class] = np.logical_or(final_array[cur_class], temp_array)
    except:
        continue

  return final_array