from models.seg_hrnet_ocr import get_seg_model
import yaml
import torch
import os
from make_dataset import Segmentation2D
from transforms import *
from torch.utils.data import DataLoader
from utills import seed_everything, calculate_weight
from train_vali_test import *

from torch.utils.tensorboard import SummaryWriter

vali_json_path = '/content/validation/labels'  # 폴더 경로
vali_image_path = '/content/validation/image'
vali_json_files = sorted([f for f in os.listdir(vali_json_path) if f.endswith('.json')])
vali_image_files = sorted([f for f in os.listdir(vali_image_path) if f.endswith('.jpg')])


if __name__=="__main__":
    logdir= 'runs'
    logger = SummaryWriter(log_dir=logdir)
    
    yaml_file_path = "/content/segmentation_20230822/train_configs.yaml"
    with open(yaml_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    seed_everything(config)
    
    model = get_seg_model(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params : {num_params}\n")
    
    print("\nmodel testing...")
    x = torch.randn((1,3,256,256))
    y = model(x)
    print("model testing done!\n")
    
    
    print("make dataset...")
    train_dataset = Segmentation2D(vali_json_path,vali_image_path,vali_json_files,vali_image_files, transform=train_transform, is_test=False)
    vali_dataset = Segmentation2D(vali_json_path,vali_image_path,vali_json_files,vali_image_files, transform=test_transform, is_test=False)
    test_dataset = Segmentation2D(vali_json_path,vali_image_path,vali_json_files,vali_image_files, transform=test_transform, is_test=False)
    
    train_dataloader = DataLoader(vali_dataset, batch_size= config['BATCH_SIZE'] , shuffle=True, num_workers=2)
    vali_dataloader = DataLoader(vali_dataset, batch_size= config['BATCH_SIZE'], shuffle=False, num_workers=2)
    test_dataloader = DataLoader(vali_dataset, batch_size= config['BATCH_SIZE'], shuffle=False, num_workers=2)
    print("dataset created\n")
    
    print(f"train dataset : {len(train_dataset)}")
    print(f"vali dataset : {len(vali_dataset)}")
    print(f"test dataset : {len(test_dataset)}\n")
    
    print("checking error...")
    train_dataset[1]
    vali_dataset[1]
    test_dataset[1]
    print("done!\n")
    
    
    print("calculate class weight...")
    weight = calculate_weight(train_dataset)
    print("done!")
    
    torch.cuda.empty_cache()
    print("train start!\n\n")
    train(config,model,logger, train_dataloader,vali_dataloader,weight)
    
    
    
    