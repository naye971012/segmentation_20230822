from models.seg_hrnet_ocr import get_seg_model
import yaml
import torch
import os
from make_dataset import Segmentation2D
from transforms import *
from torch.utils.data import DataLoader
from utills import seed_everything, calculate_weight
import segmentation_models_pytorch as smp

from torch.utils.tensorboard import SummaryWriter

vali_json_path = '/content/validation/labels'  # 폴더 경로
vali_image_path = '/content/validation/image'
vali_json_files = sorted([f for f in os.listdir(vali_json_path) if f.endswith('.json')])
vali_image_files = sorted([f for f in os.listdir(vali_image_path) if f.endswith('.jpg')])


from tqdm import tqdm
from utills import compute_miou
import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
    
        self.pos_weight = 1
        
    def forward(self, score, target):

        ce_loss = - self.pos_weight * target * torch.log(torch.sigmoid(score) + 1e-8 ) - (1 - target) * torch.log(1 - torch.sigmoid(score) + 1e-8)
        
        #weight = self.weight.view( 1, self.weight.size(0), 1, 1)
        
        loss = ce_loss.mean()
        #loss = (ce_loss * weight) .mean()

        return loss

def train(config, model, logger, train_dataloader, vali_dataloader, weight=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    criterion = CrossEntropy(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.9 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
    
    model = model.to(device)
    for epoch in range(config['EPOCH']):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        iou_list = torch.zeros(config['DATASET']['NUM_CLASSES']+1) #각 class별 IOU
        for i, (images, masks) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            images = images.to(torch.float)  
            masks = masks.to(torch.long)  
            
            images = images.to(device)
            masks = masks.to(device)
            
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            iou_list += compute_miou(outputs,masks, idx=i, is_validation=False)
            epoch_loss += loss.item()
            if(i%10==5):
                logger.add_scalar('train loss step', epoch_loss/i , epoch * len(train_dataloader) + i )
                logger.add_scalar('train mIOU step', iou_list[config['DATASET']['NUM_CLASSES']]/i , epoch * len(train_dataloader) + i )
                for j in range(config['DATASET']['NUM_CLASSES']):
                    logger.add_scalar(f'train IOU class {j} step', iou_list[j]/i , epoch * len(train_dataloader) + i )
        
        scheduler.step()
        
        print("lr: ", optimizer.param_groups[0]['lr'])
        logger.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        
        print("loss: ", epoch_loss/len(train_dataloader) , epoch)
        logger.add_scalar('train loss total', epoch_loss/len(train_dataloader) , epoch )
        logger.add_scalar('train mIOU total', iou_list[config['DATASET']['NUM_CLASSES']]/len(train_dataloader) , epoch )
        for j in range(config['DATASET']['NUM_CLASSES']):
            logger.add_scalar(f'train IOU class {j} total', iou_list[j]/len(train_dataloader) , epoch )

        vali(config, model, logger, vali_dataloader,epoch,weight)

def vali(config,model, logger, vali_dataloader,epoch,weight):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = CrossEntropy(weight=weight)
        
        epoch_loss = 0.0
        iou_list = torch.zeros(config['DATASET']['NUM_CLASSES']+1) #각 class별 IOU
        for i, (images, masks) in tqdm(enumerate(vali_dataloader), total=len(vali_dataloader)):
            
            images = images.to(torch.float)  
            masks = masks.to(torch.long)  
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)

            iou_list += compute_miou(outputs[1],masks, idx=i, is_validation=True)
            epoch_loss += loss.item()
        
        
        print("loss: ", epoch_loss/len(vali_dataloader) , epoch)
        logger.add_scalar('vali loss total', epoch_loss/len(vali_dataloader) , epoch )
        logger.add_scalar('vali mIOU total', iou_list[config['DATASET']['NUM_CLASSES']]/len(vali_dataloader) , epoch )
        for j in range(config['DATASET']['NUM_CLASSES']):
            logger.add_scalar(f'vali IOU class {j} total', iou_list[j]/len(vali_dataloader) , epoch )
        
        return
    
    

if __name__=="__main__":
    logdir= 'runs'
    logger = SummaryWriter(log_dir=logdir)
    
    yaml_file_path = "/content/segmentation_20230822/train_configs.yaml"
    with open(yaml_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    seed_everything(config)
    
    #Encoder = 'timm-resnest101e' #denset201
    Weights = 'imagenet'
    Encoder = 'timm-res2next50'

    prep_fun = smp.encoders.get_preprocessing_fn(
        Encoder,
        Weights
    )

    model = smp.MAnet(
        encoder_name = Encoder,
        encoder_weights = Weights,
        in_channels = 3,
        classes = 1,
        activation = None
    )
    
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
    
    torch.cuda.empty_cache()
    print("train start!\n\n")
    train(config,model,logger, train_dataloader,vali_dataloader)