import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
from torch.nn import functional as F

def seed_everything(config):
    random.seed(0)
    torch.manual_seed(0)    
    cudnn.benchmark = config['CUDNN']['BENCHMARK']
    cudnn.deterministic = config['CUDNN']['DETERMINISTIC']
    cudnn.enabled = config['CUDNN']['ENABLED']
    print("seed set to 0")
    

def compute_miou(pred, true_labels, num_classes=26):
    """
    전체 IOU list는 26번째에, 각 index마다 각각 class의 IOU

    Args:
        pred_labels (_type_): _description_
        true_labels (_type_): _description_
        num_classes (int, optional): _description_. Defaults to 26.

    Returns:
        _type_: _description_
    """
    ph, pw = pred.size(2), pred.size(3)
    h, w = true_labels.size(2), true_labels.size(3)
    if ph != h or pw != w:
        pred = F.interpolate(input=pred, size=( #여기서 크기 조정하네
                h, w), mode='bilinear', align_corners=True)
    
    
    pred_labels = (pred > 0.5).to(torch.int)
    true_labels = true_labels.to(torch.int)
    
    mean_iou_list =0.0
    epsilon = 1e-6  # 분모가 0이 되는 것을 방지하기 위한 작은 값

    iou_list = torch.zeros(num_classes+1) #각 class별 IOU
    
    for batch in range(pred_labels.shape[0]):
        iou_per_batch = 0.0
        for c in range(num_classes):  
            true_mask = true_labels[batch][c]
            pred_mask = pred_labels[batch][c]
            
            intersection = torch.logical_and(true_mask, pred_mask).sum()
            union = torch.logical_or(true_mask, pred_mask).sum()
            
            iou_per_class = (intersection + epsilon) / (union + epsilon)
            
            iou_per_batch += iou_per_class
            iou_list[c] += iou_per_class
            
        iou_list[num_classes] += (iou_per_batch/num_classes)

    mean_iou_list = iou_list / pred_labels.shape[0]
    return mean_iou_list