# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

NUM_OUTPUTS = 2

class DiceLoss(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(DiceLoss, self).__init__()
        self.ignore_label = ignore_label
        
        #if weight:
        #    self.weight = weight.to('cuda')

        
    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        #h, w = target.size(1), target.size(2)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            #score = F.interpolate(input=score, size=( #메모리 초과로 target를 바꾸기로
            #    h, w), mode='bilinear', align_corners=True)
            target = target.to(torch.float)
            target = F.interpolate(target, size=(ph, pw), mode='bilinear', align_corners=True)
     
        smooth = 1.0  # 작은 값 추가하여 0으로 나누기 방지
        intersection = torch.sum(score * target)
        union = torch.sum(score) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice
    
    def forward(self, score, target):

        if NUM_OUTPUTS == 1:
            score = [score]
        
        weights = [0.4 , 1.0 ]
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)]) # [ 0.4 , 1 ] [a , b], 뒤에거가 진짜 pred, 앞에는 형체예측
    
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        
        self.criterion = None
        
        #self.criterion = nn.CrossEntropyLoss(
        #    weight=weight,
        #    ignore_index=ignore_label
        #)
        self.weight = weight.to('cuda')
        self.pos_weight = 5
        
    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        #h, w = target.size(1), target.size(2)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            #score = F.interpolate(input=score, size=( #메모리 초과로 target를 바꾸기로
            #    h, w), mode='bilinear', align_corners=True)
            target = target.to(torch.float)
            target = F.interpolate(target, size=(ph, pw), mode='bilinear', align_corners=True)

        ce_loss = - self.pos_weight * target * torch.log(torch.sigmoid(score) + 1e-8 ) - (1 - target) * torch.log(1 - torch.sigmoid(score) + 1e-8)
        
        weight = self.weight.view( 1, self.weight.size(0), 1, 1)
        
        loss = (ce_loss * weight) .mean()

        return loss

    def forward(self, score, target):

        if NUM_OUTPUTS == 1:
            score = [score]
        
        weights = [0.4 , 1.0 ]
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)]) # [ 0.4 , 1 ] [a , b], 뒤에거가 진짜 pred, 앞에는 형체예측


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if NUM_OUTPUTS == 1:
            score = [score]

        weights = [0,4 , 1.0 ]
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])
