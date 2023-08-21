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


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        
        self.criterion = None
        
        self.alpha = 2
        self.gamma = 10
        #self.criterion = nn.CrossEntropyLoss(
        #    weight=weight,
        #    ignore_index=ignore_label
        #)

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        #h, w = target.size(1), target.size(2)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            #score = F.interpolate(input=score, size=( #메모리 초과로 target를 바꾸기로
            #    h, w), mode='bilinear', align_corners=True) 
            target = F.interpolate(target, size=(ph, pw), mode='nearest')

        ce_loss = -target * torch.log(torch.sigmoid(score) + 1e-8 ) - (1 - target) * torch.log(1 - torch.sigmoid(score) + 1e-8)
        pt = torch.exp(-ce_loss)  # 예측의 확률 값
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Focal Loss 계산
        loss = focal_loss.mean()

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
