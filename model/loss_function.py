# !/usr/bin/env python
# coding=utf-8

from enum import Enum
from turtle import forward
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

def build_CrossentropyLoss_ContrastiveLoss(opt):
    return CrossentropyLoss_ContrastiveLoss(opt["distance_metric"], opt["margin"], opt["loss_trade"])

def build_BCELoss(opt):
    return bceLoss()

def build_CrossEntropyLoss(opt):
    return Crossentropy_Loss()

def build_CrossEntropyLoss_weighted(opt):
    return Crossentropy_Loss_weighted()

def build_FocalLoss(opt):
    return FocalLoss()

class bceLoss(nn.Module):
    def __init__(self):
        super(bceLoss, self).__init__()
        self.bce = torch.nn.BCELoss()
    def forward(self, pre, label, rep_anchor, rep_candidate):
        loss = self.bce(pre, label)
        return loss

class Crossentropy_Loss(nn.Module):
    def __init__(self):
        super(Crossentropy_Loss, self).__init__()
        self.crossentropyLoss = torch.nn.CrossEntropyLoss()
    def forward(self, pre, label, rep_anchor, rep_candidate):
        loss = self.crossentropyLoss(pre, label)
        return loss

class Crossentropy_Loss_weighted(nn.Module):
    def __init__(self):
        super(Crossentropy_Loss_weighted, self).__init__()
        weight=torch.from_numpy(np.array([1.0, 1.2])).float()
        self.crossentropyLoss = torch.nn.CrossEntropyLoss(weight=weight)
    def forward(self, pre, label, rep_anchor, rep_candidate):

        loss = self.crossentropyLoss(pre, label)
        return loss

class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)

distance_dict = {
    "EUCLIDEAN":SiameseDistanceMetric.EUCLIDEAN,
    "MANHATTAN":SiameseDistanceMetric.MANHATTAN,
    "COSINE_DISTANCE":SiameseDistanceMetric.COSINE_DISTANCE}

class CrossentropyLoss_ContrastiveLoss(nn.Module):
    def __init__(self, distance_metric="COSINE_DISTANCE", margin: float = 0.5, loss_trade: float = 0.5):
        super(CrossentropyLoss_ContrastiveLoss, self).__init__()

        self.distance_metric = distance_dict[distance_metric]
        self.margin = margin
        self.crossentropyLoss = torch.nn.CrossEntropyLoss()
        self.loss_trade = loss_trade

    def forward(self, pre, label, rep_anchor, rep_candidate):
        # rep_anchor: [batch_size, hidden_dim] denotes the representations of anchors
        # rep_candidate: [batch_size, hidden_dim] denotes the representations of positive / negative
        # label: [batch_size, hidden_dim] denotes the label of each anchor - candidate pair

        distances = self.distance_metric(rep_anchor, rep_candidate)
        loss_contra = (0.5 * ((1 - label).float() * distances.pow(2) +  label.float() * F.relu(self.margin - distances).pow(2))).mean()
        loss = self.loss_trade * loss_contra + self.crossentropyLoss(pre, label)
        return  loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    @:param distance_metric: The distance metric function
    @:param margin: (float) The margin distance
    @:param size_average: (bool) Whether to get averaged loss

    Input example of forward function:
        rep_anchor: [[0.2, -0.1, ..., 0.6], [0.2, -0.1, ..., 0.6], ..., [0.2, -0.1, ..., 0.6]]
        rep_candidate: [[0.3, 0.1, ...m -0.3], [-0.8, 1.2, ..., 0.7], ..., [-0.9, 0.1, ..., 0.4]]
        label: [0, 1, ..., 1]

    Return example of forward function:
        0.015 (averged)
        2.672 (sum)
    """

    def __init__(self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5, size_average:bool = False):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average

    def forward(self, rep_anchor, rep_candidate, label: Tensor):
        # rep_anchor: [batch_size, hidden_dim] denotes the representations of anchors
        # rep_candidate: [batch_size, hidden_dim] denotes the representations of positive / negative
        # label: [batch_size, hidden_dim] denotes the label of each anchor - candidate pair

        distances = self.distance_metric(rep_anchor, rep_candidate)
        losses = 0.5 * (label.float() * distances.pow(2) + (1 - label).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()

from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=3, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, rep_anchor, rep_candidate):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


