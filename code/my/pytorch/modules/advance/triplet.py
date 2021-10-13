#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
   2021-07-30 4:58 下午
   
Author:
   huayang
   
Subject:
   
"""
from my.pytorch.backend.distance_fn import euclidean_distance_nosqrt
from my.pytorch.modules import EncoderWrapper
from my.pytorch.loss.TripletLoss import TripletLoss


class TripletNet(EncoderWrapper):
    """Triplet 网络"""

    def __init__(self, encoder, distance_fn=euclidean_distance_nosqrt, encode_wrapper=None, margin=2.0):
        super(TripletNet, self).__init__(encoder, encode_wrapper)

        self.margin = margin
        self.distance_fn = distance_fn
        self.triplet_loss = TripletLoss(distance_fn=distance_fn, margin=margin)

    def forward(self, anchors, positives, negatives):
        anchors = self.encode(anchors)
        positives = self.encode(positives)
        negatives = self.encode(negatives)

        loss = self.triplet_loss(anchors, positives, negatives)
        return anchors, positives, negatives, loss

    def get_embedding(self, x):
        return self.encode(x)