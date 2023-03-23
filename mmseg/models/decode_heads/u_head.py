import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Dblock(nn.Module):
    def __init__(self, in_channels, dilations=[2, 3]):
        super().__init__()
        self.dil1 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=dilations[0],
            dilation=dilations[0],
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.dil2 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=dilations[1],
            dilation=dilations[1],
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.fuse1 = ConvModule(
            in_channels=in_channels*2,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

    def forward(self, x):
        x1 = self.dil1(x)
        x_ = self.dil2(x1)
        x4 = torch.cat((x_, x), dim=1)
        x5 = self.fuse1(x4)
        return x5

@HEADS.register_module()
class UHead(BaseDecodeHead):
    def __init__(self, feature_strides, **kwargs):
        super(UHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.fuse1 = ConvModule(
            in_channels=embedding_dim*2,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.fuse2 = ConvModule(
            in_channels=embedding_dim*2,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.fuse3 = ConvModule(
            in_channels=embedding_dim*2,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.conv1 = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.conv2 = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.conv3 = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.db3 = Dblock(c3_in_channels)
        self.db4 = Dblock(c4_in_channels)
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        c3 = self.db3(c3)
        c4 = self.db4(c4)
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c4 = resize(_c4, size=c3.size()[2:],mode='bilinear',align_corners=False)
        _c3 = self.conv3(self.fuse3(torch.cat([_c4, _c3], dim=1)))
        _c3 = resize(_c3, size=c2.size()[2:],mode='bilinear',align_corners=False)
        _c2 = self.conv2(self.fuse2(torch.cat([_c3, _c2], dim=1)))
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c1 = self.conv1(self.fuse1(torch.cat([_c2, _c1], dim=1)))
        _c = self.linear_fuse(_c1)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
