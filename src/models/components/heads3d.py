# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

import torch
from torch import nn
from torch.nn import functional as F
from src.models.utils import log_transform


class TSDFHeadSimple(nn.Module):
    def __init__(self, input_dim):
        super(TSDFHeadSimple, self).__init__()

        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Feature tensor of shape (B, N, C)
        
        Returns:
            tsdf: Tensor of shape (B, N, 1) with regressed TSDF values
        """
        y = self.fc(x)
        tsdf = torch.tanh(y)  # 1.05=label_smoothing
        
        #print("tsdf-shape", tsdf.shape)
        #print("tsdf", tsdf[0, 100:130, :])
        #tsdf = 1.7159 * torch.tanh(2/3 * y)
        return tsdf
    
    
class TSDFHead(nn.Module):
    """ Main head that regresses the TSDF"""

    def __init__(self, cfg, channels, voxel_size):
        super().__init__()
        self.label_smoothing = cfg.label_smoothing
        self.split_loss = cfg.loss_split
        self.sparse_threshold = cfg.sparse_threshold
        self.multi_scale = cfg.multi_scale
        self.loss_log_transform = cfg.loss_log_transform
        self.loss_log_transform_shift = cfg.loss_log_transform_shift
        self.loss_weight = cfg.loss_weight
        self.channels = channels
        
        scales = len(self.channels)-1
        final_size = int(voxel_size*100)
        
        if self.multi_scale:
            self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
            decoders = [nn.Conv3d(c, 1, 1, bias=False) 
                        for c in self.channels[:-1]][::-1]
        else:
            self.voxel_sizes = [final_size]
            decoders = [nn.Conv3d(self.channels[0], 1, 1, bias=False)]
        
        
        self.decoders = nn.ModuleList(decoders)


    def forward(self, xs):
        output = {}
        mask_surface_pred = []

        if not self.multi_scale:
            xs = xs[-1:]

        for i, (decoder, x) in enumerate(zip(self.decoders, xs)):
            # regress the TSDF
            tsdf = torch.tanh(decoder(x)) * self.label_smoothing

            # use previous scale to sparsify current scale
            if self.split_loss=='pred' and i>0:
                tsdf_prev = output['vol_%02d_tsdf'%self.voxel_sizes[i-1]]
                tsdf_prev = F.interpolate(tsdf_prev, scale_factor=2)
                # FIXME: when using float16, why is interpolate casting to float32?
                tsdf_prev = tsdf_prev.type_as(tsdf)
                mask_surface_pred_prev = tsdf_prev.abs()<self.sparse_threshold[i-1]
                # .999 so we don't close surfaces during mc
                tsdf[~mask_surface_pred_prev] = tsdf_prev[~mask_surface_pred_prev].sign()*.999
                mask_surface_pred.append(mask_surface_pred_prev)

            output['vol_%02d_tsdf'%self.voxel_sizes[i]] = tsdf
        
        output['mask_surface_pred'] = mask_surface_pred  # for loss calculation

        return output

    def calculate_loss(self, outputs, targets):
        losses = {}
        for i, voxel_size in enumerate(self.voxel_sizes):
            key = 'vol_%02d_tsdf'%voxel_size
            tsdf_pred = outputs[key]
            tsdf = targets[key]

            mask_observed = tsdf<1
            mask_outside  = (tsdf==1).all(-1, keepdim=True)

            # TODO: extend mask_outside (in heads:loss) to also include 
            # below floor... maybe modify padding_mode in tsdf.transform... 
            # probably cleaner to look along slices similar to how we look
            # along columns for outside.

            if self.loss_log_transform:
                tsdf_pred = log_transform(tsdf_pred, self.loss_log_transform_shift)
                tsdf = log_transform(tsdf, self.loss_log_transform_shift)

            loss = F.l1_loss(tsdf_pred, tsdf, reduction='none') * self.loss_weight

            if self.split_loss=='none':
                losses[key] = loss[mask_observed | mask_outside].mean()

            elif self.split_loss=='pred':
                if i==0:
                    # no sparsifing mask for first resolution
                    losses[key] = loss[mask_observed | mask_outside].mean()
                else:
                    mask = outputs['mask_surface_pred'][i-1] & (mask_observed | mask_outside)
                    if mask.sum()>0:
                        losses[key] = loss[mask].mean()
                    else:
                        losses[key] = 0*loss.sum()

            else:
                raise NotImplementedError(f"TSDF loss split [{self.split_loss}] not supported")
        
        return sum(losses.values())

'''
class SemSegHead(nn.Module):
    """ Predicts voxel semantic segmentation"""

    def __init__(self, cfg):
        super().__init__()

        self.multi_scale = cfg.MODEL.HEADS3D.MULTI_SCALE
        self.loss_weight = cfg.MODEL.HEADS3D.SEMSEG.LOSS_WEIGHT

        scales = len(cfg.MODEL.BACKBONE3D.CHANNELS)-1
        final_size = int(cfg.VOXEL_SIZE*100)

        classes = cfg.MODEL.HEADS3D.SEMSEG.NUM_CLASSES
        if self.multi_scale:
            self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
            decoders = [nn.Conv3d(c, classes, 1, bias=False) 
                        for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1]
        else:
            self.voxel_sizes = [final_size]
            decoders = [nn.Conv3d(cfg.MODEL.BACKBONE3D.CHANNELS[0], classes, 1, bias=False)]

        self.decoders = nn.ModuleList(decoders)

    def forward(self, xs, targets=None):
        output = {}
        losses = {}

        if not self.multi_scale:
            xs = xs[-1:] # just use final scale

        for voxel_size, decoder, x in zip(self.voxel_sizes, self.decoders, xs):
            # compute semantic labels
            key = 'vol_%02d_semseg'%voxel_size
            output[key] = decoder(x)

            # compute losses
            if targets is not None and key in targets:
                pred = output[key]
                trgt = targets[key]
                mask_surface = targets['vol_%02d_tsdf'%voxel_size].squeeze(1).abs() < 1

                loss = F.cross_entropy(pred, trgt, reduction='none', ignore_index=-1)
                if mask_surface.sum()>0:
                    loss = loss[mask_surface].mean()
                else:
                    loss = 0 * loss.mean()
                losses[key] = loss * self.loss_weight

        return output, losses
'''

'''
class ColorHead(nn.Module):
    """ Predicts voxel color"""

    def __init__(self, cfg):
        super().__init__()

        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1, 1)

        self.multi_scale = cfg.MODEL.HEADS3D.MULTI_SCALE
        self.loss_weight = cfg.MODEL.HEADS3D.COLOR.LOSS_WEIGHT

        scales = len(cfg.MODEL.BACKBONE3D.CHANNELS)-1
        final_size = int(cfg.VOXEL_SIZE*100)

        if self.multi_scale:
            self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
            decoders = [nn.Conv3d(c, 3, 1, bias=False) 
                        for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1]
        else:
            self.voxel_sizes = [final_size]
            decoders = [nn.Conv3d(cfg.MODEL.BACKBONE3D.CHANNELS[0], 3, 1, bias=False)]

        self.decoders = nn.ModuleList(decoders)

    def forward(self, xs, targets=None):
        output = {}
        losses = {}

        if not self.multi_scale:
            xs = xs[-1:] # just use final scale

        for voxel_size, decoder, x in zip(self.voxel_sizes, self.decoders, xs):
            key = 'vol_%02d_color'%voxel_size
            pred = torch.sigmoid(decoder(x)) * 255
            output[key] = pred

            # compute losses
            if targets is not None and key in targets:
                pred = output[key]
                trgt = targets[key]
                mask_surface = targets['vol_%02d_tsdf'%voxel_size].squeeze(1).abs() < 1

                loss = F.l1_loss(pred, trgt, reduction='none').mean(1)
                if mask_surface.sum()>0:
                    loss = loss[mask_surface].mean()
                else:
                    loss = 0 * loss.mean()
                losses[key] = loss * self.loss_weight / 255

        return output, losses
'''
