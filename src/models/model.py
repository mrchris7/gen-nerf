# Adapted from: https://github.com/magicleap/Atlas/blob/master/atlas/model.py

import itertools
import torch
import lightning as L
from src.models.utils import farthest_point_sample
from src.data.tsdf import TSDF
from src.models.components.heads3d import TSDFHead
from src.models.components.encoder import Encoder


class GenNerf(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg

        # networks
        self.f_teacher = None  # TODO
        self.f_encoder = Encoder.from_cfg(cfg.encoder, self.f_teacher)
        #self.g_geo = build_backbone3d(cfg)
        self.head_geo = TSDFHead(cfg.head_geo, cfg.backbone3d.channels, cfg.voxel_size)
        #self.g_sem = None  # TODO
        #self.head_sem = NONE  # TODO
        
        self.origin = torch.tensor([0,0,0]).view(1,3)
        self.voxel_sizes = [int(cfg.voxel_size*100)*2**i for i in 
                            range(len(cfg.backbone3d.layers_down)-1)]

        self.initialize_volume()


    def initialize_volume(self):
        """ Reset the accumulators.
        
        self.volume is a voxel volume containg the accumulated features
        self.valid is a voxel volume containg the number of times a voxel has
            been seen by a camera view frustrum
        """

        self.volume = 0 ##
        self.valid = 0 ##

    def encode(self, projection, image, depth):
        """ Encodes image and pointcloud into a 3D feature volume and 
        accumulates them. This is the first half of the network which
        is run on every frame.

        Args:
            projection: bx4x4 pose matrix
            image: bx3xhxw RGB image
            depth: bxhxw
            #feature: bxcxh'xw' feature map (h'=h/stride, w'=w/stride)

        Feature volume is accumulated into self.volume and self.valid
        """
        
        self.initialize_volume()

        # transpose batch and time so we can accumulate sequentially
        # (b, seq_len, c, h, w) -> (seq_len, b, c, h, w)
        images = image.transpose(0,1)
        projections = projection.transpose(0,1)

        # accumulate volume
        for image, projection in zip(images, projections):

            xyz = calculate_pcd(image, depth, projection)  # TODO
            sparse_xyz = farthest_point_sample(xyz, self.cfg['num_points'])

            image = self.normalizer(image)
            self.encode(image, sparse_xyz, projection)

            # encode input
            feat_encoder = self.f_encoder(image, xyz)

            # build volume
            volume, valid = self.build_volume(feat_encoder)  # TODO
            
            # accumulate volume
            self.volume = self.volume + volume
            self.valid = self.valid + valid


    def inference_geo(self, xyz):
        """ Refines accumulated features and regresses output TSDF.
        Decoding part of the network. It should be run once after
        all frames have been accumulated.
        """

        volume = self.volume/self.valid

        # remove nans (where self.valid==0)
        volume = volume.transpose(0,1)
        volume[:,self.valid.squeeze(1)==0]=0
        volume = volume.transpose(0,1)

        output = self.g_geo(volume, xyz)
        return output
    

    def inference_sem(self, xyz):
        """ Refines accumulated features and regresses semantic properties.
        Decoding part of the network. It should be run once after
        all frames have been accumulated.
        """

        volume = self.volume/self.valid

        # remove nans (where self.valid==0)
        volume = volume.transpose(0,1)
        volume[:,self.valid.squeeze(1)==0]=0
        volume = volume.transpose(0,1)

        x = self.g_sem(volume, xyz)
        output = self.head_sem(x)
        return output


    def forward(self, xyz):
        """
        Predict (feat_geo, feat_sem) at world space query point xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        
        output_geo = self.inference_geo(xyz)
        output_sem = self.inference_sem(xyz)

        outputs = {**output_geo, **output_sem}
        return outputs
    

    def postprocess(self, outputs):
        """ Wraps the network output into a TSDF data structure
        
        Args:
            batch: dict containg network inputs and targets

        Returns:
            list of TSDFs (one TSDF per scene in the batch)
        """
        key = 'vol_%02d'%self.voxel_sizes[0] # only get vol of final resolution
        tsdf_data = []
        batch_size = len(outputs[key+'_tsdf'])

        for i in range(batch_size):
            tsdf = TSDF(self.voxel_size, 
                        self.origin,
                        outputs[key+'_tsdf'][i].squeeze(0))
            '''
            # add semseg vol
            if ('semseg' in self.voxel_types) and (key+'_semseg' in outputs):
                semseg = outputs[key+'_semseg'][i]
                if semseg.ndim==4:
                    semseg = semseg.argmax(0)
                tsdf.attribute_vols['semseg'] = semseg

            # add color vol
            if 'color' in self.voxel_types:
                color = outputs[key+'_color'][i]
                tsdf.attribute_vols['color'] = color
            '''
            tsdf_data.append(tsdf)

        return tsdf_data

    def calculate_loss(self, outputs, targets):
        return self.head_geo.calculate_loss(outputs, targets)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        image = inputs['image']
        depth = inputs['depth']
        projection = inputs['projection']  # world2image
        query_xyz = inputs['query_xyz']

        self.encode(projection, image, depth)        
        outputs = self.forward(query_xyz)

        # calculate loss
        loss = self.calculate_loss(outputs, targets)
        self.log('loss', loss)
        return loss
    

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs, targets)

        # save validation meshes
        pred_tsdfs = self.postprocess(outputs)
        trgt_tsdfs = self.postprocess(targets)
        self.logger.experiment1.save_mesh(pred_tsdfs[0], inputs['scene'][0]+'_pred.ply')
        self.logger.experiment1.save_mesh(trgt_tsdfs[0], inputs['scene'][0]+'_trgt.ply')

        # calculate loss
        loss = self.calculate_loss(outputs, targets)
        return loss


    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        # allow for different learning rates between pretrained layers 
        # (resnet backbone) and new layers (everything else).
        ###params_backbone2d = self.backbone2d[0].parameters()
        modules_rest = [self.head_geo,
                        #self.backbone3d,
                        self.f_encoder]
        params_rest = itertools.chain(*(module.parameters() 
                                        for module in modules_rest))
        
        # optimzer
        if self.cfg.optimizer.type == 'Adam':
            lr = self.cfg.optimizer.lr
            ###lr_backbone2d = lr * self.cfg.OPTIMIZER.BACKBONE2D_LR_FACTOR
            optimizer = torch.optim.Adam([
                ###{'params': params_backbone2d, 'lr': lr_backbone2d},
                {'params': params_rest, 'lr': lr}],
                weight_decay=self.cfg.optimizer.weight_decay)
            optimizers.append(optimizer)

        else:
            raise NotImplementedError(
                f'optimizer {self.cfg.optimizer.type} not supported')

        # scheduler
        if self.cfg.scheduler.type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.cfg.scheduler.step_size,
                gamma=self.cfg.scheduler.gamma)
            schedulers.append(scheduler)

        elif self.cfg.scheduler.type != 'None':
            raise NotImplementedError(
                f'scheduler {self.cfg.scheduler.type} not supported')
                
        return optimizers, schedulers
