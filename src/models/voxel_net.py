# Adapted from: https://github.com/magicleap/Atlas/blob/master/atlas/model.py

import itertools
import os
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
import time
from src.models.components.pointnet import LocalPoolPointnet
from src.models.components.spatial_encoder import SpatialEncoder
from src.models.components.backbone3d import EncoderDecoder
from src.models.components.heads3d import VoxelHeads
from src.models.components.plane_merger import FeaturePlaneMerger
from src.models.utils import *
from src.data.tsdf import TSDF
from src.utils.visuals import *

#o3d.visualization.rendering.OffscreenRenderer.enable_headless_mode(True)
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # use EGL for rendering
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  # force software rendering


log = RankedLogger(__name__)


class VoxelNet(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg  # TODO: excange self.cfg with self.hparams

        # teacher net
        self.f_teacher = None  # TODO

        # encoders
        encoder_latent = 0  # total dimension from all encoders
        if cfg.encoder.use_spatial:
            self.spatial = SpatialEncoder.from_conf(cfg.encoder.spatial, cfg.backbone3d.channels[0])
            #encoder_latent += [0, 64, 128, 256, 512, 1024][cfg.encoder.spatial.num_layers]  # for resnet34
            encoder_latent += 1856  # if resnet50

        #if cfg.encoder.use_pointnet:
        #    self.pointnet = LocalPoolPointnet.from_conf(self.cfg.encoder.pointnet)
        #    self.merger = FeaturePlaneMerger.from_conf(cfg.encoder.plane_merger, c_dim=cfg.encoder.pointnet.c_dim)
        #    encoder_latent += cfg.encoder.pointnet.c_dim
        #if cfg.encoder.use_auxiliary:
        #    encoder_latent += 0 # self.cfg.f_teacher.feature_dim
        
        self.backbone3d = EncoderDecoder.from_conf(cfg.backbone3d)
        self.heads3d = VoxelHeads(cfg.heads, cfg.backbone3d.channels, cfg.voxel_size)

        # other params
        self.origin = torch.tensor([0,0,0]).view(1,3)
        self.voxel_size = cfg.voxel_size
        self.voxel_sizes = [int(cfg.voxel_size*100)*2**i for i in 
                            range(len(cfg.backbone3d.layers_down)-1)]

        self.initialize_volume()

    def initialize_volume(self):
        """ Reset the accumulators.
        
        self.volume is a voxel volume containg the accumulated features
        self.valid is a voxel volume containg the number of times a voxel has
            been seen by a camera view frustrum
        """

        # spatial encoder
        self.volume = None
        self.valid = None

        # pointnet encoder
        self.c_plane = None
    
    def encode(self, projection, image, depth):
        """ Encodes image and corresponding pointcloud into a 3D feature volume and 
        accumulates them. This is the first half of the network which
        is run on T frames. Can be called multiple times.

        Args:
            projection: (B, T, 4, 4) pose matrix
            image: (B, T, 3, H, W) conditioning rgb-image
            depth: (B, T, H, W) conditioning depth map

        Feature volume is accumulated into self.volume and self.valid
        """
        B = projection.size(0)
       
        # transpose batch and time so we can accumulate sequentially
        # (B, T, 3, H, W) -> (T, B, C, H, W)
        images = image.transpose(0,1)
        depths = depth.transpose(0,1)
        projections = projection.transpose(0,1)

        accum_sparse_xyz = torch.empty(B, 0, 3, device=self.device)  # accumulate point cloud for PointNet
        
        # go through every observation
        for image, depth, projection in zip(images, depths, projections):
            
            # accumulate 3D volume using spatial encoder on 2D data:
            B, C, H, W = image.size()
            feat_2d = torch.empty(B, 0, H, W, device=self.device)  # feature map from spatial encoder is halved
            #image = self.normalizer(image) # TODO: normalize?

            if self.cfg.encoder.use_spatial:
                feat_spatial = self.spatial(image)
                feat_2d = torch.cat((feat_2d, feat_spatial), dim=1)  # concat along feature dim C
            
            #if self.cfg.encoder.use_auxiliary:
            #    feat_aux = self.f_teacher(image)
            #    feat_2d = torch.cat((feat_2d, feat_aux), dim=1) # concat along feature dim C

            if self.training:
                voxel_dim = self.cfg.voxel_dim_train
            else:
                voxel_dim = self.cfg.voxel_dim_val
            
            if self.cfg.encoder.use_spatial: #or self.cfg.encoder.use_auxiliary:
                volume, valid = backproject(voxel_dim, self.cfg.voxel_size, self.origin, projection, feat_2d)
                if self.volume == None:
                    self.volume = volume
                    self.valid = valid
                else:
                    self.volume = self.volume + volume
                    self.valid = self.valid + valid


            # accumulate a sparse 3D point cloud (later passed into PointNet):
            #if self.cfg.encoder.use_pointnet:
            #    xyz_map = get_3d_points(depth, projection)
            #    xyz = xyz_map.reshape(B, H*W, 3)
            #    
            #    sparse_xyz, _ = farthest_point_sample(xyz, self.cfg.encoder.pointnet.num_sparse_points) # [B, npoint, 3]
            #    accum_sparse_xyz = torch.cat((accum_sparse_xyz, sparse_xyz), dim=1)

        # build volume using PointNet
        #if self.cfg.encoder.use_pointnet:
        #    c_plane_new = self.pointnet(accum_sparse_xyz) # dict with keys 'xy', 'yz', 'xz' 
        #                                                  # each (B, c_dim=512?, plane_reso=128, plane_reso=128)
        #    if self.c_plane == None:
        #        self.c_plane = c_plane_new
        #    else:
        #        self.c_plane = self.merger(c_plane_new, self.c_plane)


    def forward(self, targets=None):
        """ Refines accumulated features and regresses output TSDF.

        This is the second half of the network. It should be run once after
        all frames have been accumulated. It may also be run more fequently
        to visualize incremental progress.

        Args:
            targets: used to compare network output to ground truth

        Returns:
            tuple of dicts ({outputs}, {losses})
                if targets is None, losses is empty
        """

        if self.cfg.encoder.use_spatial:
            volume = self.volume/self.valid

            # remove nans (where self.valid==0)
            volume = volume.transpose(0,1)
            volume[:,self.valid.squeeze(1)==0]=0
            volume = volume.transpose(0,1)

            x = self.backbone3d(volume)
        
        #elif self.cfg.encoder.use_pointnet:
        #    x = TODO()

        return self.heads3d(x, targets)
            

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        modules = [
            self.backbone3d,
            self.heads3d,
        ]
        if self.cfg.encoder.use_spatial:
            modules.append(self.spatial)
        #if self.cfg.encoder.use_pointnet:
        #    modules.append(self.pointnet)
        #    modules.append(self.merger)
        
        params = itertools.chain(*(module.parameters() 
                                        for module in modules))
        
        # optimzer
        if self.cfg.optimizer.type == 'Adam':
            lr = self.cfg.optimizer.lr
            optimizer = torch.optim.Adam([
                {'params': params, 'lr': lr}],
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
    
    
    def extract_mesh(self, tsdf_list):
        """ Extract a mesh from the TSDF volume.
        
        Args:
            tsdf_list: tsdf volume (B, nx, ny, nz)

        Returns:
            list of trimesh.Trimesh (one mesh per scene in the batch)
        """
        meshes = []

        for tsdf in tsdf_list:
            mesh = tsdf.get_mesh()
            meshes.append(mesh)

        return meshes
    
    def postprocess(self, batch):
        """ Wraps the network output into a TSDF data structure
        
        Args:
            batch: dict containg network outputs

        Returns:
            list of TSDFs (one TSDF per scene in the batch)
        """
        
        key = 'vol_%02d'%self.voxel_sizes[0] # only get vol of final resolution
        out = []
        batch_size = len(batch[key+'_tsdf'])

        for i in range(batch_size):
            tsdf = TSDF(self.voxel_size, 
                        self.origin,
                        batch[key+'_tsdf'][i].squeeze(0))
            out.append(tsdf)

        return out

    def log_loss(self, losses, B, mode):
        if len(losses.keys()) == 0:
            return

        if self.cfg.heads.use_tsdf:
            loss_dict = {}
            for key, val in losses.items():
                loss_dict[f'{mode}_{key}'] = val
            self.log_dict(loss_dict, batch_size=B, sync_dist=True, on_epoch=True)
            


    ############ DEBUGGING #############
    # used when testing only GenNerf model

    def training_step(self, batch, batch_idx):
        image = batch['image'] # (B, T, 3, H, W)
        depth = batch['depth'] # (B, T, H, W)
        projection = batch['projection']  # (B, T, 3, 4) world2image
        B = image.shape[0]

        self.initialize_volume()
        self.encode(projection, image, depth)  # encode images of whole sequence at once
        # TODO: test merging vs processing all observations at once

        #self.origin = batch['origin']
        outputs, losses = self.forward(batch)

        loss = sum(losses.values())
        losses['tsdf_loss'] = loss

        self.log_loss(losses, B, 'train')

        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image'] # (B, T, 3, H, W)
        depth = batch['depth'] # (B, T, H, W)
        projection = batch['projection']  # (B, T, 3, 4) world2image
        B = image.shape[0]

        self.initialize_volume()
        self.encode(projection, image, depth)  # encode images of whole sequence at once
        # TODO: test merging vs processing all observations at once

        #self.origin = batch['origin']
        outputs, losses = self.forward(batch)

        loss = sum(losses.values())
        losses['tsdf_loss'] = loss

        self.log_loss(losses, B, 'val')
        
        # only on 1 gpu: visualize the prediction of the final batch and log it
        if self.global_rank == 0:
            is_last_batch = (batch_idx == len(self.trainer.val_dataloaders) - 1)
            if is_last_batch:
                b = 0
                self.initialize_volume()
                self.encode(batch['projection'][b:b+1], batch['image'][b:b+1], batch['depth'][b:b+1])  # encode images of whole sequence at once
                self.geometric_reconstruction("val", batch, outputs, b_idx=b)
                
        return loss

    def test_step(self, batch, batch_idx):
        image = batch['image'] # (B, T, 3, H, W)
        depth = batch['depth'] # (B, T, H, W)
        projection = batch['projection']  # (B, T, 3, 4) world2image
        B = image.shape[0]

        self.initialize_volume()
        self.encode(projection, image, depth)  # encode images of whole sequence at once
        # TODO: test merging vs processing all observations at once

        #self.origin = batch['origin']
        outputs, losses = self.forward(batch)

        loss = sum(losses.values())
        losses['tsdf_loss'] = loss

        self.log_loss(losses, B, 'test')
        
        ## only on 1 gpu: visualize the prediction of the final batch and log it
        #if self.global_rank == 0:  # Ensure this condition for GPU 0 only
        #    is_last_batch = (batch_idx == len(self.trainer.test_dataloaders) - 1)
        #    if is_last_batch:
        #        b = 0
        #        self.initialize_volume()
        #        self.encode(batch['projection'][b:b+1], batch['image'][b:b+1], batch['depth'][b:b+1])  # encode images of whole sequence at once
        #        self.geometric_reconstruction("test", batch, outputs, b_idx=b)

        return loss
    
    def predict_step(self, batch, batch_idx):
        
        image = batch['image'] # (B, T, 3, H, W)
        depth = batch['depth'] # (B, T, H, W)
        projection = batch['projection']  # (B, T, 3, 4) world2image
        B = image.shape[0]
        assert B == 1

        self.initialize_volume()
        torch.cuda.empty_cache()

        self.encode(projection, image, depth)  # encode images of whole sequence at once
        # TODO: test merging vs processing all observations at once

        outputs, losses = self.forward()

        self.geometric_reconstruction("pred", batch, outputs, b_idx=0)

        tsdf_pred = self.postprocess(outputs)[0]

        # TODO: set origin in model... make consistent with offset above?
        #print("offset:", batch['offset'])
        #tsdf_pred.origin = batch['offset'][0]
    
        mesh_pred = tsdf_pred.get_mesh()

        scene = batch['scene'][0]
        tsdf_pred.save(os.path.join(self.cfg.output_dir, f'{scene}.npz'))
        mesh_pred.export(os.path.join(self.cfg.output_dir, f'{scene}.ply'))

        return

    def geometric_reconstruction(self, mode, batch, outputs, b_idx=0):
        
        # save validation meshes
        pred_tsdfs = self.postprocess(outputs)
        trgt_tsdfs = self.postprocess(batch)

        # get meshes
        meshes_pred = self.extract_mesh(pred_tsdfs) # batch['origin']
        meshes_trgt = self.extract_mesh(trgt_tsdfs) # batch['origin']
        
        # log locally (for debugging)
        self.logger.local.log_tsdf(pred_tsdfs[0], f'test_tsdf/test_pred_tsdf')
        self.logger.local.log_tsdf(trgt_tsdfs[0], f'test_tsdf/test_trgt_tsdf')
        self.logger.local.log_mesh(meshes_pred[0], f'test_mesh/test_pred_mesh')
        self.logger.local.log_mesh(meshes_trgt[0], f'test_mesh/test_trgt_mesh')
        
        # log to wandb
        #self.logger.log_mesh(meshes_pred[0], 'test_pred_mesh')
        #self.logger.log_mesh(meshes_trgt[0], 'test_trgt_mesh')
        self.log_rendered_images(meshes_pred[0], meshes_trgt[0], mode, batch, b_idx, num_logged_frames=1)

    
    def log_rendered_images(self, meshe_pred, meshe_trgt, mode, batch, b_idx=0, num_logged_frames=10):
        image = batch['image'] # (B, T, 3, H, W)
        depth = batch['depth'] # (B, T, H, W)
        pose = batch['pose']  # (B, T, 4, 4) camera2world
        intrinsics = batch['intrinsics']  # (B, T, 3, 3)
        B, T, _, H, W = image.shape

        # transpose batch and time so we can go through sequentially
        # (B, T, 3, H, W) -> (T, B, C, H, W)
        images = image.transpose(0,1)
        #depths = depth.transpose(0,1)
        poses = pose.transpose(0,1)
        intrinsicss = intrinsics.transpose(0,1)

        overview_pose = compute_camera_pose(meshe_trgt, intrinsicss[0][b_idx], W, H, margin=0.8)
        renderer_pred, scene_pred = get_renderer(meshe_pred, W, H, color=(0.75, 0.75, 0.75), light_pose=overview_pose)
        renderer_trgt, scene_trgt = get_renderer(meshe_trgt, W, H, color=(0.75, 0.75, 0.75), light_pose=overview_pose)

        caption = ['image', 'trgt_mesh', 'pred_mesh']

        for i, (image, pose, intrinsics) in enumerate(zip(images, poses, intrinsicss)):
            
            if i >= num_logged_frames:
                #log.info("Stopped logging reconstruction results.")
                break  # prevent too much logging

            if i == 0:
                # add overview image of meshes
                color_ov_img_pred, _ = render(renderer_pred, scene_pred, intrinsics[b_idx], overview_pose)
                color_ov_img_trgt, _ = render(renderer_trgt, scene_trgt, intrinsics[b_idx], overview_pose)
                self.logger.log_image(key=f"{mode}_{batch['scene'][b_idx]}", images=[color_ov_img_trgt, color_ov_img_pred], caption=caption[1:3])
            
            # add near images of meshes
            color_img_pred, _ = render(renderer_pred, scene_pred, intrinsics[b_idx], pose[b_idx])
            color_img_trgt, _ = render(renderer_trgt, scene_trgt, intrinsics[b_idx], pose[b_idx])
            self.logger.log_image(key=f"{mode}_{batch['scene'][b_idx]}_frame{i}", images=[image[b_idx], color_img_trgt, color_img_pred], caption=caption)
