# Adapted from: https://github.com/magicleap/Atlas/blob/master/atlas/model.py

import itertools
import os
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from src.models.components.positional_encoding import PositionalEncoding
from src.models.components.pointnet import LocalPoolPointnet
from src.models.components.spatial_encoder import SpatialEncoder
from src.models.components.resnetfc import ResnetFC
from src.models.components.heads3d import TSDFHeadSimple
from src.models.components.plane_merger import FeaturePlaneMerger
from src.models.utils import *
from src.data.tsdf import TSDF
from src.utils.visuals import *

#o3d.visualization.rendering.OffscreenRenderer.enable_headless_mode(True)
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # use EGL for rendering
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  # force software rendering



class GenNerf(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # teacher net
        self.f_teacher = None  # TODO

        # encoders
        encoder_latent = 0  # total dimension from all encoders
        if cfg.encoder.use_spatial:
            self.spatial = SpatialEncoder.from_conf(cfg.encoder.spatial)
            encoder_latent += [0, 64, 128, 256, 512, 1024][cfg.encoder.spatial.num_layers]
        if cfg.encoder.use_pointnet:
            self.pointnet = LocalPoolPointnet.from_conf(self.cfg.encoder.pointnet)
            self.merger = FeaturePlaneMerger.from_conf(cfg.encoder.plane_merger, c_dim=cfg.encoder.pointnet.c_dim)
            encoder_latent += cfg.encoder.pointnet.c_dim
        if cfg.encoder.use_auxiliary:
            encoder_latent += 0 # self.cfg.f_teacher.feature_dim
        
        # decoders
        d_in = 3  # xyz
        if cfg.use_code:
            self.code = PositionalEncoding.from_conf(cfg.code, d_in=d_in)
            d_in = self.code.d_out

        self.mlp = ResnetFC.from_conf(cfg.mlp, d_in=encoder_latent, d_latent=d_in)
        self.head_geo = TSDFHeadSimple(cfg.mlp.d_out_geo)
        self.cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        # other params
        self.origin = torch.tensor([0,0,0]).view(1,3)
        self.voxel_sizes = [int(cfg.voxel_size*100)]

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
    
    def encode(self, projection, image, depth, mode):
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

            if self.cfg.encoder.use_auxiliary:
                feat_aux = self.f_teacher(image)
                feat_2d = torch.cat((feat_2d, feat_aux), dim=1) # concat along feature dim C

            if self.training:
                voxel_dim = self.cfg.voxel_dim_train
            else:
                voxel_dim = self.cfg.voxel_dim_val
            
            if self.cfg.encoder.use_spatial or self.cfg.encoder.use_auxiliary:
                volume, valid = backproject(voxel_dim, self.cfg.voxel_size, self.origin, projection, feat_2d)
                if self.volume == None:
                    self.volume = volume
                    self.valid = valid
                else:
                    self.volume = self.volume + volume
                    self.valid = self.valid + valid


            # accumulate a sparse 3D point cloud (later passed into PointNet):
            if self.cfg.encoder.use_pointnet:
                xyz_map = get_3d_points(depth, projection)
                xyz = xyz_map.reshape(B, H*W, 3)
                
                sparse_xyz, _ = farthest_point_sample(xyz, self.cfg.encoder.pointnet.num_sparse_points) # [B, npoint, 3]
                accum_sparse_xyz = torch.cat((accum_sparse_xyz, sparse_xyz), dim=1)

        if mode == 'test' and self.cfg.encoder.use_pointnet:
            #self.logger.local.log_tensor(accum_sparse_xyz, 'sparse_points/sparse_points')
            #self.logger.local.log_tensor(xyz, 'sparse_points/dense_points')
            pass

        # build volume using PointNet
        if self.cfg.encoder.use_pointnet:
            c_plane_new = self.pointnet(accum_sparse_xyz) # dict with keys 'xy', 'yz', 'xz' 
                                                          # each (B, c_dim=512?, plane_reso=128, plane_reso=128)
            if self.c_plane == None:
                self.c_plane = c_plane_new
            else:
                self.c_plane = self.merger(c_plane_new, self.c_plane)


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.cfg.encoder.pointnet.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize (-1, 1)
        if self.cfg.loss.use_eikonal or self.cfg.loss.use_gradient:
            c = grid_sample_2d(c, vgrid)
        else:
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.cfg.encoder.pointnet.sample_mode)
        return c.squeeze(-1)

    def map_features(self, xyz):
        """
        Map to every point the corresponding feature from the
        encoded feature representations.
        
        Args:
            xyz (B, N, 3): sampled query points (world space)
                N number of points
            
        Returns:
            feat (B, N, C): combined features
                C = c_dim_spatial + c_dim_pointnet + c_dim_auxiliary
        """
        B, N, _ = xyz.size()

        # combine features from different encodings
        feat = torch.empty(B, N, 0, device=self.device)
        if self.cfg.encoder.use_pointnet:
            plane_type = list(self.c_plane.keys())
            feat_pointnet = 0
            #if 'grid' in plane_type:
            #    c_pointnet += self.sample_grid_feature(xyz, c_plane['grid'])
            if 'xz' in plane_type:
                feat_pointnet += self.sample_plane_feature(xyz, self.c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                feat_pointnet += self.sample_plane_feature(xyz, self.c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                feat_pointnet += self.sample_plane_feature(xyz, self.c_plane['yz'], plane='yz')
            feat_pointnet = feat_pointnet.transpose(1, 2)
            feat = torch.cat((feat, feat_pointnet), dim=-1)

        if self.cfg.encoder.use_spatial or self.cfg.encoder.use_auxiliary:
            volume = self.volume/self.valid
            # remove nans (where self.valid==0)
            volume = volume.transpose(0,1)
            volume[:,self.valid.squeeze(1)==0]=0
            volume = volume.transpose(0,1)

            volume_ = volume.permute(0, 2, 3, 4, 1)
            feat_spatial = trilinear_interpolation(volume_, xyz, self.origin.squeeze(), self.cfg.voxel_size)
            feat = torch.cat((feat, feat_spatial), dim=-1)
        return feat


    def forward(self, xyz):
        """
        Predict (feat_geo, feat_sem) at world space query points xyz.
        Please call encode first!
        
        Args:
            xyz (B, N, 3): sampled query points (world space)
                
        Returns:
            output (dict): outputs-dict from head_geo
                           'feat_geo': (B, N, d_out_geo)
                           'feat_sem': (B, N, d_out_sem)
                           'tsdf': (B, N, 1)
        """
        B, N, _ = xyz.shape
        d_out_geo = self.cfg.mlp.d_out_geo
        d_out_sem = self.cfg.mlp.d_out_sem

        feat = self.map_features(xyz)  # [B, N, d_latent=encoder_latent]
        
        with torch.set_grad_enabled(True):  # gradient for eikonal loss
            if self.cfg.use_code:
                xyz = xyz.reshape(-1, 3)  # (B*N, 3)
                xyz = self.code(xyz)
                xyz = xyz.reshape(B, N, -1)

            mlp_input = torch.cat((xyz, feat), dim=-1)
            mlp_output = self.mlp(mlp_input)
            mlp_output = mlp_output.reshape(B, N, d_out_geo + d_out_sem)

            feat_geo = mlp_output[...,           : d_out_geo            ]
            feat_sem = mlp_output[..., d_out_geo : d_out_geo + d_out_sem]
            
            tsdf = self.head_geo(feat_geo)

        outputs = {}
        outputs['feat_geo'] = feat_geo  # torch.identity(feat_geo)  # necessary?
        outputs['feat_sem'] = feat_sem  # torch.relu(feat_sem)    
        outputs['tsdf'] = tsdf
        outputs['feat'] = feat

        return outputs

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        modules = [
            self.mlp,
            self.head_geo,
            self.cosSim
        ]
        if self.cfg.encoder.use_spatial:
            modules.append(self.spatial)
        if self.cfg.encoder.use_pointnet:
            modules.append(self.pointnet)
            modules.append(self.merger)
        
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
    
    def postprocess_tsdf(self, tsdf_vol, origin):
        """ Apply postprocessing to the TSDF volume and wrap it into a TSDF object.
        
        Args:
            tsdf_vol: tsdf volume (B, nx, ny, nz)

        Returns:
            list of TSDF objects (one object per scene in the batch)
        """
        tsdf_objs = []
        batch_size = len(tsdf_vol)

        for i in range(batch_size):
            # optionally apply post processing here...
            tsdf = TSDF(self.cfg.voxel_size, origin, tsdf_vol[i].squeeze(0))
            tsdf_objs.append(tsdf)

        return tsdf_objs


    def loss_tsdf(self, outputs, targets):
        # Loss for TSDF values
        # Adapted from Atlas: https://arxiv.org/abs/2003.10432

        pred = outputs['tsdf']  # [B, N, 1]
        trgt = targets['tsdf']  # [B, N, 1]
        
        # optionally: create mask
        #mask_observed = trgt < 1
        #mask_outside  = (trgt == 1).all(-1, keepdim=True) # does not work in our case (vertical column is equal to 1)
                                                           # because trgt is not structured as a 3D grid but a point set
        if self.cfg.loss.tsdf.transform == 'log':
            pred = log_transform(pred, self.cfg.loss.tsdf.shift)
            trgt = log_transform(trgt, self.cfg.loss.tsdf.shift)
        elif self.cfg.loss.tsdf.transform == 'smooth_log':
            pred = smooth_log_transform(pred, self.cfg.loss.tsdf.shift, self.cfg.loss.tsdf.smoothness)
            trgt = smooth_log_transform(trgt, self.cfg.loss.tsdf.shift, self.cfg.loss.tsdf.smoothness)
        elif self.cfg.loss.tsdf.transform == 'none':
            pass
        else:
            raise NotImplementedError(f"Usage of unknown log_trans mode: {self.cfg.loss.tsdf.transform}")

        loss = F.l1_loss(pred, trgt, reduction='none')
        
        # optionally: apply mask
        #loss = loss[mask_observed | mask_outside]  # penalize observed areas and areas behind walls (outside) to prevent artifacts behind walls
        
        return loss
    
    def loss_isdf(self, outputs, targets):
        # Combined loss originally used for TSDF values
        # Adapted from iSDF: https://arxiv.org/abs/2204.02296

        pred = outputs['tsdf']  # [B, N, 1]
        trgt = targets['tsdf']  # [B, N, 1]

        # free space loss
        factor = self.cfg.loss.isdf.free_space_factor
        term1 = torch.exp(-factor * pred) - 1.0 
        term2 = pred - trgt
        loss_free_space = torch.max(
            torch.nn.functional.relu(term1),  # ensure non-negative
            term2
        )
        
        # near surface loss
        loss_near_surf = F.l1_loss(pred, trgt, reduction='none')
        loss_near_surf *= self.cfg.loss.isdf.trunc_weight

        # combine losses
        mask = (trgt <= 1.0).float()
        loss = mask * loss_near_surf + (1 - mask) * loss_free_space
        return loss

    def loss_eikonal(self, outputs, targets):
        # Eikonal term to encourage unit gradient norm
        # Adapted from: https://arxiv.org/abs/2002.10099

        tsdf = targets['tsdf']  # (B, N, 1)
        gradients = outputs['grad']  # (B, N, 3)

        gradient_norm = gradients.norm(2, dim=-1)  # (B, N)
        loss = (torch.abs(gradient_norm - 1)).unsqueeze(-1) # (B, N, 1)
        dist = self.cfg.loss.eikonal.apply_distance
        loss[tsdf < dist] = 0.
        #loss[tsdf > dist] = 0.  # TODO: add this too?
        return loss
    
    def loss_gradient(self, outputs, targets):
        # Penalise the cosine distance between 
        # approximated gradient (surface normal) and the tsdf gradient
        
        sampled_normals = targets['sampled_normals']  # (B, n_rays, 3)
        B = sampled_normals.shape[0]

        grad_vec = targets['grad_vec']  # (B, n_rays, N+M, 3)
        grad = outputs['grad'].view(B, self.cfg.ray.num_rays, -1, 3)  # (B, n_rays, 1+N+M, 3)
        
        # at surface points (B, n_rays, 1)
        surf_loss = 1 - self.cosSim(sampled_normals, grad[:, :, 0])  # (B, n_rays)

        # replace invalid gradients with normals
        grad_is_nan = torch.where(grad_vec[..., 0].isnan())  # tuple(b_idxs, ray_idxs, ray-sample_idxs)
        #                       (B, n_rays, 3)     b_idxs         ray_idxs
        grad_vec[grad_is_nan] = sampled_normals[grad_is_nan[0], grad_is_nan[1]].float()

        # at all (other) points (n_rays, M+N)
        grad_loss = 1 - self.cosSim(grad_vec, grad[:, :, 1:])  # (B, n_rays, N+M)

        # combine loss matrix
        loss = torch.cat((surf_loss[:, :, None], grad_loss), dim=2)  # (B, n_rays, 1+N+M)
        loss = loss.view(B, -1, 1)
        return loss


    def loss_feat(self, outputs, targets):
        feat = outputs['feat']  # (B, N, d_endoder)
        feat_contribution = torch.norm(feat, dim=-1).mean()
        loss = (1 / feat_contribution)
        return loss

    def calculate_loss(self, outputs, targets):
        losses = {}
        loss_mat = None
        
        # main losses
        assert self.cfg.loss.use_tsdf or self.cfg.loss.use_isdf
        if self.cfg.loss.use_tsdf:
            loss_tsdf_mat = self.loss_tsdf(outputs, targets)
            losses['tsdf'] = loss_tsdf_mat.mean()
            loss_tsdf_mat_w = self.cfg.loss.tsdf.weight * loss_tsdf_mat
            if loss_mat == None:
                loss_mat = loss_tsdf_mat_w
            else:
                loss_mat += loss_tsdf_mat_w
        if self.cfg.loss.use_isdf:
            loss_isdf_mat = self.loss_isdf(outputs, targets)
            losses['isdf'] = loss_isdf_mat.mean()
            loss_isdf_mat_w = self.cfg.loss.isdf.weight * loss_isdf_mat
            if loss_mat == None:
                loss_mat = loss_isdf_mat_w
            else:
                loss_mat += loss_isdf_mat_w
        
        # additional losses and regularizations
        if self.cfg.loss.use_eikonal:
            eik_loss_mat = self.loss_eikonal(outputs, targets)
            losses['eikonal'] = eik_loss_mat.mean()
            loss_mat += self.cfg.loss.eikonal.weight * eik_loss_mat
        if self.cfg.loss.use_gradient:
            grad_loss_mat = self.loss_gradient(outputs, targets)
            losses['gradient'] = grad_loss_mat.mean()
            loss_mat += self.cfg.loss.gradient.weight * grad_loss_mat
        if self.cfg.loss.use_feature:
            feat_loss_mat = self.loss_feat(outputs, targets)
            losses['feature'] = feat_loss_mat.mean()
            loss_mat += self.cfg.loss.feature.weight * feat_loss_mat
        
        losses['combined'] = loss_mat.mean()
        return losses

    def log_loss(self, loss, B, mode):
        if len(loss.keys()) == 0:
            return

        self.log(f'{mode}_loss', loss['combined'], batch_size=B, sync_dist=True)

        if self.cfg.loss.use_tsdf:
            self.log(f'{mode}_loss_tsdf', loss['tsdf'], batch_size=B, sync_dist=True)
        
        if self.cfg.loss.use_isdf:
            self.log(f'{mode}_loss_isdf', loss['isdf'], batch_size=B, sync_dist=True)

        if self.cfg.loss.use_eikonal:
            self.log(f'{mode}_loss_eikonal', loss['eikonal'], batch_size=B, sync_dist=True)

        if self.cfg.loss.use_gradient:
            self.log(f'{mode}_loss_gradient', loss['gradient'], batch_size=B, sync_dist=True)

        if self.cfg.loss.use_feature:
            self.log(f'{mode}_loss_feature', loss['feature'], batch_size=B, sync_dist=True)



    ############ DEBUGGING #############
    # used when testing only GenNerf model

    def training_step(self, batch, batch_idx):

        #self.origin = batch['origin']
        total_loss = self.process_step(batch, 'train')
        B = batch['image'].shape[0]
        self.log_loss(total_loss, B, 'train')

        return total_loss['combined']
    

    def validation_step(self, batch, batch_idx):

        #self.origin = batch['origin']
        total_loss = self.process_step(batch, 'val')

        B = batch['image'].shape[0]
        self.log_loss(total_loss, B, 'val')

        # visualize the prediction of the final batch and log it
        #is_last_batch = (batch_idx == len(self.trainer.val_dataloaders) - 1)
        #if is_last_batch:
        #    b = 0
        #    self.initialize_volume()
        #    self.encode(batch['projection'][b:b+1], batch['image'][b:b+1], batch['depth'][b:b+1], 'val')  # encode images of whole sequence at once
        #    self.geometric_reconstruction(batch, b_idx=b)

        return total_loss['combined']

    def test_step(self, batch, batch_idx):
        
        #self.origin = batch['origin']
        is_last_batch = (batch_idx == len(self.trainer.test_dataloaders) - 1)
        total_loss = self.process_step(batch, 'test', is_last_batch=is_last_batch)

        B = batch['image'].shape[0]
        self.log_loss(total_loss, B, 'test')
        
        # visualize the prediction of the final batch and log it
        if is_last_batch:
            b = 0
            self.initialize_volume()
            self.encode(batch['projection'][b:b+1], batch['image'][b:b+1], batch['depth'][b:b+1], 'val')  # encode images of whole sequence at once
            self.geometric_reconstruction(batch, b_idx=b)

        return #total_loss['combined']


    def process_step(self, batch, mode, is_last_batch=False):
        image = batch['image'] # (B, T, 3, H, W)
        depth = batch['depth'] # (B, T, H, W)
        pose = batch['pose']  # (B, T, 4, 4) camera2world
        projection = batch['projection']  # (B, T, 3, 4) world2image
        intrinsics = batch['intrinsics']  # (B, T, 3, 3)
        tsdf_vol = batch['vol_%02d_tsdf'%self.voxel_sizes[0]]  # (B, 1, nx, ny, nz)
        # nx, ny, nz depend on current config (transformed from loaded ground-truth)
        B, T, _, H, W = image.shape

        self.initialize_volume()
        self.encode(projection, image, depth, mode)  # encode images of whole sequence at once
        # TODO: test merging vs processing all observations at once

        # transpose batch and time so we can go through sequentially
        # (B, T, 3, H, W) -> (T, B, C, H, W)
        images = image.transpose(0,1)
        depths = depth.transpose(0,1)
        poses = pose.transpose(0,1)
        projections = projection.transpose(0,1)
        intrinsicss = intrinsics.transpose(0,1)

        total_loss = {}
        for i, (image, depth, pose, projection, intrinsics) in enumerate(zip(images, depths, poses, projections, intrinsicss)):          
  
            surface_map = get_3d_points(depth, projection) # all surface points

            if self.cfg.loss.use_gradient and self.cfg.sampling_mode != 'ray':
                    raise NotImplementedError(f"Gradient loss not implemented for sampling_mode: {self.cfg.sampling_mode}")

            if self.cfg.sampling_mode == 'ray':
                
                if self.cfg.loss.use_gradient:

                    normals = [estimate_pointcloud_normals(surface_map[b]) for b in range(B)]
                    normals = torch.stack(normals, dim=0) # (B, H, W, 3)

                    b_idxs, h_idxs, w_idxs = sample_valid_pixels(depth, normals, self.cfg.ray.num_rays) # [B, 1] # [B, n_rays] # [B, n_rays]
                    sampled_normals = normals[b_idxs, h_idxs, w_idxs]  # (B, n_rays, 3)
                    surface_xyz = surface_map[b_idxs, h_idxs, w_idxs]  # (B, n_rays, 3) # TODO: not required (can be removed)
                    #assert ~torch.isnan(sampled_normals).any()

                else:
                    #b_idxs, h_idxs, w_idxs = sample_pixels(B, H, W, n_rays, self.device) # [B, 1] # [B, n_rays] # [B, n_rays] 
                    b_idxs, h_idxs, w_idxs = sample_valid_depth_pixels(depth, self.cfg.ray.num_rays) # [B, 1] # [B, n_rays] # [B, n_rays] 
                
                # select depth values at sampled pixel coordinates
                sampled_depth = depth[b_idxs, h_idxs, w_idxs]  # Shape: (B, n_rays)
                #assert ~torch.isnan(sampled_depth).any()

                # sampled_xyz, surface_xyz, z =
                sampled_xyz, z = sample_points_on_rays(
                    h_idxs,
                    w_idxs,
                    sampled_depth,
                    intrinsics,
                    pose,
                    N=self.cfg.ray.N,
                    M=self.cfg.ray.M,
                    delta=self.cfg.ray.delta,
                    min_dist=self.cfg.ray.d_min,
                    sigma=self.cfg.ray.sigma)
                
                if self.cfg.loss.use_gradient:
                    bounds, grad_vec = bounds_pc_batch(sampled_xyz, z, sampled_depth)  # (B, n_rays, 1+N+M), (B, n_rays, N+M, 3)

            elif self.cfg.sampling_mode == 'frustum':

                N_free = self.cfg.frustum.N_free
                N_near = self.cfg.frustum.N_near
                N_surf = self.cfg.frustum.N_surf
                
                N = N_surf + N_near + N_free

                #b_idxs, h_idxs, w_idxs = sample_pixels(B, H, W, N, self.device) # [B, 1] # [B, N] # [B, N] 
                b_idxs, h_idxs, w_idxs = sample_valid_depth_pixels(depth, N) # [B, 1] # [B, N] # [B, N] 

                # select depth values at sampled pixel coordinates
                sampled_depth = depth[b_idxs, h_idxs, w_idxs]  # Shape: (B, N)
                assert ~torch.isnan(sampled_depth).any()

                h_idxs_free = h_idxs[:, 0:N_free]
                w_idxs_free = w_idxs[:, 0:N_free]
                
                h_idxs_near = h_idxs[:, N_free:(N_free+N_near)]
                w_idxs_near = w_idxs[:, N_free:(N_free+N_near)]

                h_idxs_surf = h_idxs[:, (N_free+N_near):N]
                w_idxs_surf = w_idxs[:, (N_free+N_near):N]
                
                free_xyz, z = sample_points_in_frustum(
                    h_idxs_free,
                    w_idxs_free,
                    intrinsics,
                    pose,
                    min_dist=self.cfg.frustum.d_min,
                    max_dist=self.cfg.frustum.d_max)

                surface_xyz = surface_map[b_idxs, h_idxs_surf, w_idxs_surf]
                near_xyz = surface_map[b_idxs, h_idxs_near, w_idxs_near]
                noise = torch.normal(mean=0.0, std=self.cfg.frustum.sigma, size=near_xyz.shape, device=self.device)
                near_xyz += noise
                sampled_xyz = torch.cat((surface_xyz, near_xyz, free_xyz), dim=1)  # (B, N, 3)
                
            else:
                raise NotImplementedError(f"Usage of unknown sampling_mode: {self.cfg.sampling_mode}")            

            sampled_xyz = sampled_xyz.view(B, -1, 3)
            sampled_xyz.requires_grad_(True)
            outputs = self.forward(sampled_xyz)
            
            if mode=='test' and is_last_batch:
                #self.logger.local.log_tensor(xyz, f'frustum_sampling/all_points_{i}')
                self.logger.local.log_tensor(sampled_xyz, f'frustum_sampling/sampled_points_{i}')
                self.logger.local.log_tensor(pose, f'frustum_sampling/pose_{i}')
                self.logger.local.log_tensor(intrinsics, f'frustum_sampling/intrinsics_{i}')
                self.logger.local.log_image(image[0], f'frustum_sampling/image_{i}')
                self.logger.local.log_image(depth[0], f'frustum_sampling/depth_{i}')

            targets = {}
            targets['tsdf'] = trilinear_interpolation(tsdf_vol.permute(0, 2, 3, 4, 1), sampled_xyz, self.origin.squeeze(), self.cfg.voxel_size)
            #targets['tsdf'] = -bounds.reshape(B, -1, 1)  # for using bounds

            if self.cfg.loss.use_gradient:
                targets['sampled_normals'] = sampled_normals
                targets['grad_vec'] = -grad_vec  # flip to correct direction (like grad and sampled_normals)

            if mode != 'test' and (self.cfg.loss.use_eikonal or self.cfg.loss.use_gradient):
                tsdf = outputs['tsdf']  # (B, N, 1)
                grad = calculate_grad(sampled_xyz, tsdf)  # (B, N, 3)
                outputs['grad'] = grad  # (B, n_rays, 1+N+M, 3)

            #if mode == 'val' and self.current_epoch >= 100:
            #    show_normals(sampled_xyz, grad, title=f"Visualization of sampled_xyz and grad in epoch: {self.current_epoch}")
            #    show_normals(surface_xyz, sampled_normals, title=f"Visualization of surface_xyz and sampled_normals in epoch: {self.current_epoch}")
            #    pc = sampled_xyz.clone().view(B, self.cfg.ray.num_rays, -1, 3)[:, :, 1:].reshape(B, -1, 3)
            #    show_normals(pc, grad_vec.reshape(B, -1, 3), title=f"Visualization of sampled_xyz and grad_vec in epoch: {self.current_epoch}")

            if mode == 'test' and (self.cfg.loss.use_gradient or self.cfg.loss.use_eikonal):
                # eikonal loss not working due to missing grad_fn during test mode
                loss = {}
            else:
                loss = self.calculate_loss(outputs, targets)
            total_loss = add_dicts(total_loss, loss)
        return total_loss
    
    def geometric_reconstruction(self, batch, b_idx=0):
        
        # get tsdfs
        tsdf_pred = self.predict_tsdf(batch, b_idx) # (B, nx, ny, nz)
        tsdf_trgt = batch['vol_%02d_tsdf'%self.voxel_sizes[0]].squeeze(1)  # (B, nx, ny, nz)
        tsdf_trgt = tsdf_trgt[b_idx:b_idx+1, ...]  # select batch (1, nx, ny, nz)

        # get tsdf objs
        tsdf_pred = self.postprocess_tsdf(tsdf_pred, self.origin)
        tsdf_trgt = self.postprocess_tsdf(tsdf_trgt, self.origin)

        # get meshes
        meshes_pred = self.extract_mesh(tsdf_pred) # batch['origin']
        meshes_trgt = self.extract_mesh(tsdf_trgt) # batch['origin']
        
        # log locally (for debugging)
        self.logger.local.log_tsdf(tsdf_pred[0], f'test_tsdf/test_pred_tsdf')
        self.logger.local.log_tsdf(tsdf_trgt[0], f'test_tsdf/test_trgt_tsdf')
        self.logger.local.log_mesh(meshes_pred[0], f'test_mesh/test_pred_mesh')
        self.logger.local.log_mesh(meshes_trgt[0], f'test_mesh/test_trgt_mesh')
        
        # log to wandb
        #self.logger.log_mesh(pred_mesh, 'test_pred_mesh')
        #self.logger.log_mesh(trgt_mesh, 'test_trgt_mesh')
        self.log_rendered_images(meshes_pred[0], meshes_trgt[0], batch, b_idx)

    
    def log_rendered_images(self, meshe_pred, meshe_trgt, batch, b_idx=0):
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
            
            if i == 0:
                # add overview image of meshes
                color_ov_img_pred, _ = render(renderer_pred, scene_pred, intrinsics[b_idx], overview_pose)
                color_ov_img_trgt, _ = render(renderer_trgt, scene_trgt, intrinsics[b_idx], overview_pose)
                self.logger.log_image(key=batch['scene'][b_idx], images=[color_ov_img_trgt, color_ov_img_pred], caption=caption[1:3])
            
            # add near images of meshes
            color_img_pred, _ = render(renderer_pred, scene_pred, intrinsics[b_idx], pose[b_idx])
            color_img_trgt, _ = render(renderer_trgt, scene_trgt, intrinsics[b_idx], pose[b_idx])
            self.logger.log_image(key=f'frame{i}', images=[image[b_idx], color_img_trgt, color_img_pred], caption=caption)


    def predict_tsdf(self, batch, b_idx):

        # get dimensions from target tsdf
        tsdf_trgt = batch['vol_%02d_tsdf'%self.voxel_sizes[0]].squeeze(1)  # (B, nx, ny, nz)
        tsdf_trgt = tsdf_trgt[b_idx:b_idx+1, ...]  # select batch (1, nx, ny, nz)

        _, nx, ny, nz = tsdf_trgt.shape
        volume_size = self.cfg.voxel_size * np.array(self.cfg.voxel_dim_test)
        
        # sample grid points
        grid_xyz = get_grid_coordinates(nx, ny, nz, volume_size, self.origin, device=self.device)  # (nx, ny, nz, 3)
        grid_xyz = grid_xyz.reshape(-1, 3)  # (N, 3)
        grid_xyz = grid_xyz.unsqueeze(0)  # (B=1, N, 3)
        #grid_xyz = grid_xyz.repeat(B, 1, 1)  # (B, N, 3)  # opt. repeat B times
        grid_xyz.requires_grad_(True)

        # get predicted tsdf        
        chunk_size = 10000
        chunks = torch.split(grid_xyz, chunk_size, dim=1)  # split along N dimension
        with torch.no_grad():
            chunk_list = []
            for chunk in chunks:
                output = self.forward(chunk)
                tsdf_pred = output['tsdf'].detach().cpu()
                chunk_list.append(tsdf_pred)
        tsdf_pred = torch.cat(chunk_list, dim=1)
        #outputs = self.forward(grid_xyz)
        #tsdf_pred = outputs['tsdf']  # (B, N, 1)
        tsdf_pred = tsdf_pred.reshape(1, nx, ny, nz)  # (1, nx, ny, nz)

        # debug logging
        #print("vol-dims:", nx, ny, nz)
        #print("vol-size:", volume_size)
        #print("self.origin:", self.origin)
        corner_xyz = get_corner_coordinates(volume_size, self.origin, device=self.device)
        self.logger.local.log_tensor(corner_xyz, f'test_mesh/corner_points')
        #self.logger.local.log_tensor(grid_xyz, f'test_mesh/grid_points')
        
        return tsdf_pred
