# Adapted from: https://github.com/magicleap/Atlas/blob/master/atlas/model.py

import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.models.components.positional_encoding import PositionalEncoding
import torch
import torch.nn.functional as F
import lightning as L
from src.models.components.pointnet import LocalPoolPointnet
from src.models.components.spatial_encoder import SpatialEncoder
from src.models.components.resnetfc import ResnetFC
from src.models.components.heads3d import TSDFHeadSimple
from src.models.utils import *
from src.data.tsdf import TSDF
from src.utils.debug_logger import DebugLogger
from torch_cluster import knn


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
            encoder_latent += cfg.encoder.pointnet.c_dim
        if cfg.encoder.use_auxiliary:
            encoder_latent += 0 # self.cfg.f_teacher.feature_dim
        
        # decoders
        d_in = 3  # xyz
        if cfg.use_code:
            self.code = PositionalEncoding.from_conf(cfg.code, d_in=d_in)
            d_in = self.code.d_out

        self.mlp = ResnetFC.from_conf(cfg.mlp, d_in=d_in, d_latent=encoder_latent)
        #self.head_geo = TSDFHead(cfg.head_geo, cfg.backbone3d.channels, cfg.voxel_size)  # # simpler head required that regresses tsdf-value from feature of point (instead of feature of whole volume)
        self.head_geo = TSDFHeadSimple(cfg.mlp.d_out_geo)
        
        # other params
        self.origin = torch.tensor([0,0,0]).view(1,3)
        self.voxel_sizes = [int(cfg.voxel_size*100)]

        self.initialize_volume()

        self.debug_logger = DebugLogger(cfg.debug_dir)
        self.debug_logger.clear_data()

    def initialize_volume(self):
        """ Reset the accumulators.
        
        self.volume is a voxel volume containg the accumulated features
        self.valid is a voxel volume containg the number of times a voxel has
            been seen by a camera view frustrum
        """

        # spatial encoder + f_teacher features -> backproject into voxel volume
        self.volume = 0 
        self.valid = 0

        # pointnet encoder
        self.c_plane = 0


    # TODO: ultimately this function should be callable multiple times allowing to
    # accumulate information every time it is called
    # -> currently PointNet does not support dynamic accumulation:
    #    if encode() is run again, the initially used pointcloud is gone
    def encode(self, projection, image, depth, mode):
        """ Encodes image and corresponding pointcloud into a 3D feature volume and 
        accumulates them. This is the first half of the network which
        is run on F frames.

        Args:
            projection: (B, T, 4, 4) pose matrix
            image: (B, T, 3, H, W) conditioning rgb-image
            depth: (B, T, H, W) conditioning depth map

        Feature volume is accumulated into self.volume and self.valid
        """
        B = projection.size(0)
        device = image.device
       
        # transpose batch and time so we can accumulate sequentially
        # (B, T, 3, H, W) -> (T, B, C, H, W)
        images = image.transpose(0,1)
        depths = depth.transpose(0,1)
        projections = projection.transpose(0,1)

        # TODO: declare dimension T in advance (momory-efficient)
        accum_sparse_xyz = torch.empty(B, 0, 3, device=device)  # accumulate point cloud for PointNet
                                                                # (make it a persistent pointcloud with self.sparse_xyz -> memory intensive)
        
        # go through every observation
        for image, depth, projection in zip(images, depths, projections):
            
            # accumulate 3D volume using spatial encoder on 2D data:
            B, C, H, W = image.size()
            feat_2d = torch.empty(B, 0, H, W, device=device)  # feature map from spatial encoder is halved
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
                self.volume = self.volume + volume
                self.valid = self.valid + valid


            # accumulate a sparse 3D point cloud (later passed into PointNet):
            if self.cfg.encoder.use_pointnet:
                xyz = get_3d_points(image, depth, projection)
                
                sparse_xyz = farthest_point_sample(xyz, self.cfg.encoder.pointnet.num_sparse_points)
                #sparse_xyz = self.normalizer(sparse_xyz)  # TODO: normalize?
                accum_sparse_xyz = torch.cat((accum_sparse_xyz, sparse_xyz), dim=1)

        if mode == 'test':
            self.debug_logger.log_tensor("sparse_points", "sparse_points", accum_sparse_xyz)
            self.debug_logger.log_tensor("sparse_points", "dense_points", xyz)

        # build volume using PointNet (currently it does not support dynamic accumulation)
        if self.cfg.encoder.use_pointnet:
            self.c_plane = self.pointnet(accum_sparse_xyz)  # dict with keys 'xy', 'yz', 'xz' 
                                                            # each (B, c_dim=512?, plane_reso=128, plane_reso=128)


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.cfg.encoder.pointnet.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        #c_check = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.cfg.encoder.pointnet.sample_mode).squeeze(-1)
        c = grid_sample_2d(c, vgrid)
        #assert(c_check.shape == c.shape)
        #assert(torch.allclose(c_check, c, atol=1e-3, rtol=1e-4))
        return c

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
        device = xyz.device

        # combine features from different encodings
        feat = torch.empty(B, N, 0, device=device)
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

            mlp_input = torch.cat((feat, xyz), dim=-1)
            mlp_output = self.mlp(mlp_input)
            mlp_output = mlp_output.reshape(B, N, d_out_geo + d_out_sem)

            feat_geo = mlp_output[...,           : d_out_geo            ]
            feat_sem = mlp_output[..., d_out_geo : d_out_geo + d_out_sem]
            
            tsdf = self.head_geo(feat_geo)

        outputs = {}
        outputs['feat_geo'] = feat_geo  # torch.identity(feat_geo)  # necessary?
        outputs['feat_sem'] = feat_sem  # torch.relu(feat_sem)    
        outputs['tsdf'] = tsdf

        return outputs

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        modules = [
            self.mlp,
            self.head_geo
        ]
        if self.cfg.encoder.use_spatial:
            modules.append(self.spatial)
        if self.cfg.encoder.use_pointnet:
            modules.append(self.pointnet)
        
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
    
    
    def postprocess(self, tsdf_vol):
        """ Wraps the network output into a TSDF data structure
        
        Args:
            tsdf_vol: tsdf volume (B, nx, ny, nz)

        Returns:
            list of TSDFs (one TSDF per scene in the batch) (length == batchsize)
        """
        tsdf_data = []
        batch_size = len(tsdf_vol)

        for i in range(batch_size):
            tsdf = TSDF(self.cfg.voxel_size, self.origin, tsdf_vol[i].squeeze(0))
            tsdf_data.append(tsdf)

        return tsdf_data

    def loss_tsdf(self, outputs, targets):
        pred = outputs['tsdf']  # [B, N, 1]
        trgt = targets['tsdf']  # [B, N, 1]
        
        #mask_observed = trgt < 1
        #mask_outside  = (trgt == 1).all(-1, keepdim=True)
        
        if self.cfg.tsdf_loss.log_transform: # breaks gradient for eikonal loss calculation
            pred = smooth_log_transform(pred, self.cfg.tsdf_loss.log_transform_shift, self.cfg.tsdf_loss.log_transform_beta)
            trgt = smooth_log_transform(trgt, self.cfg.tsdf_loss.log_transform_shift, self.cfg.tsdf_loss.log_transform_beta)

        loss = F.l1_loss(pred, trgt, reduction='none') * self.cfg.tsdf_loss.weight
        #loss = loss[mask_observed | mask_outside]
        loss = loss.mean()
        
        return loss

    '''
    def loss_smooth(self, outputs, targets):
        pred = outputs['tsdf']  # [B, N, 1]
        sampled_xyz = targets['sampled_xyz']
        
        device = pred.device
        B, N, _ = pred.shape
        loss = 0.0
        for b in range(B):
            p = sampled_xyz[b].detach().cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=self.cfg.tsdf_loss.smoothness_reg.k, algorithm='ball_tree').fit(p)
            distances, indices = nbrs.kneighbors(sampled_xyz[b].detach().cpu().numpy())

            # calculate the difference in tsdf values between neighbors
            tsdf_values = pred[b].squeeze(1)  # (N,)
            for i in range(N):
                tsdf_i = tsdf_values[i]  # tsdf value of the ith point
                neighbors_tsdf = tsdf_values[indices[i]]  # tsdf values of the neighbors
                loss += torch.mean((tsdf_i - neighbors_tsdf) ** 2)
        
        loss = loss / (B * N)
        return loss.to(device)
    '''
    
    def loss_smooth(self, outputs, targets): 
        pred = outputs['tsdf']  # [B, N, 1]
        sampled_xyz = targets['sampled_xyz']  # [B, N, 3]
    
        device = pred.device
        B, N, _ = pred.shape
        k = self.cfg.tsdf_loss.smoothness_reg.k

        # flatten the batched tensors to treat them as one large point cloud
        p = sampled_xyz.view(-1, 3).float().cpu()  # [B * N, 3]  # use cpu fallback, cuda not supported
        tsdf_values = pred.view(-1).cpu()  # [B * N]  # use cpu fallback, cuda not supported

        # create a batch index tensor to track the batch of each point
        batch_idx = torch.arange(B, device=p.device).repeat_interleave(N)  # [B * N]

        # use torch-cluster's knn to find the k-nearest neighbors for each point
        neighbors_idx = knn(p, p, k=k, batch_x=batch_idx, batch_y=batch_idx)

        # output of knn: (2, num_edges)
        source_points = neighbors_idx[0]  # indices of the original points
        neighbor_points = neighbors_idx[1]  # indices of their neighbors

        # gather the tsdfs of the neighbors
        neighbor_tsdf_values = tsdf_values[neighbor_points]  # [B * N * k]

        # compute the difference between each point's tsdf value and its neighbors
        tsdf_diff = tsdf_values[source_points] - neighbor_tsdf_values

        # mean squared difference
        loss = torch.mean(tsdf_diff ** 2)

        return loss.to(device)


    def loss_eikonal(self, outputs, targets):
        # Eikonal term to encourage unit gradient norm
        # tsdf: (B, N, 1)
        # query_points: (B, N, 3)

        pred = outputs['tsdf']  # [B, N, 1]
        sampled_xyz = targets['sampled_xyz']
                   
        # Make sure tsdf_values have the right shape for grad_outputs
        grad_outputs = torch.ones_like(pred)

        # Compute the gradient of the TSDF with respect to the input points
        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=sampled_xyz,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # (B, N, 3)
        
        # norm of each gradient vector for all points in the batch
        # gradients.norm(2, dim=-1) computes the L2 norm across the last dimension (x, y, z)
        gradient_norm = gradients.norm(2, dim=-1)  # (B, N)
        
        # eikonal loss: |∇f(x)| - 1)^2, averaged over all points and batches
        loss = ((gradient_norm - 1) ** 2).mean()        
        return loss

    def calculate_loss(self, outputs, targets):
        losses = {}
        losses['tsdf'] = self.loss_tsdf(outputs, targets)
        losses['combined'] = losses['tsdf']

        if self.cfg.tsdf_loss.use_smoothness_reg:
            losses['smooth'] = self.loss_smooth(outputs, targets)
            losses['combined'] += self.cfg.tsdf_loss.smoothness_reg.weight * losses['smooth']
        if self.cfg.tsdf_loss.use_eikonal_reg:
            losses['eikonal'] = self.loss_eikonal(outputs, targets)
            losses['combined'] += self.cfg.tsdf_loss.eikonal_reg.weight * losses['eikonal']

        return losses

    def log_loss(self, loss, B, mode):
        self.log(f'{mode}_loss_tsdf', loss['tsdf'], batch_size=B, sync_dist=True)
        self.log(f'{mode}_loss', loss['combined'], batch_size=B, sync_dist=True)
        
        if self.cfg.tsdf_loss.use_smoothness_reg:
            self.log(f'{mode}_loss_smooth', loss['smooth'], batch_size=B, sync_dist=True)

        if self.cfg.tsdf_loss.use_eikonal_reg:
            self.log(f'{mode}_loss_eikonal', loss['eikonal'], batch_size=B, sync_dist=True)




    ############ DEBUGGING #############
    # used when testing only GenNerf model

    def training_step(self, batch, batch_idx):
        '''
        # required if "FrameDataset" is used!
        if batch_idx == 0:
            L.seed_everything(0, workers=True)
            #print("train rand:", np.random.random(1))
        '''

        total_loss = self.process_step(batch, 'train')
        B = batch['image'].shape[0]
        self.log_loss(total_loss, B, 'train')

        return total_loss['combined']
    

    def validation_step(self, batch, batch_idx):
        '''
        # required if "FrameDataset" is used!
        if batch_idx == 0:
            L.seed_everything(1, workers=True)
            #print("val rand:", np.random.random(1))
        '''
        
        total_loss = self.process_step(batch, 'val')

        B = batch['image'].shape[0]
        self.log_loss(total_loss, B, 'val')

        return total_loss['combined']

    def test_step(self, batch, batch_idx):
        B = batch['image'].shape[0]
        device = batch['image'].device

        # loss not working due to missing grad_fn during training mode
        total_loss = self.process_step(batch, 'test')
        self.log_loss(total_loss, B, 'test')

        # get target tsdf
        tsdf_trgt = batch['vol_%02d_tsdf'%self.voxel_sizes[0]]  # (B, 1, nx, ny, nz)
        tsdf_trgt = tsdf_trgt.squeeze(1)  # (B, nx, ny, nz)

        # get predicted tsdf
        _, nx, ny, nz = tsdf_trgt.shape
        volume_size = self.cfg.voxel_size * np.array(self.cfg.voxel_dim_test)
        print("vol-dims:", nx, ny, nz)
        print("vol-size:", volume_size)
        corner_xyz = get_corner_coordinates(volume_size, device=device)  # TODO: take grid_origin into account!
        grid_xyz = get_grid_coordinates(nx, ny, nz, volume_size, device=device)  # (nx, ny, nz, 3)
        grid_xyz = grid_xyz.reshape(-1, 3)  # (N, 3)
        grid_xyz = grid_xyz.unsqueeze(0)  # (B=1, N, 3)
        #grid_points = grid_points.repeat(B, 1, 1)  # (B, N, 3)  # opt. repeat B times
        grid_xyz.requires_grad_(True)

        outputs = self.forward(grid_xyz)
        tsdf_pred = outputs['tsdf']  # (B, N, 1)
        tsdf_pred = tsdf_pred.reshape(B, nx, ny, nz)  # (B, nx, ny, nz)
        
        # get meshes
        pred_tsdfs = self.postprocess(tsdf_pred)
        trgt_tsdfs = self.postprocess(tsdf_trgt)

        pred_mesh = pred_tsdfs[0].get_mesh()
        trgt_mesh = trgt_tsdfs[0].get_mesh()

        # log to wandb
        #log_mesh_to_wandb(pred_mesh, 'test_pred_mesh')
        #log_mesh_to_wandb(trgt_mesh, 'test_trgt_mesh')
        #log_image_to_wandb(batch['image'][0, 0, :, :, :], 'test_image')

        # Log to disk (for debugging)
        self.debug_logger.log_tensor('test_tsdf', 'test_pred_tsdf', tsdf_pred)
        self.debug_logger.log_tensor('test_tsdf', 'test_trgt_tsdf', tsdf_trgt)
        self.debug_logger.log_mesh('test_mesh', 'test_pred_mesh', pred_mesh)
        self.debug_logger.log_mesh('test_mesh', 'test_trgt_mesh', trgt_mesh)
        self.debug_logger.log_tensor("test_mesh", "corner_points", corner_xyz)
        #self.debug_logger.log_tensor("test_mesh", "grid_points", grid_xyz)

        return #total_loss['combined']


    def process_step(self, batch, mode):
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

        # transpose batch and time so we can go through sequentially
        # (B, T, 3, H, W) -> (T, B, C, H, W)
        images = image.transpose(0,1)
        depths = depth.transpose(0,1)
        poses = pose.transpose(0,1)
        projections = projection.transpose(0,1)
        intrinsicss = intrinsics.transpose(0,1)

        total_loss = {}
        for i, (image, depth, pose, projection, intrinsics) in enumerate(zip(images, depths, poses, projections, intrinsicss)):          
            
            if self.cfg.sampling_mode == 'ray':
                sampled_xyz = sample_points_on_rays(intrinsics, pose, depth,
                                                    num_samples=self.cfg.ray.num_rays,
                                                    N=self.cfg.ray.N, M=self.cfg.ray.M,
                                                    delta=self.cfg.ray.delta,
                                                    min_dist=self.cfg.ray.d_min,
                                                    sigma=self.cfg.ray.sigma)
                
            elif self.cfg.sampling_mode == 'frustum':
                free_xyz = sample_points_in_frustum(intrinsics, pose,
                                                    num_samples=self.cfg.frustum.N,
                                                    min_dist=self.cfg.frustum.d_min,
                                                    max_dist=self.cfg.frustum.d_max,
                                                    img_width=W,
                                                    img_height=H)
                xyz = get_3d_points(image, depth, projection)
                surface_xyz = farthest_point_sample(xyz, self.cfg.frustum.M)
                noise = torch.normal(mean=0.0, std=self.cfg.frustum.sigma, size=surface_xyz.shape, device=surface_xyz.device)
                surface_xyz += noise
                sampled_xyz = torch.cat((free_xyz, surface_xyz), dim=1)

            else:
                raise NotImplementedError(f"Usage of unknown sampling_mode: {self.cfg.sampling_mode}")

            sampled_xyz.requires_grad_(True)

            if mode=='test':
                #xyz = get_3d_points(image, depth, projection)                
                #self.debug_logger.log_tensor('frustum_sampling', f'all_points_{i}', xyz)
                self.debug_logger.log_tensor('frustum_sampling', f'sampled_points_{i}', sampled_xyz)
                self.debug_logger.log_tensor('frustum_sampling', f'pose_{i}', pose)
                self.debug_logger.log_tensor('frustum_sampling', f'intrinsics_{i}', intrinsics)
                self.debug_logger.log_tensor('frustum_sampling', f'image_{i}', image)
                self.debug_logger.log_tensor('frustum_sampling', f'depth_{i}', depth)
            
            #log_image_to_wandb(batch['image'][0, i, :, :, :], f'{mode}_image_{i}')

            outputs = self.forward(sampled_xyz)
            targets = {}
            targets['tsdf'] = trilinear_interpolation(tsdf_vol.permute(0, 2, 3, 4, 1), sampled_xyz, self.origin.squeeze(), self.cfg.voxel_size)
            targets['sampled_xyz'] = sampled_xyz
            loss = self.calculate_loss(outputs, targets)
            total_loss = add_dicts(total_loss, loss)
        #raise Exception()
        return total_loss
    
    def test_grad(self, input, output=None):
        print("test gradients:")
        
        x_b = input  # (N, 3)
        print("x:", x_b.shape)
        print("x_grad:", x_b.grad_fn)
        print("x_grad:", x_b.requires_grad)
        
        if output == None:
            return
        y_b = output  # (N, 1)
        print("y:", y_b.shape)
        print("y_grad:", y_b.grad_fn)
        print("y_grad:", y_b.requires_grad)
        
        gradients = torch.autograd.grad(
            outputs=y_b,  # prediction
            inputs=x_b,  # query
            grad_outputs=torch.ones_like(y_b),  # Grad output for autograd
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]
        print("result gradients:", gradients)
