# Adapted from: https://github.com/magicleap/Atlas/blob/master/atlas/model.py

import itertools
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from src.models.components.pointnet import LocalPoolPointnet
from src.models.components.spatial_encoder import SpatialEncoder
from src.models.components.resnetfc import ResnetFC
from src.models.components.heads3d import TSDFHeadSimple
from src.models.utils import add_dicts, farthest_point_sample, get_3d_points, normalize_coordinate
from src.data.tsdf import TSDF, coordinates


def backproject(voxel_dim, voxel_size, origin, projection, features):
    """ Takes 2d features and fills them along rays in a 3d volume

    This function implements eqs. 1,2 in https://arxiv.org/pdf/2003.10432.pdf
    Each pixel in a feature image corresponds to a ray in 3d.
    We fill all the voxels along the ray with that pixel's features.

    Args:
        voxel_dim: size of voxel volume to construct (nx, ny, nz)
        voxel_size: metric size of each voxel (ex: .04m)
        origin: origin of the voxel volume (xyz position of voxel (0,0,0))
        projection (B, 4, 3): projection matrices (intrinsics@extrinsics)
        features (B, C, H, W): 2d feature tensor to be backprojected into 3d

    Returns:
        volume (B, C, nx, ny, nz): 3d feature volume
        valid (B, 1, nx, ny, nz): boolean volume, each voxel contains a 1 if it projects to a
                                  pixel and 0 otherwise (not in view frustrum of the camera)
    """

    B = features.size(0)
    C = features.size(1)
    device = features.device
    nx, ny, nz = voxel_dim

    coords = coordinates(voxel_dim, device).unsqueeze(0).expand(B,-1,-1) # (B, 3, H, W, D)
    world = coords.type_as(projection) * voxel_size + origin.to(device).unsqueeze(2)
    world = torch.cat((world, torch.ones_like(world[:,:1]) ), dim=1)
    
    camera = torch.bmm(projection, world)
    px = (camera[:,0,:]/camera[:,2,:]).round().type(torch.long)
    py = (camera[:,1,:]/camera[:,2,:]).round().type(torch.long)
    pz = camera[:,2,:]

    # voxels in view frustrum
    height, width = features.size()[2:]
    valid = (px >= 0) & (py >= 0) & (px < width) & (py < height) & (pz>0) # bxhwd

    # put features in volume
    volume = torch.zeros(B, C, nx*ny*nz, dtype=features.dtype, 
                         device=device)
    for b in range(B):
        volume[b,:,valid[b]] = features[b,:,py[b,valid[b]], px[b,valid[b]]]

    volume = volume.view(B, C, nx, ny, nz)
    valid = valid.view(B, 1, nx, ny, nz)

    return volume, valid


def trilinear_interpolation(voxel_volume, xyz, origin, voxel_size):
    """
    Perform trilinear interpolation to map 3D world points to features in the voxel volume.
    
    Args:
        voxel_volume (B, C, nx, ny, nz): voxel volume
        xyz (B, N, 3): 3D world points
        origin (3,): world coordinates of voxel (0, 0, 0)
        voxel_size: size of each voxel
    
    Returns:
        features (B, N, C): interpolated features
    """
    B, C, nx, ny, nz = voxel_volume.shape
    B, N, _ = xyz.shape
    device = xyz.device

    # normalize points to voxel grid coords and scale points to voxel grid dim
    xyz_normalized = (xyz - origin.to(device).view(1, 1, 3)) / voxel_size
    scaled_xyz = 2 * (xyz_normalized / torch.tensor([nx-1, ny-1, nz-1], device=xyz.device)) - 1

    # reshape (B, N, 1, 1, 3) to (B, 1, N, 1, 3)
    scaled_xyz = scaled_xyz.view(B, N, 1, 1, 3).permute(0, 3, 1, 2, 4)
    
    features = F.grid_sample(voxel_volume, scaled_xyz, align_corners=True)
    features = features.view(B, C, N).permute(0, 2, 1) # reshape to (B, N, C)
    
    return features


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
        # self.d_latent + self.d_in == x.shape(-1)
        self.mlp = ResnetFC.from_conf(cfg.mlp, d_in=3, d_latent=encoder_latent)  # TODO: check: d_in=dim_points=3, d_latent=dim_encoded_feature
        #self.head_geo = TSDFHead(cfg.head_geo, cfg.backbone3d.channels, cfg.voxel_size)  # # simpler head required that regresses tsdf-value from feature of point (instead of feature of whole volume)
        self.head_geo = TSDFHeadSimple(cfg.mlp.d_out_geo)
        
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

        # spatial encoder + f_teacher features -> backproject into voxel volume
        self.volume = 0 
        self.valid = 0

        # pointnet encoder
        self.c_plane = 0


    # TODO: ultimately this function should be callable multiple times allowing to
    # accumulate information every time it is called
    # -> currently PointNet does not support dynamic accumulation:
    #    if encode() is run again, the initially used pointcloud is gone
    def encode(self, projection, image, depth):
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


        accum_sparse_xyz = torch.empty(B, 0, 3, device=device)  # accumulate point cloud for PointNet
                                                                  # (make it a persistent pointcloud with self.sparse_xyz -> memory intensive)
        
        # go through every observation
        for image, depth, projection in zip(images, depths, projections):
            
            # accumulate 3D volume using spatial encoder on 2D data:
            B, C, H, W = image.size()
            feat_2d = torch.empty(B, 0, int(H/2), int(W/2), device=device)  # feature map from spatial encoder is halved
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
            volume, valid = backproject(voxel_dim, self.cfg.voxel_size, self.origin, projection, feat_2d)
            self.volume = self.volume + volume
            self.valid = self.valid + valid


            # accumulate a sparse 3D point cloud (later passed into PointNet):
            if self.cfg.encoder.use_pointnet:
                xyz = get_3d_points(image, depth, projection)
                centroids = farthest_point_sample(xyz, self.cfg.encoder.pointnet.num_sparse_points)
                sparse_xyz = xyz[torch.arange(B)[:, None], centroids]
                #sparse_xyz = self.normalizer(sparse_xyz)  # TODO: normalize?
                accum_sparse_xyz = torch.cat((accum_sparse_xyz, sparse_xyz), dim=1)
        
        # build volume using PointNet (currently it does not support dynamic accumulation)
        if self.cfg.encoder.use_pointnet:
            self.c_plane = self.pointnet(accum_sparse_xyz)  # dict with keys 'xy', 'yz', 'xz' 
                                                            # each (B, c_dim=512?, plane_reso=128, plane_reso=128)


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.cfg.encoder.pointnet.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.cfg.encoder.pointnet.sample_mode).squeeze(-1)
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

            feat_spatial = trilinear_interpolation(volume, xyz, self.origin, self.cfg.voxel_size)
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
        """
        B, N, _ = xyz.size()
        d_out_geo = self.cfg.mlp.d_out_geo
        d_out_sem = self.cfg.mlp.d_out_sem
        
        feat = self.map_features(xyz)  # [B, N, d_latent=encoder_latent]
        mlp_input = torch.cat((feat, xyz), dim=-1)

        mlp_output = self.mlp(mlp_input)
        mlp_output = mlp_output.reshape(B, N, d_out_geo + d_out_sem)

        feat_geo = mlp_output[...,           : d_out_geo            ]
        feat_sem = mlp_output[..., d_out_geo : d_out_geo + d_out_sem]
               
        outputs = {}
        outputs['feat_geo'] = feat_geo  # torch.identity(feat_geo)  # necessary?
        outputs['feat_sem'] = feat_sem  # torch.relu(feat_sem)

        tsdf = self.head_geo(feat_geo)
        outputs['tsdf'] = tsdf

        return outputs
    
    '''
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
            
            ## add semseg vol
            #if ('semseg' in self.voxel_types) and (key+'_semseg' in outputs):
            #    semseg = outputs[key+'_semseg'][i]
            #    if semseg.ndim==4:
            #        semseg = semseg.argmax(0)
            #    tsdf.attribute_vols['semseg'] = semseg
            #
            ## add color vol
            #if 'color' in self.voxel_types:
            #    color = outputs[key+'_color'][i]
            #    tsdf.attribute_vols['color'] = color
            #
            tsdf_data.append(tsdf)

        return tsdf_data
    '''

    def loss_tsdf(self, outputs, targets):
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(outputs['tsdf'], targets['tsdf'])
        return loss

    def calculate_loss(self, outputs, targets):
        losses = {}
        losses['tsdf'] = self.loss_tsdf(outputs, targets)
        losses['combined'] = losses['tsdf']
        return losses







    ############ DEBUGGING #############
    # used when testing only GenNerf model

    def training_step(self, batch, batch_idx):
        image = batch['image'] # (B, T, 3, H, W)
        depth = batch['depth'] # (B, T, H, W)
        projection = batch['projection']  # (B, T, 3, 4) world2image
        tsdf_vol = batch['vol_%02d_tsdf'%self.voxel_sizes[0]]  # (B, 1, 256, 256, 96)
        B, T, _, H, W = image.shape

        self.initialize_volume()
        self.encode(projection, image, depth)  # encode images of whole sequence at once

        # transpose batch and time so we can go through sequentially
        # (B, T, 3, H, W) -> (T, B, C, H, W)
        images = image.transpose(0,1)
        depths = depth.transpose(0,1)
        projections = projection.transpose(0,1)

        total_loss = {}
        for i, (image, depth, projection) in enumerate(zip(images, depths, projections)):

            ##########
            ## Save depth image
            #print("image shape", image.shape)
            #depth_image_norm = cv2.normalize(depth[0, :, :].cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            #cv2.imwrite(f'/home/atuin/g101ea/g101ea13/debug/depth_image_{i}.png', depth_image_norm)
            #
            ## Save color image
            #color_image = image[0, :, :, :].cpu().numpy()
            ## Ensure the image is in HWC format and normalize to 0-255
            #color_image = np.transpose(color_image, (1, 2, 0))  # Convert CHW to HWC
            #color_image = cv2.normalize(color_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ## OpenCV expects color images in BGR format
            #color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(f'/home/atuin/g101ea/g101ea13/debug/color_image_{i}.png', color_image_bgr)
            ###########

            xyz = get_3d_points(image, depth, projection)
            centroids = farthest_point_sample(xyz, 512)
            sparse_xyz = xyz[torch.arange(B)[:, None], centroids]
            outputs = self.forward(sparse_xyz)
            targets = {}
            targets['tsdf'] = trilinear_interpolation(tsdf_vol, sparse_xyz, self.origin, self.cfg.voxel_size)  # TODO either calculate dynamically or computed in advance
            loss = self.calculate_loss(outputs, targets)
            total_loss = add_dicts(total_loss, loss)

        self.log('loss_tsdf', total_loss['tsdf'], batch_size=B)
        self.log('loss', total_loss['combined'], batch_size=B)
        return total_loss['combined']
    

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)


    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        # allow for different learning rates between pretrained layers 
        # (resnet backbone) and new layers (everything else).
        ###params_backbone2d = self.backbone2d[0].parameters()
        modules = [self.mlp,
                   self.head_geo,
                   #self.f_teacher
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
            ###lr_backbone2d = lr * self.cfg.OPTIMIZER.BACKBONE2D_LR_FACTOR
            optimizer = torch.optim.Adam([
                ###{'params': params_backbone2d, 'lr': lr_backbone2d},
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
