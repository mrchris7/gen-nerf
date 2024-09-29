# Adapted from: https://github.com/magicleap/Atlas/blob/master/atlas/model.py

import itertools
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.models.components.positional_encoding import PositionalEncoding
import torch
import torch.nn.functional as F
import lightning as L
import wandb
import tempfile
from scipy.interpolate import RegularGridInterpolator
from src.models.components.pointnet import LocalPoolPointnet
from src.models.components.spatial_encoder import SpatialEncoder
from src.models.components.resnetfc import ResnetFC
from src.models.components.heads3d import TSDFHeadSimple
from src.models.utils import add_dicts, farthest_point_sample, get_3d_points,\
    normalize_coordinate, sample_points_in_frustum, log_transform
from src.data.tsdf import TSDF, coordinates


def backproject(voxel_dim, voxel_size, origin, projection, features):
    """ Takes 2d features and fills them along rays in a 3d volume

    This function implements eqs. 1,2 in https://arxiv.org/pdf/2003.10432.pdf
    Each pixel in a feature image corresponds to a ray in 3d.
    We fill all the voxels along the ray with that pixel's features.

    Args:
        voxel_dim: size of voxel volume to construct (nx, ny, nz)
        voxel_size: metric size of each voxel (ex: .04m)
        origin (1, 3): origin of the voxel volume (xyz position of voxel (0,0,0))
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

# TODO: find an alternative with gpu-support
def trilinear_interpolation(voxel_volume, xyz, origin, voxel_size):
    """
    Perform trilinear interpolation to map 3D world points to features in the voxel volume.
    
    Args:
        voxel_volume (B, nx, ny, nz, C): voxel volume
        xyz (B, N, 3): 3D world points
        origin (3,): world coordinates of voxel (0, 0, 0)
        voxel_size: size of each voxel
    
    Returns:
        features (B, N, C): interpolated features
    """
    device = voxel_volume.device
    B, nx, ny, nz, C = voxel_volume.shape
    N = xyz.shape[1]
    
    x = torch.linspace(0.0, nx*voxel_size, nx) + origin[0]
    y = torch.linspace(0.0, ny*voxel_size, ny) + origin[1]
    z = torch.linspace(0.0, nz*voxel_size, nz) + origin[2]
    points = (x, y, z) # x=(nx,) y=(ny,) z=(nz,)

    features = torch.empty(B, N, C, device=device)
    for batch in range(B):

        interpolator = RegularGridInterpolator(points, voxel_volume[batch].detach().cpu(),
                                               bounds_error=False, fill_value=None) # extrapolate outside bounds
        interpolated_features = interpolator(xyz[batch].detach().cpu())
        features[batch] = torch.from_numpy(interpolated_features).to(device)
    
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
            feat_2d = F.interpolate(feat_2d, scale_factor=2, mode='bilinear', align_corners=False)
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
        
        ######
        #point_cloud_np = accum_sparse_xyz[0].cpu().numpy()
        #point_cloud_o3d = o3d.geometry.PointCloud()
        #point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np)
        #o3d.io.write_point_cloud("/home/atuin/g101ea/g101ea13/debug/point_cloud_x2.ply", point_cloud_o3d)
        ######

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
        B, N, _ = xyz.size()
        d_out_geo = self.cfg.mlp.d_out_geo
        d_out_sem = self.cfg.mlp.d_out_sem
        
        feat = self.map_features(xyz)  # [B, N, d_latent=encoder_latent]
        
        if self.cfg.use_code:
            B, N, _ = xyz.shape
            xyz = xyz.reshape(-1, 3)  # (B*N, 3)
            xyz = self.code(xyz)
            xyz = xyz.reshape(B, N, -1)

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
        
        mask_observed = trgt < 1
        mask_outside  = (trgt == 1).all(-1, keepdim=True)
        
        if self.cfg.tsdf_loss.log_transform: # breaks gradient for eikonal loss calculation
            pred = log_transform(pred, self.cfg.tsdf_loss.log_transform_shift)
            trgt = log_transform(trgt, self.cfg.tsdf_loss.log_transform_shift)

        loss = F.l1_loss(pred, trgt, reduction='none') * self.cfg.tsdf_loss.weight
        loss = loss[mask_observed | mask_outside].mean()
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

        if self.cfg.tsdf_loss.smoothness_reg:
            losses['smooth'] = self.loss_smooth(outputs, targets)
            losses['combined'] += self.cfg.tsdf_loss.smoothness_reg.weight * losses['smooth']
        if self.cfg.tsdf_loss.eikonal_reg:
            losses['eikonal'] = self.loss_eikonal(outputs, targets)
            losses['combined'] += self.cfg.tsdf_loss.eikonal_reg.weight * losses['eikonal']

        return losses

    def log_loss(self, loss, B, mode):
        self.log(f'{mode}_loss_tsdf', loss['tsdf'], batch_size=B, sync_dist=True)
        self.log(f'{mode}_loss', loss['combined'], batch_size=B, sync_dist=True)
        
        if self.cfg.tsdf_loss.use_smoothness_reg:
            self.log(f'{mode}_loss_eikonal', loss['eikonal'], batch_size=B, sync_dist=True)
        if self.cfg.tsdf_loss.use_eikonal_reg:
            self.log(f'{mode}_loss_smooth', loss['smooth'], batch_size=B, sync_dist=True)





    ############ DEBUGGING #############
    # used when testing only GenNerf model

    def training_step(self, batch, batch_idx):
        # required if "FrameDataset" is used!
        if batch_idx == 0:
            L.seed_everything(0, workers=True)
            #print("train rand:", np.random.random(1))

        total_loss = self.process_step(batch, 'train')
        B = batch['image'].shape[0]
        self.log_loss(total_loss, B, 'train')

        return total_loss['combined']
    

    def validation_step(self, batch, batch_idx):
        # required if "FrameDataset" is used!
        if batch_idx == 0:
            L.seed_everything(1, workers=True)
            #print("val rand:", np.random.random(1))

        total_loss = self.process_step(batch, 'val')

        B = batch['image'].shape[0]
        self.log_loss(total_loss, B, 'val')

        return total_loss['combined']

    def test_step(self, batch, batch_idx):
        total_loss = self.process_step(batch, 'test')

        B = batch['image'].shape[0]
        device = batch['image'].device

        ''' # loss not working due to missing grad_fn during training mode
        total_loss = self.process_step(batch, 'test')
        self.log('test_loss_tsdf', total_loss['tsdf'], batch_size=B, sync_dist=True)
        #self.log('test_loss', total_loss['combined'], batch_size=B, sync_dist=True)
        '''

        # get target tsdf
        tsdf_trgt = batch['vol_%02d_tsdf'%self.voxel_sizes[0]]  # (B, 1, nx, ny, nz)
        tsdf_trgt = tsdf_trgt.squeeze(1)  # (B, nx, ny, nz)
        print("tsdf_trgt:", tsdf_trgt.shape)

        # get predicted tsdf
        _, nx, ny, nz = tsdf_trgt.shape
        volume_size = self.cfg.voxel_size*self.cfg.voxel_dim_test
        print("vol-dims:", nx, ny, nz)
        print("vol-size:", volume_size)
        x = torch.linspace(0, volume_size[0], nx, device=device)
        y = torch.linspace(0, volume_size[1], ny, device=device)
        z = torch.linspace(0, volume_size[2], nz, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

        # stack the grid coordinates and reshape to match input shape (B, N, 3)
        grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (nx, ny, nz, 3)
        
        grid_xyz = grid_xyz.reshape(-1, 3)  # flatten to (N, 3)
        grid_xyz = grid_xyz.unsqueeze(0)  # (B=1, N, 3)
        # optionally repeat for every batch (-> not necessary)
        #grid_points = grid_points.repeat(B, 1, 1)  # (B, N, 3)
        
        # debugging:
        debug_folder = '/home/atuin/g101ea/g101ea13/debug/test_mesh'
        torch.save(grid_xyz, f'{debug_folder}/grid_points.pt')

        outputs = self.forward(grid_xyz)
        tsdf_pred = outputs['tsdf']  # (B, N, 1)
        tsdf_pred = tsdf_pred.reshape(B, nx, ny, nz)  # (B, nx, ny, nz)

        debug_folder = '/home/atuin/g101ea/g101ea13/debug/test_tsdf'
        torch.save(tsdf_pred, f'{debug_folder}/test_pred_tsdf.pt')
        torch.save(tsdf_trgt, f'{debug_folder}/test_trgt_tsdf.pt')
        
        pred_tsdfs = self.postprocess(tsdf_pred)
        trgt_tsdfs = self.postprocess(tsdf_trgt)

        pred_mesh = pred_tsdfs[0].get_mesh()
        trgt_mesh = trgt_tsdfs[0].get_mesh()

        # Log image
        #image = batch['image'][0, 0, :, :, :].cpu().numpy()
        #image = np.transpose(image, (1, 2, 0))  # convert CHW to HWC
        #wandb.log({"test_image": wandb.Image(image)})

        # Log the meshes to wandb
        self.log_mesh_to_wandb(pred_mesh, "test_pred_mesh")
        self.log_mesh_to_wandb(trgt_mesh, "test_trgt_mesh")

        return total_loss['combined']

    def log_mesh_to_wandb(self, mesh, name):
        # create a temporary file to store the mesh
        with tempfile.NamedTemporaryFile(suffix=".obj") as tmpfile:
            mesh.export(tmpfile.name, file_type='obj')
            debug_folder = '/home/atuin/g101ea/g101ea13/debug/test_mesh'
            mesh.export(os.path.join(debug_folder, f"{name}.obj"))
            wandb.log({name: wandb.Object3D(tmpfile.name)})

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        # allow for different learning rates between pretrained layers 
        # (resnet backbone) and new layers (everything else).
        ###params_backbone2d = self.backbone2d[0].parameters()
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
        self.encode(projection, image, depth)  # encode images of whole sequence at once

        # transpose batch and time so we can go through sequentially
        # (B, T, 3, H, W) -> (T, B, C, H, W)
        images = image.transpose(0,1)
        depths = depth.transpose(0,1)
        poses = pose.transpose(0,1)
        projections = projection.transpose(0,1)
        intrinsicss = intrinsics.transpose(0,1)

        total_loss = {}
        for i, (image, depth, pose, projection, intrinsics) in enumerate(zip(images, depths, poses, projections, intrinsicss)):
            # maybe not necessary to go through all frames but only a subset?            
            sampled_xyz = sample_points_in_frustum(intrinsics, pose, self.cfg.num_points, min_dist=0.5, max_dist=4.0, img_width=W, img_height=H)

            if mode=='test':
                # save to view locally
                debug_folder = '/home/atuin/g101ea/g101ea13/debug/frustum_sampling'
                xyz = get_3d_points(image, depth, projection)
                torch.save(xyz, f'{debug_folder}/all_points_{i}.pt')
                torch.save(sampled_xyz, f'{debug_folder}/sampled_points_{i}.pt')
                torch.save(pose, f'{debug_folder}/pose_{i}.pt')
                torch.save(intrinsics, f'{debug_folder}/intrinsics_{i}.pt')
                torch.save(image, f'{debug_folder}/image_{i}.pt')
                torch.save(depth, f'{debug_folder}/depth_{i}.pt')

            outputs = self.forward(sampled_xyz)
            targets = {}
            targets['tsdf'] = trilinear_interpolation(tsdf_vol.permute(0, 2, 3, 4, 1), sampled_xyz, self.origin.squeeze(), self.cfg.voxel_size)
            targets['sampled_xyz'] = sampled_xyz
            loss = self.calculate_loss(outputs, targets)
            total_loss = add_dicts(total_loss, loss)
        return total_loss
