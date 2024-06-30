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

from PIL import Image, ImageOps
import numpy as np
import torch



class Compose(object):
    """ Apply a list of transforms sequentially"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

class ToTensor(object):
    """ Convert to torch tensors"""
    def __call__(self, data):
        for frame in data['frames']:
            image = np.array(frame['image'])
            frame['image'] = torch.as_tensor(image).float().permute(2, 0, 1)
            frame['intrinsics'] = torch.as_tensor(frame['intrinsics'])
            frame['pose'] = torch.as_tensor(frame['pose'])

            if 'depth' in frame:
                frame['depth'] = torch.as_tensor(np.array(frame['depth']))

            if 'instance' in frame:
                instance = np.array(frame['instance'])
                frame['instance'] = torch.as_tensor(instance).long()
        return data

class IntrinsicsPoseToProjection(object):
    """ Convert intrinsics and extrinsics matrices to a single projection matrix"""
    def __call__(self, data):
        for frame in data['frames']:
            intrinsics = frame.pop('intrinsics')
            pose = frame.pop('pose')
            frame['projection'] = intrinsics @ pose.inverse()[:3,:]
        return data


def pad_scannet(frame):
    """ Scannet images are 1296x968 but 1296x972 is 4x3
    so we pad vertically 4 pixels to make it 4x3
    """

    w,h = frame['image'].size
    if w==1296 and h==968:
        frame['image'] = ImageOps.expand(frame['image'], border=(0,2))
        frame['intrinsics'][1, 2] += 2
        if 'instance' in frame and frame['instance'] is not None:
            frame['instance'] = ImageOps.expand(frame['instance'], border=(0,2))
    return frame


class ResizeImage(object):
    """ Resize everything to given size.

    Intrinsics are assumed to refer to image prior to resize.
    After resize everything (ex: depth) should have the same intrinsics
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        for frame in data['frames']:
            pad_scannet(frame)

            w,h = frame['image'].size
            frame['image'] = frame['image'].resize(self.size, Image.BILINEAR)
            frame['intrinsics'][0, :] /= (w / self.size[0])
            frame['intrinsics'][1, :] /= (h / self.size[1])

            if 'depth' in frame:
                frame['depth'] = frame['depth'].resize(self.size, Image.NEAREST)

            if 'instance' in frame and frame['instance'] is not None:
                frame['instance'] = frame['instance'].resize(self.size, Image.NEAREST)
            #if 'semseg' in frame:
            #    frame['semseg'] = frame['semseg'].resize(self.size, Image.NEAREST)

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def transform_space(data, transform, voxel_dim, origin):
    """ Apply a 3x4 linear transform to the world coordinate system.

    This affects pose as well as TSDFs.
    """

    for frame in data['frames']:
        frame['pose'] = transform.inverse() @ frame['pose']

    voxel_sizes = [int(key[4:]) for key in data if key[:3]=='vol']

    for voxel_size in voxel_sizes:
        # compute voxel_dim for this voxel_size
        scale = voxel_size/min(voxel_sizes)
        vd = [int(vd/scale) for vd in voxel_dim]
        key = 'vol_%02d'%voxel_size

        # do transform
        data[key] = data[key].transform(transform, vd, origin)

    return data


class TransformSpace(object):
    """ See transform_space"""

    def __init__(self, transform, voxel_dim, origin):
        self.transform = transform
        self.voxel_dim = voxel_dim
        self.origin = origin

    def __call__(self, data):
        return transform_space(data, self.transform, self.voxel_dim, self.origin)

    def __repr__(self):
        return self.__class__.__name__


class RandomTransformSpace(object):
    """ Apply a random 3x4 linear transform to the world coordinate system."""

    def __init__(self, voxel_dim, random_rotation=True, random_translation=True,
                 paddingXY=1.5, paddingZ=.25, origin=[0,0,0]):
        """
        Args:
            voxel_dim: tuple of 3 ints (nx,ny,nz) specifying 
                the size of the output volume
            random_rotation: wheater or not to apply a random rotation
            random_translation: wheater or not to apply a random translation
            paddingXY: amount to allow croping beyond maximum extent of TSDF
            paddingZ: amount to allow croping beyond maximum extent of TSDF
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
        """

        self.voxel_dim = voxel_dim
        self.origin = origin
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.padding_start = torch.tensor([paddingXY, paddingXY, paddingZ])
        # no need to pad above (bias towards floor in volume)
        self.padding_end = torch.tensor([paddingXY, paddingXY, 0])

    def __call__(self, data):
        voxel_sizes = [int(key[4:]) for key in data if key[:3]=='vol']
        if len(voxel_sizes)==0:
            return data

        # pick a single tsdf to compute our transform
        voxel_size = min(voxel_sizes)
        tsdf = data['vol_%02d'%voxel_size]

        # construct rotaion matrix about z axis
        if self.random_rotation:
            r = torch.rand(1) * 2*np.pi
        else:
            r = 0
        # first construct it in 2d so we can rotate bounding corners in the plane
        R = torch.tensor([[np.cos(r), -np.sin(r)],
                          [np.sin(r), np.cos(r)]], dtype=torch.float32)

        # get corners of bounding volume
        voxel_dim = torch.tensor(tsdf.tsdf_vol.shape) * tsdf.voxel_size
        xmin, ymin, zmin = tsdf.origin[0]
        xmax, ymax, zmax = tsdf.origin[0] + voxel_dim
        corners2d = torch.tensor([[xmin, xmin, xmax, xmax],
                                  [ymin, ymax, ymin, ymax]])

        # rotate corners in plane
        corners2d = R @ corners2d

        # get new bounding volume (add padding for data augmentation)
        xmin = corners2d[0].min()
        xmax = corners2d[0].max()
        ymin = corners2d[1].min()
        ymax = corners2d[1].max()
        zmin = zmin
        zmax = zmax

        # randomly sample a crop
        start = torch.tensor([xmin, ymin, zmin]) - self.padding_start
        end = (-torch.as_tensor(self.voxel_dim) * tsdf.voxel_size +
                torch.tensor([xmax, ymax, zmax]) + self.padding_end)
        if self.random_translation:
            t = torch.rand(3)
        else:
            t = .5
        t = t*start + (1-t)*end
            
        T = torch.eye(4)
        T[:2,:2] = R
        T[:3,3] = -t

        # TODO: scale augmentation

        return transform_space(data, T.inverse(), self.voxel_dim, self.origin)

    def __repr__(self):
        return self.__class__.__name__


class FlattenTSDF(object):
    """ Take data out of TSDF data structure so we can collate into a batch"""
    def __call__(self, data):
        for key in list(data.keys()):
            if key[:3]=='vol':
                tsdf = data.pop(key)
                data['vol_'+key[4:]+'_tsdf'] = tsdf.tsdf_vol.unsqueeze(0)
                for attr in tsdf.attribute_vols.keys():
                    data['vol_'+key[4:]+'_'+attr] = tsdf.attribute_vols[attr]
        return data

    def __repr__(self):
        return self.__class__.__name__
