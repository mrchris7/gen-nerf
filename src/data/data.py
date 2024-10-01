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

import io
import os
import json
import tarfile
import numpy as np
from PIL import Image
import torch
import trimesh
from src.data.tsdf import TSDF


DEPTH_SHIFT = 1000

def load_info_json(json_file):
    """ Open a json info_file and do a bit of preprocessing"""

    info = json.load(open(json_file,'r'))
    
    """
    not needed:
    if 'instances' in info and info['instances'] is not None:
        # json doesn't store keys as ints so we cast here
        info['instances'] = {int(k):v for k,v in info['instances'].items()}
    else:
        info['instances'] = None
    """
    return info
    

def map_frame(frame, frame_types=[], prepare=False):
    """ Load images and metadata for a single frame.

    Given an info json we use this to load the images, etc for a single frame

    Args:
        frame: dict with metadata and paths to image files
            (see datasets/README)
        frame_types: which images to load (ex: depth, semseg, etc)

    Returns:
        dict containg metadata plus the loaded image
    """

    data = {key:value for key, value in frame.items()}
    
    if prepare:
        data['image'] = Image.open(frame['file_name_image_prep'])
        if 'depth' in frame_types:
            depth = Image.open(frame['file_name_depth_prep'])
    else:
        data['image'] = open_from_archive(frame['file_name_image'])
        if 'depth' in frame_types:
            depth = open_from_archive(frame['file_name_depth'])
            
    depth = np.array(depth, dtype=np.float32) / DEPTH_SHIFT
    data['depth'] = Image.fromarray(depth)
    data['intrinsics'] = np.array(frame['intrinsics'], dtype=np.float32)
    data['pose'] = np.array(frame['pose'], dtype=np.float32)
    
    return data


def map_frames(frames, frame_ids, frame_types=[], prepare=False):
    """ Load images and metadata for frame_ids of frames.

    Given an info json we use this to load the images, etc for a all frames

    Args:
        frames: dicts with metadata and paths to image files
            (see datasets/README)
        frame_ids: frame ids to load
        frame_types: which images to load (ex: depth, semseg, etc)

    Returns:
        dict containg metadata plus the loaded images
    """

    # copy data of given frame ids
    frames_data = []
    for i in frame_ids:
        frames_data.append(frames[i].copy())

    if prepare:
        # images are stored unpacked (not used here)
        for data in frames_data:
            data['image'] = Image.open(data['file_name_image_prep'])
            if 'depth' in frame_types:
                depth = Image.open(data['file_name_depth_prep'])
                depth = np.array(depth, dtype=np.float32) / DEPTH_SHIFT
                data['depth'] = Image.fromarray(depth)
    else:
        # images are stored in an archive
        add_images(frames_data, is_depth=False)
        if 'depth' in frame_types:
            add_images(frames_data, is_depth=True)
    
    for data in frames_data:
        data['intrinsics'] = np.array(data['intrinsics'], dtype=np.float32)
        data['pose'] = np.array(data['pose'], dtype=np.float32)
    
    return frames_data


def map_tsdf(info, data, voxel_types, voxel_sizes):
    """ Load TSDFs from paths in info.

    Args:
        info: dict with paths to TSDF files (see datasets/README)
        data: dict to add TSDF data to
        voxel_types: list of voxel attributes to load with the TSDF
        voxel_sizes: list of voxel sizes to load

    Returns:
        dict with TSDFs included
    """

    if len(voxel_types)>0:
        for scale in voxel_sizes:
            data['vol_%02d'%scale] = TSDF.load(info['file_name_vol_%02d'%scale],
                                               voxel_types)
    return data

def open_from_archive(full_path):
    """ Load the frame from the tar archive."""
    # first extract tar_path and frame_name from the full_path
    # i.e. 'scene/color/1.jpg' -> ('scene/color/color.tar', '1.jpg')
    dir_path, frame_name = os.path.split(full_path)
    base_dir = os.path.basename(dir_path)
    tar_path = os.path.join(dir_path, base_dir + '.tar')

    # load frame from tar
    with tarfile.open(tar_path, 'r') as tar_file:
        image_member = tar_file.getmember(frame_name)
        image_file = tar_file.extractfile(image_member)
        image = Image.open(io.BytesIO(image_file.read()))

    return image

def add_images(frames_data, is_depth=False):
    """ Load all frames from the tar archive using the data dict."""
    # first construct the tar_path from the path to one frame
    # i.e. 'scene/color/1.jpg' -> 'scene/color/color.tar'
    
    if is_depth:
        first_frame_name = frames_data[0]['file_name_depth']  # assume all frames are in same tar
    else:
        first_frame_name = frames_data[0]['file_name_image']  # assume all frames are in same tar
    dir_path, _ = os.path.split(first_frame_name)  # -> ('scene/color', '1.jpg')
    base_dir = os.path.basename(dir_path)  # -> 'color'
    tar_path = os.path.join(dir_path, base_dir + '.tar')

    # load frames from tar
    with tarfile.open(tar_path, 'r') as tar_file:
        for data in frames_data:
            if is_depth:
                file_name = data['file_name_depth']
            else:
                file_name = data['file_name_image']
            frame_name = os.path.split(file_name)[1] # x.jpg
            image_member = tar_file.getmember(frame_name)
            image_file = tar_file.extractfile(image_member)
            image = Image.open(io.BytesIO(image_file.read()))

            # add to data dict
            if is_depth:
                image = np.array(image, dtype=np.float32) / DEPTH_SHIFT
                data['depth'] = Image.fromarray(image)
            else:
                data['image'] = image

def find_first_higher_index(list, val):
        # find the index of the first element that is higher than val
        for i, x in enumerate(list):
            if x > val:
                return i


class SceneDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, info_file, transform=None, frame_types=[],
                 voxel_types=[], voxel_sizes=[], num_frames=-1, prepare=False):
        """
        Args:
            info_file: path to json file (format described in datasets/README)
            transform: transform object to preprocess data
            frame_types: which images to load (ex: depth, semseg, etc)
            voxel_types: list of voxel attributes to load with the TSDF
            voxel_sizes: list of voxel sizes to load
            num_frames: number of evenly spaced frames to use (-1 for all)
        """

        self.info = load_info_json(info_file)
        self.transform = transform
        self.frame_types = frame_types
        self.voxel_types = voxel_types
        self.voxel_sizes = voxel_sizes
        self.prepare = prepare

        # select evenly spaced subset of frames
        if num_frames>-1:
            length = len(self.info['frames'])
            inds = np.linspace(0, length-1, num_frames, dtype=int)
            self.info['frames'] = [self.info['frames'][i] for i in inds]


    def __len__(self):
        return len(self.info['frames'])

    def __getitem__(self, i):
        """
        Returns:
            dict of meta data and images for a single frame
        """

        frame = map_frame(self.info['frames'][i], self.frame_types, self.prepare)

        # put data in common format so we can apply transforms
        data = {'dataset': self.info['dataset'],
                #'instances': self.info['instances'],
                'frames': [frame]}
        if self.transform is not None:
            data = self.transform(data)
        # remove data from common format and return the single frame
        data = data['frames'][0]

        return data

    def get_tsdf(self):
        """
        Returns:
            dict with TSDFs
        """

        # put data in common format so we can apply transforms
        data = {'dataset': self.info['dataset'],
                #'instances': self.info['instances'],
                'frames': [],
               }

        # load tsdf volumes
        data = map_tsdf(self.info, data, self.voxel_types, self.voxel_sizes)

        # apply transforms
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_mesh(self):
        # TODO: also get vertex instances/semantics
        return trimesh.load(self.info['file_name_mesh_gt'], process=False)


class ScenesDataset(torch.utils.data.Dataset):
    """ Pytorch Dataset for a multiple scenes
    
    getitem loads a sequence of frames from a scene
    along with the corresponding TSDF for the scene
    """

    def __init__(self, info_files, num_frames, transform=None, frame_types=[],
                 frame_selection='random', voxel_types=[], voxel_sizes=[]):
        """
        Args:
            info_files: list of info_json files
            num_frames: number of frames in the sequence to load
            transform: apply preprocessing transform to images and TSDF
            frame_types: which images to load (ex: depth, semseg, etc)
            frame_selection: how to choose the frames in the sequence
            voxel_types: list of voxel attributes to load with the TSDF
            voxel_sizes: list of voxel sizes to load
        """

        self.info_files = info_files
        self.num_frames = num_frames
        self.transform = transform
        self.frame_types = frame_types
        self.frame_selection = frame_selection
        self.voxel_types = voxel_types
        self.voxel_sizes = voxel_sizes

    def __len__(self):
        return len(self.info_files)

    def __getitem__(self, i):
        """ Load images and TSDF for scene i"""

        info = load_info_json(self.info_files[i])

        frame_ids = self.get_frame_ids(info)
        # print(frame_ids)
        frames = [map_frame(info['frames'][i], self.frame_types)
                  for i in frame_ids]  # TODO: move loop into extraction process to only open tar-file only once

        data = {'dataset': info['dataset'],
                'scene': info['scene'],
                #'instances': info['instances'],
                'frames': frames}

        # load tsdf volumes
        data = map_tsdf(info, data, self.voxel_types, self.voxel_sizes)

        # apply transforms
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_frame_ids(self, info):
        """ Get the ids of the frames to load"""

        if self.frame_selection=='random':
            # select num_frames random frames from the scene
            return torch.randint(len(info['frames']), size=[self.num_frames])
        else:
            raise NotImplementedError('frame selection %s'%self.frame_selection)
        

class ScenesSequencesDataset(torch.utils.data.Dataset):
    """ Pytorch Dataset for a multiple scenes and multiple sequences per scene
    
    getitem loads a sequence of frames from a scene
    along with the corresponding TSDF for the scene
    """

    def __init__(self, info_files, sequence_amount, sequence_length, sequence_locations,
                 sequence_order, num_frames, frame_locations, frame_order,
                 transform=None, frame_types=[], voxel_types=[], voxel_sizes=[]):
        """
        Args:
            info_files: list of info_json files
            sequence_amount: [0.0, 1.0] controls the number of sequences scene-denpendently
            sequence_length: number of raw frames to be considered as one sequence
            sequence_locations: the location of sequences in a scene
            sequence_order: choose the sequence order
            num_frames: number of frames in the sequence to load
            frame_locations: the location of frames in a sequence
            frame_order: choose the sequence order
            transform: apply preprocessing transform to images and TSDF
            frame_types: which images to load (ex: depth, semseg, etc)
            voxel_types: list of voxel attributes to load with the TSDF
            voxel_sizes: list of voxel sizes to load
        """

        self.info_files = info_files
        self.sequence_amount = sequence_amount
        self.sequence_length = sequence_length
        self.sequence_locations = sequence_locations
        self.sequence_order = sequence_order
        self.num_frames = num_frames
        self.frame_locations = frame_locations
        self.frame_order = frame_order
        self.transform = transform
        self.frame_types = frame_types
        self.voxel_types = voxel_types
        self.voxel_sizes = voxel_sizes
        
        start_idxs_list = []
        num_sequences_list = []
        delete_scenes_idxs = []
        for i, info_file in enumerate(self.info_files):
            # calculate num sequences for each scene
            info = load_info_json(info_file)
            num_scene_frames = len(info['frames'])
            num_sequences = int(self.sequence_amount * (num_scene_frames / self.sequence_length))
            #print("num_seq", num_sequences)

            if num_scene_frames < self.sequence_length:
                # exclude scenes that have not enough frames
                delete_scenes_idxs.append(i)
                continue
            
            num_sequences_list.append(num_sequences)

            # calculate start indices for each sequence of each scene
            start_idxs = self.calculate_start_idxs(num_scene_frames, num_sequences)
            #print("start_idxs", start_idxs)
            #start_idxs = [5466]

            if self.sequence_order=='random':
                pass  # already random
            elif self.sequence_order=='sorted':
                start_idxs.sort()
            else:
                raise NotImplementedError(f"sequence_order: {self.sequence_order}")

            start_idxs_list.append(start_idxs)

        # exclude scenes that have not enough frames
        for i in sorted(delete_scenes_idxs, reverse=True):
            info = load_info_json(self.info_files[i])
            print(f"exclude scene {i} ({info['scene']})")
            del self.info_files[i]

        self.num_sequences_list = num_sequences_list
        self.start_idxs_list = start_idxs_list

    def __len__(self):
        return sum(self.num_sequences_list)

    def __getitem__(self, i):
        """ Load images and TSDF for scene i"""
        assert(i>=0)

        scene_idx, sequence_idx = self.get_indices(i)
        info = load_info_json(self.info_files[scene_idx])
        frame_ids = self.get_frame_ids(scene_idx, sequence_idx)

        if self.frame_order=='random':
            pass  # already random
        elif self.sequence_order=='sorted':
            frame_ids.sort()
        else:
            raise NotImplementedError(f"sequence_order: {self.sequence_order}")

        #print("frame_ids", frame_ids)
        frames = map_frames(info['frames'], frame_ids, self.frame_types)

        data = {'dataset': info['dataset'],
                'scene': info['scene'],
                #'instances': info['instances'],
                'frames': frames}

        # load tsdf volumes
        data = map_tsdf(info, data, self.voxel_types, self.voxel_sizes)

        # apply transforms
        if self.transform is not None:
            data = self.transform(data)

        return data

    def calculate_start_idxs(self, num_scene_frames, num_sequences):
        if self.sequence_locations == 'free':
            # freely select sequences across all frames (with potential overlap)
            num_start_idxs = num_scene_frames-self.sequence_length
            free_idxs = np.random.choice(num_start_idxs, num_sequences, replace=False)
            return free_idxs
        
        elif self.sequence_locations == 'fixed':
            # divide all frames into fixed sequences and select randomly (without overlap)
            max_num_sequences = num_scene_frames // self.sequence_length
            fixed_idxs = np.random.choice(max_num_sequences, num_sequences, replace=False)
            fixed_idxs *= self.sequence_length
            return fixed_idxs
        
        elif self.sequence_locations == 'evenly_spaced':
            # select sequences evenly spaced accross all frames of the scene (without overlap)
            if num_sequences == 1:
                evenly_spaced_idxs = np.array([(num_scene_frames-self.sequence_length)//2])  # select middle sequence
            else:
                evenly_spaced_idxs = np.linspace(0, num_scene_frames-self.sequence_length, num=num_sequences).astype(int)
            np.random.shuffle(evenly_spaced_idxs)
            return evenly_spaced_idxs
        
        else:
            raise NotImplementedError(f"sequence_locations: {self.sequence_locations}")

    def get_indices(self, item_idx):
        """
        Finds the scene_idx and sequence_idx of a single 
        item_idx (i of __getitem__) based on a list that contains 
        the number of sequences for every scene (self.num_sequences_list)
        """
        cum_num_sequences_list = np.cumsum(self.num_sequences_list)
        scene_idx = find_first_higher_index(cum_num_sequences_list, item_idx)
        if scene_idx-1 < 0:
            prev = 0
        else:
            prev = cum_num_sequences_list[scene_idx-1]
        sequence_idx = item_idx - prev
        return scene_idx, sequence_idx

    def get_frame_ids(self, scene_idx, sequence_idx):
        """ Get the ids of the frames to load"""
                
        # get start and end frame of the given sequence
        low = self.start_idxs_list[scene_idx][sequence_idx]
        high = low + self.sequence_length

        if self.frame_locations=='random':
            # select num_frames random frames from the sequence (without replacement)
            sequence = torch.arange(low, high, dtype=float)
            selected_idxs = torch.multinomial(sequence, self.num_frames)
            return np.array(sequence[selected_idxs].int())
        elif self.frame_locations=='evenly_spaced':
            selected_idxs = np.linspace(low, high, num=self.num_frames).astype(int)
            np.random.shuffle(selected_idxs)
            return selected_idxs
        else:
            raise NotImplementedError(f"frame_locations: {self.frame_locations}")


def collate_fn(data_list):
    """ Flatten a set of items from ScenesDataset into a batch.

    Pytorch dataloader has memory issues with nested and complex 
    data structures. This flattens the data into a dict of batched tensors.
    Frames are batched temporally as well (bxtxcxhxw)
    """

    keys = list(data_list[0].keys())
    if len(data_list[0]['frames'])>0:
        frame_keys = list(data_list[0]['frames'][0].keys()) 
    else:
        frame_keys = []
    keys.remove('frames')

    out = {key:[] for key in keys+frame_keys}
    for data in data_list:
        for key in keys:
            out[key].append(data[key])

        for key in frame_keys:
            if torch.is_tensor(data['frames'][0][key]):
                out[key].append( torch.stack([frame[key] 
                                              for frame in data['frames']]) )
            else:
                # non tensor metadata may not exist for every frame
                # (ex: instance_file_name)
                out[key].append( [frame[key] if key in frame else None 
                                  for frame in data['frames']] )

    for key in out.keys():
        if torch.is_tensor(out[key][0]):
            out[key] = torch.stack(out[key])

    return out


def parse_splits_list(splits, data_dir=None):
    """ Returns a list of info_file paths
    Args:
        splits (list of strings): each item is a path to a .json file 
            or a path to a .txt file containing a list of paths to .json's.
    """
    if isinstance(splits, str):
        splits = splits.split()
    info_files = []
    for split in splits:
        split_path = os.path.join(data_dir, split.lstrip('/')) if data_dir else split
        ext = os.path.splitext(split)[1]
        if ext=='.json':
            info_files.append(split_path)
        elif ext=='.txt':
            info_files += [info_file.rstrip() for info_file in open(split_path, 'r')]
        else:
            raise NotImplementedError('%s not a valid info_file type'%split)
    return info_files



class FrameDataset(torch.utils.data.Dataset):
    """ Pytorch Dataset for a single frame in a scene
    
    getitem loads the same frame of a scene
    along with the corresponding TSDF for the scene
    """

    def __init__(self, info_files, frame_idx, length, scene_idx=0, transform=None,
                 frame_types=[], voxel_types=[], voxel_sizes=[]):
        """
        Args:
            info_files: list of info_json files
            fram_idx: index of frame in the scene to load
            length: length of dataset
            scene_idx: the index of the scene in info-files to load
            transform: apply preprocessing transform to images and TSDF
            frame_types: which images to load (ex: depth, semseg, etc)
            frame_selection: how to choose the frames in the sequence
            voxel_types: list of voxel attributes to load with the TSDF
            voxel_sizes: list of voxel sizes to load
        """

        self.info_files = info_files
        self.frame_idx = frame_idx
        self.length = length
        self.scene_idx = scene_idx
        self.transform = transform
        self.frame_types = frame_types
        self.voxel_types = voxel_types
        self.voxel_sizes = voxel_sizes
        self.info = load_info_json(self.info_files[scene_idx])

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        """ Load the same frame (image and TSDF) the i-th time"""
        frames = [map_frame(self.info['frames'][self.frame_idx], self.frame_types)]

        data = {'dataset': self.info['dataset'],
                'scene': self.info['scene'],
                #'instances': info['instances'],
                'frames': frames}

        # load tsdf volumes
        data = map_tsdf(self.info, data, self.voxel_types, self.voxel_sizes)

        # apply transforms
        #print("XXXXXXXXXX before tsdf_vol", data['vol_04'].tsdf_vol.shape) # [265, 280, 132] (when created) -> min:-1, max:1, mean:0.77
        if self.transform is not None:
            data = self.transform(data)
        #print("XXXXXXXXXX after tsdf_vol", data['vol_04_tsdf'].shape) # [1, 200, 2510 64] = voxel_dim_val

        return data


class OneSceneDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for multiple frames in a single scene. getitem loads individual frames"""

    def __init__(self, info_file, transform=None, frame_types=[],
                 voxel_types=[], voxel_sizes=[], frames=[], prepare=False):
        """
        Args:
            info_file: path to json file (format described in datasets/README)
            transform: transform object to preprocess data
            frame_types: which images to load (ex: depth, semseg, etc)
            voxel_types: list of voxel attributes to load with the TSDF
            voxel_sizes: list of voxel sizes to load
            num_frames: number of evenly spaced frames to use (-1 for all)
        """

        self.info = load_info_json(info_file)
        self.transform = transform
        self.frame_types = frame_types
        self.voxel_types = voxel_types
        self.voxel_sizes = voxel_sizes
        self.prepare = prepare

        self.info['frames'] = [self.info['frames'][i] for i in frames]


    def __len__(self):
        return len(self.info['frames'])

    def __getitem__(self, i):
        """
        Returns:
            dict of meta data and images for a single frame
        """

        frame = map_frame(self.info['frames'][i], self.frame_types, self.prepare)

        # put data in common format so we can apply transforms
        data = {'dataset': self.info['dataset'],
                #'instances': self.info['instances'],
                'frames': [frame]}
        data = map_tsdf(self.info, data, self.voxel_types, self.voxel_sizes)

        if self.transform is not None:
            data = self.transform(data)
        # remove data from common format and return the single frame
        #data = data['frames'][0]

        return data

    def get_tsdf(self):
        """
        Returns:
            dict with TSDFs
        """

        # put data in common format so we can apply transforms
        data = {'dataset': self.info['dataset'],
                #'instances': self.info['instances'],
                'frames': [],
               }

        # load tsdf volumes
        data = map_tsdf(self.info, data, self.voxel_types, self.voxel_sizes)

        # apply transforms
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_mesh(self):
        # TODO: also get vertex instances/semantics
        return trimesh.load(self.info['file_name_mesh_gt'], process=False)
