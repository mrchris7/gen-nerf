
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

import argparse
import json
import os

from src.data.data import SceneDataset, parse_splits_list
from src.models.metrics import eval_tsdf
import src.data.transforms as transforms
from src.data.tsdf import TSDF



def process(info_file, save_path):

    # gt depth data loader
    width, height = 640, 480
    transform = transforms.Compose([
        transforms.ResizeImage((width,height)),
        transforms.ToTensor(),
    ])
    dataset = SceneDataset(info_file, transform, frame_types=['depth'], from_archive=True)
    
    scene = dataset.info['scene']

    # get info about tsdf
    file_tsdf_pred = os.path.join(save_path, 'test_tsdf/test_pred_tsdf.npz')
    temp = TSDF.load(file_tsdf_pred)
    voxel_size = int(temp.voxel_size*100)
    

    # eval tsdf
    #file_tsdf_trgt = dataset.info['file_name_vol_%02d'%voxel_size]  # use here own target tsdf
    file_tsdf_trgt = os.path.join(save_path, 'test_tsdf/test_trgt_tsdf.npz')
    metrics_tsdf = eval_tsdf(file_tsdf_pred, file_tsdf_trgt)

    metrics = {**metrics_tsdf}
    print(metrics)

    rslt_file = os.path.join(save_path, '%s_metrics_tsdf.json'%scene)
    json.dump(metrics, open(rslt_file, 'w'))

    return scene, metrics



def main():
    parser = argparse.ArgumentParser(description="GenNerf Testing")
    parser.add_argument("--model", default='/home/atuin/g101ea/g101ea13/debug', metavar="FILE",
                        help="path to debug folder")
    parser.add_argument("--scenes", default="/home/atuin/g101ea/g101ea13/data/scannet/scans/scene0244_01/info.json",
                        help="which scene(s) to run on")
    args = parser.parse_args()

    # get all the info_file.json's from the command line
    # .txt files contain a list of info_file.json'
    print("scenes:", args.scenes)
    info_files = parse_splits_list(args.scenes)
    # info_files=[info_files[0]]

    metrics = {}
    for i, info_file in enumerate(info_files):
        print("i", i, " info_file:", info_file)
        # run model on each scene
        scene, temp = process(info_file, args.model)
        metrics[scene] = temp

    print(metrics)
    '''
    rslt_file = os.path.join(args.model, 'metrics.json')
    json.dump(metrics, open(rslt_file, 'w'))

    # display results
    visualize(rslt_file)
    '''

if __name__ == "__main__":
    main()
