import os
import time
import numpy as np
import argparse
from reader import process_sens_file
import tqdm
from os.path import join
from os import listdir
import numpy as np
import multiprocessing

# paths
# path_in is the root of the raw scannet dataset (i.e. path_in/scans/scene0000_00)
# path_out is the root of the processed scannet dataset (i.e. path_out/scene/color/1.jpg)
path_in = '/home/atuin/g101ea/g101ea13/data/scannet_raw'
path_out = '/home/atuin/g101ea/g101ea13/data/scannet'

# result
export_color_images = True
export_depth_images = True
export_poses = True
export_intrinsics = True

# options
test_only = False
archive_result = True
specific_scenes = None  # i.e. ['scans/scene0000_00', 'scans_test/scene0000_01']
max_scenes = 10

if __name__ == '__main__':

    path_out = join(path_out, 'scans')
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        print("created directory:", path_out)

    if specific_scenes:
        scenes = specific_scenes
    else:
        scenes = []
        if not test_only:
            scenes += sorted([os.path.join('scans', scene) 
                            for scene in os.listdir(os.path.join(path_in, 'scans'))])
        scenes += sorted([os.path.join('scans_test', scene)
                        for scene in os.listdir(os.path.join(path_in, 'scans_test'))])

    if max_scenes != None:
        print(f"Processing only the first {max_scenes} scenes")
        scenes = scenes[:max_scenes]

    pbar = tqdm.tqdm()
    pool = multiprocessing.Pool(processes=8)
    for scene in scenes:
        scene_id = scene.split('/')[1]
        filename = os.path.join(path_in, scene, scene_id + ".sens")
        path_out_i = os.path.join(path_out, scene_id)
        pool.apply_async(
            process_sens_file,
            args=(filename, path_out_i, export_depth_images, export_color_images,
                  export_poses, export_intrinsics, archive_result),
            callback=lambda _: pbar.update()
        )
        pbar.update()
    pool.close()
    pool.join()
