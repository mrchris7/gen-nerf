import os
import argparse
import shutil
import tarfile
import tempfile
import zipfile
import tqdm
import multiprocessing


# default paths
PATH_TARGET = '$TMPDIR/data/scannet'  # path where to build the dataset (TMPDIR file system)
PATH_RAW = '$WORK/data/scannet_raw'  # path to the raw scannet dataset
PATH_ARCHIVE = '$WORK/data/scannet'  # path to the scannet data that was exported and archived from the .sens file

"""
Builds the scannet dataset inside PATH_TARGET using the data from PATH_RAW and PATH_ARCHIVE
Used to unpack files of all frames inside a node-local file system (i.e. TMPDIR)

PATH_TARGET
└───scannetv2-labels.combined.tsv
└───scannetv2_test.txt
└───scannetv2_train.txt
└───scannetv2_val.txt
└───scans
|   └───scene0000_00
|   |   └───scene0000_00.aggregation.json
|   |   └───scene0000_00.txt
|   |   └───scene0000_00_vh_clean_2.0.010000.segs.json
|   |   └───scene0000_00_vh_clean_2.ply
|   |   └───intrinsics
|   |   │   └───extrinsic_color.txt     
|   |   │   └───extrinsic_depth.txt
|   |   │   └───intrinsic_color.txt
|   |   │   └───intrinsic_depth.txt
|   |   └───color
|   |   │   └───0.jpg |--> default: color.tar
|   |   │   └───...   |
|   |   └───depth
|   |   │   └───0.txt |--> default: depth.tar
|   |   │   └───...   |
|   |   └───poses
|   |   │   └───0.txt |--> default: poses.tar
|   |   │   └───...   |
|   |   └───instance-filt
|   |       └───0.png
|   |       └───...
|   └───...
└───scans_test
    └───scene700_00
    └───...
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build the scannet dataset")
    parser.add_argument('--path_target', default=PATH_TARGET, help="Path where to build the dataset")
    parser.add_argument('--path_raw', default=PATH_RAW, help="Path to the raw scannet dataset")
    parser.add_argument('--path_archive', default=PATH_ARCHIVE, help="Path to the scannet data that was exported and archived from the .sens file")
    parser.add_argument('--test_only', action='store_true', help="Only build the test set (if you dont plan to train)")
    parser.add_argument('--scenes', nargs='+', default=None, help="List of directories of specific scenes to build i.e. scans/scene0000_00, scans_test/scene0000_01, ...")
    parser.add_argument('--num_scenes', default=-1, type=int, help="Number of scenes to build")
    parser.add_argument('--extract_archives', action='store_true', help="Extract the .tar files for color, depth and poses")
    return parser.parse_args()


def build_scene(scene, path_target, path_raw, path_archive, extract_archives):
    print("Build scene:", scene)
    scene_f, scene_id = scene.split('/')  
    # _: scans or scans_test
    # scene_id: scene0000_00

    path_scene_target = os.path.join(path_target, scene)
    os.makedirs(path_scene_target)

    # copy raw files from path_raw to path_target
    path_scene_raw = os.path.join(path_raw, scene)
    shutil.copy(os.path.join(path_scene_raw, scene_id+'_vh_clean_2.ply'), path_scene_target)
    shutil.copy(os.path.join(path_scene_raw, scene_id+'.txt'), path_scene_target)
    if scene_f == 'scans': 
        shutil.copy(os.path.join(path_scene_raw, scene_id+'_vh_clean_2.0.010000.segs.json'), path_scene_target)
        shutil.copy(os.path.join(path_scene_raw, scene_id+'.aggregation.json'), path_scene_target)
        #...
    
    # extract zip files from path_raw to path_target (and remove outer directory-layer)
    if scene_f == 'scans':
        for type in [f'{scene_id}_2d-instance-filt']:
            zip_file = os.path.join(path_scene_raw, f'{type}.zip')
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_file, 'r') as zip:
                    zip.extractall(temp_dir)
                # remove outer layer
                extracted_dir = os.path.join(temp_dir)
                child_name = os.listdir(extracted_dir)[0]
                extracted_dir_child = os.path.join(temp_dir, child_name)
                shutil.move(extracted_dir_child, path_scene_target)
                shutil.rmtree(extracted_dir)
    
    path_scene_tar = os.path.join(path_archive, scene_f, scene_id)
    if os.path.exists(path_scene_tar):
        # copy raw files from path_archive to path_target
        for folder in ['intrinsics']:
            dir = os.path.join(path_scene_target, folder)
            os.makedirs(dir)
            for file in ['extrinsic_color.txt', 'extrinsic_depth.txt', 'intrinsic_color.txt', 'intrinsic_depth.txt']:
                shutil.copy(os.path.join(path_scene_tar, folder, file), dir)
        
        # extract tars from path_archive to path_target
        for folder in ['color', 'depth', 'poses']:
            tar_file = os.path.join(path_scene_tar, folder, f'{folder}.tar')

            dir = os.path.join(path_scene_target, folder)
            os.makedirs(dir)
            if extract_archives:
                with tarfile.open(tar_file, 'r') as tar:
                    tar.extractall(path=dir, filter='data')
            else:
                shutil.copy(tar_file, dir)
    else:
        print(f"Could not build frames of scene {scene} (the .sens file has not yet been read and extracted)")
        return


def main():
    args = parse_arguments()

    # support env-variables inside paths
    path_target = os.path.expandvars(args.path_target)
    path_raw = os.path.expandvars(args.path_raw)
    path_archive = os.path.expandvars(args.path_archive)

    os.makedirs(path_target, exist_ok=True) 

    # copy scannetv2-labels.combined.tsv
    shutil.copy(os.path.join(path_raw, 'scannetv2-labels.combined.tsv'), path_target)

    # copy splits
    shutil.copy(os.path.join(path_raw, 'scannetv2_train.txt'), path_target)
    shutil.copy(os.path.join(path_raw, 'scannetv2_test.txt'), path_target)
    shutil.copy(os.path.join(path_raw, 'scannetv2_val.txt'), path_target)

    # make subdirectories
    os.makedirs(os.path.join(path_target, 'scans'))
    os.makedirs(os.path.join(path_target, 'scans_test'))
    #os.makedirs(os.path.join(path_target, "tasks"))

    # collect scenes
    scenes = []
    if args.scenes:
        print(f"Building only specific scenes: {args.scenes}")
        scenes = args.scenes
    else:
        if not args.test_only:
            scenes += sorted([os.path.join('scans', scene) 
                            for scene in os.listdir(os.path.join(path_raw, 'scans'))])
        scenes += sorted([os.path.join('scans_test', scene)
                        for scene in os.listdir(os.path.join(path_raw, 'scans_test'))])

    if args.num_scenes > -1:
        print(f"Building only the first {args.num_scenes} scenes")
        scenes = scenes[:args.num_scenes]
    
    # scenes contain: "scans/scene0000_00" or "scans_test/scene0000_00"
    pbar = tqdm.tqdm()
    pool = multiprocessing.Pool(processes=8)
    for scene in scenes:
        pool.apply_async(build_scene, args=(scene, path_target, path_raw, path_archive, args.extract_archives), callback=lambda _: pbar.update())
        pbar.update()
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
