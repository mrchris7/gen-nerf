import os
import shutil
import tarfile
import tempfile
import zipfile
import tqdm
import multiprocessing


"""
Builds the scannet dataset in the TMPDIR file system (node-local!)

$TMPDIR
└───scannet
|   └───scannetv2-labels.combined.tsv
|   └───scannetv2_test.txt
|   └───scannetv2_train.txt
|   └───scannetv2_val.txt
│   └───scans
│   |   └───scene0000_00
│   |   |   └───scene0000_00.aggregation.json
│   |   |   └───scene0000_00.txt
│   |   |   └───scene0000_00_vh_clean_2.0.010000.segs.json
│   |   |   └───scene0000_00_vh_clean_2.ply
│   |   |   └───color
│   |   |   │   └───0.jpg
│   |   |   │   └───...
│   |   |   └───depth
│   |   |   │   └───0.txt
│   |   |   │   └───...
│   |   |   └───pose
│   |   |   │   └───0.txt
│   |   |   │   └───...
│   |   |   └───instance-filt
│   |   |       │   0.png
│   |   |       └───...
│   |   └───...
│   └───scans_test
│       └───scene700_00
│       └───...
"""

# paths
# path_target is the root of the scannet dataset in the TMPDIR file system
# path_raw is the root of the raw scannet dataset (i.e. path_in/scans/scene0000_00)
# path_tar is the root of the processed scannet dataset (i.e. path_out/scene/color/color.tar)
path_target = os.path.expandvars('$TMPDIR/data/scannet')
path_raw = '/home/atuin/g101ea/g101ea13/data/scannet_raw'
path_tar = '/home/atuin/g101ea/g101ea13/data/scannet'

# options
test_only = False
specific_scenes = None  # i.e. ['scans/scene0000_00', 'scans_test/scene0000_01']
max_scenes = 10



def build_scene(scene):
    print("Build scene:", scene)
    _, scene_id = scene.split('/')  
    # _: scans or scans_test
    # scene_id: scene0000_00

    path_scene_target = os.path.join(path_target, scene)
    os.makedirs(path_scene_target)

    # copy raw files from path_raw to path_target
    path_scene_raw = os.path.join(path_raw, scene)
    shutil.copy(os.path.join(path_scene_raw, scene_id+'_vh_clean_2.0.010000.segs.json'), path_scene_target)
    shutil.copy(os.path.join(path_scene_raw, scene_id+'.aggregation.json'), path_scene_target)
    shutil.copy(os.path.join(path_scene_raw, scene_id+'_vh_clean_2.ply'), path_scene_target)
    shutil.copy(os.path.join(path_scene_raw, scene_id+'.txt'), path_scene_target)
    #...
    
    # extract zip files from path_raw to path_target (and remove outer directory-layer)
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

    # extract tars from path_tar to path_target
    path_scene_tar = os.path.join(path_tar, "scans", scene_id)  # test scenes are also inside scans
    for type in ['color', 'depth', 'pose']:
        tar_file = os.path.join(path_scene_tar, type, f'{type}.tar')

        dir = os.path.join(path_scene_target, type)
        os.makedirs(dir)
        with tarfile.open(tar_file, 'r') as tar:
            tar.extractall(path=dir, filter='data')


if __name__ == "__main__":

    os.makedirs(path_target)    

    # copy scannetv2-labels.combined.tsv
    shutil.copy(os.path.join(path_raw, 'scannetv2-labels.combined.tsv'), path_target)

    # copy splits
    shutil.copy(os.path.join(path_tar, 'scannetv2_train.txt'), path_target)
    shutil.copy(os.path.join(path_tar, 'scannetv2_test.txt'), path_target)
    shutil.copy(os.path.join(path_tar, 'scannetv2_val.txt'), path_target)

    # make subdirectories
    os.makedirs(os.path.join(path_target, 'scans'))
    os.makedirs(os.path.join(path_target, 'scans_test'))
    #os.makedirs(os.path.join(path_target, "tasks"))


    # collect scenes
    scenes = []
    if specific_scenes:
            scenes = specific_scenes
    else:
        if not test_only:
            scenes += sorted([os.path.join('scans', scene) 
                            for scene in os.listdir(os.path.join(path_raw, 'scans'))])
        scenes += sorted([os.path.join('scans_test', scene)
                        for scene in os.listdir(os.path.join(path_raw, 'scans_test'))])

    if max_scenes != None:
        print(f"Building only the first {max_scenes} scenes")
        scenes = scenes[:max_scenes]
    
    # scenes contain: "scans/scene0000_00" or "scans_test/scene0000_00"
    pbar = tqdm.tqdm()
    pool = multiprocessing.Pool(processes=8)
    for scene in scenes:
        pool.apply_async(build_scene, args=(scene,), callback=lambda _: pbar.update())
        pbar.update()
    pool.close()
    pool.join()
