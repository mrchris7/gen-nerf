import os
import argparse
import tqdm
import multiprocessing
from src.data.prepare.SensorData import SensorData


# default paths
PATH_IN = '$WORK/data/scannet_raw'  # path to the raw scannet dataset
PATH_OUT = '$WORK/data/scannet'  # path where to build the dataset

"""
Exports the sensor data of the scannet dataset to PATH_OUT using the raw scannet data from PATH_IN
Used to export color images, depth images, poses and intrinsics of the .sens files inside PATH_IN
and optionally saves it as an archive instead of individual files

PATH_OUT
└───scans
|   └───scene0000_00
|   |   └───color
|   |   │   └───0.jpg |
|   |   │   └───1.jpg |--> optionally: color.tar
|   |   |   └───...   |
|   |   └───depth
|   |   │   └───0.txt
|   |   │   └───...
|   |   └───poses
|   |   │   └───0.txt
|   |   │   └───...
|   |   └───intrinsics
|   |       └───extrinsic_color.txt     
|   |       └───extrinsic_depth.txt 
|   |       └───intrinsic_color.txt
|   |       └───intrinsic_depth.txt
|   └───...
└───scans_test
    └───scene700_00
    └───...
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default=PATH_IN, help="Path to input folder")
    parser.add_argument('--path_out', default=PATH_OUT, help="Path to output folder")
    parser.add_argument('--export_depth', action='store_true', help="Whether to export depth images")
    parser.add_argument('--export_color', action='store_true', help="Whether to export color images")
    parser.add_argument('--export_poses', action='store_true', help="Whether to export poses")
    parser.add_argument('--export_intrinsics', action='store_true', help="Whether to export camera intrinsics")
    parser.add_argument('--export_all', action='store_true', help="Whether to export all data")
    parser.add_argument('--archive_result', action='store_true', help="Whether to pack the files of all frames into an archive")
    parser.add_argument('--test_only', action='store_true', help="Only export the test set (if you dont plan to train)")
    parser.add_argument('--scenes', nargs='+', default=None, help="List of directories of specific scenes to read i.e. scans/scene0000_00, scans_test/scene0000_01, ...")
    parser.add_argument('--scenes_file', default=None, help="Text file that contains a list of directories of specific scenes to read i.e. scans/scene0000_00, scans_test/scene0000_01, ...")
    parser.add_argument('--num_scenes', default=-1, type=int, help="Number of scenes to read")
    return parser.parse_args()


def process_sens_file(filename, output_path, export_depth, export_color, export_poses, export_intrinsics, archive_result):
    print(f"Reading scene: {filename}")
    sd = SensorData(filename, archive_result)
    
    if export_depth:
        sd.export_depth_images(os.path.join(output_path, 'depth'))
    if export_color:
        sd.export_color_images(os.path.join(output_path, 'color'))
    if export_poses:
        sd.export_poses(os.path.join(output_path, 'poses'))
    if export_intrinsics:
        sd.export_intrinsics(os.path.join(output_path, 'intrinsics'))


def main():
    args = parse_arguments()

    if args.export_all:
        args.export_depth = True
        args.export_color = True
        args.export_poses = True
        args.export_intrinsics = True

    if not (args.export_depth or args.export_color or
            args.export_poses or args.export_intrinsics):
        print("Aborted: nothing to export.")
        return

    # support env-variables inside paths
    path_in = os.path.expandvars(args.path_in)
    path_out = os.path.expandvars(args.path_out)

    os.makedirs(path_out, exist_ok=True) 
    
    # make subdirectories
    os.makedirs(os.path.join(path_out, 'scans'), exist_ok=True)
    os.makedirs(os.path.join(path_out, 'scans_test'), exist_ok=True)


    # collect scenes
    scenes = []
    if args.scenes:
        print(f"Reading specific scenes: {args.scenes}")
        scenes += args.scenes

    if args.scenes_file:
        scenes_file = os.path.expandvars(args.scenes_file)
        print(f"Reading scenes from file: {args.scenes_file}")
        ext = os.path.splitext(scenes_file)[1]
        if ext=='.txt':
            scenes += [scene.rstrip() for scene in open(scenes_file, 'r')]
        else:
            raise NotImplementedError(f"{ext} not a valid scenes_file type")

    if args.num_scenes > -1:
        print(f"Reading the first {args.num_scenes} scenes")
        all_scenes = []
        all_scenes += sorted([os.path.join('scans', scene) 
                                for scene in os.listdir(os.path.join(path_in, 'scans'))])
        all_scenes += sorted([os.path.join('scans_test', scene)
                        for scene in os.listdir(os.path.join(path_in, 'scans_test'))])
        scenes += all_scenes[:args.num_scenes]
    
    if not args.scenes and not args.scenes_file and not args.num_scenes:
        if not args.test_only:
            print(f"Reading all scenes")
            scenes += sorted([os.path.join('scans', scene) 
                                for scene in os.listdir(os.path.join(path_in, 'scans'))])
        else:
            print(f"Reading only the test scenes")
            
        scenes += sorted([os.path.join('scans_test', scene)
                        for scene in os.listdir(os.path.join(path_in, 'scans_test'))])
    else:
        if args.test_only:
            print("Flag \"--test_only\" has no effect")
    
    # remove duplicates and sort
    scenes = sorted(list(dict.fromkeys(scenes)))
    print("Scenes to read:", len(scenes))

    for scene in scenes:
        print(scene)

    # scenes contain: "scans/scene0000_00" or "scans_test/scene0000_00"
    pbar = tqdm.tqdm()
    pool = multiprocessing.Pool(processes=8)
    for scene in scenes:
        scene_id = scene.split('/')[1]
        filename = os.path.join(path_in, scene, scene_id + '.sens')
        path_out_scene = os.path.join(path_out, scene)
        pool.apply_async(
            process_sens_file,
            args=(filename, path_out_scene, args.export_depth, args.export_color,
                  args.export_poses, args.export_intrinsics, args.archive_result),
            callback=lambda _: pbar.update()
        )
        pbar.update()
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
    