import argparse
import multiprocessing
import os
import shutil
import tarfile
import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Copy the scannet dataset")
    parser.add_argument('--path_src', help="Path to the source directory containing the dataset")
    parser.add_argument('--path_des', help="Path to the destination directory where the files will be saved")
    parser.add_argument('--test_only', action='store_true', help="Only copy the test set (if you dont plan to train)")
    parser.add_argument('--scenes', nargs='+', default=None, help="List of directories of specific scenes to copy i.e. scans/scene0000_00, scans_test/scene0000_01, ...")
    parser.add_argument('--scenes_file', default=None, help="A file that contains a list of specific scenes to copy i.e. scans/scene0000_00, test_scans/scene0000_01, ...")
    parser.add_argument('--num_scenes', default=-1, type=int, help="Number of scenes to copy")
    parser.add_argument('--extract_archives', action='store_true', help="Extract the .tar files")
    return parser.parse_args()

def replace_in_file(file_path, search_text, replace_text):
    # read and modify the file content
    with open(file_path, "r") as file:
        data = file.read()
    data = data.replace(search_text, replace_text)

    # write back the modified content
    with open(file_path, "w") as file:
        file.write(data)
    #print(f"Updated file: {file_path}")


def stage_scene(scene, path_src, path_des, extract_archives):

    print("Stage scene:", scene)

    path_scene_src = os.path.join(path_src, scene)
    path_scene_des = os.path.join(path_des, scene)

    for root, dirs, files in os.walk(path_scene_src):
        # Create the corresponding destination directory
        relative_path = os.path.relpath(root, path_scene_src)
        target_path = os.path.join(path_scene_des, relative_path)
        os.makedirs(target_path, exist_ok=True)
        
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_path, file)
            
            if extract_archives and file.endswith(".tar"):
                # Extract .tar file
                with tarfile.open(source_file) as tar:
                    tar.extractall(path=target_path)
                #print(f"Extracted: {source_file} to {target_path}")
            else:
                # Copy other files
                shutil.copy2(source_file, target_file)
                #print(f"Copied: {source_file} to {target_file}")
            
            # Check for info.json and modify its content
            if file == "info.json":
                replace_in_file(target_file, path_scene_src, path_scene_des)



def main():
    args = parse_arguments()

    # support env-variables inside paths
    path_src = os.path.expandvars(args.path_src)
    path_des = os.path.expandvars(args.path_des)
    
    os.makedirs(path_des, exist_ok=True) 

    # copy files
    for item in os.listdir(path_src):
        source_item = os.path.join(path_src, item)
        if os.path.isfile(source_item):
            dest_item = shutil.copy2(source_item, path_des)

            if source_item.endswith(".txt"): # split files
                replace_in_file(dest_item, path_src, path_des)


    # make subdirectories for scenes
    os.makedirs(os.path.join(path_des, 'scans'), exist_ok=True)
    os.makedirs(os.path.join(path_des, 'scans_test'), exist_ok=True)

    # collect scenes
    scenes = []
    if args.scenes:
        print(f"Staging specific scenes: {args.scenes}")
        scenes += args.scenes

    if args.scenes_file:
        scenes_file = os.path.expandvars(args.scenes_file)
        print(f"Staging scenes from file: {args.scenes_file}")
        ext = os.path.splitext(scenes_file)[1]
        if ext=='.txt':
            scenes += [scene.rstrip() for scene in open(scenes_file, 'r')]
        else:
            raise NotImplementedError(f"{ext} not a valid scenes_file type")

    if args.num_scenes > -1:
        print(f"Staging the first {args.num_scenes} scenes")
        all_scenes = []
        all_scenes += sorted([os.path.join('scans', scene) 
                                for scene in os.listdir(os.path.join(path_src, 'scans'))])
        all_scenes += sorted([os.path.join('scans_test', scene)
                        for scene in os.listdir(os.path.join(path_src, 'scans_test'))])
        scenes += all_scenes[:args.num_scenes]
    
    if not args.scenes and not args.scenes_file and args.num_scenes <= 0:
        if not args.test_only:
            print(f"Staging all scenes")
            scenes += sorted([os.path.join('scans', scene) 
                                for scene in os.listdir(os.path.join(path_src, 'scans'))])
        else:
            print(f"Staging only the test scenes")
            
        scenes += sorted([os.path.join('scans_test', scene)
                        for scene in os.listdir(os.path.join(path_src, 'scans_test'))])
    else:
        if args.test_only:
            print("Flag \"--test_only\" has no effect")
    
    # remove duplicates and sort
    scenes = sorted(list(dict.fromkeys(scenes)))
    print("Scenes to stage:", len(scenes))

    for scene in scenes:
        print(scene)

    # scenes contain: "scans/scene0000_00" or "scans_test/scene0000_00"
    pbar = tqdm.tqdm()
    pool = multiprocessing.Pool(processes=16)
    for scene in scenes:
        pool.apply_async(stage_scene, args=(scene, path_src, path_des, args.extract_archives), callback=lambda _: pbar.update())
        pbar.update()
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()