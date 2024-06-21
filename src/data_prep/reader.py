# Adapted from: https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py

import argparse
import os
import sys
from SensorData import SensorData

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True, help='path to sens file to read')
    parser.add_argument('--output_path', required=True, help='path to output folder')
    parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
    parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
    parser.add_argument('--export_poses', dest='export_poses', action='store_true')
    parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
    parser.add_argument('--archive_result', dest='archive_result', action='store_true')

    parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False, archive_result=False)
    return parser.parse_args()

def process_sens_file(filename, output_path, export_depth_images, export_color_images, export_poses, export_intrinsics, archive_result):
    print(f'loading {filename}...')
    sd = SensorData(filename, archive_result)
    print("loaded!")
    if export_depth_images:
        sd.export_depth_images(os.path.join(output_path, 'depth'))
    if export_color_images:
        sd.export_color_images(os.path.join(output_path, 'color'))
    if export_poses:
        sd.export_poses(os.path.join(output_path, 'pose'))
    if export_intrinsics:
        sd.export_intrinsics(os.path.join(output_path, 'intrinsic'))

def main():
    opt = parse_arguments()
    print(opt)
    process_sens_file(
        opt.filename,
        opt.output_path,
        opt.export_depth_images,
        opt.export_color_images,
        opt.export_poses,
        opt.export_intrinsics
    )

if __name__ == '__main__':
    main()