# Adapted from: https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py

import os
import numpy as np
import zlib
import imageio
import cv2
import png
import struct
import tarfile

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}


class RGBDFrame:

    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise ValueError("Invalid compression type")

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        else:
            raise ValueError("Invalid compression type")

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:

    def __init__(self, filename, archive_result):
        self.version = 4
        self.load(filename)
        self.archive_result = archive_result

    def load(self, filename):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = f.read(strlen).decode('utf-8')
            self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1, skip_existing=True):
        output_dir = os.path.abspath(output_path)
        if os.path.exists(output_dir):
            if skip_existing:
                print("Skip depth for scene:", os.path.basename(os.path.dirname(output_dir)))
                return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.archive_result:
            archive_name = os.path.basename(output_dir)
            print('Exporting', len(self.frames)//frame_skip, 'depth frames to', output_dir, "(as a .tar)")
            archive_filename = os.path.join(output_dir, archive_name + '.tar')
            with tarfile.open(archive_filename, 'w') as tar:
                for f in range(0, len(self.frames), frame_skip):
                    depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
                    depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
                    if image_size is not None:
                        depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
                    depth_filename = os.path.join(output_dir, str(f) + '.png')
                    with open(depth_filename, 'wb') as png_file:
                        writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
                        depth_list = depth.reshape(-1, depth.shape[1]).tolist()
                        writer.write(png_file, depth_list)
                    tar.add(depth_filename, arcname=os.path.basename(depth_filename))
                    os.remove(depth_filename)
        else:    
            print('exporting', len(self.frames)//frame_skip, ' depth frames to', output_dir)
            for f in range(0, len(self.frames), frame_skip):
                depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
                depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
                if image_size is not None:
                    depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
                depth_filename = os.path.join(output_dir, str(f) + '.png')
                with open(depth_filename, 'wb') as png_file:
                    writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
                    depth_list = depth.reshape(-1, depth.shape[1]).tolist()
                    writer.write(png_file, depth_list)

    def export_color_images(self, output_path, image_size=None, frame_skip=1, skip_existing=True):
        output_dir = os.path.abspath(output_path)
        if os.path.exists(output_dir):
            if skip_existing:
                print("Skip color for scene:", os.path.basename(os.path.dirname(output_dir)))
                return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if self.archive_result:
            archive_name = os.path.basename(output_dir)
            print('exporting', len(self.frames)//frame_skip, 'color frames to', output_dir, "(as a .tar)")
            archive_filename = os.path.join(output_dir, archive_name + '.tar')
            with tarfile.open(archive_filename, 'w') as tar:
                for f in range(0, len(self.frames), frame_skip):
                    color = self.frames[f].decompress_color(self.color_compression_type)
                    if image_size is not None:
                        color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
                    image_filename = os.path.join(output_dir, str(f) + '.jpg')
                    imageio.imwrite(image_filename, color)
                    tar.add(image_filename, arcname=os.path.basename(image_filename))
                    os.remove(image_filename)
        else:
            print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
            for f in range(0, len(self.frames), frame_skip):
                color = self.frames[f].decompress_color(self.color_compression_type)
                if image_size is not None:
                    color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
                imageio.imwrite(os.path.join(output_dir, str(f) + '.jpg'), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')

    def export_poses(self, output_path, frame_skip=1, skip_existing=True):
        output_dir = os.path.abspath(output_path)
        if os.path.exists(output_dir):
            if skip_existing:
                print("Skip poses for scene:", os.path.basename(os.path.dirname(output_dir)))
                return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if self.archive_result:
            archive_name = os.path.basename(output_dir)
            print('exporting', len(self.frames)//frame_skip, 'camera poses to', output_dir, "(as a .tar)")
            tar_filename = os.path.join(output_dir, archive_name + '.tar')
            with tarfile.open(tar_filename, 'w') as tar:
                for f in range(0, len(self.frames), frame_skip):
                    file_to_save = os.path.join(output_dir, str(f) + '.txt')
                    self.save_mat_to_file(self.frames[f].camera_to_world, file_to_save)
                    tar.add(file_to_save, arcname=os.path.basename(file_to_save))
                    os.remove(file_to_save)
        else:
            print('exporting', len(self.frames)//frame_skip, 'camera poses to', output_dir)
            for f in range(0, len(self.frames), frame_skip):
                file_to_save = os.path.join(output_dir, str(f) + '.txt')
                self.save_mat_to_file(self.frames[f].camera_to_world, file_to_save)

    def export_intrinsics(self, output_path, skip_existing=True):
        output_dir = os.path.abspath(output_path)
        if os.path.exists(output_dir):
            if skip_existing:
                print("Skip intrinsics for scene:", os.path.basename(os.path.dirname(output_dir)))
                return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('exporting camera intrinsics to', output_dir)
        self.save_mat_to_file(self.intrinsic_color, os.path.join(output_dir, 'intrinsic_color.txt'))
        self.save_mat_to_file(self.extrinsic_color, os.path.join(output_dir, 'extrinsic_color.txt'))
        self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_dir, 'intrinsic_depth.txt'))
        self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_dir, 'extrinsic_depth.txt'))