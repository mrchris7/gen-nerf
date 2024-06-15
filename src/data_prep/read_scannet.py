from reader import process_sens_file

if __name__ == '__main__':
    
    filename = '../../ScanNetData/data/scans/scene0000_00/scene0000_00.sens'
    output_path = '../../MaskCLIP/data/ScanNetv2/imgs/'
    export_depth_images = True
    export_color_images = True
    export_poses = True
    export_intrinsics = True

    process_sens_file(
        filename,
        output_path,
        export_depth_images,
        export_color_images,
        export_poses,
        export_intrinsics
    )