from matplotlib import pyplot as plt
import pyvista as pv
import numpy as np
import torch
from src.data.tsdf import TSDF

print(pv.__version__)


def plot_tsdf_slice(file, use_tsdf_obj=False):
    
    if use_tsdf_obj:
        tsdf = TSDF.load(file)
        tsdf_tensor = tsdf.tsdf_vol
        tsdf_np = tsdf_tensor.squeeze().numpy().astype(np.float32)
        print("origin:", tsdf.origin)
        print("tsdf_vol:", tsdf_np.shape)

    else:
        tsdf_tensor = torch.load(file, map_location=torch.device('cpu'))
        tsdf_np = tsdf_tensor.squeeze().numpy().astype(np.float32)
        print("tsdf_vol", tsdf_np.shape)
    
    Z=25
    tsdf_slice = tsdf_np[:, :, Z]

    plt.imshow(tsdf_slice, cmap='coolwarm', origin='lower')  
    plt.colorbar(label='Truncated Signed Distance [m]', orientation="horizontal")
    #plt.title(f"TSDF Slice at z={Z}")
    #plt.xlabel("X axis")
    #plt.ylabel("Y axis")
    plt.show()

data_folder = '/home/master/Main/iRobMan-Lab2/workspace/gen-nerf/data' 
base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_pointnet_500ep_seed0'                  # method
#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_eikonal_500ep_seed0'                  # with L_eik
#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_nologtrans_500ep_seed0'               # without h(x)
#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_noposenc_500ep_seed0'                 # without \gamma(x)
#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_pointnetspatialnoblur_500ep_seed1'    # f_feat & f_vol
#base_folder = f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_spatialnoblur_500ep_seed0'            # f_vol


tsdf_trgt_file = f'{base_folder}/test_tsdf/test_trgt_tsdf.npz'
tsdf_pred_file = f'{base_folder}/test_tsdf/test_pred_tsdf.npz'
plot_tsdf_slice(tsdf_trgt_file, True)
plot_tsdf_slice(tsdf_pred_file, True)
