from matplotlib import pyplot as plt
import pyvista as pv
import numpy as np
import torch
from src.data.tsdf import TSDF

print(pv.__version__)


def plot_tsdf_slices(file_list, use_tsdf_obj=False):
    
    tsdf_volumes = []
    for file in file_list:
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
        
        tsdf_volumes.append(tsdf_np)
    
    slice_indices = [5, 20, 35, 45]  # Z-axis slices to visualize

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=len(tsdf_volumes), ncols=len(slice_indices), figsize=(12, 4 * len(tsdf_volumes)))
    #fig.suptitle("TSDF Slices for Each Volume", fontsize=16)

    # Plot each slice
    for i, tsdf_volume in enumerate(tsdf_volumes):
        for j, z in enumerate(slice_indices):
            tsdf_slice = tsdf_volume[:, :, z]
            
            # Visualize the slice in the appropriate subplot
            ax = axes[i, j] if len(tsdf_volumes) > 1 else axes[j]  # Handle single row case
            cax = ax.imshow(tsdf_slice, cmap='coolwarm', origin='lower') # PiYG, RdBu
            #ax.set_title(f"Volume {i+1}, Slice Z={z}")

            # Add colorbar to the right of each row
            #fig.colorbar(cax, ax=ax, orientation="horizontal", fraction=0.02, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Adjust layout to make room for the main title
    plt.show()


data_folder = '/home/master/Main/iRobMan-Lab2/workspace/gen-nerf/data' 

file_list = []
file_list.append(f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_pointnet_500ep_seed0/test_tsdf/test_trgt_tsdf.npz')               # gt
file_list.append(f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_pointnet_500ep_seed0/test_tsdf/test_pred_tsdf.npz')               # method
file_list.append(f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_spatialnoblur_500ep_seed0/test_tsdf/test_pred_tsdf.npz')            # f_vol
file_list.append(f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_pointnetspatialnoblur_500ep_seed1/test_tsdf/test_pred_tsdf.npz')    # f_feat & f_vol
file_list.append(f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_nologtrans_500ep_seed0/test_tsdf/test_pred_tsdf.npz')               # without h(x)
file_list.append(f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_eikonal_500ep_seed0/test_tsdf/test_pred_tsdf.npz')                  # with L_eik
file_list.append(f'{data_folder}/backups_eval/X_seq1_0244_01_frames8_evenspaced_noposenc_500ep_seed0/test_tsdf/test_pred_tsdf.npz')                 # without \gamma(x)

plot_tsdf_slices(file_list, True)
