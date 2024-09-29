
import os
import tempfile
import torch


class DebugLogger:

    def __init__(self, dir):
        self.dir = dir

    def log_tensor(self, folder_name, file_name, data):
        file = os.path.join(self.dir, folder_name, f'{file_name}.pt')
        torch.save(data, file)

    def log_mesh(self, folder_name, file_name, mesh):
        # create a temporary file to store the mesh
        with tempfile.NamedTemporaryFile(suffix=".obj") as tmpfile:
            mesh.export(tmpfile.name, file_type='obj')
            mesh.export(os.path.join(self.dir, folder_name, f"{file_name}.obj"))

    def clear_data(self, folder_list=[]):
        folder_list += ["sparse_points", "frustum_sampling", "test_mesh", "test_tsdf"]
        for folder in folder_list:
            folder_dir = os.path.join(self.dir, folder)
            for f in os.listdir(folder_dir):
                os.remove(os.path.join(folder_dir, f))
            print("deleted contents of:", folder_dir)