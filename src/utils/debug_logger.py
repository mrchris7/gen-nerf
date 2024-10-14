
import os
import tempfile
import torch


class DebugLogger:

    def __init__(self, dir, tag):
        if tag == '' or tag == None:
            self.dir = f"{dir}/logs"  # default dir if no tag is provided
        else:
            self.dir = f"{dir}/logs_{tag}"
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            

    def log_tensor(self, folder_name, file_name, data):
        file = os.path.join(self.dir, folder_name, f'{file_name}.pt')
        torch.save(data, file)
        return file

    def log_mesh(self, folder_name, file_name, mesh):
        # create a temporary file to store the mesh
        #with tempfile.NamedTemporaryFile(suffix=".obj") as tmpfile:
        #    mesh.export(tmpfile.name, file_type='obj')
        #    mesh.export(os.path.join(self.dir, folder_name, f"{file_name}.obj"))

        # from Atlas
        file = os.path.join(os.path.join(self.dir, folder_name), f"{file_name}.ply")
        mesh.export(file)
        return file

    def log_tsdf(self, folder_name, file_name, tsdf):
        file = os.path.join(os.path.join(self.dir, folder_name), f"{file_name}.npz")
        tsdf.save(file)
        return file

    def clear_data(self, folder_list=[]):
        folder_list += ["sparse_points", "frustum_sampling", "test_mesh", "test_tsdf", "eval_metrics"]
        for folder in folder_list:
            folder_dir = os.path.join(self.dir, folder)
            if os.path.exists(folder_dir):
                for f in os.listdir(folder_dir):
                    os.remove(os.path.join(folder_dir, f))
            else:
                os.makedirs(folder_dir)
            print("deleted contents of:", folder_dir)