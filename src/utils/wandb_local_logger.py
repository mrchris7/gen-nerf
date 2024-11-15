import os
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch
import wandb


class LocalWriter:
    """ Saves media locally during training """

    def __init__(self, save_path, mute):
        self._save_path = os.path.join(save_path, "local")
        self._mute = mute
        os.makedirs(self._save_path, exist_ok=True)

    def log_mesh(self, mesh, path): # TODO: include epoch # and train/val
        if self._mute:
            return
        file = os.path.join(self._save_path, f'{path}.ply')
        os.makedirs(os.path.dirname(file), exist_ok=True)
        mesh.export(file)

    def log_tensor(self, tensor, path):
        if self._mute:
            return
        file = os.path.join(self._save_path, f'{path}.pt')
        os.makedirs(os.path.dirname(file), exist_ok=True)
        torch.save(tensor, file)

    def log_tsdf(self, tsdf, path):
        if self._mute:
            return
        file = os.path.join(self._save_path, f'{path}.npz')
        os.makedirs(os.path.dirname(file), exist_ok=True)
        tsdf.save(file)   
    
    def log_image(self, image, path):
        # tries to auto convert image into a PIL image
        if self._mute:
            return
        file = os.path.join(self._save_path, f'{path}.png')
        pil_image = wandb.Image(image).image
        os.makedirs(os.path.dirname(file), exist_ok=True)
        pil_image.save(file)
    
class WandbLocalLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        mute_local = kwargs.pop('mute_local', False)
        super().__init__(*args, **kwargs)

        self._local = LocalWriter(self.save_dir, mute_local)

    @property
    def local(self) -> LocalWriter:
        return self._local


    @rank_zero_only
    def log_pointcloud(self, pc, name):
        """
        pc (x, y, z, color): pointcloud 
        """
        wandb.log({name: wandb.Object3D(pc)})

    '''
    def log_mesh_custom(self, mesh, name):
        with tempfile.NamedTemporaryFile(suffix=".obj") as tmpfile:
            mesh.export(tmpfile.name, file_type='obj')
            wandb.log({name: wandb.Object3D(tmpfile.name)})
    '''

    '''
    def log_image_custom(self, image_tensor, name):
        img = image_tensor.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # convert CHW to HWC
        wandb.log({name: wandb.Image(img)})
    '''
