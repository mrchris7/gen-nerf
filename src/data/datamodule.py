import torch
from typing import Any, Dict, Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.data import ScenesDataset, collate_fn, parse_splits_list
import src.data.transforms as transforms


class ScannetDataModule(LightningDataModule):
    """`LightningDataModule` for the ScanNet dataset"""

    def __init__(
        self,
        data_dir: str,
        datasets_train: list[str],
        datasets_val: list[str],
        num_workers_train: int,
        num_workers_val: int,
        pin_memory: bool,
        batch_size_train: int,
        num_frames_train: int,
        num_frames_val: int,
        frame_selection: str,
        random_rotation_3d: bool,
        random_translation_3d: bool,
        pad_xy_3d: float,
        pad_z_3d: float,
        voxel_size: float,
        voxel_types: list[str],
        voxel_dim_train: list[int],
        voxel_dim_val: list[int],
        voxel_dim_test: list[int],
        layers_down: list[int]
    ) -> None:
        """Initialize a `ScannetDataModule`"""
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        '''
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        '''

        self.batch_size_per_device = batch_size_train

        self.frame_types = ['depth']  # color is always loaded
        self.voxel_types = self.hparams.voxel_types
        self.voxel_sizes = [int(self.hparams.voxel_size*100)*2**i for i in 
                            range(len(self.hparams.layers_down)-1)]


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        # -> modified so that it is loaded 

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size_train % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size_train}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size_train // self.trainer.world_size
            
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        transform = self.get_transform(True)
        info_files = parse_splits_list(self.hparams.datasets_train, self.hparams.data_dir)
        dataset = ScenesDataset(
            info_files, self.hparams.num_frames_train, transform, self.frame_types,
            self.hparams.frame_selection, self.voxel_types, self.voxel_sizes
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size_per_device, num_workers=self.hparams.num_workers_train,
            collate_fn=collate_fn, shuffle=True, drop_last=True, pin_memory=self.hparams.pin_memory
        )
        return dataloader

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        transform = self.get_transform(False)
        info_files = parse_splits_list(self.hparams.datasets_val, self.hparams.data_dir)
        dataset = ScenesDataset(
            info_files, self.hparams.num_frames_val, transform, self.frame_types,
            self.hparams.frame_selection, self.voxel_types, self.voxel_sizes
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=self.hparams.num_workers_val, collate_fn=collate_fn,
            shuffle=False, drop_last=False, pin_memory=self.hparams.pin_memory
        )
        return dataloader

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        # TODO: see evaluate.py of conv_onets
        return None 

    def get_transform(self, is_train):
        """ Gets a transform to preprocess the input data."""

        if is_train:
            voxel_dim = self.hparams.voxel_dim_train
            random_rotation = self.hparams.random_rotation_3d
            random_translation = self.hparams.random_translation_3d
            paddingXY = self.hparams.pad_xy_3d
            paddingZ = self.hparams.pad_z_3d
        else:
            # center volume
            voxel_dim = self.hparams.voxel_dim_val
            random_rotation = False
            random_translation = False
            paddingXY = 0
            paddingZ = 0

        transform = []
        transform += [transforms.ResizeImage((640,480)),
                      transforms.ToTensor(),
                      #transforms.InstanceToSemseg('nyu40'),
                      transforms.RandomTransformSpace(
                          voxel_dim, random_rotation, random_translation,
                          paddingXY, paddingZ),
                      transforms.FlattenTSDF(),
                      transforms.IntrinsicsPoseToProjection(),
                     ]

        return transforms.Compose(transform)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = ScannetDataModule()