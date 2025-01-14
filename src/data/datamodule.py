import hydra
import psutil
import torch
from typing import Any, Dict, Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data.data import ScenesSequencesDataset, FrameDataset, OneSceneDataset, ScenesDataset, collate_fn, parse_splits_list
import src.data.transforms as transforms


class ScannetDataModule(LightningDataModule):
    """`LightningDataModule` for the ScanNet dataset"""

    def __init__(
        self,
        data_dir: str,
        datasets_train: list[str],
        datasets_val: list[str],
        datasets_test: list[str],
        from_archive: bool,
        num_workers_train: int,
        num_workers_val: int,
        num_workers_test: int,
        persistent_workers: bool,
        pin_memory: bool,
        batch_size: int,
        shuffle_train: bool,
        shuffle_val: bool,
        shuffle_test: bool,
        random_rotation_3d: bool,
        random_translation_3d: bool,
        pad_xy_3d: float,
        pad_z_3d: float,
        voxel_size: float,
        voxel_types: list[str],
        voxel_dim_train: list[int],
        voxel_dim_val: list[int],
        voxel_dim_test: list[int],
        layers_down: int,
        dataset_type: str,
        # additionally for ScenesSequencesDataset:
        sequence_amount_train: float = 0.0,
        sequence_amount_val: float = 0.0,
        sequence_amount_test: float = 0.0,
        sequence_length: int = 0,
        sequence_locations: int = 0,
        sequence_order: str = '',
        frame_locations: str = '',
        frame_order: str = '',
        num_frames_train: int = 0,
        num_frames_val: int = 0,
        num_frames_test: int = 0,
        # additionally for FrameDataset:
        length_train: int = 0,
        length_val: int = 0,
        length_test: int = 0,
        frame_idx: int = 0,
        scene_idx: int = 0,
        # additionally for OneSceneDataset:
        frames_train: int = 0,
        frames_val: int = 0,
        frames_test: int = 0,
        # additionally for ScenesDataset: /
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

        self.batch_size_per_device = self.hparams.batch_size

        self.frame_types = ['depth']  # color is always loaded
        self.voxel_types = self.hparams.voxel_types
        if self.hparams.layers_down:
            self.voxel_sizes = [int(self.hparams.voxel_size*100)*2**i for i in 
                                range(len(self.hparams.layers_down)-1)]
        else:
            self.voxel_sizes = [int(self.hparams.voxel_size*100)]


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
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
            
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        transform = self.get_transform('train')
        info_files = parse_splits_list(self.hparams.datasets_train, self.hparams.data_dir)
        dataset = self.choose_dataset(
            info_files, transform, self.hparams.sequence_amount_train, 
            self.hparams.length_train, self.hparams.num_frames_train,
            self.hparams.frames_train, self.voxel_sizes, self.hparams.voxel_dim_train
        )
        print(f"Train Dataset len: {len(dataset)} (scenes: {len(dataset.info_files)})")

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size_per_device, num_workers=self.hparams.num_workers_train,
            persistent_workers=self.hparams.persistent_workers, collate_fn=collate_fn, 
            shuffle=self.hparams.shuffle_train, drop_last=True, pin_memory=self.hparams.pin_memory
        )
        return dataloader

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        transform = self.get_transform('val')
        info_files = parse_splits_list(self.hparams.datasets_val, self.hparams.data_dir)
        dataset = self.choose_dataset(
            info_files, transform, self.hparams.sequence_amount_val,
            self.hparams.length_val, self.hparams.num_frames_val,
            self.hparams.frames_val, self.voxel_sizes, self.hparams.voxel_dim_val
        )
        print(f"Validation Dataset len: {len(dataset)} (scenes: {len(dataset.info_files)})")

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size_per_device, num_workers=self.hparams.num_workers_val,
            persistent_workers=self.hparams.persistent_workers, collate_fn=collate_fn,
            shuffle=self.hparams.shuffle_val, drop_last=False, pin_memory=self.hparams.pin_memory
        )
        return dataloader

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        transform = self.get_transform('test')
        info_files = parse_splits_list(self.hparams.datasets_test, self.hparams.data_dir)
        dataset = self.choose_dataset(
            info_files, transform, self.hparams.sequence_amount_test, 
            self.hparams.length_test, self.hparams.num_frames_test,
            self.hparams.frames_test, self.voxel_sizes, self.hparams.voxel_dim_test
        )
        print(f"Test dataset len: {len(dataset)} (scenes: {len(dataset.info_files)})")

        dataloader = DataLoader(
            dataset, batch_size=1, num_workers=self.hparams.num_workers_test, collate_fn=collate_fn,
            shuffle=self.hparams.shuffle_test, drop_last=False, pin_memory=self.hparams.pin_memory
        )
        return dataloader

    def predict_dataloader(self):
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        transform = self.get_transform('test') # TODO: should be "predict" for having it equal to Atlas inference
        info_files = parse_splits_list(self.hparams.datasets_test, self.hparams.data_dir)
        dataset = self.choose_dataset(
            info_files, transform, self.hparams.sequence_amount_test, 
            self.hparams.length_test, self.hparams.num_frames_test,
            self.hparams.frames_test, [self.voxel_sizes[0]], self.hparams.voxel_dim_test
        )
        print(f"Predict dataset len: {len(dataset)} (scenes: {len(dataset.info_files)})")

        dataloader = DataLoader(
            dataset, batch_size=1, num_workers=self.hparams.num_workers_test, collate_fn=collate_fn,
            shuffle=self.hparams.shuffle_test, drop_last=False, pin_memory=self.hparams.pin_memory
        )
        return dataloader

    def choose_dataset(self, info_files, transform, sequence_amount, length, num_frames, frames, voxel_sizes, voxel_dim=None):
        if self.hparams.dataset_type=='frame':
            dataset = FrameDataset(
                info_files, self.hparams.frame_idx, length, self.hparams.scene_idx, 
                transform, self.frame_types, self.voxel_types, self.voxel_sizes, self.hparams.from_archive
            )
        elif self.hparams.dataset_type=='scene':
            dataset = OneSceneDataset(
                info_files[0], transform, self.frame_types, self.voxel_types, self.voxel_sizes, 
                frames, self.hparams.from_archive
            )
        elif self.hparams.dataset_type=='scenes':
            dataset = ScenesDataset(
                info_files, num_frames, self.hparams.frame_locations, self.hparams.frame_order, transform,
                self.frame_types, self.hparams.voxel_types, voxel_sizes, self.hparams.from_archive, 
                voxel_dim)
        elif self.hparams.dataset_type=='sequences':
            dataset = ScenesSequencesDataset(
                info_files, sequence_amount, self.hparams.sequence_length, 
                self.hparams.sequence_locations, self.hparams.sequence_order, num_frames,
                self.hparams.frame_locations, self.hparams.frame_order, transform, self.frame_types,
                self.voxel_types, self.voxel_sizes, self.hparams.from_archive
            )
        else:
            raise NotImplementedError(f"Usage of unknown mode: {self.hparams.dataset_type}")
        
        return dataset

    def get_transform(self, mode):
        """ Gets a transform to preprocess the input data."""

        if mode == "predict":
            return None

        if mode == "train":
            voxel_dim = self.hparams.voxel_dim_train
            random_rotation = self.hparams.random_rotation_3d
            random_translation = self.hparams.random_translation_3d
            paddingXY = self.hparams.pad_xy_3d
            paddingZ = self.hparams.pad_z_3d
        else:
            # center volume
            if mode == "val":
                voxel_dim = self.hparams.voxel_dim_val
            elif mode == "test":
                voxel_dim = self.hparams.voxel_dim_test
            else:
                raise NotImplementedError(f"Usage of unknown mode: {mode}")
                
            random_rotation = False
            random_translation = False
            paddingXY = 0
            paddingZ = 0

        transform = []
        transform += [transforms.ResizeImage((640,480)),  # TODO: 640,480 good size? -> define in config?
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