# Project Setup on NHR@FAU
The following instructions apply specifically to the [Erlangen National High Performance Computing Center (NHR@FAU)](https://hpc.fau.de/). However they can be adopted for usage on any HPC.


### Set location for environments
Before creating your environment you can configure where packages and environments are stored. Store them in ```$WORK``` instead of ```$HOME``` in order to save space in the latter:
```
if [ ! -f ~/.bash_profile ]; then
  echo "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi" > ~/.bash_profile
fi
module add python
conda config --add pkgs_dirs $WORK/software/private/conda/pkgs
conda config --add envs_dirs $WORK/software/private/conda/envs
```

More information: https://doc.nhr.fau.de/environment/python-env/#first-time-only-initialization

## Installation
Load modules:
```
module load python
module load gcc
```

Create a conda environment:

```
conda create -n gen-nerf-cuda118 python=3.9
conda activate gen-nerf-cuda118
```
Install necessary packages in the following order:

```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install lightning
pip install hydra-core --upgrade
pip install rootutils
pip install matplotlib==3.8
pip install scikit-image
pip install trimesh>=3.7.6
pip install opencv-python
pip install open3d>=0.10.0.0
pip install wandb tensorboard
pip install rich pypng pyrender
pip install torch-cluster  # before: module load gcc
```

Clone this repository to the ```$HOME``` filesystem and install it as a package:
```
git clone https://github.com/mrchris7/gen-nerf.git
cd gen-nerf
pip install -e .
```


## Data Preparation

### Download ScanNet

First, you need to download the ScanNet dataset to a directory ```PATH_RAW```, following the instructions from http://www.scan-net.org/.
For the target directory you can use ```PATH_RAW=$WORK/data/scannet_raw```:
```
mkdir -p $WORK/data/scannet_raw
python download-scannet.py -o $WORK/data/scannet_raw --skip_existing
```
Downloading takes a while and can sometimes lead to a cancellation of the process if performed on the login node. In this case delete the partially downloaded directory and start again from where it ended (use flag ```--skip_existing```).

Then add the splits from the repository to the ```PATH_RAW``` directory:
```
cd gen-nerf
scp -r splits/ $WORK/data/scannet_raw
```

### Prepare Data

Now the raw data needs to be transformed into a usable structure and format.
First, read the raw data format (.sens) and transforms the RGB-D frames and camera information as archives (.tar) to another directory ```PATH_DATA=$WORK/data/scannet```:
```
sbatch job_scripts/read_scannet.sh PATH_RAW PATH_DATA 
```

Then create scene files (.info) that store all references to scene data, generates ground truth TSDF-values and add them to the directory ```PATH_DATA```. Note that internally it is done by extracting all archives from all scenes from ```PATH_DATA``` to temporary, node-local space (```$TMPDIR/data/scannet```), calculating the TSDF-values, and then copying them to ```PATH_DATA```.

```
sbatch job_scripts/generate_tsdf.sh PATH_RAW PATH_DATA
```

If you only need to prepare "living room" scenes use the following scripts:

```
sbatch job_scripts/read_scannet_living.sh
sbatch job_scripts/generate_tsdf_living.sh
```


The final directory structure should look as follows:
```
PATH_DATA
└───scannet_test.txt
└───scannet_train.txt
└───scannet_val.txt
└───scannet_living_test.txt
└───scannet_living_train.txt
└───scannet_living_val.txt
└───scans
|   └───scene0000_00
|   |   └───info.json
|   |   └───color
|   |   │   └───color.tar
|   |   └───depth
|   |   │   └───depth.tar
|   |   └───poses
|   |   │   └───poses.tar
|   |   └───intrinsics
|   |   │   └───extrinsic_color.txt     
|   |   │   └───extrinsic_depth.txt 
|   |   │   └───intrinsic_color.txt
|   |   │   └───intrinsic_depth.txt
|   |   └───mesh_04.ply
|   |   └───...
|   |   └───tsdf_04.npz
|   |   └───...
|   └───...
└───scans_test
    └───scene700_00
    └───...
```


## Quick Demo

If you just want to run a demo and do not want to install the whole dataset, we provide a prepared data for training on one scene.

### Download Prepared Data
Download ```scannet_demo.tar``` [here](https://drive.google.com/file/d/1HlkqURV0shaQg06PfP0qmaWoM-A4dvcw/view?usp=drive_link) and unpack it in the data directory ```DATA_DIR``` (i.e. ```/home/atuin/gxxxxx/gxxxxx00/data```):
```
mkdir DATA_DIR
tar -xf scannet_demo.tar -C DATA_DIR
```

Create a info.json that contains the absolute path to your data directory ```DATA_DIR```:
```
cd DATA_DIR/scannet/scans/scene0244_01/
sed 's|XXX|DATA_DIR|g' info_template.json > info.json
```

Adjust the paths in the config file ```configs/paths/cluster.yaml``` (set ```data_dir=DATA_DIR/scannet```).


### Run Experiment

Start an interactive job:
```
salloc --gres=gpu:a40:1 --time=0:15:0
export HTTPS_PROXY=http://proxy:80
module load python
conda activate gen-nerf-cuda118
```

Run the experiment:
```
python src/train.py experiment=seq1_0244_01_frames8_evenspaced_pointnet_cluster.yaml logger=wandb_local
```

The result of the run will be stored in the ```log_dir``` specified in the paths config file.

### Visualize Result
First, you need to transfer the results of the desired run in ```[log_dir]/train/runs``` to your local machine.
You can vizualize the result locally using the script ```scripts/local/visualize_all.py```.