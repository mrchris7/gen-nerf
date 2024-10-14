# Generalizable Neural Fields

Learning scene-level generalizable neural fields using NeRFs and feature distillation from pre-trained Vision Language Models, creating a unified scene representation that captures geometric and semantic properties.


# Installation

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
pip install -e .
pip install matplotlib==3.8
pip install scikit-image
pip install trimesh>=3.7.6
pip install opencv-python
pip install open3d>=0.10.0.0
pip install rich wandb pypng pyrender
```

# Configure an Experiment

We use the configuration framework Hydra that allows for structured and hierarchical orginization of configuration files.

# Run Training

Run an experiment by using the configuration yaml "exp1":
```
python src/train.py experiment=exp1
```


# Evaluate

Run evaluation:

```
python src/models/evaluation.py --result FILE
```