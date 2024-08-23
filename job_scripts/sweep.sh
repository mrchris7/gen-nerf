#!/bin/bash -l

# Name
#SBATCH --job-name=sweep_tsdf

# Settings
#SBATCH --mail-type=NONE
#SBATCH --export=NONE

# Output files
#SBATCH --output=/home/hpc/g101ea/g101ea13/job_out/slurm_job_%A.out
#SBATCH --error=/home/hpc/g101ea/g101ea13/job_out/slurm_job_%A.err

# Hardware
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8  # too much: 64

# SBATCH --exclusive
# SBATCH --mem-per-cpu=3250M  # outdated

# Max time (hh:mm:ss)
#SBATCH --time=02:30:00

# load modules
unset SLURM_EXPORT_ENV
module purge
module load python/3.9-anaconda
#module load intel/2022.1
#module load cuda/11.7.1

# set number of threads to requested cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 

# enable connection to internet for wandb-logger
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# manually set log directory for sweeps
export WANDB_DIR=/home/atuin/g101ea/g101ea13/logs/train/sweep

# activate env
conda activate gen-nerf-cuda118

echo --------------- START --------------------
echo nproc=$(nproc)
echo CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
echo slurm_gpus_on_node="$SLURM_GPUS_ON_NODE"
echo ------------------------------------------

# manually run before:
# > export WANDB_DIR=/home/atuin/g101ea/g101ea13/logs/train/sweep
# > wandb sweep --entity X --project X $HOME/workspace/gennerf/gen-nerf/configs/sweeps/tsdf_one_frame.yaml

# execute
CMD="wandb agent irosa-ias/gen-nerf/ub96dvux --count 20"
echo ${CMD}
${CMD}
EXITCODE=$?

# wait until all processes finish and then return exitcode
wait
echo "Execution done"
exit $EXITCODE