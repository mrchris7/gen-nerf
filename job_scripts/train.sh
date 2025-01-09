#!/bin/bash -l

# Name
#SBATCH --job-name=train_geometric

# Settings
#SBATCH --mail-type=NONE
#SBATCH --export=NONE

# Output files
#SBATCH --output=/home/hpc/g101ea/g101ea13/job_out/slurm_job_%A.out
#SBATCH --error=/home/hpc/g101ea/g101ea13/job_out/slurm_job_%A.err

# Hardware
#SBATCH --gres=gpu:a40:4
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8  # too much: 64

# SBATCH --exclusive
# SBATCH --mem-per-cpu=3250M  # outdated

# Max time (hh:mm:ss)
#SBATCH --time=02:00:00

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

# activate env
conda activate gen-nerf-cuda118

echo --------------- START --------------------
echo nproc=$(nproc)
echo CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
echo slurm_gpus_on_node="$SLURM_GPUS_ON_NODE"
echo ------------------------------------------

PROJECT=. # i.e. $HOME/workspace/gennerf/gen-nerf
ARGS=${@:1}

PATH_DATA=/anvme/workspace/g101ea13-databox/data/scannet

# execute
CMD="python $PROJECT/scripts/staging.py\
 --path_src $PATH_DATA\
 --path_des $TMPDIR/data/scannet\
 --extract_archives
 --scenes_file /home/hpc/g101ea/g101ea13/workspace/gennerf/gen-nerf/splits/scenes_file_living.txt"
 
echo ${CMD}
${CMD}

echo "Dataset staged!"


# execute
CMD="python $PROJECT/src/train.py $ARGS"

echo ${CMD}
${CMD}
EXITCODE=$?

# wait until all processes finish and then return exitcode
wait
echo "Execution done"
exit $EXITCODE