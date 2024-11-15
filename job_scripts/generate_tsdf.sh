#!/bin/bash -l

# Name
#SBATCH --job-name=generate_tsdf

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

#SBATCH --cpus-per-task=16  # too much: 64

# SBATCH --exclusive
# SBATCH --mem-per-cpu=3250M  # outdated

# Max time (hh:mm:ss)
#SBATCH --time=01:00:00

# load modules
unset SLURM_EXPORT_ENV
module purge
module load python/3.9-anaconda
#module load intel/2022.1
#module load cuda/11.7.1

# set number of threads to requested cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 

# activate env
conda activate gen-nerf

echo --------------- START --------------------
echo nproc=$(nproc)
echo CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
echo slurm_gpus_on_node="$SLURM_GPUS_ON_NODE"
echo ------------------------------------------

# execute
CMD="python $HOME/workspace/gennerf/gen-nerf/scripts/build_scannet.py\
 --path_target $TMPDIR/data/scannet\
 --path_raw $WORK/data/scannet_raw\
 --path_archive $WORK/data/scannet\
 --extract_archives\
 --num_scenes 10"
echo ${CMD}
${CMD}

echo "Dataset built!"

CMD="python $HOME/workspace/gennerf/gen-nerf/src/data/prepare/prepare_data.py\
 --path $TMPDIR/data/scannet\
 --path_meta $WORK/data/scannet" # --skip_existing
echo ${CMD}
${CMD}
EXITCODE=$?

# wait until all processes finish and then return exitcode
wait
echo "Execution done"
exit $EXITCODE