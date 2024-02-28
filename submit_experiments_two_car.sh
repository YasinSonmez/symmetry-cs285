#!/bin/bash
#SBATCH --job-name=two-cars-cos
#SBATCH --account=fc_control
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Processors per task:
# Always at least twice the number of GPUs (savio2_gpu and GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu and A5000 in savio4_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=4
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=48:00:00
#SBATCH --qos=savio_normal
#SBATCH --array=0-3

#### SBATCH --job-name=two-cars-cos
#### SBATCH --account=fc_control
#### SBATCH --partition=savio3
#### SBATCH --nodes=1
#### SBATCH --time=48:00:00
#### SBATCH --qos=savio_normal
#### SBATCH --array=0-3

# ================================
module load python/3.10.10
source activate mbrl3

export TASK_ID=$SLURM_ARRAY_TASK_ID
export USE_GPU=1

(export SEED=0; python state_projection.py) &
(export SEED=1; python state_projection.py) &

wait
conda deactivate
echo "all done"