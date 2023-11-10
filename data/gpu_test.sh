#!/bin/bash
# Job name:
#SBATCH --job-name=rl-symmetry-100-iters
#
# Account:
#SBATCH --account=fc_control
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Always at least twice the number of GPUs (savio2_gpu and GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu and A5000 in savio4_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:1
#
# Wall clock limit:
#SBATCH --time=08:00:00
#
## Command(s) to run (example):
module load cuda/10.2
module load python
source activate cs285
python --version
python cs285/scripts/run_hw4.py -cfg experiments/symmetry_model/reacher.yaml --sac_config_file experiments/symmetry_sac/reacher.yaml &
python cs285/scripts/run_hw4.py -cfg experiments/symmetry_model/reacher_no_symmetry.yaml --sac_config_file experiments/symmetry_sac/reacher.yaml &
# python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_cem_2iters.yaml &
# python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_cem.yaml &
wait
echo "all done"
