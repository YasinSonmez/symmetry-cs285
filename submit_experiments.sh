#!/bin/bash
# Job name:
#SBATCH --job-name=mbpo-reacher-seed2
#
# Account:
#SBATCH --account=fc_control
#
# Partition:
#SBATCH --partition=savio4_gpu
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
#SBATCH --cpus-per-task=4
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:A5000:1
#
# Wall clock limit:
#SBATCH --time=8:00:00
#
## Command(s) to run (example):

module load python
source activate mbrl2
python --version

python -m mbrl.examples.main algorithm=mbpo seed=2 overrides=custom_mbpo_reacher_no_symm experiment=no_symmetry &
python -m mbrl.examples.main algorithm=mbpo seed=2 overrides=custom_mbpo_reacher_symm experiment=symmetry dynamics_model.in_size=5 &

# python -m mbrl.examples.main algorithm=mbpo seed=2 overrides=custom_mbpo_rwd_asymm_inv_pend_no_symm experiment=no_symmetry &
# python -m mbrl.examples.main algorithm=mbpo seed=2 overrides=custom_mbpo_rwd_asymm_inv_pend_symm experiment=symmetry dynamics_model.in_size=4 &

# python -m mbrl.examples.main algorithm=pets overrides=custom_pets_rwd_asymm_inv_pend_no_symm experiment=no_symmetry &
# python -m mbrl.examples.main algorithm=pets overrides=custom_pets_rwd_asymm_inv_pend_symm experiment=symmetry dynamics_model.in_size=4 &

wait
echo "all done"
