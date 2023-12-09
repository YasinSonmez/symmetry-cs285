#!/bin/bash
# Job name:
#SBATCH --job-name=mbrl-lib
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
#SBATCH --time=7:00:00
#
## Command(s) to run (example):

#pip install glfw
#pip install mujoco
#pip install "cython<3"
#conda install -c conda-forge glew
#conda install -c conda-forge mesalib
#conda install -c menpo glfw3
#echo 'export CPATH=$CONDA_PREFIX/include' >> ~/.bashrc

#echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/home/users/yasinsonmez/.mujoco/mujoco210/bin' >> ~/.bashrc
#echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
#source ~/.bashrc

python --version
python -m mbrl.examples.main algorithm=mbpo overrides=mbpo_ant
#python -m mbrl.examples.main algorithm=mbpo overrides=mbpo_walker
#python -m mbrl.examples.main algorithm=pets overrides=pets_halfcheetah 
#python -m mbrl.examples.main algorithm=pets overrides=pets_inv_pendulum