#!/bin/bash
# Job name:
#SBATCH --job-name=rl-symmetry-1
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
#SBATCH --cpus-per-task=3
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:1
#
# Wall clock limit:
#SBATCH --time=10:00:00
#
## Command(s) to run (example):
#module load cuda/10.2
#module load python
#source activate symmetry
#conda install pip
#pip install -r requirements.txt
#sudo apt-get install libglew-dev
#cd d3rlpy
#pip install -e .
#cd ..
python --version

#conda install -c conda-forge glew
#conda install -c conda-forge mesalib
#conda install -c menpo glfw3
#echo 'export CPATH=$CONDA_PREFIX/include' >> ~/.bashrc

#echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/home/users/yasinsonmez/.mujoco/mujoco210/bin' >> ~/.bashrc
#echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
#source ~/.bashrc
# Run multiple commands in parallel with the "&" at the end
# This makes it so we're more fully utilizing the GPU.
#python discreteSymmetry.py --cfg cfg/augmentation.yaml &
#python discreteSymmetry.py --cfg cfg/noAugmentation.yaml &
#python discreteSymmetry.py --cfg cfg/reduction.yaml &
#python discreteSymmetry.py --cfg cfg/stochasticAugmentation.yaml &
#python discreteSymmetry.py --cfg cfg/reductionAugmentation.yaml &

# Define the range of seeds
start_seed=2
N_runs=1
end_seed=$((start_seed + N_runs - 1))

# Loop through the range
for seed in $(seq $start_seed $end_seed)
do
    echo "Running script with seed $seed"
    python discreteSymmetry.py --cfg cfg/noAugmentation.yaml --seed $seed --dynamics &
    python discreteSymmetry.py --cfg cfg/reduction.yaml --seed $seed --dynamics &
    python discreteSymmetry.py --cfg cfg/stochasticAugmentation.yaml --seed $seed --dynamics &
    wait
    echo "Script with seed $seed ended"
done

#python discreteSymmetry.py --cfg cfg/reduction.yaml --dynamics --COMBO --combo_symmetry &
#python discreteSymmetry.py --cfg cfg/reductionAugmentation.yaml --dynamics &
#python discreteSymmetry.py --cfg cfg/noAugmentation.yaml --dynamics --COMBO &
#python discreteSymmetry.py --cfg cfg/stochasticAugmentation.yaml --dynamics &

#python discreteSymmetry.py --cfg cfg/reduction.yaml --COMBO --combo_symmetry &
#python discreteSymmetry.py --cfg cfg/noAugmentation.yaml --COMBO &

wait # Wait for all background processes to end

echo "all done"