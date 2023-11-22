import d3rlpy
import d4rl
import gym
import h5py
import numpy as np
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
import gym
from gym.wrappers import TransformObservation
import encoders
import os
import yaml
import argparse


def read_hdf5_to_dict(file_name):
    data = {}

    def recursive_read(group, prefix=''):
        for key in group.keys():
            if key not in ['observations', 'actions', 'terminals', 'rewards', 'timeouts']:
                continue
            item = group[key]
            if isinstance(item, h5py.Dataset):
                # This is a dataset
                data[prefix + key] = np.array(item)
            elif isinstance(item, h5py.Group):
                # This is a group, so recurse into it
                new_prefix = prefix + key + '/'
                recursive_read(item, new_prefix)

    with h5py.File(file_name, 'r') as f:
        recursive_read(f)

    return data

def create_permutation_matrix(size, source_indices, target_indices):
    """
    Create a permutation matrix of given size that permutes rows from source_indices to target_indices.

    :param size: Size of the square permutation matrix
    :param source_indices: List of indices to be permuted
    :param target_indices: List of target indices where source indices should be moved
    :return: Permutation matrix of size 'size x size'
    """
    if len(source_indices) != len(target_indices):
        raise ValueError("Source and target indices lists must be of the same length")

    # Create an identity matrix
    perm_matrix = np.identity(size)

    # Apply the permutations
    for src, tgt in zip(source_indices, target_indices):
        perm_matrix[[src, tgt]] = perm_matrix[[tgt, src]]

    return perm_matrix

def augment_data(data_dict, state_permutation, action_permutation):
    m = data_dict['observations'].shape[1]
    n = data_dict['actions'].shape[1]

    P_s = create_permutation_matrix(m, state_permutation[0], state_permutation[1])
    P_a = create_permutation_matrix(n, action_permutation[0], action_permutation[1])
    # Vectorized transformation
    transformed_observations = np.dot(data_dict['observations'], P_s.T)  # Transpose if necessary
    transformed_actions = np.dot(data_dict['actions'], P_a.T)            # Transpose if necessary
    
    data_dict['observations'] = np.concatenate((data_dict['observations'], transformed_observations), axis=0)
    data_dict['actions'] = np.concatenate((data_dict['actions'], transformed_actions), axis=0)
    data_dict['terminals'] = np.concatenate((data_dict['terminals'], data_dict['terminals']))
    data_dict['rewards'] = np.concatenate((data_dict['rewards'], data_dict['rewards']))
    data_dict['timeouts'] = np.concatenate((data_dict['timeouts'], data_dict['timeouts']))

def main(args):
    # Read the YAML configuration file
    with open(args.cfg, 'r') as file:
        config = yaml.safe_load(file)
    augmentation = config['augmentation']

    print(gym.version.VERSION)
    seed = 1
    d3rlpy.seed(seed)
    use_gpu = True
    # prepare environment
    #env = gym.make("Walker2d-v2")
    #eval_env = gym.make("Walker2d-v2")
    #env.reset(seed=seed)
    #eval_env.reset(seed=seed)
        
    file_path = 'd3rlpy_data/walker2d_expert-v2.hdf5'  # Replace with the path to your HDF5 file
    data_dict = read_hdf5_to_dict(file_path)

    state_permutation = ([2, 3, 4, 11, 12, 13], [5, 6, 7, 14, 15, 16])
    action_permutation = ([0, 1, 2], [3, 4, 5])

    if augmentation:
        augment_data(data_dict, state_permutation, action_permutation)

    dataset = MDPDataset(data_dict['observations'], data_dict['actions'], data_dict['rewards'], np.logical_or(data_dict['terminals'], data_dict['timeouts']))
    del data_dict

    train_episodes, test_episodes = train_test_split(dataset, random_state=seed, train_size=0.1)
    dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=2e-4, use_gpu=True) # Baseline
    # same as algorithms
    dynamics.fit(train_episodes,
                eval_episodes=test_episodes,
                n_epochs=500,
                n_steps_per_epoch=1000,
                scorers={
                    'observation_error': d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
                    'reward_error': d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
                    'variance': d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
                },
                tensorboard_dir='tensorboard_logs/dynamics',
                experiment_name='augmentation_'+str(augmentation)+'_exp1')


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Read configuration from a YAML file.")
    parser.add_argument("--cfg", required=True, help="Path to the YAML configuration file")

    # Parse arguments
    args = parser.parse_args()

    # Pass the value to the main function
    main(args)