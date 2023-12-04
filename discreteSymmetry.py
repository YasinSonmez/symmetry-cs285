import d3rlpy
import d4rl
import gym
import h5py
import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import COMBO
from sklearn.model_selection import train_test_split
import gym
from gym.wrappers import TransformObservation
import encoders
import os
import yaml
import argparse
import json


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

def create_train_merge_matrix(m, source_indices, target_indices):
    """
    Create two matrices that prepares the data to train with symmetry and merge afterwards

    :param m: Size of the state space
    :param source_indices: List of indices to be permuted
    :param target_indices: List of target indices where source indices should be moved
    :return: train_matrix: removes the rows with target indices from the data
    merge_matrix: takes input of size 2(m-len*(target_indices)) and merges them into 
    next observations data of size m by averaging the common parts and merging different parts
    """
    if len(source_indices) != len(target_indices):
        raise ValueError("Source and target indices lists must be of the same length")

    # Create an identity matrix
    train_matrix = np.identity(m)
    train_matrix = np.delete(train_matrix, target_indices, axis=0)

    t = m - len(target_indices) # size of new data
    merge_matrix = np.zeros((m, 2*t))
    reduced_idx=0
    for i in range(m):
        if i not in source_indices and i not in target_indices:
            # common states, average
            merge_matrix[i, reduced_idx] = 0.5
            merge_matrix[i, t + reduced_idx] = 0.5
            reduced_idx += 1
        elif i in source_indices:
            permutation_idx = np.where(np.array(source_indices) == i)[0][0]
            target_i = target_indices[permutation_idx]
            merge_matrix[i, reduced_idx] = 1
            merge_matrix[target_i, t + reduced_idx] = 1
            reduced_idx += 1

    return train_matrix, merge_matrix

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

def experiment_COMBO_training(dataset, eval_env, experiment_name, save_name, models_dir, seed=1, use_gpu=True):
    model_paths = [filename for filename in os.listdir(models_dir) if filename.startswith(experiment_name)]
    model_paths = [models_dir + model_paths_i for model_paths_i in model_paths]
    model_paths.sort()
    print(model_paths)

    symmetry_reduced_paths = []
    default_paths = []
    for model_path_i in model_paths:
        f = open(model_path_i +'/params.json')
        model_path_i_params = json.load(f)
        if(model_path_i_params["reduction"]==True and model_path_i_params["augmentation"]==False):
            symmetry_reduced_paths.append(model_path_i)
        elif(model_path_i_params["augmentation"]==True):
            continue
        else:
            default_paths.append(model_path_i)
    print("Default_paths:", default_paths, "Symmetry reduced paths: ", symmetry_reduced_paths)

    # load trained dynamics model
    for i in range(1):
        #for type, dynamics_model_path in zip(['symmetry', 'default'],[symmetry_reduced_paths[i], default_paths[i]]):
        #for type, dynamics_model_path in zip(['default', 'symmetry'],[default_paths[i], symmetry_reduced_paths[i]]):
        # use the same seeds for default and symmetric runs
        if args.combo_symmetry:
            type='symmetry'
            dynamics_model_path = symmetry_reduced_paths[i]
        else:
            type='default'
            dynamics_model_path = default_paths[i]
            
        train_episodes, test_episodes = train_test_split(dataset, random_state=seed+i, train_size=0.9)
        """if type == 'symmetry':
            #state_encoder_factory = encoders.SymmetryEncoderFactory(project=symmetry_project, projection_size=projection_size)
            dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=2e-4, use_gpu=use_gpu)
            dynamics.build_with_dataset(dataset)
        else:
            dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics.from_json(dynamics_model_path + '/params.json')"""
        dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics.from_json(dynamics_model_path + '/params.json')

        filenames = os.listdir(dynamics_model_path)
        latest_model_path = dynamics_model_path + '/model_' +  str(max([int(filename.strip('model_.pt')) for filename in filenames if filename.endswith(".pt")])) + '.pt'
        dynamics.load_model(latest_model_path)
        print("Loaded model: ", latest_model_path)
        
        #encoder = d3rlpy.models.encoders.DefaultEncoderFactory(dropout_rate=0.2)
        encoder = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[256, 256, 256], dropout_rate=0.1)
        # give COMBO as the generator argument.
        combo = COMBO(dynamics=dynamics, critic_encoder_factory=encoder, actor_encoder_factory=encoder, use_gpu=use_gpu, 
                      critic_learning_rate=0.0001, actor_learning_rate=0.00001,
                      rollout_horizon=1, real_ratio=0.8)
        combo.fit(dataset = train_episodes, eval_episodes=test_episodes, n_steps=500000, n_steps_per_epoch=1000,
                    tensorboard_dir="tensorboard_logs",
                    scorers={
                    'environment': d3rlpy.metrics.scorer.evaluate_on_environment(eval_env)
                },
                    experiment_name=save_name + "_" + type,
                    save_interval=10)

        # Save memory?
        combo = None; dynamics = None; train_episodes = None; test_episodes = None
        del combo; del dynamics; del train_episodes; del test_episodes


def main(args):
    # Read the YAML configuration file
    with open(args.cfg, 'r') as file:
        config = yaml.safe_load(file)
    augmentation = config['augmentation']
    reduction = config['reduction']
    state_permutation = np.array(config['state_permutation'])
    action_permutation = np.array(config['action_permutation'])
    env_name = config['env_name']
    file_path = config['file_path']
    EXP_NAME = 'exp_ddasdasdas' + env_name
    if args.COMBO is None:
        args.COMBO = False
    
    print(gym.version.VERSION)
    seed = 1
    if args.seed is not None:
        seed = int(args.seed)
    print('seed: ', seed)
    d3rlpy.seed(seed)
    use_gpu = True
        
    data_dict = read_hdf5_to_dict(file_path)
    # Use the same test episodes in each
    dataset = MDPDataset(data_dict['observations'], data_dict['actions'], data_dict['rewards'], np.logical_or(data_dict['terminals'], data_dict['timeouts']))
    train_episodes, test_episodes = train_test_split(dataset, random_state=seed, train_size=0.9)

    del data_dict

    if reduction == True or augmentation==True:
        permutation_indices = (state_permutation, action_permutation)
    else:
        permutation_indices = None

    small_encoder = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[256, 256], dropout_rate=0.2)
    large_encoder = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[200, 200, 200, 200], dropout_rate=0.2)

    dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=3e-4, use_gpu=use_gpu, n_ensembles=3, 
                                                             state_encoder_factory=large_encoder, reward_encoder_factory=small_encoder, 
                                                             permutation_indices=permutation_indices, augmentation=augmentation, 
                                                             reduction=reduction)
    # same as algorithms
    if args.dynamics:
        dynamics.fit(train_episodes,
                    eval_episodes=test_episodes,
                    n_steps=1000000,
                    n_steps_per_epoch=10000,
                    scorers={
                        'observation_error': d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
                        'reward_error': d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
                        'variance': d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
                    },
                    save_interval=10,
                    tensorboard_dir='tensorboard_logs/dynamics',
                    experiment_name= EXP_NAME + '_seed_' + str(seed) + '_reduction_'+str(reduction)+ '_augmentation_'+str(augmentation))
    if args.COMBO:
        # Run offline RL using COMBO
        env = gym.make(env_name)
        eval_env = gym.make(env_name)

        env.reset(seed=seed)
        eval_env.reset(seed=seed)
        experiment_COMBO_training(dataset, eval_env, EXP_NAME, save_name= 'COMBO_'+ EXP_NAME+ '_seed_' + str(seed), models_dir='d3rlpy_logs/', seed=seed, use_gpu=use_gpu)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Read configuration from a YAML file.")
    parser.add_argument("--cfg", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--seed", required=False)
    parser.add_argument("--COMBO", required=False, action='store_true', help="Whether to run offline RL instead of dynamics")
    parser.add_argument("--dynamics", required=False, action='store_true', help="Whether to run dynamics")
    parser.add_argument("--combo_symmetry", required=False, action='store_true', help="Whether to run offline RL instead of dynamics")

    # Parse arguments
    args = parser.parse_args()

    # Pass the value to the main function
    main(args)