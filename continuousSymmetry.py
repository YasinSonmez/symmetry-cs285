import d3rlpy
from d3rlpy.algos import COMBO
from sklearn.model_selection import train_test_split
import gym
from gym.wrappers import TransformObservation
import numpy as np
import encoders
import os
import json
import yaml
import argparse

def inverted_pendulum_project(x):
    return x[:, 1:]

def reacher_project(x):
    return x[:, [1,4,5]]

def observation_edit1(obs):
    new_obs = np.zeros(6)
    new_obs[0] = np.arctan2(obs[2], obs[0])
    new_obs[1] = np.arctan2(obs[3], obs[1])
    new_obs[2:] = obs[4:-3]
    return new_obs

def main(args):
    # Read the YAML configuration file
    with open(args.cfg, 'r') as file:
        config = yaml.safe_load(file)
    reduction = config['reduction']
    env_name = config['env_name']
    file_path = config['file_path']
    EXP_NAME = 'exp_13_' + env_name
    if args.COMBO is None:
        args.COMBO = False
    
    print(gym.version.VERSION)
    seed = 1
    if args.seed is not None:
        seed = int(args.seed)
    print('seed: ', seed)
    d3rlpy.seed(seed)
    use_gpu = True

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed)

    env1 = TransformObservation(env, observation_edit1)
    env1.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype= np.float64 )
    print(env1.reset(seed=seed))

    eval_env1 = TransformObservation(eval_env, observation_edit1)
    eval_env1.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype= np.float64 )
    print(env1.reset(seed=seed))

    # Dataset
    dataset = d3rlpy.dataset.MDPDataset.load(file_path)
        
    # Use the same test episodes in each
    train_episodes, test_episodes = train_test_split(dataset, random_state=seed, train_size=0.9)

    small_encoder = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[256, 256], dropout_rate=0.2, activation='swish')
    large_encoder = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[200,200, 200], dropout_rate=0.2, activation='swish')
    dynamics_optim = d3rlpy.models.optimizers.AdamFactory(weight_decay=2.5e-5)

    if reduction == True:
        symmetry_large_encoder = encoders.SymmetryEncoderFactory(project=reacher_project, projection_size=3, hidden_units=[200,200, 200], dropout_rate=0.2, activation='swish')
        dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=3e-4, use_gpu=use_gpu, 
                                                                 state_encoder_factory=symmetry_large_encoder, 
                                                                 reward_encoder_factory=small_encoder, 
                                                                 n_ensembles=3,
                                                                 optim_factory=dynamics_optim,
                                                                 )
    else:
        dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=3e-4, use_gpu=use_gpu, 
                                                                 state_encoder_factory=large_encoder, 
                                                                 reward_encoder_factory=small_encoder, 
                                                                 n_ensembles=3,
                                                                 optim_factory=dynamics_optim,
                                                                 )
    dynamics.fit(train_episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=5000,
            scorers={
            'observation_error': d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
            'reward_error': d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
            'variance': d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
            },
        tensorboard_dir='tensorboard_logs/dynamics',
        experiment_name=EXP_NAME + '_seed_' + str(seed) + '_reduction_'+str(reduction),
        save_interval=10)

    if args.COMBO:
        print('Combo')
        # Run offline RL using COMBO
        #experiment_COMBO_training(dataset, eval_env, EXP_NAME, save_name= 'COMBO_'+ EXP_NAME+ '_seed_' + str(seed), models_dir='d3rlpy_logs/', seed=seed, use_gpu=use_gpu)

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