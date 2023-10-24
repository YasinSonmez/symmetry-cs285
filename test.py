import d3rlpy
from d3rlpy.algos import COMBO
from sklearn.model_selection import train_test_split
import gym
from gym.wrappers import TransformObservation
import numpy as np
import encoders
import os
import json

def inverted_pendulum_project(x):
    return x[:, 1:]

def reacher_project(x):
    return x[:, [1,4,5]]

def observation_edit1(obs):
    new_obs = np.zeros(8)
    new_obs[0] = np.arccos(obs[0])
    new_obs[1] = np.arccos(obs[1])
    new_obs[2:] = obs[4:-1]
    return new_obs

def experiment_dynamics_training(dataset, symmetry_project, projection_size, n_runs, experiment_name, seed=1, use_gpu=True):
    for i in range(n_runs):
        for exp_type in ['symmetry', 'default']:
            # use the same seeds for default and symmetric runs
            train_episodes, test_episodes = train_test_split(dataset, random_state=seed+i)
            if exp_type == 'symmetry':
                state_encoder_factory = encoders.SymmetryEncoderFactory(project=symmetry_project, projection_size=projection_size)
                dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=use_gpu, state_encoder_factory=state_encoder_factory)
            else:
                dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=use_gpu)
            dynamics.fit(train_episodes,
                 eval_episodes=test_episodes,
                 n_epochs=1,
                 scorers={
                    'observation_error': d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
                    'reward_error': d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
                    'variance': d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
                 },
                tensorboard_dir='tensorboard_logs/dynamics',
                experiment_name=experiment_name + '_' + exp_type,
                save_interval=10)

            # Save memory?
            dynamics = None; train_episodes = None; test_episodes = None
            del dynamics; del train_episodes; del test_episodes

def experiment_COMBO_training(dataset, eval_env, experiment_name, save_name, models_dir, symmetry_project, projection_size, seed=1, use_gpu=True):
    model_paths = [filename for filename in os.listdir(models_dir) if filename.startswith(experiment_name+'_dynamics')]
    model_paths = [models_dir + model_paths_i for model_paths_i in model_paths]
    model_paths.sort()
    print(model_paths)

    symmetry_reduced_paths = []
    default_paths = []
    for model_path_i in model_paths:
        f = open(model_path_i +'/params.json')
        model_path_i_params = json.load(f)
        if(model_path_i_params["state_encoder_factory"]['type']=='symmetry'):
            symmetry_reduced_paths.append(model_path_i)
        elif(model_path_i_params["state_encoder_factory"]['type']=='default'):
            default_paths.append(model_path_i)
    print("Default_paths:", default_paths, "Symmetry reduced paths: ", symmetry_reduced_paths)

    # load trained dynamics model
    for i in range(len(default_paths)):
        for type, dynamics_model_path in zip(['symmetry', 'default'],[symmetry_reduced_paths[i], default_paths[i]]):
        #for type, dynamics_model_path in zip(['default', 'symmetry'],[default_paths[i], symmetry_reduced_paths[i]]):
            # use the same seeds for default and symmetric runs
            train_episodes, test_episodes = train_test_split(dataset, random_state=seed+i)
            if type == 'symmetry':
                state_encoder_factory = encoders.SymmetryEncoderFactory(project=symmetry_project, projection_size=projection_size)
                dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=use_gpu, state_encoder_factory=state_encoder_factory)
                dynamics.build_with_dataset(dataset)
            else:
                dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics.from_json(dynamics_model_path + '/params.json')

            filenames = os.listdir(dynamics_model_path)
            latest_model_path = dynamics_model_path + '/model_' +  str(max([int(filename.strip('model_.pt')) for filename in filenames if filename.endswith(".pt")])) + '.pt'
            dynamics.load_model(latest_model_path)
            print("Loaded model: ", latest_model_path)
            
            encoder = d3rlpy.models.encoders.DefaultEncoderFactory(dropout_rate=0.2)
            # give COMBO as the generator argument.
            combo = COMBO(dynamics=dynamics, critic_encoder_factory=encoder, actor_encoder_factory=encoder, use_gpu=use_gpu,
                    actor_learning_rate=0.00003, critic_learning_rate=0.0001, conservative_weight=5)
            combo.fit(dataset = train_episodes, eval_episodes=test_episodes, n_steps=10, n_steps_per_epoch=10,
                      tensorboard_dir="tensorboard_logs",
                     scorers={
                        'environment': d3rlpy.metrics.scorer.evaluate_on_environment(eval_env)
                    },
                     experiment_name=save_name + "_" + type,
                     save_interval=10)

            # Save memory?
            combo = None; dynamics = None; train_episodes = None; test_episodes = None
            del combo; del dynamics; del train_episodes; del test_episodes

# Parameters
seed = 1
d3rlpy.seed(seed)
use_gpu = True

# prepare environment
#env = gym.make("InvertedPendulum-v2")
#eval_env = gym.make("InvertedPendulum-v2")
env = gym.make("Reacher-v2")
eval_env = gym.make("Reacher-v2")
env.reset(seed=seed)
eval_env.reset(seed=seed)

env1 = TransformObservation(env, observation_edit1)
env1.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype= np.float64 )
print(env1.reset(seed=seed))

eval_env1 = TransformObservation(eval_env, observation_edit1)
eval_env1.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype= np.float64 )
print(env1.reset(seed=seed))

# Dataset
dataset = d3rlpy.dataset.MDPDataset.load('d3rlpy_data/reacherv2_atan2.h5')

experiment_dynamics_training(dataset=dataset, symmetry_project=reacher_project, projection_size=3, n_runs=3, experiment_name="exp_6_dynamics_reacher", use_gpu=True)
experiment_COMBO_training(dataset, eval_env1, 'exp_6', save_name='exp_6_COMBO_reacher', models_dir='d3rlpy_logs/', symmetry_project=reacher_project, projection_size=3, seed=1, use_gpu=True)