import d3rlpy
from d3rlpy.algos import COMBO
from sklearn.model_selection import train_test_split
#import gymnasium as gym
import gym
from gym.wrappers import TransformObservation
import numpy as np
import encoders
import os
import json

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

def observation_edit1(obs):
    new_obs = np.zeros(8)
    new_obs[0] = np.arccos(obs[0])
    new_obs[1] = np.arccos(obs[1])
    new_obs[2:] = obs[4:-1]
    return new_obs

env1 = TransformObservation(env, observation_edit1)
env1.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype= np.float64 )
print(env1.reset(seed=seed))

eval_env1 = TransformObservation(eval_env, observation_edit1)
eval_env1.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype= np.float64 )
print(env1.reset(seed=seed))

dataset = d3rlpy.dataset.MDPDataset.load('d3rlpy_data/reacherv2.h5')

def inverted_pendulum_project(x):
    return x[:, 1:]
projection_size = 3

def reacher_project(x):
    return x[:, [1,4,5]]
projection_size = 3

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
            combo.fit(dataset = train_episodes, eval_episodes=test_episodes, n_steps=1000000, n_steps_per_epoch=1000,
                      tensorboard_dir="tensorboard_logs",
                     scorers={
                        'environment': d3rlpy.metrics.scorer.evaluate_on_environment(eval_env)
                    },
                     experiment_name=save_name + "_" + type,
                     save_interval=50)

experiment_COMBO_training(dataset, eval_env1, 'exp_5', save_name='exp_5_COMBO_reacher', models_dir='d3rlpy_logs/', symmetry_project=reacher_project, projection_size=3, seed=1, use_gpu=True)