import json
import os

# import gymnasium as gym
import gym
import numpy as np
from gym.wrappers import TimeLimit, TransformObservation
from sklearn.model_selection import train_test_split

import d3rlpy
import encoders
import environments
from d3rlpy.algos import COMBO

print(gym.version.VERSION)

seed = 0
d3rlpy.seed(seed)
use_gpu = True

env = TimeLimit(environments.RewardAssymetricInvertedPendulum(), max_episode_steps=300)
eval_env = TimeLimit(
    environments.RewardAssymetricInvertedPendulum(), max_episode_steps=300
)

env.reset(seed=seed)
eval_env.reset(seed=seed)

actor_encoder = d3rlpy.models.encoders.DefaultEncoderFactory(dropout_rate=0.2)
# setup algorithm
sac = d3rlpy.algos.SAC(
    batch_size=256,
    actor_encoder_factory=actor_encoder,
    actor_learning_rate=1e-3,
    critic_learning_rate=1e-3,
    temp_learning_rate=1e-3,
    use_gpu=use_gpu,
)

# prepare utilities
buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=10000, env=env)

# start training
sac.fit_online(
    env,
    buffer,
    eval_env=eval_env,
    n_steps=100000,
    n_steps_per_epoch=1000,
    update_interval=2,
    update_start_step=1000,
    tensorboard_dir="tensorboard_logs",
    experiment_name="test7",
)

dataset = buffer.to_mdp_dataset()
dataset.dump("d3rlpy_data/TEST.h5")

scorer = d3rlpy.metrics.scorer.evaluate_on_environment(eval_env, render=True)
mean_episode_return = scorer(sac)

# 5: goal 0, lr 3e-4, update interval 2
# 6: goal 1, lr 3e-4, update interval 2
# 7: goal 1, lr 1e-3, update interval 2, time limit 1000
