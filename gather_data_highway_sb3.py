# Environment
import gymnasium as gym
import highway_env
import pprint
# Agent
from stable_baselines3 import SAC

import sys
sys.path.insert(0, '/HighwayEnv/scripts/')

from utils import record_videos, show_videos

# Suppress warning
import warnings

# Define a custom warning filter
def custom_filter(message, category, filename, lineno, file=None, line=None):
    return None

# Add the custom filter
warnings.showwarning = custom_filter

#print(gym.version.VERSION)

seed = 42

#@title Training

LEARNING_STEPS = 5e4 # @param {type: "number"}
# Set the device to 'cpu'
#device = 'cpu'
#print('device: ', device )
env = gym.make('parking-v0')
#her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')

CONFIGURE=False

if CONFIGURE:
  env.configure({
      "controlled_vehicles": 2,
      'show_trajectories': True,
      "observation": {
          "type": "MultiAgentObservation",
          "observation_config": {'features': ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                                  'normalize': False,
                                  'scales': [100, 100, 5, 5, 1, 1],
                                  'type': 'KinematicsGoal'}
          },
      "action": {
      "type": "MultiAgentAction",
      "action_config": {
        "type": "ContinuousAction",
      }
    }
  })
pprint.pprint(env.config)

model = SAC('MultiInputPolicy', 
            env,
            #replay_buffer_class=HerReplayBuffer,
            #replay_buffer_kwargs=her_kwargs,
            verbose=1,
            tensorboard_log="tensorboard_logs/parking",
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=1024, tau=0.05,
            policy_kwargs=dict(net_arch=[512, 512, 512]),
            #device=device
            )

model.learn(int(LEARNING_STEPS))
model.save("sac_parking_params")
model.save_replay_buffer('sac_parking_replay_buffer')

N_EPISODES = 10  # @param {type: "integer"}

env = gym.make('parking-v0', render_mode='rgb_array')
env = record_videos(env)
for episode in N_EPISODES:
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
env.close()
show_videos()