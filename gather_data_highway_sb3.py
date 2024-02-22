# Environment
import gymnasium as gym
import highway_env
import pprint

# Agent
from stable_baselines3 import SAC

import sys

sys.path.insert(0, "/HighwayEnv/scripts/")

from utils import record_videos, show_videos

# Suppress warning
import warnings


# Define a custom warning filter
def custom_filter(message, category, filename, lineno, file=None, line=None):
    return None


# Add the custom filter
warnings.showwarning = custom_filter

import numpy as np
from collections import OrderedDict

class TwoCarScenarioWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Correctly adapt the observation space
        self.observation_space = self._adapt_observation_space(env.observation_space)
        self.action_space = self._adapt_action_space(env.action_space)
    
    def _adapt_action_space(self, action_space):
        # Assuming the action space is identical for both vehicles
        low = np.concatenate([action_space.low, action_space.low])
        high = np.concatenate([action_space.high, action_space.high])
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _adapt_observation_space(self, observation_space):
        # Correctly adapt the observation space for the combined scenario
        print(observation_space)
        adapted_spaces = {}
        for key in observation_space.keys():  # Assuming the first element of the tuple for structure
            low = np.concatenate([observation_space[key].low, observation_space[key].low])
            high = np.concatenate([observation_space[key].high, observation_space[key].high])
            adapted_spaces[key] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return gym.spaces.Dict(adapted_spaces)

    def step(self, action):
        # Split and pass actions, adapt observations
        action1, action2 = np.split(action, 2)
        # print(action, action1, action2)
        # print(self.env.step((action1, action2)))
        observation, reward, termination, done, info = self.env.step((action1, action2))
        new_observation = self._adapt_observation(observation)
        return new_observation, reward, termination, done, info

    def reset(self, **kwargs):
        # Reset and adapt initial observation
        observation, info = self.env.reset(**kwargs)
        new_observation = self._adapt_observation(observation)
        return new_observation, info

    def _adapt_observation(self, observation):
        # Assuming observation is a tuple of tuples, where each inner tuple corresponds to
        # the observations from each environment. This is typical with environments wrapped by DummyVecEnv.
        
        # if isinstance(observation, tuple):
        # Process each element in the tuple to adapt the observation
        # print(obs)
        # print({type(obs)})
        # Now, assuming each obs is a dictionary with the structure we expect
        adapted_obs = {}
        # print(observation)
        # for key, value in obs.items():
        #     # Assuming value is a NumPy array; adjust logic if it's not
        #     # Here, you might concatenate, stack, or otherwise combine the observations
        #     # For simplicity, this example just collects the values directly
        #     adapted_obs[key] = value  # This might be a direct assignment, concatenation, etc., depending on your needs
        # new_observations.append(adapted_obs)
        # new_observations.append(new_observation)

        for key in observation[0].keys():  # Assuming the first element of the tuple for structure
            adapted_obs[key] = np.concatenate([observation[0][key], observation[1][key]])
        
        # Convert the list of adapted observations back into a tuple or another suitable structure
        # print(adapted_obs)
        return OrderedDict(adapted_obs)
        # else:
        #     # Handle non-tuple observations or raise an error
        #     raise TypeError(f"Unsupported observation structure: {type(observation)}")
# Assume `env` is your two-vehicle Gym environment
# wrapped_env = TwoCarScenarioWrapper(env)

# print(gym.version.VERSION)

seed = 42

# @title Training

LEARNING_STEPS = 100  # 5e4  # @param {type: "number"}
# Set the device to 'cpu'
# device = 'cpu'
# print('device: ', device )
env = gym.make("parking-v0")
# her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')

CONFIGURE = True

if CONFIGURE:
    # custom_config = {
    #     "controlled_vehicles": 2,
    #     "action": {
    #         "type": "MultiAgentAction",
    #         "action_config": {"type": "ContinuousAction"},
    #     },
    #     "observation": {
    #         "type": "MultiAgentObservation",
    #         "observation_config": {
    #             "type": "Kinematics",
    #             "vehicles_count": 4,
    #             "features": ["x", "y", "vx", "vy", "heading"],
    #             "features_range": {
    #                 "x": [-100, 100],
    #                 "y": [-100, 100],
    #                 "vx": [-20, 20],
    #                 "vy": [-20, 20],
    #             },
    #             "absolute": False,
    #             "order": "sorted",
    #         },
    #     },
    #     "render_mode": "rgb_array",
    # }
    # custom_config = {
    #     "observation": {
    #         "features": ["x", "y", "vx", "vy", "heading"],
    #         "normalize": False,
    #         "scales": [100, 100, 5, 5, 1],
    #         "type": "KinematicsGoal",
    #     }
    # }
    # env = gym.make("parking-v0", config=custom_config)
    env.configure(
        {
            "controlled_vehicles": 2,
            'show_trajectories': True,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "normalize": False,
                    "scales": [100, 100, 5, 5, 1, 1],
                    "type": "KinematicsGoal",
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "ContinuousAction",
                },
            },
        }
    )
pprint.pprint(env.config)
wrapped_env = TwoCarScenarioWrapper(env)


model = SAC(
    "MultiInputPolicy",
    wrapped_env,
    # replay_buffer_class=HerReplayBuffer,
    # replay_buffer_kwargs=her_kwargs,
    verbose=1,
    tensorboard_log="tensorboard_logs/parking",
    buffer_size=int(1e6),
    learning_rate=1e-3,
    gamma=0.95,
    batch_size=1024,
    tau=0.05,
    policy_kwargs=dict(net_arch=[512, 512, 512]),
    # device=device
)

model.learn(int(LEARNING_STEPS))
# model.save("sac_parking_params")
# model.save_replay_buffer("sac_parking_replay_buffer")

# model.load("sac_parking_params")

N_EPISODES = 2  # @param {type: "integer"}

env = gym.make("parking-v0", render_mode="rgb_array")
env.configure(
        {
            "controlled_vehicles": 2,
            'show_trajectories': True,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "normalize": False,
                    "scales": [100, 100, 5, 5, 1, 1],
                    "type": "KinematicsGoal",
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "ContinuousAction",
                },
            },
        }
    )
wrapped_env = TwoCarScenarioWrapper(env)
env = record_videos(wrapped_env)
for episode in range(N_EPISODES):
    obs, info = env.reset()
    print(obs)
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
env.close()
show_videos()
