# Environment
import gymnasium as gym
import highway_env
import pprint
import numpy as np
from collections import OrderedDict
# Agent
from stable_baselines3 import HerReplayBuffer, SAC
import torch
import sys
from utils import record_videos, show_videos
# Suppress warning
import warnings
# Define a custom warning filter
def custom_filter(message, category, filename, lineno, file=None, line=None):
    return None
# Add the custom filter
warnings.showwarning = custom_filter

sys.path.insert(0, "/HighwayEnv/scripts/")

class TwoCarScenarioWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Correctly adapt the observation space
        self.observation_space = self._adapt_observation_space(env.observation_space)
        self.action_space = self._adapt_action_space(env.action_space)
        self.env = env
    
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
        observation, reward, termination, done, info = self.env.step((action1, action2))
        observation[1]['desired_goal'] = self.second_goal.copy()

        # Modify reward
        reward += sum(
            self.compute_reward(
                agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
            )
            for agent_obs in observation
        )

        # Modify termination
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in observation
        )
        
        new_observation = self._adapt_observation(observation)

        return new_observation, reward, bool(termination or success), done, info
    
    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.env.config["success_goal_reward"]
        )
    
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.env.config["reward_weights"]),
            ),
            p,
        )

    def reset(self, **kwargs):
        # Reset and adapt initial observation
        observation2, info = self.env.reset(**kwargs)
        # Extract seed from kwargs if it exists
        seed = kwargs.pop('seed', None)
        # If seed is provided, increase it by 1
        if seed is not None:
            seed += 1
        else:
            seed = np.random.randint(0, 10000)
        # Update kwargs with the increased seed
        kwargs['seed'] = seed
        observation, _ = self.env.reset(**kwargs)
        self.second_goal = observation2[1]['desired_goal'].copy()
        observation[1]['desired_goal'] = self.second_goal.copy()
        new_observation = self._adapt_observation(observation)
        return new_observation, info

    def _adapt_observation(self, observation):
        # Assuming observation is a tuple of tuples, where each inner tuple corresponds to
        # the observations from each environment. This is typical with environments wrapped by DummyVecEnv.
        adapted_obs = {}

        for key in observation[0].keys():  # Assuming the first element of the tuple for structure
            adapted_obs[key] = np.concatenate([observation[0][key], observation[1][key]])

        return OrderedDict(adapted_obs)
seed = 1

# @title Training

LEARNING_STEPS = 1e6  # 5e4  # @param {type: "number"}
env = gym.make("parking-v0")
#env = gym.make("parking_many_cars-v0")
# her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')

CONFIGURE = True

if CONFIGURE:
    env.configure(
        {
            "controlled_vehicles": 2,
            "collision_reward": -5,
            'show_trajectories': True,
            "screen_width": 600,
            "screen_height": 600,
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

policy_kwargs = {
    'net_arch': [512, 512, 512],               # Two hidden layers with 64 units each
    'activation_fn': torch.nn.GELU,            # GELU activation function
    'n_critics': 2,    
    'use_sde': True,                    # Use State Dependent Exploration
    'use_expln': True,                  # Use Exponential Linear Units (ELUs) for activation
}

model = SAC(
    "MultiInputPolicy",
    wrapped_env,
    # replay_buffer_class=HerReplayBuffer,
    # replay_buffer_kwargs=her_kwargs,
    verbose=1,
    tensorboard_log="tensorboard_logs/2_car_parking/modified_reward_modified_termination2",
    buffer_size=int(1e6),
    learning_rate=1e-3,
    gamma=0.95,
    batch_size=1024,
    tau=0.05,
    # policy_kwargs=dict(net_arch=[512, 512]),
    policy_kwargs=policy_kwargs,
    # device='cpu'
)
model.learn(int(LEARNING_STEPS))
model.save("sac_parking_params_4_car_reset_modified_reward_modified_termination2_fancy_network")
model.save_replay_buffer("sac_parking_replay_buffer_4_car_reset_modified_reward_modified_termination2_fancy_network")

N_EPISODES = 10  # @param {type: "integer"}

env = gym.make("parking-v0", render_mode="rgb_array")
env.configure(
        {
            "controlled_vehicles": 2,
            "collision_reward": -5,
            'show_trajectories': True,
            "screen_width": 600,
            "screen_height": 600,
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
env = record_videos(wrapped_env, video_folder='videos_fancy_network_modified_termination2')
for episode in range(N_EPISODES):
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(obs)
env.close()
show_videos()
