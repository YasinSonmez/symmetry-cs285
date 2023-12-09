import torch
from .mpc_config import mpc_config
import numpy as np
import gym
from gym.envs.mujoco.reacher_v4 import ReacherEnv
from gym.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv

class InvertedPendulumWrapper(InvertedPendulumEnv):

    goal_pos = 1.0

    def step(self, action):
        ob, _reward, _terminated, truncated, info = super().step(action)

        reward, _ = self.get_reward(ob, action)

        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > 0.79))

        return ob, reward, terminated, truncated, info
    
    def get_reward(self, observations, actions):
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, axis=0)
            actions = np.expand_dims(actions, axis=0)
            batch_mode = False
        else:
            batch_mode = True

        xs = observations[:, 0]
        dists = xs - self.goal_pos

        b = 2
        c = 1.5
        reward_pos = b/(np.power(c, dists) + np.power(c, -dists))

        angs = observations[:, 1]
        reward_upright = b/(np.power(c, angs) + np.power(c, -angs))

        rewards = reward_pos + reward_upright

        dones = np.zeros((observations.shape[0],))

        if not batch_mode:
            return rewards[0], dones[0]
        return rewards, dones
    
def inverted_pendulum_symmetry(
    env_name,
    exp_name,
    use_projector = False,
    **kwargs
):
    print("Using inverted pendulum symmetry config")
    config = mpc_config(env_name, exp_name, **kwargs)


    def make_env(render: bool = False):
        assert env_name == "InvertedPendulum-v4"
        env = InvertedPendulumWrapper(render_mode="single_rgb_array" if render else None)
        env = gym.wrappers.StepAPICompatibility(env, new_step_api=False)
        return env

    def projector(obs):
        """Takes in a batch of obs and returns a batch of reduced observations."""
        # Input size (batch_size, ob_dim)
        # Output size (batch_size, reduced_size)
        # Remove the position of the cart (state 0)
        if obs.ndim == 2:
            return obs[:, [1, 2, 3]]
        elif obs.ndim == 1:
            return obs[[1, 2, 3]]
        else:
            assert False

    reduced_size = 3

    config["make_env"] = make_env
    config["ep_len"] = 200
    if use_projector:
        config["agent_kwargs"]["projector"] = projector
        config["agent_kwargs"]["reduced_size"] = reduced_size
    else:
        config["agent_kwargs"]["projector"] = None
        config["agent_kwargs"]["reduced_size"] = None 

    return config

class ReacherWrapper(ReacherEnv):

    def step(self, action):
        ob, _reward, x, y, info = super().step(action)

        reward, _ = self.get_reward(ob, action)

        return ob, reward, x, y, info

    def get_reward(self, observations, actions):
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, axis=0)
            actions = np.expand_dims(actions, axis=0)
            batch_mode = False
        else:
            batch_mode = True

        # Get the states corresponding to distances between finger tip and target.
        if observations.shape[1] == 6:
            # If calling outside of env, after observation transformation wrapper
            theta1 = observations[:, 0]
            theta2 = observations[:, 1]
            target = observations[:, [4, 5]]
            # dists = observations[:, [6, 7]]
        elif observations.shape[1] == 11:
            # If calling inside env, before observation transformation wrapper
            theta1 = np.arctan2(observations[:, 2], observations[:, 0])
            theta2 = np.arctan2(observations[:, 3], observations[:, 1])
            target = observations[:, [4, 5]]
            # dists = observations[:, [8, 9]]
        else:
            assert False, f"invalid ob shape: {observations.shape}"

        # I believe these are the lengths of the arms based on the xml file.
        l1 = 0.1 
        l2 = 0.1
        fingertip_pos = np.hstack((
            np.reshape(l1*np.cos(theta1) + l2*np.cos(theta1 + theta2), (-1, 1)),
            np.reshape(l1*np.sin(theta1) + l2*np.sin(theta1 + theta2), (-1, 1))
        ))

        dists = fingertip_pos - target

        rewards_dist = -np.linalg.norm(dists, axis=1)
        rewards_ctrl = -np.square(actions).sum(axis=1)
        rewards = rewards_dist + rewards_ctrl

        dones = np.zeros((observations.shape[0],))

        if not batch_mode:
            return rewards[0], dones[0]
        return rewards, dones

def reacher_symmetry(
    env_name,
    exp_name,
    use_projector = False,
    **kwargs
):
    print("Using reacher symmetry config")
    config = mpc_config(env_name, exp_name, **kwargs)

    def observation_edit(obs):
        assert len(obs) == 11
        # new_obs = np.zeros(8)
        # new_obs[0] = np.arctan2(obs[2], obs[0])
        # new_obs[1] = np.arctan2(obs[3], obs[1])
        # new_obs[2:] = obs[4:-1]
        new_obs = np.zeros(6)
        new_obs[0] = np.arctan2(obs[2], obs[0]) # Angle of first arm
        new_obs[1] = np.arctan2(obs[3], obs[1]) # Angle of second arm
        new_obs[2] = obs[6] # Angular velocity of first arm
        new_obs[3] = obs[7] # Angular velocity of second arm
        new_obs[4] = obs[4] # x-coordinate of target
        new_obs[5] = obs[5] # y-coordinate of target
        return new_obs

    def make_env(render: bool = False):
        assert env_name == "Reacher-v4"
        env = ReacherWrapper() # gym.make(env_name)
        new_env = gym.wrappers.TransformObservation(env, observation_edit)
        new_env.observation_space = gym.spaces.Box(
            # -np.inf, np.inf, shape=(8,), dtype= np.float64
            -np.inf, np.inf, shape=(6,), dtype= np.float64
        )
        # TODO: make a reward function, transform the env to use it, and 
        # add a get_reward function to the env
        new_env = gym.wrappers.StepAPICompatibility(new_env, new_step_api=False)
        return new_env

    def projector(obs):
        """Takes in a batch of obs and returns a batch of reduced observations."""
        # Input size (batch_size, ob_dim)
        # Output size (batch_size, reduced_size)
        # return obs[:, [1, 4, 5]]
        if obs.ndim == 2:
            return obs[:, [1, 2, 3]] # angle of second arm and both angular velocities
        elif obs.ndim == 1:
            return obs[[1, 2, 3]]
        else:
            assert False

    reduced_size = 3

    config["make_env"] = make_env
    config["ep_len"] = 200
    if use_projector:
        config["agent_kwargs"]["projector"] = projector
        config["agent_kwargs"]["reduced_size"] = reduced_size
    else:
        config["agent_kwargs"]["projector"] = None
        config["agent_kwargs"]["reduced_size"] = None 

    return config