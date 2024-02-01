# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)


def cartpole_pets(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    x0 = next_obs[:, :1]
    theta = next_obs[:, 1:2]
    ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6**2))
    act_cost = -0.01 * torch.sum(act**2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act**2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)


## Custom -- Neelay


def reward_asymmetric_inverted_pendulum(
    act: torch.Tensor, next_obs: torch.Tensor
) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    goal_pos = 1.0
    x = next_obs[:, :1]
    theta = next_obs[:, 1:2]

    pos_reward = torch.exp(-((x - goal_pos) ** 2))
    upright_reward = torch.exp(-(theta**2))
    reward = pos_reward + upright_reward

    return reward.view(-1, 1)


def reacher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    assert next_obs.shape[1] == 6  # After observation_edit
    l1 = 0.1
    l2 = 0.1

    # Estimate fingertip position
    theta1 = next_obs[:, 0:1]
    theta2 = next_obs[:, 1:2]
    fingertipx = l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
    fingertipy = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)

    targetx = next_obs[:, 4:5]
    targety = next_obs[:, 5:6]

    dist_squared = (fingertipx - targetx) ** 2 + (fingertipy - targety) ** 2
    reward_pos = torch.exp(-dist_squared)

    reward_ctrl = 0.5 * torch.exp(-act.square().sum(axis=1))

    reward = reward_pos + reward_ctrl

    return reward.view(-1, 1)

def car(_act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    reward = torch.exp(-next_obs.square().sum(axis=1))
    return reward.view(-1, 1)