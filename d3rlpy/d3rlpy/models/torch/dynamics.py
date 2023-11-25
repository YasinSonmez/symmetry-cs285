# pylint: disable=protected-access

from typing import List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import spectral_norm

from .encoders import EncoderWithAction
import numpy as np

def _compute_ensemble_variance(
    observations: torch.Tensor,
    rewards: torch.Tensor,
    variances: torch.Tensor,
    variance_type: str,
) -> torch.Tensor:
    if variance_type == "max":
        return variances.max(dim=1).values
    elif variance_type == "data":
        data = torch.cat([observations, rewards], dim=2)
        return (data.std(dim=1) ** 2).sum(dim=1, keepdim=True)
    raise ValueError(f"invalid variance_type: {variance_type}")


def _apply_spectral_norm_recursively(model: nn.Module) -> None:
    for _, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for m in module:
                _apply_spectral_norm_recursively(m)
        else:
            if "weight" in module._parameters:
                spectral_norm(module)


def _gaussian_likelihood(
    x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor
) -> torch.Tensor:
    inv_var = torch.exp(-2.0 * logstd)
    return 0.5 * (((mu - x) ** 2) * inv_var).sum(dim=1, keepdim=True)


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

    perm_matrix = torch.from_numpy(perm_matrix).to(torch.float)
    perm_matrix.requires_grad = False
    return perm_matrix

class ProbabilisticDynamicsModel(nn.Module):  # type: ignore
    """Probabilistic dynamics model.

    References:
        * `Janner et al., When to Trust Your Model: Model-Based Policy
          Optimization. <https://arxiv.org/abs/1906.08253>`_
        * `Chua et al., Deep Reinforcement Learning in a Handful of Trials
          using Probabilistic Dynamics Models.
          <https://arxiv.org/abs/1805.12114>`_

    """

    _encoder: EncoderWithAction
    _mu: nn.Linear
    _logstd: nn.Linear
    _max_logstd: nn.Parameter
    _min_logstd: nn.Parameter

    def __init__(self, state_encoder: EncoderWithAction, 
                reward_encoder: EncoderWithAction,
                permutation_indices = None,
                augmentation = None,
                reduction = None,
                ):
        super().__init__()
        # apply spectral normalization except logstd encoder.
        _apply_spectral_norm_recursively(cast(nn.Module, state_encoder))
        _apply_spectral_norm_recursively(cast(nn.Module, reward_encoder))
        self._state_encoder = state_encoder
        self._reward_encoder = reward_encoder
        self._permutation_indices = permutation_indices
        self._augmentation = augmentation
        self._reduction = reduction

        state_feature_size = state_encoder.get_feature_size() # TODO: do I need to enforce feature and observation size are common between the state and reward encoders?
        reward_feature_size = reward_encoder.get_feature_size()
        observation_size = state_encoder.observation_shape[0]
        action_size = state_encoder.action_size
        if permutation_indices is not None:
            self._P_s = create_permutation_matrix(observation_size, self._permutation_indices[0][0], self._permutation_indices[0][1])
            self._P_a = create_permutation_matrix(action_size, self._permutation_indices[1][0], self._permutation_indices[1][1])
        else:
            self._P_s = None
            self._P_a = None
        # out_size = observation_size + 1
        out_size = observation_size

        # TODO: handle image observation
        self._state_mu = spectral_norm(nn.Linear(state_feature_size, out_size))
        self._state_logstd = nn.Linear(state_feature_size, out_size)

        self._reward_mu = spectral_norm(nn.Linear(reward_feature_size, 1))
        self._reward_logstd = nn.Linear(reward_feature_size, 1)

        # logstd bounds
        state_init_max = torch.empty(1, out_size, dtype=torch.float32).fill_(2.0)
        state_init_min = torch.empty(1, out_size, dtype=torch.float32).fill_(-10.0)
        self._state_max_logstd = nn.Parameter(state_init_max)
        self._state_min_logstd = nn.Parameter(state_init_min)

        reward_init_max = torch.empty(1, 1, dtype=torch.float32).fill_(2.0)
        reward_init_min = torch.empty(1, 1, dtype=torch.float32).fill_(-10.0)
        self._reward_max_logstd = nn.Parameter(reward_init_max)
        self._reward_min_logstd = nn.Parameter(reward_init_min)

    def compute_stats(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state_h = self._state_encoder(x, action)
        reward_h = self._reward_encoder(x, action)

        state_mu = self._state_mu(state_h)
        reward_mu = self._reward_mu(reward_h)

        # log standard deviation with bounds
        state_logstd = self._state_logstd(state_h)
        state_logstd = self._state_max_logstd - F.softplus(self._state_max_logstd - state_logstd)
        state_logstd = self._state_min_logstd + F.softplus(state_logstd - self._state_min_logstd)

        reward_logstd = self._reward_logstd(reward_h)
        reward_logstd = self._reward_max_logstd - F.softplus(self._reward_max_logstd - reward_logstd)
        reward_logstd = self._reward_min_logstd + F.softplus(reward_logstd - self._reward_min_logstd)

        # return mu, logstd
        return state_mu, reward_mu, state_logstd, reward_logstd

    def forward(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_with_variance(x, action)[:2]

    def predict_with_variance(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # mu, logstd = self.compute_stats(x, action)
        state_mu, reward_mu, state_logstd, reward_logstd = self.compute_stats(x, action)
        if self._P_s is not None:
            # transform before input
            state_mu2, _, state_logstd2, _ = self.compute_stats(x@self._P_s.T, action@self._P_a.T)
            # reverse transform after input
            state_mu2 = state_mu2@self._P_s.T
            state_logstd2 = state_logstd2@self._P_s.T
            state_mu = (state_mu + state_mu2)/2
            state_logstd = (state_logstd + state_logstd2)/2

        state_dist = Normal(state_mu, state_logstd.exp())
        state_pred = state_dist.rsample()

        reward_dist = Normal(reward_mu, reward_logstd.exp())
        reward_pred = reward_dist.rsample()
        # residual prediction
        next_x = x + state_pred
        next_reward = reward_pred.view(-1, 1)
        return next_x, next_reward, state_dist.variance.sum(dim=1, keepdims=True) + reward_dist.variance.sum(dim=1, keepdims=True)

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> torch.Tensor:
        # mu, logstd = self.compute_stats(x, action)
        state_mu, reward_mu, state_logstd, reward_logstd = self.compute_stats(observations, actions)
        if self._P_s is not None:
            device = observations.device
            self._P_s = self._P_s.to(device)
            self._P_a = self._P_a.to(device)
            # transform before input
            state_mu2, _, state_logstd2, _ = self.compute_stats(observations@self._P_s.T, actions@self._P_a.T)
            # reverse transform after input
            state_mu2 = state_mu2@self._P_s.T
            state_logstd2 = state_logstd2@self._P_s.T
            state_mu = (state_mu + state_mu2)/2
            state_logstd = (state_logstd + state_logstd2)/2
        # residual prediction
        mu_x = observations + state_mu
        mu_reward = reward_mu.view(-1, 1)
        logstd_x = state_logstd
        logstd_reward = reward_logstd.view(-1, 1)

        # gaussian likelihood loss
        likelihood_loss = _gaussian_likelihood(
            next_observations, mu_x, logstd_x
        )
        likelihood_loss += _gaussian_likelihood(
            rewards, mu_reward, logstd_reward
        )

        # penalty to minimize standard deviation
        penalty = state_logstd.sum(dim=1, keepdim=True) + reward_logstd.sum(dim=1, keepdim=True)

        # minimize logstd bounds
        bound_loss = self._state_max_logstd.sum() - self._state_min_logstd.sum() + self._reward_max_logstd.sum() - self._reward_min_logstd.sum()

        loss = likelihood_loss + penalty + 1e-2 * bound_loss

        return loss.view(-1, 1)

        # mu, logstd = self.compute_stats(observations, actions)

        # # residual prediction
        # mu_x = observations + mu[:, :-1]
        # mu_reward = mu[:, -1].view(-1, 1)
        # logstd_x = logstd[:, :-1]
        # logstd_reward = logstd[:, -1].view(-1, 1)

        # # gaussian likelihood loss
        # likelihood_loss = _gaussian_likelihood(
        #     next_observations, mu_x, logstd_x
        # )
        # likelihood_loss += _gaussian_likelihood(
        #     rewards, mu_reward, logstd_reward
        # )

        # # penalty to minimize standard deviation
        # penalty = logstd.sum(dim=1, keepdim=True)

        # # minimize logstd bounds
        # bound_loss = self._max_logstd.sum() - self._min_logstd.sum()

        # loss = likelihood_loss + penalty + 1e-2 * bound_loss

        # return loss.view(-1, 1)


class ProbabilisticEnsembleDynamicsModel(nn.Module):  # type: ignore
    _models: nn.ModuleList

    def __init__(self, models: List[ProbabilisticDynamicsModel]):
        super().__init__()
        self._models = nn.ModuleList(models)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_with_variance(x, action, indices=indices)[:2]

    def __call__(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return cast(
            Tuple[torch.Tensor, torch.Tensor],
            super().__call__(x, action, indices),
        )

    def predict_with_variance(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        variance_type: str = "data",
        indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations_list: List[torch.Tensor] = []
        rewards_list: List[torch.Tensor] = []
        variances_list: List[torch.Tensor] = []

        # predict next observation and reward
        for model in self._models:
            obs, rew, var = model.predict_with_variance(x, action)
            observations_list.append(obs.view(1, x.shape[0], -1))
            rewards_list.append(rew.view(1, x.shape[0], 1))
            variances_list.append(var.view(1, x.shape[0], 1))

        # (ensemble, batch, -1) -> (batch, ensemble, -1)
        observations = torch.cat(observations_list, dim=0).transpose(0, 1)
        rewards = torch.cat(rewards_list, dim=0).transpose(0, 1)
        variances = torch.cat(variances_list, dim=0).transpose(0, 1)

        variances = _compute_ensemble_variance(
            observations=observations,
            rewards=rewards,
            variances=variances,
            variance_type=variance_type,
        )

        if indices is None:
            return observations, rewards, variances

        # pick samples based on indices
        partial_observations = observations[torch.arange(x.shape[0]), indices]
        partial_rewards = rewards[torch.arange(x.shape[0]), indices]
        return partial_observations, partial_rewards, variances

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_sum = torch.tensor(
            0.0, dtype=torch.float32, device=observations.device
        )
        for i, model in enumerate(self._models):
            loss = model.compute_error(
                observations, actions, rewards, next_observations
            )
            assert loss.shape == (observations.shape[0], 1)

            # create mask if necessary
            if masks is None:
                mask = torch.randint(
                    0, 2, size=loss.shape, device=observations.device
                )
            else:
                mask = masks[i]

            loss_sum += (loss * mask).mean()

        return loss_sum

    @property
    def models(self) -> nn.ModuleList:
        return self._models
