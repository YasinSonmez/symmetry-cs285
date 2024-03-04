# pylint: disable=protected-access

from typing import List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, MultivariateNormal
from torch.nn.utils import spectral_norm

from .encoders import EncoderWithAction
import numpy as np
from itertools import combinations


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
    likelihood = 0.5 * (((mu - x) ** 2) * inv_var).sum(dim=1, keepdim=True)
    return likelihood


def _gaussian_likelihood_cov(
    x: torch.Tensor, mu: torch.Tensor, L: torch.Tensor
) -> torch.Tensor:
    # Given x (states), mu (mean of distribution), and L (constructs covariance matrix as Sigma = L L^T)
    # constructs gaussian likelihood loss

    # First term from eq 1 of https://arxiv.org/pdf/1805.12114.pdf,
    # "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models"
    demeaned_x = x - mu
    assert demeaned_x.shape == x.shape

    y = torch.linalg.solve(L, demeaned_x)
    assert y.shape == x.shape

    likelihood = (y**2).sum(dim=1, keepdim=True)
    assert likelihood.shape == (x.shape[0], 1)

    return likelihood


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


def create_pairwise_permutation_matrices(size, perm_indices, pairs):
    """
    Create a permutation matrix of given size that permutes rows from source_indices to target_indices.

    :param size: Size of the square permutation matrix
    :param perm_indices: List of List of indices to be permuted
    :return: Permutation matrix of size 'size x size'
    """

    perm_matrices = []
    for pair in pairs:
        perm_matrices.append(
            create_permutation_matrix(
                size, perm_indices[pair[0]], perm_indices[pair[1]]
            )
        )

    return perm_matrices


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

    def __init__(
        self,
        state_encoder: EncoderWithAction,
        reward_encoder: EncoderWithAction,
        permutation_indices=None,
        augmentation=None,
        reduction=None,
        # Cartan's Moving Frame method stuff (Neelay)
        cartans_deterministic=False,
        cartans_stochastic=False,
        cartans_rho=None,
        cartans_phi=None,
        cartans_psi=None,
        cartans_R=None,
        cartans_gamma=None,
        cartans_group_inv=None,
        cartans_full_state_encoder: Optional[EncoderWithAction] = None,
        cartans_reduced_state_encoder: Optional[EncoderWithAction] = None,
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
        self._transferred_to_device = False

        # Cartan's moving frame method stuff (Neelay)
        assert not (cartans_deterministic and cartans_stochastic)
        self.cartans_deterministic = cartans_deterministic
        self.cartans_stochastic = cartans_stochastic
        if self.cartans_deterministic:
            assert cartans_rho is not None
            assert cartans_phi is not None
            assert cartans_psi is not None
            assert cartans_gamma is not None
            assert cartans_group_inv is not None
            assert cartans_full_state_encoder is not None
            assert cartans_reduced_state_encoder is not None
            self.cartans_rho = cartans_rho
            self.cartans_phi = cartans_phi
            self.cartans_psi = cartans_psi
            self.cartans_gamma = cartans_gamma
            self.cartans_group_inv = cartans_group_inv
            self.cartans_full_state_encoder = cartans_full_state_encoder
            self.cartans_reduced_state_encoder = cartans_reduced_state_encoder
            # Ensure these aren't used if using cartan's impl
            self._state_encoder = None
            self._reward_encoder = None
        if self.cartans_stochastic:
            assert cartans_rho is not None
            assert cartans_psi is not None
            assert cartans_gamma is not None
            assert cartans_full_state_encoder is not None
            assert cartans_reduced_state_encoder is not None
            self.cartans_rho = cartans_rho
            self.cartans_psi = cartans_psi
            self.cartans_R = cartans_R
            self.cartans_gamma = cartans_gamma
            self.cartans_group_inv = cartans_group_inv
            self.cartans_full_state_encoder = cartans_full_state_encoder
            self.cartans_reduced_state_encoder = cartans_reduced_state_encoder
            # Ensure these aren't used if using cartan's impl
            self._state_encoder = None
            self._reward_encoder = None

        state_mu_feature_size = state_encoder.get_feature_size()  # TODO: do I need to enforce feature and observation size are common between the state and reward encoders?
        state_logstd_feature_size = state_mu_feature_size
        reward_feature_size = reward_encoder.get_feature_size()
        observation_size = state_encoder.observation_shape[0]
        action_size = state_encoder.action_size

        if self.cartans_deterministic or self.cartans_stochastic:
            state_mu_feature_size = (
                self.cartans_reduced_state_encoder.get_feature_size()
            )
            reward_feature_size = self.cartans_full_state_encoder.get_feature_size()
            observation_size = self.cartans_full_state_encoder.observation_shape[0]
            action_size = self.cartans_full_state_encoder.action_size
        if self.cartans_deterministic:
            state_logstd_feature_size = (
                self.cartans_full_state_encoder.get_feature_size()
            )
        if self.cartans_stochastic:
            state_logstd_feature_size = (
                self.cartans_reduced_state_encoder.get_feature_size()
            )

        if permutation_indices is not None:
            m = len(self._permutation_indices)
            pairs = np.array(list(combinations(list(range(m)), 2)))
            self._P_s = create_pairwise_permutation_matrices(
                observation_size, self._permutation_indices[0], pairs
            )
            self._P_a = create_pairwise_permutation_matrices(
                action_size, self._permutation_indices[1], pairs
            )
        else:
            self._P_s = None
            self._P_a = None
        # out_size = observation_size + 1
        out_size = observation_size

        # TODO: handle image observation
        self._state_mu = spectral_norm(nn.Linear(state_mu_feature_size, out_size))
        # If doing cartans_stochastic, this actually outputs the square root of a diagonal covariance matrix for the transformed space.
        # This will then be transformed into the square root of a covariance matrix, L such that Sigma = L L^T.
        # If not using cartans_stochastic, outputs the log of the square root of a diagonal covariance matrix.
        self._state_logstd = nn.Linear(state_logstd_feature_size, out_size)

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
        if not (self.cartans_deterministic or self.cartans_stochastic):
            state_mu_h = self._state_encoder(x, action)
            state_logstd_h = state_mu_h
            reward_h = self._reward_encoder(x, action)
        else:
            gammax = self.cartans_gamma(x)
            xbar = self.cartans_rho(x, gammax=gammax)
            actionbar = self.cartans_psi(gammax, action)
            reduced_h = self.cartans_reduced_state_encoder(xbar, actionbar)
            full_h = self.cartans_full_state_encoder(x, action)
            state_mu_h = reduced_h
            if self.cartans_deterministic:
                state_logstd_h = full_h
            if self.cartans_stochastic:
                state_logstd_h = reduced_h
            reward_h = full_h

            gammaxinv = self.cartans_group_inv(gammax, x=x)

        state_mu0 = self._state_mu(state_mu_h)
        if self.cartans_deterministic:
            state_mu = state_mu0 + self.cartans_phi(gammax, x)
            state_mu = self.cartans_phi(gammaxinv, state_mu)
            state_mu = state_mu - x
        elif self.cartans_stochastic:
            Rgammaxinv = self.cartans_R(gammaxinv)
            state_mu = torch.mm(Rgammaxinv, state_mu0.t()).t()
        else:
            state_mu = state_mu0
        reward_mu = self._reward_mu(reward_h)

        # log standard deviation with bounds
        state_logstd = self._state_logstd(state_logstd_h)
        state_logstd = self._state_max_logstd - F.softplus(
            self._state_max_logstd - state_logstd
        )
        state_logstd = self._state_min_logstd + F.softplus(
            state_logstd - self._state_min_logstd
        )
        if self.cartans_stochastic:
            # This is L in var(DeltaF(x,u)) = LL^T, assuming state_logstd is Lbar such that var(DeltaFbar(xbar, ubar)) = Lbar Lbar^T
            state_logstd = state_logstd**2 + 1e-6 # Make positive definite
            state_logstd = Rgammaxinv @ state_logstd.diag_embed()

        reward_logstd = self._reward_logstd(reward_h)
        reward_logstd = self._reward_max_logstd - F.softplus(
            self._reward_max_logstd - reward_logstd
        )
        reward_logstd = self._reward_min_logstd + F.softplus(
            reward_logstd - self._reward_min_logstd
        )

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
        if self._reduction:
            state_mus = [state_mu]
            state_logstds = [state_logstd]
            for P_s, P_a in zip(self._P_s, self._P_a):
                # transform before input
                state_mu2, _, state_logstd2, _ = self.compute_stats(
                    x @ P_s.T, action @ P_a.T
                )
                # reverse transform after input
                state_mu2 = state_mu2 @ P_s.T
                state_logstd2 = state_logstd2 @ P_s.T
                state_mus.append(state_mu2)
                state_logstds.append(state_logstd2)
                state_mu = torch.stack(state_mus).mean(axis=0)
                state_logstd = torch.stack(state_logstds).mean(axis=0)

        if self.cartans_stochastic:
            state_dist = MultivariateNormal(state_mu, scale_tril=state_logstd)
        else:
            state_dist = Normal(state_mu, state_logstd.exp())
        state_pred = state_dist.rsample()

        reward_dist = Normal(reward_mu, reward_logstd.exp())
        reward_pred = reward_dist.rsample()
        # residual prediction
        next_x = x + state_pred
        next_reward = reward_pred.view(-1, 1)
        return (
            next_x,
            next_reward,
            state_dist.variance.sum(dim=1, keepdims=True)
            + reward_dist.variance.sum(dim=1, keepdims=True),
        )

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> torch.Tensor:
        # Only transfer once using boolean self._transferred_to_device
        if self._P_s is not None and not self._transferred_to_device:
            device = observations.device
            for i in range(len(self._P_s)):
                self._P_s[i] = self._P_s[i].to(device)
                self._P_a[i] = self._P_a[i].to(device)
            self._transferred_to_device = True

        if self._augmentation:
            symmetry_count = len(self._P_s)
            if np.random.rand() > 1 / (1 + symmetry_count):
                i = np.random.randint(symmetry_count)
                observations = observations @ self._P_s[i].T
                next_observations = next_observations @ self._P_s[i].T
                actions = actions @ self._P_a[i].T

        # mu, logstd = self.compute_stats(x, action)
        state_mu, reward_mu, state_logstd, reward_logstd = self.compute_stats(
            observations, actions
        )
        if self._reduction:
            state_mus = [state_mu]
            state_logstds = [state_logstd]
            for P_s, P_a in zip(self._P_s, self._P_a):
                # transform before input
                state_mu2, _, state_logstd2, _ = self.compute_stats(
                    observations @ P_s.T, actions @ P_a.T
                )
                # reverse transform after input
                state_mu2 = state_mu2 @ P_s.T
                state_logstd2 = state_logstd2 @ P_s.T
                state_mus.append(state_mu2)
                state_logstds.append(state_logstd2)
                state_mu = torch.stack(state_mus).mean(axis=0)
                state_logstd = torch.stack(state_logstds).mean(axis=0)
        # residual prediction
        mu_x = observations + state_mu
        mu_reward = reward_mu.view(-1, 1)
        logstd_x = state_logstd
        logstd_reward = reward_logstd.view(-1, 1)

        # gaussian likelihood loss
        if not self.cartans_stochastic:
            likelihood_loss = _gaussian_likelihood(next_observations, mu_x, logstd_x)
        else:
            likelihood_loss = _gaussian_likelihood_cov(
                next_observations, mu_x, logstd_x
            )
        likelihood_loss += _gaussian_likelihood(rewards, mu_reward, logstd_reward)

        # penalty to minimize standard deviation
        if not self.cartans_stochastic:
            penalty = state_logstd.sum(dim=1, keepdim=True)
        else:
            penalty = state_logstd.logdet().reshape((-1, 1))
        penalty += reward_logstd.sum(dim=1, keepdim=True)

        # minimize logstd bounds
        bound_loss = (
            self._state_max_logstd.sum()
            - self._state_min_logstd.sum()
            + self._reward_max_logstd.sum()
            - self._reward_min_logstd.sum()
        )

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
        loss_sum = torch.tensor(0.0, dtype=torch.float32, device=observations.device)
        for i, model in enumerate(self._models):
            loss = model.compute_error(
                observations, actions, rewards, next_observations
            )
            assert loss.shape == (observations.shape[0], 1)

            # create mask if necessary
            if masks is None:
                mask = torch.randint(0, 2, size=loss.shape, device=observations.device)
            else:
                mask = masks[i]

            loss_sum += (loss * mask).mean()

        return loss_sum

    @property
    def models(self) -> nn.ModuleList:
        return self._models
