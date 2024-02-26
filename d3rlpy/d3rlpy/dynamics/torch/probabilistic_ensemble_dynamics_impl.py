from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import create_probabilistic_ensemble_dynamics_model
from ...models.encoders import EncoderFactory, VectorEncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.torch import ProbabilisticEnsembleDynamicsModel
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .base import TorchImplBase


class ProbabilisticEnsembleDynamicsImpl(TorchImplBase):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _n_ensembles: int
    _variance_type: str
    _discrete_action: bool
    _use_gpu: Optional[Device]
    _dynamics: Optional[ProbabilisticEnsembleDynamicsModel]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        state_encoder_factory: EncoderFactory,
        reward_encoder_factory: EncoderFactory,
        n_ensembles: int,
        variance_type: str,
        discrete_action: bool,
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        use_gpu: Optional[Device],
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
        cartans_submanifold_dim=None,
        cartans_encoder_factory=VectorEncoderFactory(),
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._state_encoder_factory = state_encoder_factory
        self._reward_encoder_factory = reward_encoder_factory
        self._n_ensembles = n_ensembles
        self._variance_type = variance_type
        self._discrete_action = discrete_action
        self._use_gpu = use_gpu
        self._permutation_indices=permutation_indices
        self._augmentation=augmentation
        self._reduction=reduction

        # Cartan's moving frame method stuff (Neelay)
        assert not (cartans_deterministic and cartans_stochastic)
        self.cartans_deterministic = cartans_deterministic
        self.cartans_stochastic = cartans_stochastic
        self.cartans_rho = cartans_rho
        self.cartans_phi = cartans_phi
        self.cartans_psi = cartans_psi
        self.cartans_R = cartans_R
        self.cartans_gamma = cartans_gamma
        self.cartans_group_inv = cartans_group_inv
        self.cartans_submanifold_dim = cartans_submanifold_dim
        self.cartans_encoder_factory = cartans_encoder_factory
        if self.cartans_deterministic:
            assert self.cartans_rho is not None
            assert self.cartans_phi is not None
            assert self.cartans_psi is not None
            assert self.cartans_submanifold_dim is not None
        if self.cartans_stochastic:
            assert self.cartans_rho is not None
            assert self.cartans_psi is not None
            assert self.cartans_submanifold_dim is not None

        # initialized in build
        self._dynamics = None
        self._optim = None

    def build(self) -> None:
        self._build_dynamics()

        self.to_cpu()
        if self._use_gpu:
            self.to_gpu(self._use_gpu)

        self._build_optim()

    def _build_dynamics(self) -> None:
        self._dynamics = create_probabilistic_ensemble_dynamics_model(
            self._observation_shape,
            self._action_size,
            self._state_encoder_factory,
            self._reward_encoder_factory,
            n_ensembles=self._n_ensembles,
            discrete_action=self._discrete_action,
            permutation_indices = self._permutation_indices,
            augmentation = self._augmentation,
            reduction=self._reduction,
            # Cartan's Moving Frame method stuff (Neelay)
            cartans_deterministic=self.cartans_deterministic,
            cartans_stochastic=self.cartans_stochastic,
            cartans_rho=self.cartans_rho,
            cartans_phi=self.cartans_phi,
            cartans_psi=self.cartans_psi,
            cartans_R=self.cartans_R,
            cartans_gamma=self.cartans_gamma,
            cartans_group_inv=self.cartans_group_inv,
            cartans_submanifold_dim=self.cartans_submanifold_dim,
            cartans_encoder_factory=self.cartans_encoder_factory,
        )

    def _build_optim(self) -> None:
        assert self._dynamics is not None
        self._optim = self._optim_factory.create(
            self._dynamics.parameters(), lr=self._learning_rate
        )

    def _predict(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        indices: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._dynamics is not None
        if indices is None:
            indices = torch.randint(self._n_ensembles, size=(x.shape[0],))
        else:
            assert indices.shape == (x.shape[0],)
        return self._dynamics.predict_with_variance(
            x,
            action,
            variance_type=self._variance_type,
            indices=indices.long(),
        )

    @train_api
    @torch_api()
    def update(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._dynamics is not None
        assert self._optim is not None

        loss = self._dynamics.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            next_observations=batch.next_observations,
        )

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()