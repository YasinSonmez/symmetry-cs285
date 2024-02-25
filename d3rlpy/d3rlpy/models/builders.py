from typing import Optional, Sequence, cast

import torch
from torch import nn

from .encoders import EncoderFactory
from .q_functions import QFunctionFactory
from .torch import (
    CategoricalPolicy,
    ConditionalVAE,
    DeterministicPolicy,
    DeterministicRegressor,
    DeterministicResidualPolicy,
    DiscreteImitator,
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
    NonSquashedNormalPolicy,
    Parameter,
    ProbabilisticDynamicsModel,
    ProbabilisticEnsembleDynamicsModel,
    ProbablisticRegressor,
    SquashedNormalPolicy,
    ValueFunction,
)

from .encoders import VectorEncoderFactory


def create_discrete_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    n_ensembles: int = 1,
) -> EnsembleDiscreteQFunction:
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create(observation_shape)
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create(observation_shape)
        q_funcs.append(q_func_factory.create_discrete(encoder, action_size))
    return EnsembleDiscreteQFunction(q_funcs)


def create_continuous_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    n_ensembles: int = 1,
) -> EnsembleContinuousQFunction:
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create_with_action(
            observation_shape, action_size
        )
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create_with_action(
                observation_shape, action_size
            )
        q_funcs.append(q_func_factory.create_continuous(encoder))
    return EnsembleContinuousQFunction(q_funcs)


def create_deterministic_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> DeterministicPolicy:
    encoder = encoder_factory.create(observation_shape)
    return DeterministicPolicy(encoder, action_size)


def create_deterministic_residual_policy(
    observation_shape: Sequence[int],
    action_size: int,
    scale: float,
    encoder_factory: EncoderFactory,
) -> DeterministicResidualPolicy:
    encoder = encoder_factory.create_with_action(observation_shape, action_size)
    return DeterministicResidualPolicy(encoder, scale)


def create_squashed_normal_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    use_std_parameter: bool = False,
) -> SquashedNormalPolicy:
    encoder = encoder_factory.create(observation_shape)
    return SquashedNormalPolicy(
        encoder,
        action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )


def create_non_squashed_normal_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    use_std_parameter: bool = False,
) -> NonSquashedNormalPolicy:
    encoder = encoder_factory.create(observation_shape)
    return NonSquashedNormalPolicy(
        encoder,
        action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )


def create_categorical_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> CategoricalPolicy:
    encoder = encoder_factory.create(observation_shape)
    return CategoricalPolicy(encoder, action_size)


def create_conditional_vae(
    observation_shape: Sequence[int],
    action_size: int,
    latent_size: int,
    beta: float,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
) -> ConditionalVAE:
    encoder_encoder = encoder_factory.create_with_action(
        observation_shape, action_size
    )
    decoder_encoder = encoder_factory.create_with_action(
        observation_shape, latent_size
    )
    return ConditionalVAE(
        encoder_encoder,
        decoder_encoder,
        beta,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
    )


def create_discrete_imitator(
    observation_shape: Sequence[int],
    action_size: int,
    beta: float,
    encoder_factory: EncoderFactory,
) -> DiscreteImitator:
    encoder = encoder_factory.create(observation_shape)
    return DiscreteImitator(encoder, action_size, beta)


def create_deterministic_regressor(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> DeterministicRegressor:
    encoder = encoder_factory.create(observation_shape)
    return DeterministicRegressor(encoder, action_size)


def create_probablistic_regressor(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
) -> ProbablisticRegressor:
    encoder = encoder_factory.create(observation_shape)
    return ProbablisticRegressor(
        encoder, action_size, min_logstd=min_logstd, max_logstd=max_logstd
    )


def create_value_function(
    observation_shape: Sequence[int], encoder_factory: EncoderFactory
) -> ValueFunction:
    encoder = encoder_factory.create(observation_shape)
    return ValueFunction(encoder)


def create_probabilistic_ensemble_dynamics_model(
    observation_shape: Sequence[int],
    action_size: int,
    state_encoder_factory: EncoderFactory,
    reward_encoder_factory: EncoderFactory,
    n_ensembles: int = 5,
    discrete_action: bool = False,
    permutation_indices = None,
    augmentation = None,
    reduction = None,
    # Cartan's Moving Frame method stuff (Neelay)
    cartans_deterministic=False,
    cartans_stochastic=False,
    cartans_rho=None,
    cartans_phi=None,
    cartans_psi=None,
    cartans_R=None,
    cartans_submanifold_dim:Optional[int]=None,
    cartans_encoder_factory=VectorEncoderFactory(),
) -> ProbabilisticEnsembleDynamicsModel:
    models = []
    for _ in range(n_ensembles):
        state_encoder = state_encoder_factory.create_with_action(
            observation_shape=observation_shape,
            action_size=action_size,
            discrete_action=discrete_action,
        )
        reward_encoder = reward_encoder_factory.create_with_action(
            observation_shape=observation_shape,
            action_size=action_size,
            discrete_action=discrete_action,
        )

        full_state_encoder = cartans_encoder_factory.create_with_action(
            observation_shape=observation_shape,
            action_size=action_size,
            discrete_action=discrete_action,
        )
        reduced_state_encoder = cartans_encoder_factory.create_with_action(
            observation_shape=cartans_submanifold_dim,
            action_size=action_size,
            discrete_action=discrete_action,
        )
        model = ProbabilisticDynamicsModel(
            state_encoder, 
            reward_encoder,
            permutation_indices = permutation_indices,
            augmentation = augmentation,
            reduction = reduction,
            # Cartan's Moving Frame method stuff (Neelay)
            cartans_deterministic=cartans_deterministic,
            cartans_stochastic=cartans_stochastic,
            cartans_rho=cartans_rho,
            cartans_phi=cartans_phi,
            cartans_psi=cartans_psi,
            cartans_R=cartans_R,
            cartans_full_state_encoder=full_state_encoder,
            cartans_reduced_state_encoder=reduced_state_encoder,
        )
        models.append(model)
    return ProbabilisticEnsembleDynamicsModel(models)


def create_parameter(shape: Sequence[int], initial_value: float) -> Parameter:
    data = torch.full(shape, initial_value, dtype=torch.float32)
    return Parameter(data)