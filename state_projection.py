import gym
from sklearn.model_selection import train_test_split

import d3rlpy
import encoders
import environments
from d3rlpy.algos import COMBO, MOPO
import torch
from d3rlpy.models.encoders import VectorEncoderFactory

print(gym.version.VERSION)

N_POLICY_STEPS = 50000  # 100000
N_EPOCHS = 5
N_RUNS = 3
EXPERIMENT_NAME = "TEST"


use_gpu = True
seed = 0
d3rlpy.seed(seed)

env = environments.RewardAssymetricInvertedPendulum()
eval_env = environments.RewardAssymetricInvertedPendulum()
env.reset(seed=seed)
eval_env.reset(seed=seed)
dataset = d3rlpy.dataset.MDPDataset.load("d3rlpy_data/rwd_assym_inv_pend_v8.h5")
train_episodes, test_episodes = train_test_split(dataset, random_state=seed)

symmetry = False

if symmetry:
    dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(
        learning_rate=1e-4,
        use_gpu=use_gpu,
        cartans_deterministic=True,
        cartans_stochastic=False,
        cartans_rho=env.rho,
        cartans_phi=env.phi,
        cartans_psi=env.psi,
        cartans_R=env.R,
        cartans_gamma=env.gamma,
        cartans_group_inv=env.group_inv,
        cartans_submanifold_dim=env.submanifold_dim,
        cartans_encoder_factory=VectorEncoderFactory(),
    )
else:
    dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(
        learning_rate=1e-4,
        use_gpu=use_gpu,
    )

dynamics.fit(
    train_episodes,
    eval_episodes=test_episodes,
    n_epochs=N_EPOCHS,
    scorers={
        "observation_error": d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
        "reward_error": d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
        "variance": d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
    },
    tensorboard_dir="tensorboard_logs/dynamics",
    experiment_name=f"{EXPERIMENT_NAME}_SEED{seed}",
)

# encoder_factory = d3rlpy.models.encoders.DefaultEncoderFactory(dropout_rate=0.2)
# policy = MOPO(
#     dynamics=dynamics,
#     critic_encoder_factory=encoder_factory,
#     actor_encoder_factory=encoder_factory,
#     use_gpu=use_gpu,
# )

# policy.fit(
#     dataset=train_episodes,
#     eval_episodes=test_episodes,
#     n_steps=N_POLICY_STEPS,
#     n_steps_per_epoch=1000,
#     tensorboard_dir="tensorboard_logs",
#     scorers={"environment": d3rlpy.metrics.scorer.evaluate_on_environment(eval_env)},
#     save_interval=50,
#     experiment_name=f"{EXPERIMENT_NAME}_SEED{SEED}",
# )

# scorer = d3rlpy.metrics.scorer.evaluate_on_environment(eval_env, render=True)
# mean_episode_return = scorer(policy)


# def inverted_pendulum_project(x):
#     return x[:, 1:]

# projection_size = 3

# for run_i in range(N_RUNS):
#     print(f"\nSTARTING RUN {run_i}\n")
#     seed = seed0 + run_i

#     for dyn_type in ["no_sym", "dyn_sym", "dyn_rwd_sym"]:
#         print(f"\nRUN {run_i} WITH {dyn_type}\n")

#         d3rlpy.seed(seed)
#         env = environments.RewardAssymetricInvertedPendulum()
#         eval_env = environments.RewardAssymetricInvertedPendulum()
#         env.reset(seed=seed)
#         eval_env.reset(seed=seed)
#         dataset = d3rlpy.dataset.MDPDataset.load("d3rlpy_data/inverted_pendulum2.h5")
#         train_episodes, test_episodes = train_test_split(dataset, random_state=seed)

#         if dyn_type == "no_sym":
#             state_encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(
#                 hidden_units=[64, 64]
#             )
#             reward_encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(
#                 hidden_units=[64, 64]
#             )
#         elif dyn_type == "dyn_sym":
#             state_encoder_factory = encoders.SymmetryEncoderFactory(
#                 project=inverted_pendulum_project,
#                 projection_size=projection_size,
#                 hidden_units=[64, 64],
#             )
#             reward_encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(
#                 hidden_units=[64, 64]
#             )
#         elif dyn_type == "dyn_rwd_symm":
#             state_encoder_factory = encoders.SymmetryEncoderFactory(
#                 project=inverted_pendulum_project,
#                 projection_size=projection_size,
#                 hidden_units=[64, 64],
#             )
#             reward_encoder_factory = encoders.SymmetryEncoderFactory(
#                 project=inverted_pendulum_project,
#                 projection_size=projection_size,
#                 hidden_units=[64, 64],
#             )
#         else:
#             assert "Error"

#         dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(
#             learning_rate=1e-4,
#             use_gpu=use_gpu,
#             state_encoder_factory=state_encoder_factory,
#             reward_encoder_factory=reward_encoder_factory,
#         )

#         dynamics.fit(
#             train_episodes,
#             eval_episodes=test_episodes,
#             n_epochs=N_EPOCHS,
#             scorers={
#                 "observation_error": d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
#                 "reward_error": d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
#                 "variance": d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
#             },
#             tensorboard_dir="tensorboard_logs/dynamics",
#             experiment_name=f"{EXPERIMENT_NAME}_run{run_i}_dyn_{dyn_type}",
#         )

#         encoder_factory = d3rlpy.models.encoders.DefaultEncoderFactory(dropout_rate=0.2)

#         policy = MOPO(
#             dynamics=dynamics,
#             critic_encoder_factory=encoder_factory,
#             actor_encoder_factory=encoder_factory,
#             use_gpu=use_gpu,
#         )

#         policy.fit(
#             dataset=train_episodes,
#             eval_episodes=test_episodes,
#             n_steps=N_POLICY_STEPS,
#             n_steps_per_epoch=1000,
#             tensorboard_dir="tensorboard_logs",
#             scorers={
#                 "environment": d3rlpy.metrics.scorer.evaluate_on_environment(eval_env)
#             },
#             save_interval=50,
#             experiment_name=f"{EXPERIMENT_NAME}_run{run_i}_policy_{dyn_type}",
#         )

# scorer = d3rlpy.metrics.scorer.evaluate_on_environment(eval_env, render=True)
# mean_episode_return = scorer(combo)
