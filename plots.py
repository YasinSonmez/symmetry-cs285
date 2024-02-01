import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# plt.plot(x, mean, label = data_i['type'])
# plt.fill_between(x, mean-std, mean+std, alpha=0.2)
# plt.grid()
# if title is not None:
# plt.title(title)
# plt.ylabel(metric_name)
# plt.xlabel('iterations')
# plt.legend()


def plot(dirs, csv_file, x_label, y_label, legend_label):
    x = None

    ys = []
    for dir in dirs:
        df = pd.read_csv(f"{dir}/{csv_file}")

        if x is None:
            x = df[x_label]

        ys.append(df[y_label].to_numpy())

    min_len = np.min([arr.size for arr in ys])

    ys = [y[:min_len] for y in ys]
    ys = np.stack(ys, axis=-1)

    x = x[:min_len]

    mean_y = ys.mean(axis=1)
    std_y = ys.std(axis=1)

    plt.plot(x, mean_y, label=f"{legend_label}")
    plt.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.2)


# Reward Asymmetric Inverted Pendulum

symmetry_dirs = (
    [
        "exp/mbpo/symmetry/custom___RewardAsymmetricInvertedPendulum/2023.12.09/120417",  # seed 0
        "exp/mbpo/symmetry/custom___RewardAsymmetricInvertedPendulum/2023.12.09/132231",  # seed 1
        "exp/mbpo/symmetry/custom___RewardAsymmetricInvertedPendulum/2023.12.09/133050",  # seed 2
    ],
    "Symmetry",
)

no_symmetry_dirs = (
    [
        "exp/mbpo/no_symmetry/custom___RewardAsymmetricInvertedPendulum/2023.12.09/112453",  # seed 0
        "exp/mbpo/no_symmetry/custom___RewardAsymmetricInvertedPendulum/2023.12.09/132231",  # seed 1
        "exp/mbpo/no_symmetry/custom___RewardAsymmetricInvertedPendulum/2023.12.09/133050",  # seed 2
    ],
    "No Symmetry",
)

plt.figure()
for dirs, label in [symmetry_dirs, no_symmetry_dirs]:
    plot(dirs, "results.csv", "env_step", "episode_reward", label)
plt.grid()
plt.title("Reward: Symmetry vs. No Symmetry on Inverted Pendulum")
plt.xlabel("Environment Steps")
plt.ylabel("Episode Reward")
plt.legend()
plt.ylim([0, 400])
plt.savefig("figures/invpend_mbpo_reward.png", dpi=300, bbox_inches="tight")

plt.figure()
for dirs, label in [symmetry_dirs, no_symmetry_dirs]:
    plot(dirs, "model_train.csv", "step", "model_loss", label)
plt.grid()
plt.title("Model Loss: Symmetry vs. No Symmetry on Inverted Pendulum")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("figures/invpend_mbpo_model_loss.png", dpi=300, bbox_inches="tight")

plt.figure()
for dirs, label in [symmetry_dirs, no_symmetry_dirs]:
    plot(dirs, "model_train.csv", "step", "model_val_score", label)
plt.grid()
plt.title("Model Validation Score: Symmetry vs. No Symmetry on Inverted Pendulum")
plt.xlabel("Steps")
plt.ylabel("Validation Score")
plt.legend()
plt.savefig("figures/invpend_mbpo_model_validation.png", dpi=300, bbox_inches="tight")

# plt.figure()
# for dirs, label in [symmetry_dirs, no_symmetry_dirs]:
#     plot(dirs, "model_train.csv", "step", "model_best_val_score", label)
# plt.grid()
# plt.title("Model Best Validation Score: Symmetry vs. No Symmetry on Inverted Pendulum")
# plt.xlabel("Steps")
# plt.ylabel("Best Validation Score")
# plt.legend()

# # Reacher

symmetry_dirs = (
    [
        "exp/mbpo/symmetry/custom___Reacher/2023.12.09/155712", # seed 0
        "exp/mbpo/symmetry/custom___Reacher/2023.12.09/160150", # seed 1
        "exp/mbpo/symmetry/custom___Reacher/2023.12.09/160853", # seed 2
    ],
    "Symmetry",
)

no_symmetry_dirs = (
    [
        "exp/mbpo/no_symmetry/custom___Reacher/2023.12.09/155712", # seed 0
        "exp/mbpo/no_symmetry/custom___Reacher/2023.12.09/160150", # seed 1
        "exp/mbpo/no_symmetry/custom___Reacher/2023.12.09/160853", # seed 2
    ],
    "No Symmetry",
)

plt.figure()
for dirs, label in [symmetry_dirs, no_symmetry_dirs]:
    plot(dirs, "results.csv", "env_step", "episode_reward", label)
plt.grid()
plt.title("Reward: Symmetry vs. No Symmetry on Reacher")
plt.xlabel("Environment Steps")
plt.ylabel("Episode Reward")
plt.legend()
plt.ylim([0, 400])
plt.savefig("figures/reacher_mbpo_reward.png", dpi=300, bbox_inches="tight")


plt.figure()
for dirs, label in [symmetry_dirs, no_symmetry_dirs]:
    plot(dirs, "model_train.csv", "step", "model_loss", label)
plt.grid()
plt.title("Model Loss: Symmetry vs. No Symmetry on Reacher")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.ylim([-75, 50])
plt.savefig("figures/reacher_mbpo_model_loss.png", dpi=300, bbox_inches="tight")

plt.figure()
for dirs, label in [symmetry_dirs, no_symmetry_dirs]:
    plot(dirs, "model_train.csv", "step", "model_val_score", label)
plt.grid()
plt.title("Model Validation Score: Symmetry vs. No Symmetry on Reacher")
plt.xlabel("Steps")
plt.ylabel("Validation Score")
plt.legend()
plt.ylim([0, 2])
plt.savefig("figures/reacher_mbpo_model_validation.png", dpi=300, bbox_inches="tight")

# plt.show()
