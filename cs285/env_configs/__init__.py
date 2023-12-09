from .sac_config import sac_config
from .mpc_config import mpc_config
from .symmetry_configs import inverted_pendulum_symmetry, reacher_symmetry

configs = {
    "sac": sac_config,
    "mpc": mpc_config,
    "reacher_symmetry": reacher_symmetry,
    "inverted_pendulum_symmetry": inverted_pendulum_symmetry
}
