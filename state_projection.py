import os

import torch
from sklearn.model_selection import train_test_split

import d3rlpy

# import environments


# def inv_pend_symmetries():
#     env = environments.RewardAssymetricInvertedPendulum
#     return {
#         "rho": env.rho,
#         "phi": env.phi,
#         "psi": env.psi,
#         "R": env.R,
#         "gamma": env.gamma,
#         "group_inv": env.group_inv,
#         "submanifold_dim": env.submanifold_dim(),
#     }


def reacher_symmetries():
    def rho(x, gammax=None):
        assert x.ndim == 2
        assert x.shape[1] == 11

        costheta1 = x[:, 0:1]
        costheta2 = x[:, 1:2]
        sintheta1 = x[:, 2:3]
        sintheta2 = x[:, 3:4]
        xt = x[:, 4:5]
        yt = x[:, 5:6]
        theta1dot = x[:, 6:7]
        theta2dot = x[:, 7:8]
        xfmxt = x[:, 8:9]
        yfmyt = x[:, 9:10]
        zfmzt = x[:, 10:11]

        result = torch.hstack(
            (
                costheta2,
                sintheta2,
                costheta1 * xt + sintheta1 * yt,
                -sintheta1 * xt + costheta1 * yt,
                theta1dot,
                theta2dot,
                costheta1 * xfmxt + sintheta1 * yfmyt,
                -sintheta1 * xfmxt + costheta1 * yfmyt,
                zfmzt,
            )
        )
        assert result.shape[1] == 9
        return result

    def phi(alpha, x):
        assert alpha.shape[1] == 1
        assert x.ndim == 2
        assert x.shape[1] == 11

        costheta1 = x[:, 0:1]
        costheta2 = x[:, 1:2]
        sintheta1 = x[:, 2:3]
        sintheta2 = x[:, 3:4]
        xt = x[:, 4:5]
        yt = x[:, 5:6]
        theta1dot = x[:, 6:7]
        theta2dot = x[:, 7:8]
        xfmxt = x[:, 8:9]
        yfmyt = x[:, 9:10]
        zfmzt = x[:, 10:11]

        cosalpha = torch.cos(alpha)
        sinalpha = torch.sin(alpha)

        result = torch.hstack(
            (
                cosalpha * costheta1 - sinalpha * sintheta1,
                costheta2,
                sinalpha * costheta1 + cosalpha * sintheta1,
                sintheta2,
                cosalpha * xt - sinalpha * yt,
                sinalpha * xt + cosalpha * yt,
                theta1dot,
                theta2dot,
                cosalpha * xfmxt - sinalpha * yfmyt,
                sinalpha * xfmxt + cosalpha * yfmyt,
                zfmzt,
            )
        )
        assert result.shape[1] == 11
        return result

    def psi(alpha, u):
        return u

    def gamma(x):
        assert x.ndim == 2
        assert x.shape[1] == 11

        costheta1 = x[:, 0:1]
        sintheta1 = x[:, 2:3]

        result = torch.atan2(-sintheta1, costheta1)
        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == 1
        return result

    R = None

    def group_inv(alpha, x=None):
        # If x is not none, computes and returns inverse of gamma(x)
        assert x is not None
        assert x.ndim == 2 and x.shape[1] == 11

        costheta1 = x[:, 0:1]
        sintheta1 = x[:, 2:3]

        result = torch.atan2(sintheta1, costheta1)
        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == 1
        return result

    submanifold_dim = (9,)

    return {
        "rho": rho,
        "phi": phi,
        "psi": psi,
        "R": R,
        "gamma": gamma,
        "group_inv": group_inv,
        "submanifold_dim": submanifold_dim,
    }


def reacher_symmetries2():
    def rho(x, gammax=None):
        assert x.ndim == 2
        assert x.shape[1] == 11

        costheta1 = x[:, 0:1]
        costheta2 = x[:, 1:2]
        sintheta1 = x[:, 2:3]
        sintheta2 = x[:, 3:4]
        # xt = x[:, 4:5]
        # yt = x[:, 5:6]
        theta1dot = x[:, 6:7]
        theta2dot = x[:, 7:8]
        xfmxt = x[:, 8:9]
        yfmyt = x[:, 9:10]
        # zfmzt = x[:, 10:11]

        result = torch.hstack(
            (
                costheta2,
                sintheta2,
                # costheta1 * xt + sintheta1 * yt,
                # -sintheta1 * xt + costheta1 * yt,
                theta1dot,
                theta2dot,
                costheta1 * xfmxt + sintheta1 * yfmyt,
                -sintheta1 * xfmxt + costheta1 * yfmyt,
                # zfmzt,
            )
        )
        assert result.shape[1] == 6
        return result

    def phi(alpha, x):
        assert alpha.ndim == 2 and alpha.shape[1] == 4
        assert x.ndim == 2
        assert x.shape[1] == 11

        costheta1 = x[:, 0:1]
        costheta2 = x[:, 1:2]
        sintheta1 = x[:, 2:3]
        sintheta2 = x[:, 3:4]
        xt = x[:, 4:5]
        yt = x[:, 5:6]
        theta1dot = x[:, 6:7]
        theta2dot = x[:, 7:8]
        xfmxt = x[:, 8:9]
        yfmyt = x[:, 9:10]
        zfmzt = x[:, 10:11]

        thetaprime = alpha[:, 0:1]

        costhetaprime = torch.cos(thetaprime)
        sinthetaprime = torch.sin(thetaprime)

        result = torch.hstack(
            (
                costhetaprime * costheta1 - sinthetaprime * sintheta1,
                costheta2,
                sinthetaprime * costheta1 + costhetaprime * sintheta1,
                sintheta2,
                # cosalpha * xt - sinalpha * yt,
                # sinalpha * xt + cosalpha * yt,
                xt + alpha[:, 1:2],
                yt + alpha[:, 2:3],
                theta1dot,
                theta2dot,
                costhetaprime * xfmxt - sinthetaprime * yfmyt,
                sinthetaprime * xfmxt + costhetaprime * yfmyt,
                zfmzt + alpha[:, 3:4],
            )
        )
        assert result.shape[1] == 11
        return result

    def psi(alpha, u):
        return u

    def gamma(x):
        assert x.ndim == 2
        assert x.shape[1] == 11

        costheta1 = x[:, 0:1]
        sintheta1 = x[:, 2:3]
        xt = x[:, 4:5]
        yt = x[:, 5:6]
        zfmzt = x[:, 10:11]

        result0 = torch.atan2(-sintheta1, costheta1)
        assert result0.ndim == 2 and result0.shape == (x.shape[0], 1)
        result1 = -xt
        result2 = -yt
        result3 = -zfmzt
        result = torch.hstack((result0, result1, result2, result3))
        assert result.ndim == 2 and result.shape == (x.shape[0], 4)
        return result

    R = None

    def group_inv(alpha, x=None):
        # If x is not none, computes and returns inverse of gamma(x)
        assert x is not None
        assert x.ndim == 2 and x.shape[1] == 11

        costheta1 = x[:, 0:1]
        sintheta1 = x[:, 2:3]
        xt = x[:, 4:5]
        yt = x[:, 5:6]
        zfmzt = x[:, 10:11]

        result0 = torch.atan2(sintheta1, costheta1)
        assert result0.ndim == 2 and result0.shape == (x.shape[0], 1)
        result1 = xt
        result2 = yt
        result3 = zfmzt
        result = torch.hstack((result0, result1, result2, result3))
        assert result.ndim == 2 and result.shape == (x.shape[0], 4)
        return result

    submanifold_dim = (6,)

    return {
        "rho": rho,
        "phi": phi,
        "psi": psi,
        "R": R,
        "gamma": gamma,
        "group_inv": group_inv,
        "submanifold_dim": submanifold_dim,
    }


def reacher_symmetries3():
    print("Using Reacher Symmetries 3.")

    def rho(x, gammax=None):
        assert x.ndim == 2
        assert x.shape[1] == 11

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]
        x4 = x[:, 3:4]
        x5 = x[:, 4:5]
        x6 = x[:, 5:6]
        x7 = x[:, 6:7]
        x8 = x[:, 7:8]
        x9 = x[:, 8:9]
        x10 = x[:, 9:10]

        result = torch.hstack(
            (
                x2,
                x4,
                x7,
                x8,
                x1 * (x9 + x5) + x3 * (x10 + x6) - x5,
                -x3 * (x9 + x5) + x1 * (x10 + x6) - x6,
            )
        )
        assert result.shape[1] == 6
        return result

    def phi(alpha, x):
        assert alpha.ndim == 2 and alpha.shape[1] == 4
        assert x.ndim == 2
        assert x.shape[1] == 11

        thetaprime = alpha[:, 0:1]
        delta1 = alpha[:, 1:2]
        delta2 = alpha[:, 2:3]
        delta3 = alpha[:, 3:4]

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]
        x4 = x[:, 3:4]
        x5 = x[:, 4:5]
        x6 = x[:, 5:6]
        x7 = x[:, 6:7]
        x8 = x[:, 7:8]
        x9 = x[:, 8:9]
        x10 = x[:, 9:10]
        x11 = x[:, 10:11]

        costhetaprime = torch.cos(thetaprime)
        sinthetaprime = torch.sin(thetaprime)

        result = torch.hstack(
            (
                costhetaprime * x1 - sinthetaprime * x3,
                x2,
                sinthetaprime * x1 + costhetaprime * x3,
                x4,
                x5 + delta1,
                x6 + delta2,
                x7,
                x8,
                costhetaprime * (x9 + x5) - sinthetaprime * (x10 + x6) - x5,
                sinthetaprime * (x9 + x5) + costhetaprime * (x10 + x6) - x6,
                x11 + delta3,
            )
        )
        assert result.shape[1] == 11
        return result

    def psi(alpha, u):
        return u

    def gamma(x):
        assert x.ndim == 2
        assert x.shape[1] == 11

        x1 = x[:, 0:1]
        x3 = x[:, 2:3]
        x5 = x[:, 4:5]
        x6 = x[:, 5:6]
        x11 = x[:, 10:11]

        thetaprime = torch.atan2(-x3, x1)
        assert thetaprime.ndim == 2 and thetaprime.shape == (x.shape[0], 1)
        delta1 = -x5
        delta2 = -x6
        delta3 = -x11
        alpha = torch.hstack((thetaprime, delta1, delta2, delta3))
        assert alpha.ndim == 2 and alpha.shape == (x.shape[0], 4)
        return alpha

    R = None

    def group_inv(alpha, x=None):
        # If x is not none, computes and returns inverse of gamma(x)
        assert x is not None
        assert x.ndim == 2 and x.shape[1] == 11

        x1 = x[:, 0:1]
        x3 = x[:, 2:3]
        x5 = x[:, 4:5]
        x6 = x[:, 5:6]
        x11 = x[:, 10:11]

        thetaprime = torch.atan2(x3, x1)
        assert thetaprime.ndim == 2 and thetaprime.shape == (x.shape[0], 1)
        delta1 = x5
        delta2 = x6
        delta3 = x11
        alpha = torch.hstack((thetaprime, delta1, delta2, delta3))
        assert alpha.ndim == 2 and alpha.shape == (x.shape[0], 4)
        return alpha

    submanifold_dim = (6,)

    return {
        "rho": rho,
        "phi": phi,
        "psi": psi,
        "R": R,
        "gamma": gamma,
        "group_inv": group_inv,
        "submanifold_dim": submanifold_dim,
    }


def reacher_symmetries4():
    print("Using Reacher Symmetries 4.")

    def rho(x, gammax=None):
        assert x.ndim == 2
        assert x.shape[1] == 11

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]
        x4 = x[:, 3:4]
        x5 = x[:, 4:5]
        x6 = x[:, 5:6]
        x7 = x[:, 6:7]
        x8 = x[:, 7:8]
        x9 = x[:, 8:9]
        x10 = x[:, 9:10]

        result = torch.hstack(
            (
                x2,
                x4,
                x7,
                x8,
                x1 * (x9 + x5) + x3 * (x10 + x6),
                -x3 * (x9 + x5) + x1 * (x10 + x6),
            )
        )
        assert result.shape[1] == 6
        return result

    def phi(alpha, x):
        assert alpha.ndim == 2 and alpha.shape[1] == 4
        assert x.ndim == 2
        assert x.shape[1] == 11

        thetaprime = alpha[:, 0:1]
        delta1 = alpha[:, 1:2]
        delta2 = alpha[:, 2:3]
        delta3 = alpha[:, 3:4]

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]
        x4 = x[:, 3:4]
        x5 = x[:, 4:5]
        x6 = x[:, 5:6]
        x7 = x[:, 6:7]
        x8 = x[:, 7:8]
        x9 = x[:, 8:9]
        x10 = x[:, 9:10]
        x11 = x[:, 10:11]

        costhetaprime = torch.cos(thetaprime)
        sinthetaprime = torch.sin(thetaprime)

        result = torch.hstack(
            (
                costhetaprime * x1 - sinthetaprime * x3,
                x2,
                sinthetaprime * x1 + costhetaprime * x3,
                x4,
                costhetaprime * (x5 + delta1) - sinthetaprime * (x6 + delta2),
                sinthetaprime * (x5 + delta1) + costhetaprime * (x6 + delta2),
                x7,
                x8,
                costhetaprime * (x9 - delta1) - sinthetaprime * (x10 - delta2),
                sinthetaprime * (x9 - delta1) + costhetaprime * (x10 - delta2),
                x11 + delta3,
            )
        )
        assert result.shape[1] == 11
        return result

    def psi(alpha, u):
        return u

    def gamma(x):
        assert x.ndim == 2
        assert x.shape[1] == 11

        x1 = x[:, 0:1]
        x3 = x[:, 2:3]
        x5 = x[:, 4:5]
        x6 = x[:, 5:6]
        x11 = x[:, 10:11]

        thetaprime = torch.atan2(-x3, x1)
        assert thetaprime.ndim == 2 and thetaprime.shape == (x.shape[0], 1)
        delta1 = -x5
        delta2 = -x6
        delta3 = -x11
        alpha = torch.hstack((thetaprime, delta1, delta2, delta3))
        assert alpha.ndim == 2 and alpha.shape == (x.shape[0], 4)
        return alpha

    R = None

    def group_inv(alpha, x=None):
        # If x is not none, computes and returns inverse of gamma(x)
        assert x is not None
        assert x.ndim == 2 and x.shape[1] == 11

        x1 = x[:, 0:1]
        x3 = x[:, 2:3]
        x5 = x[:, 4:5]
        x6 = x[:, 5:6]
        x11 = x[:, 10:11]

        thetaprime = torch.atan2(x3, x1)
        assert thetaprime.ndim == 2 and thetaprime.shape == (x.shape[0], 1)
        delta1 = x1 * x5 + x3 * x6
        delta2 = -x3 * x5 + x1 * x6
        delta3 = x11
        alpha = torch.hstack((thetaprime, delta1, delta2, delta3))
        assert alpha.ndim == 2 and alpha.shape == (x.shape[0], 4)
        return alpha

    submanifold_dim = (6,)

    return {
        "rho": rho,
        "phi": phi,
        "psi": psi,
        "R": R,
        "gamma": gamma,
        "group_inv": group_inv,
        "submanifold_dim": submanifold_dim,
    }


def two_car_symmetries():
    def rho(x, gammax=None):
        assert x.ndim == 2
        assert x.shape[1] == 24
        x1 = x[:, :6]
        x2 = x[:, 6:12]
        x1g = x[:, 12:18]
        x2g = x[:, 18:24]

        # First do it for the first car
        v1x = x1[:, 2:3]
        v1y = x1[:, 3:4]
        h1x = x1[:, 4:5]
        h1y = x1[:, 5:6]
        rho1 = torch.hstack((h1x * v1x + h1y * v1y, -v1x * h1y + h1x * v1y))
        assert rho1.shape == (x.shape[0], 2)

        # Now second car
        v2x = x2[:, 2:3]
        v2y = x2[:, 3:4]
        h2x = x2[:, 4:5]
        h2y = x2[:, 5:6]
        rho2 = torch.hstack((h2x * v2x + h2y * v2y, -v2x * h2y + h2x * v2y))
        assert rho2.shape == (x.shape[0], 2)

        # Goal rho is simply empty

        rho = torch.hstack((rho1, rho2))
        assert rho.shape[0] == x.shape[0] and rho.shape[1] == 2 + 2
        return rho

    def phi(alpha, x):
        assert alpha.shape == (x.shape[0], 3 + 3 + 6 + 6)
        assert x.ndim == 2
        assert x.shape[1] == 24
        x1 = x[:, :6]
        x2 = x[:, 6:12]
        x1g = x[:, 12:18]
        x2g = x[:, 18:24]
        alpha1 = alpha[:, 0:3]
        alpha2 = alpha[:, 3:6]
        alpha1g = alpha[:, 6:12]
        alpha2g = alpha[:, 12:18]

        def _phi_car(alpha, xtilde):
            assert xtilde.ndim == 2 and alpha.ndim == 2
            assert xtilde.shape[1] == 6
            assert alpha.shape[0] == xtilde.shape[0]
            assert alpha.shape[1] == 3

            xprime = alpha[:, 0:1]
            yprime = alpha[:, 1:2]
            psiprime = alpha[:, 2:3]

            x = xtilde[:, 0:1]
            y = xtilde[:, 1:2]
            vx = xtilde[:, 2:3]
            vy = xtilde[:, 3:4]
            hx = xtilde[:, 4:5]
            hy = xtilde[:, 5:6]

            cospsiprime = torch.cos(psiprime)
            sinpsiprime = torch.sin(psiprime)

            def _rotate(x, y):
                return (
                    cospsiprime * x - sinpsiprime * y,
                    sinpsiprime * x + cospsiprime * y,
                )

            rot_x, rot_y = _rotate(x, y)
            rot_vx, rot_vy = _rotate(vx, vy)
            rot_hx, rot_hy = _rotate(hx, hy)

            res = torch.hstack(
                (rot_x + xprime, rot_y + yprime, rot_vx, rot_vy, rot_hx, rot_hy)
            )
            assert res.shape == xtilde.shape
            return res

        def _phi_goal(alpha, g):
            assert alpha.ndim == 2 and g.ndim == 2
            assert alpha.shape == g.shape and g.shape[1] == 6
            return g + alpha

        phi1 = _phi_car(alpha1, x1)
        phi2 = _phi_car(alpha2, x2)
        phi1g = _phi_goal(alpha1g, x1g)
        phi2g = _phi_goal(alpha2g, x2g)
        phi = torch.hstack((phi1, phi2, phi1g, phi2g))
        assert phi.shape == x.shape
        return phi

    def gamma(x):
        assert x.ndim == 2
        assert x.shape[1] == 24
        x1 = x[:, :6]
        x2 = x[:, 6:12]
        x1g = x[:, 12:18]
        x2g = x[:, 18:24]

        def _gamma_car(xtilde):
            assert xtilde.ndim == 2 and xtilde.shape[1] == 6
            x = xtilde[:, 0:1]
            y = xtilde[:, 1:2]
            hx = xtilde[:, 4:5]
            hy = xtilde[:, 5:6]
            gamma1 = -x * hx - y * hy
            gamma2 = x * hy - y * hx
            gamma3 = torch.atan2(-hy, hx)
            res = torch.hstack((gamma1, gamma2, gamma3))
            assert res.shape == (xtilde.shape[0], 3)
            return res

        def _gamma_goal(g):
            assert g.ndim == 2 and g.shape[1] == 6
            res = -g
            assert res.shape == (g.shape[0], 6)
            return res

        gamma1 = _gamma_car(x1)
        gamma2 = _gamma_car(x2)
        gamma1g = _gamma_goal(x1g)
        gamma2g = _gamma_goal(x2g)
        gamma = torch.hstack((gamma1, gamma2, gamma1g, gamma2g))
        assert gamma.shape == (x.shape[0], 18)

        return gamma

    def psi(alpha, u):
        return u

    R = None

    def group_inv(alpha, x=None):
        # If x is not none, computes and returns inverse of gamma(x)
        assert x is not None
        assert x.ndim == 2 and x.shape[1] == 24
        x1 = x[:, :6]
        x2 = x[:, 6:12]
        x1g = x[:, 12:18]
        x2g = x[:, 18:24]

        def _group_inv_car(xtilde):
            assert xtilde.ndim == 2 and xtilde.shape[1] == 6
            x = xtilde[:, 0:1]
            y = xtilde[:, 1:2]
            hx = xtilde[:, 4:5]
            hy = xtilde[:, 5:6]
            psiprime = torch.atan2(hy, hx)
            res = torch.hstack((x, y, psiprime))
            assert (
                res.ndim == 2 and res.shape[0] == xtilde.shape[0] and res.shape[1] == 3
            )
            return res

        def _group_inv_goal(g):
            return g

        res1 = _group_inv_car(x1)
        res2 = _group_inv_car(x2)
        res3 = _group_inv_goal(x1g)
        res4 = _group_inv_goal(x2g)
        res = torch.hstack((res1, res2, res3, res4))
        assert res.ndim == 2
        assert res.shape[0] == x.shape[0]
        assert res.shape[1] == 3 + 3 + 6 + 6

        return res

    submanifold_dim = (2 + 2 + 0 + 0,)

    return {
        "rho": rho,
        "phi": phi,
        "psi": psi,
        "R": R,
        "gamma": gamma,
        "group_inv": group_inv,
        "submanifold_dim": submanifold_dim,
    }


def no_symmetry():
    def rho(x, gammax=None):
        return x

    def phi(alpha, x):
        return x

    def gamma(x):
        return 0

    def psi(alpha, u):
        return u

    R = None

    def group_inv(alpha, x=None):
        return 0

    submanifold_dim = (11,)
    # (24,) for two_car, (11,) for reacher_symmetries and reacher_symmetries2

    return {
        "rho": rho,
        "phi": phi,
        "psi": psi,
        "R": R,
        "gamma": gamma,
        "group_inv": group_inv,
        "submanifold_dim": submanifold_dim,
    }


# N_POLICY_STEPS = 50000  # 100000
# N_RUNS = 3

# N_EPOCHS = 100
N_STEPS = 1000000
N_STEPS_PER_EPOCH = 5000

# TASK_ID = int(os.getenv("TASK_ID"))
SEED = 0  # int(os.getenv("SEED"))
USE_GPU = True  # int(os.getenv("USE_GPU"))

# assert TASK_ID in [0, 1, 2, 3]
# if TASK_ID == 0:
#     SYMMETRY = True
#     EXPERIMENT_NAME = f"TwoCarsCos_Symm_2Layer_SEED{SEED}"
#     hidden_units = [256, 256]
# elif TASK_ID == 1:
#     SYMMETRY = False
#     EXPERIMENT_NAME = f"TwoCarsCos_NoSymm_2Layer_SEED{SEED}"
#     hidden_units = [256, 256]
# elif TASK_ID == 2:
#     SYMMETRY = True
#     EXPERIMENT_NAME = f"TwoCarsCos_Symm_3Layer_SEED{SEED}"
#     hidden_units = [256, 256, 256]
# elif TASK_ID == 3:
#     SYMMETRY = False
#     EXPERIMENT_NAME = f"TwoCarsCos_NoSymm_3Layer_SEED{SEED}"
#     hidden_units = [256, 256, 256]

SYMMETRY = True
EXPERIMENT_NAME = "Test_Reacher"
hidden_units = [64]

d3rlpy.seed(SEED)

if USE_GPU == 1:
    use_gpu = True
elif USE_GPU == 0:
    use_gpu = False
else:
    raise ValueError(f"USE_GPU must be 0 or 1. Was {USE_GPU}")

print("===================================")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Symmetry: {SYMMETRY}")
print(f"Seed: {SEED}")
print(f"Gpu: {use_gpu}")
print("===================================")


# env = environments.RewardAssymetricInvertedPendulum()
# eval_env = environments.RewardAssymetricInvertedPendulum()
# env.reset(seed=seed)
# eval_env.reset(seed=seed)
# dataset = d3rlpy.dataset.MDPDataset.load("d3rlpy_data/rwd_assym_inv_pend_v8.h5")

dataset = d3rlpy.dataset.MDPDataset.load(
    # "d3rlpy_data/sac_parking_replay_buffer_2_cars_any_car_termination_d3rlpy.h5"
    "d3rlpy_data/reacher.h5"
)
train_episodes, test_episodes = train_test_split(
    dataset,
    random_state=SEED,  # , train_size=0.9
)

encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=hidden_units)

if SYMMETRY:
    print("Using symmetry")
    # symms = inv_pend_symmetries()
    # symms = two_car_symmetries()
    # symms = reacher_symmetries2()
    # symms = reacher_symmetries3()
    symms = reacher_symmetries4()
else:
    print("Not using symmetry")
    symms = no_symmetry()

dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(
    learning_rate=3e-4,
    use_gpu=use_gpu,
    n_ensembles=3,
    cartans_deterministic=True,
    cartans_stochastic=False,
    cartans_rho=symms["rho"],
    cartans_phi=symms["phi"],
    cartans_psi=symms["psi"],
    cartans_R=symms["R"],
    cartans_gamma=symms["gamma"],
    cartans_group_inv=symms["group_inv"],
    cartans_submanifold_dim=symms["submanifold_dim"],
    cartans_encoder_factory=encoder_factory,
)

dynamics.fit(
    train_episodes,
    eval_episodes=test_episodes,
    n_steps=N_STEPS,
    n_steps_per_epoch=N_STEPS_PER_EPOCH,
    scorers={
        "observation_error": d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
        "reward_error": d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
        "variance": d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
    },
    tensorboard_dir="tensorboard_logs/dynamics/reacher",
    experiment_name=EXPERIMENT_NAME,
    save_interval=20,
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
