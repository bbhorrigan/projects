#!/usr/bin/env python3
# vr_tteg_demo.py
# Empirical scaffolding for VR-TTEG convergence to local minimax
# Requirements: numpy, matplotlib

import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, Optional

# -------------------------
# Utilities
# -------------------------

def clip_norm(v, max_norm=1e6):
    n = np.linalg.norm(v)
    if n > max_norm and n > 0:
        return v * (max_norm / n)
    return v

def eigmax_sym(A):
    # Largest eigenvalue of symmetric 1x1 or 2x2 matrix; fallback to np.linalg.eigvalsh
    A = np.atleast_2d(A)
    return float(np.linalg.eigvalsh(A)[-1])

# -------------------------
# Game definitions
# -------------------------

@dataclass
class Game:
    name: str
    # f: R^dx x R^dy -> R  (here dx=dy=1 for clarity/plotting)
    f: Callable[[float, float], float]
    grad_x: Callable[[float, float], float]
    grad_y: Callable[[float, float], float]
    hyy: Callable[[float, float], float]  # ∂^2 f / ∂y^2

def make_games():
    games = {}

    # G1: Non-strict local minimax with coupling:
    # f(x,y) = eps*x*y - y^4
    # At (0,0): ∇f=0, H_yy = 0 (non-strict). For fixed x, y* = (eps*x/4)^{1/3} (unique if x!=0).
    eps = 0.5
    def f1(x, y): return eps*x*y - y**4
    def gx1(x, y): return eps*y
    def gy1(x, y): return eps*x - 4*y**3
    def hyy1(x, y): return -12*y**2
    games["non_strict_coupled"] = Game("non_strict_coupled", f1, gx1, gy1, hyy1)

    # G2: Bilinear (cycling if no timescale sep / no EG; inner max ill-posed without reg)
    def f2(x, y): return x*y
    def gx2(x, y): return y
    def gy2(x, y): return x
    def hyy2(x, y): return 0.0
    games["bilinear"] = Game("bilinear", f2, gx2, gy2, hyy2)

    # G3: Strict local minimax example:
    # f(x,y) = x^2*y - y^2. For fixed x, y* = x^2/2; φ(x)=x^4/4; LM at (0,0).
    def f3(x, y): return x**2 * y - y**2
    def gx3(x, y): return 2*x*y
    def gy3(x, y): return x**2 - 2*y
    def hyy3(x, y): return -2.0
    games["strict_lm"] = Game("strict_lm", f3, gx3, gy3, hyy3)

    return games

# -------------------------
# Algorithms
# -------------------------

@dataclass
class Schedules:
    eta0: float = 0.2      # fast y stepsize base
    alpha0: float = 0.02   # slow x stepsize base
    a: float = 0.6         # eta_k = eta0 / (k+1)^a
    b: float = 0.9         # alpha_k = alpha0 / (k+1)^b  (alpha/eta -> 0)
    rho: float = 0.2       # sigma_k = (k+1)^(-rho)
    sigma0: float = 1.0    # optional scale for sigma
    noise_std: float = 0.0 # small noise in y-updates (optional)

def vr_tteg(
    game: Game,
    x0: float,
    y0: float,
    steps: int = 20000,
    sched: Schedules = Schedules(),
    use_extragradient: bool = True,
    timescale_separation: bool = True,
    use_vanishing_reg: bool = True,
    track_every: int = 50,
) -> Dict[str, Any]:
    """
    VR-TTEG with toggles to ablate ingredients.
    Returns trajectory and diagnostics.
    """
    x, y = float(x0), float(y0)
    xs, ys, fs = [], [], []

    for k in range(steps):
        eta = sched.eta0 / ((k + 1) ** sched.a)
        alpha = sched.alpha0 / ((k + 1) ** sched.b) if timescale_separation else sched.eta0 / ((k + 1) ** sched.a)
        sigma = (sched.sigma0 / ((k + 1) ** sched.rho)) if use_vanishing_reg else 0.0

        # --- y fast dynamics (ascent on g_sigma)
        gy = game.grad_y(x, y) - sigma * y
        if use_extragradient:
            yp = y + eta * gy
            gy_p = game.grad_y(x, yp) - sigma * yp
            y = y + eta * gy_p
        else:
            y = y + eta * gy

        # optional tiny noise to mimic strict-saddle escape (off by default)
        if sched.noise_std > 0.0:
            y += np.random.normal(scale=sched.noise_std)

        # --- x slow dynamics (descent on value gradient)
        gx = game.grad_x(x, y)
        if use_extragradient:
            xp = x - alpha * gx
            gx_p = game.grad_x(xp, y)
            x = x - alpha * gx_p
        else:
            x = x - alpha * gx

        if (k % track_every) == 0 or k == steps - 1:
            xs.append(x); ys.append(y); fs.append(game.f(x, y))

    return {
        "x": np.array(xs),
        "y": np.array(ys),
        "f": np.array(fs),
        "track_every": track_every,
        "final": (x, y),
        "steps": steps,
        "sched": sched,
        "game": game.name,
        "flags": {
            "EG": use_extragradient,
            "TTS": timescale_separation,
            "REG": use_vanishing_reg,
        }
    }

# -------------------------
# Numerical LM checks
# -------------------------

def inner_argmax_y(game: Game, x: float, y_init: float, sigma: float, iters: int = 200, eta: float = 0.2):
    """Ascent on g_sigma to approximate y*(x)."""
    y = float(y_init)
    for _ in range(iters):
        gy = game.grad_y(x, y) - sigma * y
        y = y + eta * gy
    return y

def check_local_minimax(game: Game, x: float, y: float, sigma_probe: float = 1e-2, tol: float = 1e-2) -> Dict[str, Any]:
    """
    Checks:
      (1) ||∇_y f(x,y)|| small
      (2) λ_max(H_yy(x,y)) <= tol  (nonpositive curvature)
      (3) y*(x) via regularized ascent; then ||∇_x f(x, y*(x))|| small
    """
    gy = game.grad_y(x, y)
    hyy = game.hyy(x, y)
    yn = inner_argmax_y(game, x, y, sigma=sigma_probe, iters=500, eta=0.1)
    gx_val = game.grad_x(x, yn)

    cond1 = np.linalg.norm([gy]) <= tol
    cond2 = (hyy <= tol)  # scalar case
    cond3 = np.linalg.norm([gx_val]) <= tol

    return {
        "grad_y_norm": float(abs(gy)),
        "hyy": float(hyy),
        "grad_x_at_y_star_norm": float(abs(gx_val)),
        "is_local_minimax_like": bool(cond1 and cond2 and cond3),
        "x": float(x),
        "y": float(y),
        "y_star_probe": float(yn),
    }

# -------------------------
# Experiments
# -------------------------

def run_experiments(show_plots=True):
    games = make_games()
    results = []

    # Common init
    x0, y0 = 1.0, -1.0

    # Schedules tuned for stability
    base_sched = Schedules(eta0=0.2, alpha0=0.02, a=0.6, b=0.9, rho=0.2, sigma0=1.0, noise_std=0.0)

    # 1) Target: Non-strict local minimax (with coupling)
    res_ok = vr_tteg(games["non_strict_coupled"], x0, y0, steps=15000, sched=base_sched,
                     use_extragradient=True, timescale_separation=True, use_vanishing_reg=True, track_every=50)
    chk_ok = check_local_minimax(games["non_strict_coupled"], res_ok["final"][0], res_ok["final"][1])
    results.append(("non_strict_coupled / VR-TTEG", res_ok, chk_ok))

    # Ablation A: remove timescale separation (alpha ~ eta) -> expect worse stability
    res_no_tts = vr_tteg(games["non_strict_coupled"], x0, y0, steps=15000, sched=base_sched,
                         use_extragradient=True, timescale_separation=False, use_vanishing_reg=True, track_every=50)
    chk_no_tts = check_local_minimax(games["non_strict_coupled"], res_no_tts["final"][0], res_no_tts["final"][1])
    results.append(("non_strict_coupled / no TTS", res_no_tts, chk_no_tts))

    # Ablation B: remove extragradient
    res_no_eg = vr_tteg(games["non_strict_coupled"], x0, y0, steps=15000, sched=base_sched,
                        use_extragradient=False, timescale_separation=True, use_vanishing_reg=True, track_every=50)
    chk_no_eg = check_local_minimax(games["non_strict_coupled"], res_no_eg["final"][0], res_no_eg["final"][1])
    results.append(("non_strict_coupled / no EG", res_no_eg, chk_no_eg))

    # Ablation C: remove vanishing regularization (sigma=0) -> inner max can be ill-posed in some games
    res_no_reg = vr_tteg(games["non_strict_coupled"], x0, y0, steps=15000, sched=base_sched,
                         use_extragradient=True, timescale_separation=True, use_vanishing_reg=False, track_every=50)
    chk_no_reg = check_local_minimax(games["non_strict_coupled"], res_no_reg["final"][0], res_no_reg["final"][1])
    results.append(("non_strict_coupled / no REG", res_no_reg, chk_no_reg))

    # 2) Bilinear: demonstrate necessity of reg/TTS to avoid cycling/divergence
    res_bi_ok = vr_tteg(games["bilinear"], x0, y0, steps=8000, sched=base_sched,
                        use_extragradient=True, timescale_separation=True, use_vanishing_reg=True, track_every=20)
    results.append(("bilinear / VR-TTEG", res_bi_ok, check_local_minimax(games["bilinear"], res_bi_ok["final"][0], res_bi_ok["final"][1])))

    res_bi_no_tts = vr_tteg(games["bilinear"], x0, y0, steps=8000, sched=base_sched,
                            use_extragradient=True, timescale_separation=False, use_vanishing_reg=True, track_every=20)
    results.append(("bilinear / no TTS", res_bi_no_tts, check_local_minimax(games["bilinear"], res_bi_no_tts["final"][0], res_bi_no_tts["final"][1])))

    res_bi_no_reg = vr_tteg(games["bilinear"], x0, y0, steps=8000, sched=base_sched,
                            use_extragradient=True, timescale_separation=True, use_vanishing_reg=False, track_every=20)
    results.append(("bilinear / no REG", res_bi_no_reg, check_local_minimax(games["bilinear"], res_bi_no_reg["final"][0], res_bi_no_reg["final"][1])))

    # 3) Strict LM example: should be easy for most variants; good sanity check
    res_strict = vr_tteg(games["strict_lm"], x0, y0, steps=8000, sched=base_sched,
                         use_extragradient=True, timescale_separation=True, use_vanishing_reg=True, track_every=20)
    results.append(("strict_lm / VR-TTEG", res_strict, check_local_minimax(games["strict_lm"], res_strict["final"][0], res_strict["final"][1])))

    # Print diagnostics
    print("\n=== SUMMARY ===")
    for tag, res, chk in results:
        xF, yF = res["final"]
        print(f"{tag:30s} -> final (x,y)=({xF:.4e},{yF:.4e});  "
              f"LM-like={chk['is_local_minimax_like']}  "
              f"[||gy||={chk['grad_y_norm']:.2e}, hyy={chk['hyy']:.2e}, ||gx@y*||={chk['grad_x_at_y_star_norm']:.2e}]")

    # Optional plots (phase portraits)
    if show_plots:
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        plots = [
            ("non_strict_coupled / VR-TTEG", res_ok),
            ("non_strict_coupled / no TTS", res_no_tts),
            ("non_strict_coupled / no EG", res_no_eg),
            ("bilinear / VR-TTEG", res_bi_ok),
            ("bilinear / no TTS", res_bi_no_tts),
            ("bilinear / no REG", res_bi_no_reg),
        ]
        for ax, (tag, res) in zip(axs.ravel(), plots):
            ax.plot(res["x"], res["y"], linewidth=1.0)
            ax.scatter([res["x"][0]], [res["y"][0]], marker='o')       # start
            ax.scatter([res["x"][-1]], [res["y"][-1]], marker='x')     # end
            ax.set_title(tag, fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, linewidth=0.5)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_experiments(show_plots=True)
