#!/usr/bin/env python3
"""Generate all numerical outputs for the Agentic AI governance manuscript.

The script is the executable reference used to generate the figures and CSV tables
included in the LaTeX manuscript. It implements the same primitives and equations
stated in the paper:

- Two persistent reliability types H and L.
- Exponential success technology.
- Marketplace pooling benchmark with a common auditable protocol e_M(mu).
- Induced type values under the common protocol.
- Separating accountability checks based on high-type internalization and low-type non-mimicry.
- Bayesian posteriors, learning paths, marginal value decomposition, and policy diagnostics.

The companion MATLAB file mirrors the formulas and output logic for editable replication.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.8,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.8,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 320,
    "figure.dpi": 160,
})

COL = {
    "blue": "#315AA6",
    "green": "#2A8C6A",
    "orange": "#D47A2A",
    "red": "#B54A45",
    "purple": "#6D53A6",
    "gray": "#5D6673",
    "dark": "#202A35",
    "light": "#EEF2F7",
    "gold": "#C79A2B",
    "teal": "#297F8E",
}


@dataclass(frozen=True)
class Params:
    thetaH: float = 0.92
    thetaL: float = 0.62
    kappa: float = 1.35
    gamma: float = 0.82
    L: float = 0.42
    tau: float = 0.115
    delta: float = 0.92
    mu0: float = 0.50
    rhoM: float = 0.30
    rhoR: float = 0.31
    a: float = 0.95
    emax: float = 3.20
    lambdaM: float = 0.10
    lambdaR: float = 1.00
    n_mu: int = 241
    n_e: int = 220
    maxit: int = 2500
    tol: float = 1e-8

P = Params()


def success(theta: float | np.ndarray, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    return 1.0 - (1.0 - np.asarray(theta)) * np.exp(-p.kappa * np.asarray(e))


def dsuccess(theta: float | np.ndarray, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    return p.kappa * (1.0 - np.asarray(theta)) * np.exp(-p.kappa * np.asarray(e))


def expected_success(mu: float | np.ndarray, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    return np.asarray(mu) * success(p.thetaH, e, p) + (1.0 - np.asarray(mu)) * success(p.thetaL, e, p)


def dexp_success(mu: float | np.ndarray, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    return np.asarray(mu) * dsuccess(p.thetaH, e, p) + (1.0 - np.asarray(mu)) * dsuccess(p.thetaL, e, p)


def posterior_success(mu: float | np.ndarray, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    mu = np.asarray(mu)
    sh, sl = success(p.thetaH, e, p), success(p.thetaL, e, p)
    denom = mu * sh + (1.0 - mu) * sl
    return np.divide(mu * sh, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def posterior_failure(mu: float | np.ndarray, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    mu = np.asarray(mu)
    fh, fl = 1.0 - success(p.thetaH, e, p), 1.0 - success(p.thetaL, e, p)
    denom = mu * fh + (1.0 - mu) * fl
    return np.divide(mu * fh, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def pi_m_public(mu: float | np.ndarray, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    q = expected_success(mu, e, p)
    demand = p.a + p.gamma * q
    expected_exposure = p.lambdaM * p.L * (1.0 - q)
    return p.rhoM * demand**2 - p.tau * np.asarray(e)**2 - expected_exposure


def pi_m_type(mu: float | np.ndarray, theta: float, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    q_public = expected_success(mu, e, p)
    demand = p.a + p.gamma * q_public
    direct_exposure = p.lambdaM * p.L * (1.0 - success(theta, e, p))
    return p.rhoM * demand**2 - p.tau * np.asarray(e)**2 - direct_exposure


def pi_r_signal(mu_signal: float | np.ndarray, theta: float, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    q_public = expected_success(mu_signal, e, p)
    demand = p.a + p.gamma * q_public
    direct_exposure = p.lambdaR * p.L * (1.0 - success(theta, e, p))
    return p.rhoR * demand**2 - p.tau * np.asarray(e)**2 - direct_exposure


def d_pi_m_public(mu: float, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    q = expected_success(mu, e, p)
    dq = dexp_success(mu, e, p)
    return 2.0 * p.rhoM * (p.a + p.gamma * q) * p.gamma * dq - 2.0 * p.tau * np.asarray(e) + p.lambdaM * p.L * dq


def d_pi_r_signal(mu_signal: float, theta: float, e: float | np.ndarray, p: Params = P) -> np.ndarray:
    q = expected_success(mu_signal, e, p)
    dq = dexp_success(mu_signal, e, p)
    ds = dsuccess(theta, e, p)
    return 2.0 * p.rhoR * (p.a + p.gamma * q) * p.gamma * dq - 2.0 * p.tau * np.asarray(e) + p.lambdaR * p.L * ds


def grid_argmax(values: np.ndarray, grid: np.ndarray) -> Tuple[float, float]:
    idx = int(np.nanargmax(values))
    return float(grid[idx]), float(values[idx])


def static_opt(func: Callable[[np.ndarray], np.ndarray], p: Params = P, n: int = 1200) -> Tuple[float, float]:
    e = np.linspace(0.0, p.emax, n)
    return grid_argmax(func(e), e)


def solve_pooling_bellman(p: Params = P) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu_grid = np.linspace(0.001, 0.999, p.n_mu)
    e_grid = np.linspace(0.0, p.emax, p.n_e)
    W = np.zeros_like(mu_grid)
    policy = np.zeros_like(mu_grid)
    residuals = []

    for _ in range(p.maxit):
        Wn = np.empty_like(W)
        pn = np.empty_like(policy)
        for i, mu in enumerate(mu_grid):
            ps = np.clip(posterior_success(mu, e_grid, p), mu_grid[0], mu_grid[-1])
            pf = np.clip(posterior_failure(mu, e_grid, p), mu_grid[0], mu_grid[-1])
            q = expected_success(mu, e_grid, p)
            cont = q * np.interp(ps, mu_grid, W) + (1.0 - q) * np.interp(pf, mu_grid, W)
            vals = pi_m_public(mu, e_grid, p) + p.delta * cont
            idx = int(np.argmax(vals))
            Wn[i] = vals[idx]
            pn[i] = e_grid[idx]
        resid = float(np.max(np.abs(Wn - W)))
        residuals.append(resid)
        W, policy = Wn, pn
        if resid < p.tol:
            break
    return mu_grid, W, policy, np.asarray(residuals)


def induced_type_values(theta: float, mu_grid: np.ndarray, policy: np.ndarray, p: Params = P) -> Tuple[np.ndarray, np.ndarray]:
    V = np.zeros_like(mu_grid)
    residuals = []
    for _ in range(p.maxit):
        Vn = np.empty_like(V)
        for i, mu in enumerate(mu_grid):
            e = float(policy[i])
            ps = float(np.clip(posterior_success(mu, e, p), mu_grid[0], mu_grid[-1]))
            pf = float(np.clip(posterior_failure(mu, e, p), mu_grid[0], mu_grid[-1]))
            q_true = float(success(theta, e, p))
            cont = q_true * np.interp(ps, mu_grid, V) + (1.0 - q_true) * np.interp(pf, mu_grid, V)
            Vn[i] = float(pi_m_type(mu, theta, e, p)) + p.delta * cont
        resid = float(np.max(np.abs(Vn - V)))
        residuals.append(resid)
        V = Vn
        if resid < p.tol:
            break
    return V, np.asarray(residuals)


def perpetual_value(current_payoff: float, p: Params = P) -> float:
    return current_payoff / (1.0 - p.delta)


def accountability_values(p: Params = P, n: int = 1200) -> Dict[str, float]:
    e_h_r, pi_h_r = static_opt(lambda e: pi_r_signal(1.0, p.thetaH, e, p), p, n=n)
    e_l_r_mimic, pi_l_r_mimic = static_opt(lambda e: pi_r_signal(1.0, p.thetaL, e, p), p, n=n)
    e_l_m, pi_l_m = static_opt(lambda e: pi_m_type(0.0, p.thetaL, e, p), p, n=n)
    e_h_m_dev, pi_h_m_dev = static_opt(lambda e: pi_m_type(0.0, p.thetaH, e, p), p, n=n)
    return {
        "e_H_R": e_h_r,
        "pi_H_R": pi_h_r,
        "U_H_R": perpetual_value(pi_h_r, p),
        "e_L_R_mimic": e_l_r_mimic,
        "pi_L_R_mimic": pi_l_r_mimic,
        "U_L_R_mimic": perpetual_value(pi_l_r_mimic, p),
        "e_L_M": e_l_m,
        "pi_L_M": pi_l_m,
        "U_L_M": perpetual_value(pi_l_m, p),
        "e_H_M_dev": e_h_m_dev,
        "pi_H_M_dev": pi_h_m_dev,
        "U_H_M_dev": perpetual_value(pi_h_m_dev, p),
    }




def separating_ic_margins_fast(p: Params = P, n: int = 420) -> Dict[str, float]:
    acc = accountability_values(p, n=n)
    return {
        "high_accountability_margin": acc["U_H_R"] - acc["U_H_M_dev"],
        "low_antimimic_margin": acc["U_L_M"] - acc["U_L_R_mimic"],
        **acc,
    }

def dynamic_pool_deviation_margins(mu_grid: np.ndarray, VH: np.ndarray, VL: np.ndarray, p: Params = P) -> Tuple[np.ndarray, np.ndarray]:
    acc = accountability_values(p)
    return VH - acc["U_H_R"], VL - acc["U_L_R_mimic"]


def separating_ic_margins(p: Params = P) -> Dict[str, float]:
    acc = accountability_values(p)
    return {
        "high_accountability_margin": acc["U_H_R"] - acc["U_H_M_dev"],
        "low_antimimic_margin": acc["U_L_M"] - acc["U_L_R_mimic"],
        **acc,
    }


def classify_governance_static(mu: float, L: float, p: Params = P) -> int:
    """Classify local governance structure under static/perpetual ICs.

    0 = no pure recommendation / conflicting incentives
    1 = marketplace pooling robust to both R deviations (local benchmark)
    2 = separating accountability feasible (H -> R, L -> M)
    3 = both feasible; equilibrium selection is refinement-dependent
    """
    pp = replace(p, L=L)
    # Pooling deviation test uses static perpetual M values at public belief mu.
    e_h_m_pool, pi_h_m_pool = static_opt(lambda e: pi_m_type(mu, pp.thetaH, e, pp), pp, n=420)
    e_l_m_pool, pi_l_m_pool = static_opt(lambda e: pi_m_type(mu, pp.thetaL, e, pp), pp, n=420)
    acc = accountability_values(pp, n=420)
    pool_H = perpetual_value(pi_h_m_pool, pp) - acc["U_H_R"]
    pool_L = perpetual_value(pi_l_m_pool, pp) - acc["U_L_R_mimic"]
    pooling = (pool_H >= 0.0) and (pool_L >= 0.0)
    sep = (acc["U_H_R"] - acc["U_H_M_dev"] >= 0.0) and (acc["U_L_M"] - acc["U_L_R_mimic"] >= 0.0)
    if pooling and sep:
        return 3
    if pooling:
        return 1
    if sep:
        return 2
    return 0


def bayes_martingale_error(mu: float, e: np.ndarray, p: Params = P) -> np.ndarray:
    q = expected_success(mu, e, p)
    return q * posterior_success(mu, e, p) + (1.0 - q) * posterior_failure(mu, e, p) - mu


def simulate_belief_paths(theta: float, mu_grid: np.ndarray, policy: np.ndarray, p: Params = P, reps: int = 600, horizon: int = 40, seed: int = 20260518) -> np.ndarray:
    rng = np.random.default_rng(seed + (1 if theta == p.thetaH else 7))
    paths = np.zeros((reps, horizon + 1))
    paths[:, 0] = p.mu0
    for r in range(reps):
        mu = p.mu0
        for t in range(horizon):
            e = float(np.interp(mu, mu_grid, policy))
            y = rng.random() < float(success(theta, e, p))
            mu = float(posterior_success(mu, e, p) if y else posterior_failure(mu, e, p))
            mu = min(max(mu, 0.001), 0.999)
            paths[r, t + 1] = mu
    return paths


def marginal_decomposition(mu0: float, mu_grid: np.ndarray, W: np.ndarray, p: Params = P) -> Tuple[np.ndarray, np.ndarray]:
    egrid = np.linspace(0.02, p.emax, 280)
    dW = np.gradient(W, mu_grid)
    comps = np.zeros((len(egrid), 4))
    h = 1e-4
    for i, e in enumerate(egrid):
        q = float(expected_success(mu0, e, p))
        dq = float(dexp_success(mu0, e, p))
        ms = float(posterior_success(mu0, e, p))
        mf = float(posterior_failure(mu0, e, p))
        dms = float((posterior_success(mu0, e + h, p) - posterior_success(mu0, e - h, p)) / (2.0 * h))
        dmf = float((posterior_failure(mu0, e + h, p) - posterior_failure(mu0, e - h, p)) / (2.0 * h))
        current = float((pi_m_public(mu0, e + h, p) - pi_m_public(mu0, e - h, p)) / (2.0 * h))
        success_prob = p.delta * dq * (np.interp(ms, mu_grid, W) - np.interp(mf, mu_grid, W))
        posterior_location = p.delta * (q * np.interp(ms, mu_grid, dW) * dms + (1.0 - q) * np.interp(mf, mu_grid, dW) * dmf)
        comps[i, :] = [current, success_prob, posterior_location, current + success_prob + posterior_location]
    return egrid, comps


def audit_posterior(mu: float, accuracy: float, favorable: bool = True) -> float:
    # Symmetric binary audit: P(good report | H)=accuracy, P(good report | L)=1-accuracy.
    if favorable:
        num = mu * accuracy
        den = mu * accuracy + (1.0 - mu) * (1.0 - accuracy)
    else:
        num = mu * (1.0 - accuracy)
        den = mu * (1.0 - accuracy) + (1.0 - mu) * accuracy
    return num / den if den > 0 else mu


def policy_counterfactuals(p: Params = P) -> list[dict[str, float | str]]:
    scenarios = []
    configs = [
        ("Baseline", p, p.mu0, "Original primitives"),
        ("Liability mandate", replace(p, L=1.00), p.mu0, "Higher direct failure exposure"),
        ("Certified audit", p, audit_posterior(p.mu0, 0.78, True), "Favorable public audit with 78% accuracy"),
        ("Safe harbor", replace(p, lambdaR=0.72), p.mu0, "Verified internalizers bear lower residual liability"),
        ("Hybrid", replace(p, L=1.20, lambdaR=0.90), audit_posterior(p.mu0, 0.78, True), "Audit plus calibrated liability"),
    ]
    for name, pp, mu, note in configs:
        acc = accountability_values(pp)
        sep = separating_ic_margins(pp)
        e_m, pi_m = static_opt(lambda e: pi_m_public(mu, e, pp), pp)
        success_m = float(expected_success(mu, e_m, pp))
        pool_h = perpetual_value(static_opt(lambda e: pi_m_type(mu, pp.thetaH, e, pp), pp)[1], pp) - acc["U_H_R"]
        pool_l = perpetual_value(static_opt(lambda e: pi_m_type(mu, pp.thetaL, e, pp), pp)[1], pp) - acc["U_L_R_mimic"]
        scenarios.append({
            "scenario": name,
            "mu_conditioned": mu,
            "L": pp.L,
            "lambdaR": pp.lambdaR,
            "eM_static": e_m,
            "market_success": success_m,
            "high_sep_margin": sep["high_accountability_margin"],
            "low_antimimic_margin": sep["low_antimimic_margin"],
            "pool_high_margin": pool_h,
            "pool_low_margin": pool_l,
            "note": note,
        })
    return scenarios


def write_csv(path: str, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save(fig: plt.Figure, basename: str, png: bool = True) -> None:
    fig.savefig(os.path.join(FIG, basename + ".pdf"), bbox_inches="tight")
    if png:
        fig.savefig(os.path.join(FIG, basename + ".png"), bbox_inches="tight")
    plt.close(fig)


def panel_style(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18, linewidth=0.55)
    ax.margins(x=0.02)


def add_zero(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="black", lw=0.95, alpha=0.80)


def make_fig1_timing() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 7.55)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, tc: str = "white", fs: float = 8.4) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.15", facecolor=fc, edgecolor=fc, linewidth=1.1, alpha=0.98)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", color=tc, fontsize=fs, wrap=True)

    def arrow(x1: float, y1: float, x2: float, y2: float, rad: float = 0.0) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.05, color=COL["gray"], connectionstyle=f"arc3,rad={rad}"))

    ax.text(6.1, 7.25, "Agentic AI liability governance: hidden reliability, public protocol, dynamic learning", ha="center", va="center", fontsize=12.3, fontweight="bold", color=COL["dark"])
    box(0.32, 5.78, 1.95, 0.88, "Private telemetry\n$\\theta\\in\\{H,L\\}$", COL["blue"])
    box(2.72, 5.78, 2.15, 0.88, "Observed regime\n$g\\in\\{M,R\\}$", COL["orange"])
    box(5.33, 5.78, 2.35, 0.88, "Public posterior\n$\\hat\\mu_g$", COL["purple"])
    box(8.08, 5.78, 2.33, 0.88, "Safeguard protocol\n$e_g$", COL["green"])
    box(10.78, 5.78, 1.12, 0.88, "Outcome\n$y$", COL["gray"])
    for a, b in [(2.27, 2.72), (4.87, 5.33), (7.68, 8.08), (10.41, 10.78)]:
        arrow(a, 6.22, b, 6.22)

    box(0.55, 4.12, 3.02, 0.92, "Marketplace pooling benchmark\ncommon certifiable protocol $e_M(\\mu)$", COL["blue"])
    box(4.12, 4.12, 3.34, 0.92, "Accountability-separation candidate\n$H\\to R$, $L\\to M$", COL["red"])
    box(8.05, 4.12, 3.55, 0.92, "Deviation diagnostics\n$\\Delta_H^{acct}$, $\\Delta_L^{anti}$, $\\Phi_\\theta^M$", COL["gold"], tc=COL["dark"])
    arrow(1.98, 5.78, 2.06, 5.04)
    arrow(6.33, 5.78, 5.79, 5.04)
    arrow(9.22, 5.78, 9.82, 5.04)

    box(0.76, 2.28, 2.95, 0.90, "Visible outcomes update\n$\\mu^S(\\mu,e)$ and $\\mu^F(\\mu,e)$", COL["teal"])
    box(4.42, 2.28, 2.62, 0.90, "Bellman continuation\n$W_M(\\mu)$", COL["purple"])
    box(7.78, 2.28, 3.36, 0.90, "Governance topology\nconflict / pooling / separation / overlap", COL["green"])
    arrow(11.25, 5.78, 2.46, 3.18, rad=-0.20)
    arrow(3.71, 2.73, 4.42, 2.73)
    arrow(7.04, 2.73, 7.78, 2.73)
    arrow(5.73, 2.28, 5.73, 1.16)
    arrow(5.73, 1.16, 9.46, 1.16)
    arrow(9.46, 1.16, 9.46, 2.28)
    ax.text(6.17, 0.58, "Operational design question: when do liability, audits, and public protocols jointly produce credible reliability governance?", ha="center", fontsize=8.7, color=COL["gray"])
    save(fig, "fig1_timing_architecture")


def make_fig2_static_incentives(mu_grid: np.ndarray, policy: np.ndarray, p: Params = P) -> None:
    egrid = np.linspace(0.0, p.emax, 500)
    e_m, pi_m = static_opt(lambda e: pi_m_public(p.mu0, e, p), p)
    acc = accountability_values(p)
    sep = separating_ic_margins(p)

    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    vals = pi_m_public(p.mu0, egrid, p)
    ax.plot(egrid, vals, lw=2.0, color=COL["blue"], label="marketplace flow payoff")
    ax.scatter([e_m], [pi_m], color=COL["red"], s=24, zorder=4, label=f"optimum {e_m:.3f}")
    ax.axvline(e_m, color=COL["red"], lw=1.0, ls="--")
    ax.set_title("(a) Marketplace protocol payoff at $\\mu_0$")
    panel_style(ax, "effort $e$", "flow payoff")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig2a_marketplace_payoff")

    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    vals_h = pi_r_signal(1.0, p.thetaH, egrid, p)
    vals_l = pi_r_signal(1.0, p.thetaL, egrid, p)
    ax.plot(egrid, vals_h, lw=2.0, color=COL["green"], label="high type under $R$")
    ax.plot(egrid, vals_l, lw=2.0, color=COL["orange"], label="low type mimic under $R$")
    ax.scatter([acc["e_H_R"]], [acc["pi_H_R"]], color=COL["green"], s=22, zorder=4)
    ax.scatter([acc["e_L_R_mimic"]], [acc["pi_L_R_mimic"]], color=COL["orange"], s=22, zorder=4)
    ax.set_title("(b) Internalized-liability payoff shapes")
    panel_style(ax, "effort $e$", "flow payoff")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig2b_liability_payoffs")

    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    ax.plot(egrid, d_pi_m_public(p.mu0, egrid, p), lw=1.9, color=COL["blue"], label="$\\partial_e \\pi_M^{pub}$")
    ax.plot(egrid, d_pi_r_signal(1.0, p.thetaH, egrid, p), lw=1.9, color=COL["green"], label="$\\partial_e \\pi_R(H)$")
    ax.plot(egrid, d_pi_r_signal(1.0, p.thetaL, egrid, p), lw=1.9, color=COL["orange"], label="$\\partial_e \\pi_R(L)$")
    add_zero(ax)
    ax.set_title("(c) First-order residual geometry")
    panel_style(ax, "effort $e$", "marginal flow return")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig2c_foc_residuals")

    static_curve = np.array([static_opt(lambda e, mu=float(mu): pi_m_public(mu, e, p), p, n=480)[0] for mu in mu_grid])
    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    ax.plot(mu_grid, static_curve, lw=1.9, color=COL["gray"], ls="--", label="static protocol")
    ax.plot(mu_grid, policy, lw=2.0, color=COL["blue"], label="dynamic protocol")
    ax.axvline(p.mu0, color="black", lw=0.95, ls="--")
    ax.set_title("(d) Static-versus-dynamic marketplace protocol")
    panel_style(ax, "belief $\\mu$", "effort")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig2d_protocol_static_dynamic")

    Ls = np.linspace(0.05, 1.20, 120)
    eH = []; eL = []
    for L in Ls:
        pp = replace(p, L=float(L))
        a = accountability_values(pp, n=450)
        eH.append(a["e_H_R"])
        eL.append(a["e_L_R_mimic"])
    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    ax.plot(Ls, eH, lw=2.0, color=COL["green"], label="$e_H^R$")
    ax.plot(Ls, eL, lw=2.0, color=COL["orange"], label="$e_L^{R\\text{-}mimic}$")
    ax.axvline(p.L, color="black", lw=0.95, ls="--")
    ax.set_title("(e) Liability exposure and optimized safeguard effort")
    panel_style(ax, "failure exposure $L$", "optimal effort")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig2e_liability_effort")

    labels = ["$U_H^R$", "$U_H^{M-dev}$", "$U_L^M$", "$U_L^{R-mimic}$"]
    vals = [acc["U_H_R"], acc["U_H_M_dev"], acc["U_L_M"], acc["U_L_R_mimic"]]
    colors = [COL["green"], COL["blue"], COL["blue"], COL["orange"]]
    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=colors, alpha=0.92)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("(f) Baseline perpetual-value comparisons")
    panel_style(ax, None, "discounted value")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.text(0.02, 0.96, f"$\\Delta_H^{{acct}}={sep['high_accountability_margin']:.2f}$\n$\\Delta_L^{{anti}}={sep['low_antimimic_margin']:.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=8.0, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COL["gray"], alpha=0.86))
    save(fig, "fig2f_perpetual_value_comparison")


def make_fig3_learning(mu_grid: np.ndarray, policy: np.ndarray, p: Params = P) -> None:
    mu = np.linspace(0.02, 0.98, 320)
    e_levels = [0.20, 0.95, 1.70]
    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    for e in e_levels:
        ax.plot(mu, posterior_success(mu, e, p), lw=1.95, label=f"e={e:.2f}")
    ax.plot(mu, mu, ls="--", lw=1.0, color="black", label="45-degree")
    ax.set_title("(a) Posterior after a success")
    panel_style(ax, "prior belief $\\mu$", "posterior $\\mu^S$")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig3a_success_posterior")

    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    for e in e_levels:
        ax.plot(mu, posterior_failure(mu, e, p), lw=1.95, label=f"e={e:.2f}")
    ax.plot(mu, mu, ls="--", lw=1.0, color="black", label="45-degree")
    ax.set_title("(b) Posterior after a failure")
    panel_style(ax, "prior belief $\\mu$", "posterior $\\mu^F$")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig3b_failure_posterior")

    egrid = np.linspace(0.01, p.emax, 320)
    spread = posterior_success(p.mu0, egrid, p) - posterior_failure(p.mu0, egrid, p)
    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    ax.plot(egrid, spread, lw=2.0, color=COL["purple"])
    ax.set_title("(c) Posterior spread contracts with safeguards")
    panel_style(ax, "effort $e$", "$\\mu^S-\\mu^F$")
    save(fig, "fig3c_posterior_spread")

    sh, sl = success(p.thetaH, egrid, p), success(p.thetaL, egrid, p)
    kl_hl = sh * np.log(sh / sl) + (1.0 - sh) * np.log((1.0 - sh) / (1.0 - sl))
    loglr_success = np.log(sh / sl)
    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    ax.plot(egrid, kl_hl, lw=2.0, color=COL["red"], label="$D_{KL}(H||L)$")
    ax2 = ax.twinx()
    ax2.plot(egrid, loglr_success, lw=1.8, color=COL["gold"], ls="--", label="success log-LR")
    ax.set_title("(d) Signal informativeness declines")
    panel_style(ax, "effort $e$", "KL diagnosticity")
    ax2.set_ylabel("success log-likelihood ratio")
    lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    save(fig, "fig3d_signal_informativeness")

    paths_h = simulate_belief_paths(p.thetaH, mu_grid, policy, p, reps=850, horizon=45)
    paths_l = simulate_belief_paths(p.thetaL, mu_grid, policy, p, reps=850, horizon=45)
    x = np.arange(paths_h.shape[1])
    fig, ax = plt.subplots(figsize=(4.65, 3.25))
    for paths, color, label in [(paths_h, COL["green"], "true H"), (paths_l, COL["orange"], "true L")]:
        q10 = np.quantile(paths, 0.10, axis=0); q90 = np.quantile(paths, 0.90, axis=0); mean = np.mean(paths, axis=0)
        ax.fill_between(x, q10, q90, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, lw=2.0, label=label)
    ax.axhline(p.mu0, color="black", lw=1.0, ls="--")
    ax.set_ylim(0, 1)
    ax.set_title("(e) Simulated belief paths with 80% envelopes")
    panel_style(ax, "period", "belief $\\mu_t$")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig3e_belief_paths")

    error = np.abs(bayes_martingale_error(p.mu0, egrid, p))
    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    ax.semilogy(egrid, np.maximum(error, 1e-18), lw=2.0, color=COL["teal"])
    ax.set_title("(f) Bayes martingale error across effort")
    panel_style(ax, "effort $e$", "absolute error")
    ax.axhline(1e-15, color=COL["red"], lw=0.95, ls="--", label="$10^{-15}$")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig3f_martingale_error")


def make_fig4_dynamic(mu_grid: np.ndarray, W: np.ndarray, VH: np.ndarray, VL: np.ndarray, policy: np.ndarray, resW: np.ndarray, resH: np.ndarray, resL: np.ndarray, p: Params = P) -> None:
    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(mu_grid, W, lw=2.0, color=COL["blue"], label="$W_M$")
    ax.plot(mu_grid, VH, lw=1.9, color=COL["green"], label="$V_H^M$")
    ax.plot(mu_grid, VL, lw=1.9, color=COL["orange"], label="$V_L^M$")
    ax.axvline(p.mu0, color="black", lw=0.95, ls="--")
    ax.set_title("(a) Public and type-induced values")
    panel_style(ax, "belief $\\mu$", "discounted value")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig4a_values")

    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(mu_grid, policy, lw=2.0, color=COL["blue"])
    ax.axvline(p.mu0, color="black", ls="--", lw=0.95)
    ax.set_title("(b) Common marketplace protocol")
    panel_style(ax, "belief $\\mu$", "$e_M^*(\\mu)$")
    save(fig, "fig4b_protocol")

    slope = np.gradient(policy, mu_grid)
    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(mu_grid, slope, lw=2.0, color=COL["purple"])
    add_zero(ax)
    ax.axvline(p.mu0, color="black", ls="--", lw=0.95)
    ax.set_title("(c) Local slope of the dynamic protocol")
    panel_style(ax, "belief $\\mu$", "$\\mathrm{d}e_M^*/\\mathrm{d}\\mu$")
    save(fig, "fig4c_policy_slope")

    dW = np.gradient(W, mu_grid)
    curv = np.gradient(dW, mu_grid)
    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(mu_grid, curv, lw=2.0, color=COL["red"])
    add_zero(ax)
    ax.axvline(p.mu0, color="black", ls="--", lw=0.95)
    ax.set_title("(d) Curvature of the public Bellman value")
    panel_style(ax, "belief $\\mu$", "$W_M''(\\mu)$")
    save(fig, "fig4d_value_curvature")

    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.semilogy(np.arange(1, len(resW)+1), resW, lw=1.95, color=COL["blue"], label="$W_M$")
    ax.semilogy(np.arange(1, len(resH)+1), resH, lw=1.8, color=COL["green"], label="$V_H^M$")
    ax.semilogy(np.arange(1, len(resL)+1), resL, lw=1.8, color=COL["orange"], label="$V_L^M$")
    ax.set_title("(e) Fixed-point residual decay")
    panel_style(ax, "iteration", "sup-norm residual")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig4e_residuals")

    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(mu_grid, VH - VL, lw=2.0, color=COL["dark"])
    ax.axvline(p.mu0, color="black", lw=0.95, ls="--")
    ax.set_title("(f) Dynamic type-value wedge")
    panel_style(ax, "belief $\\mu$", "$V_H^M(\\mu)-V_L^M(\\mu)$")
    save(fig, "fig4f_type_wedge")


def classify_governance_static_fast(mu: float, L: float, p: Params = P, n: int = 260) -> int:
    pp = replace(p, L=L)
    e_h_m_pool, pi_h_m_pool = static_opt(lambda e: pi_m_type(mu, pp.thetaH, e, pp), pp, n=n)
    e_l_m_pool, pi_l_m_pool = static_opt(lambda e: pi_m_type(mu, pp.thetaL, e, pp), pp, n=n)
    acc = accountability_values(pp, n=n)
    pool_H = perpetual_value(pi_h_m_pool, pp) - acc["U_H_R"]
    pool_L = perpetual_value(pi_l_m_pool, pp) - acc["U_L_R_mimic"]
    pooling = (pool_H >= 0.0) and (pool_L >= 0.0)
    sep = (acc["U_H_R"] - acc["U_H_M_dev"] >= 0.0) and (acc["U_L_M"] - acc["U_L_R_mimic"] >= 0.0)
    if pooling and sep:
        return 3
    if pooling:
        return 1
    if sep:
        return 2
    return 0


def governance_map(xgrid: np.ndarray, ygrid: np.ndarray, p: Params, xkind: str, ykind: str) -> np.ndarray:
    Z = np.zeros((len(ygrid), len(xgrid)))
    for i, y in enumerate(ygrid):
        for j, x in enumerate(xgrid):
            pp = p
            mu = p.mu0
            L = p.L
            if xkind == "mu": mu = float(x)
            elif xkind == "gamma": pp = replace(pp, gamma=float(x))
            elif xkind == "thetaL": pp = replace(pp, thetaL=float(x))
            if ykind == "L": L = float(y)
            Z[i, j] = classify_governance_static_fast(mu, L, pp, n=150)
    return Z


def plot_governance_map(Z: np.ndarray, xgrid: np.ndarray, ygrid: np.ndarray, xlabel: str, ylabel: str, title: str, basename: str, marker: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(4.75, 3.45))
    cmap = plt.matplotlib.colors.ListedColormap(["#D8DDE5", "#BFD7EA", "#CFE8D9", "#EBD8A7"])
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()], cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    if marker is not None:
        ax.scatter([marker[0]], [marker[1]], s=30, color="black", marker="x", zorder=5)
    ax.set_title(title)
    panel_style(ax, xlabel, ylabel)
    cbar = fig.colorbar(im, ax=ax, ticks=[0,1,2,3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["conflict", "pooling", "separation", "overlap"])
    save(fig, basename)


def make_fig5_governance(mu_grid: np.ndarray, VH: np.ndarray, VL: np.ndarray, p: Params = P) -> None:
    Ls = np.linspace(0.05, 1.20, 180)
    high_sep = []; low_anti = []
    for L in Ls:
        margins = separating_ic_margins_fast(replace(p, L=float(L)), n=420)
        high_sep.append(margins["high_accountability_margin"])
        low_anti.append(margins["low_antimimic_margin"])
    high_sep = np.asarray(high_sep); low_anti = np.asarray(low_anti)

    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(Ls, high_sep, lw=2.0, color=COL["green"], label="H accountability")
    add_zero(ax); ax.axvline(p.L, color="black", lw=0.95, ls="--")
    ax.set_title("(a) High-type accountability IC")
    panel_style(ax, "failure exposure $L$", "$\\Delta_H^{acct}$")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig5a_high_accountability")

    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(Ls, low_anti, lw=2.0, color=COL["orange"], label="L anti-mimicry")
    add_zero(ax); ax.axvline(p.L, color="black", lw=0.95, ls="--")
    ax.set_title("(b) Low-type anti-mimicry IC")
    panel_style(ax, "failure exposure $L$", "$\\Delta_L^{anti}$")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig5b_low_antimimic")

    pool_H, pool_L = dynamic_pool_deviation_margins(mu_grid, VH, VL, p)
    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(mu_grid, pool_H, lw=2.0, color=COL["green"], label="$\\Phi_H^M$")
    ax.plot(mu_grid, pool_L, lw=2.0, color=COL["orange"], label="$\\Phi_L^M$")
    add_zero(ax); ax.axvline(p.mu0, color="black", lw=0.95, ls="--")
    ax.set_title("(c) Dynamic pooling-deviation margins")
    panel_style(ax, "belief $\\mu$", "margin")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig5c_pooling_margins")

    mu_map = np.linspace(0.05, 0.95, 28); L_map = np.linspace(0.05, 1.20, 28)
    Z = governance_map(mu_map, L_map, p, "mu", "L")
    plot_governance_map(Z, mu_map, L_map, "prior belief $\\mu$", "failure exposure $L$", "(d) Governance topology in belief-liability space", "fig5d_governance_map_mu_L", marker=(p.mu0, p.L))

    gamma_map = np.linspace(0.45, 1.15, 28)
    Zg = governance_map(gamma_map, L_map, p, "gamma", "L")
    plot_governance_map(Zg, gamma_map, L_map, "demand sensitivity $\\gamma$", "failure exposure $L$", "(e) Governance topology in demand-liability space", "fig5e_governance_map_gamma_L", marker=(p.gamma, p.L))

    thetaL_map = np.linspace(0.40, min(0.88, p.thetaH-0.03), 28)
    Zt = governance_map(thetaL_map, L_map, p, "thetaL", "L")
    plot_governance_map(Zt, thetaL_map, L_map, "low-type reliability $\\theta_L$", "failure exposure $L$", "(f) Governance topology in type-gap-liability space", "fig5f_governance_map_thetaL_L", marker=(p.thetaL, p.L))


def decomposition_at_policy(mu_grid: np.ndarray, W: np.ndarray, policy: np.ndarray, p: Params = P) -> np.ndarray:
    dW = np.gradient(W, mu_grid)
    rows = np.zeros((len(mu_grid), 4))
    h = 1e-4
    for i, mu in enumerate(mu_grid):
        e = float(policy[i])
        q = float(expected_success(mu, e, p)); dq = float(dexp_success(mu, e, p))
        ms = float(posterior_success(mu, e, p)); mf = float(posterior_failure(mu, e, p))
        dms = float((posterior_success(mu, e+h, p)-posterior_success(mu, e-h, p))/(2*h))
        dmf = float((posterior_failure(mu, e+h, p)-posterior_failure(mu, e-h, p))/(2*h))
        current = float((pi_m_public(mu, e+h, p)-pi_m_public(mu, e-h, p))/(2*h))
        prob = p.delta*dq*(np.interp(ms, mu_grid, W)-np.interp(mf, mu_grid, W))
        loc = p.delta*(q*np.interp(ms, mu_grid, dW)*dms + (1-q)*np.interp(mf, mu_grid, dW)*dmf)
        rows[i,:] = [current, prob, loc, current+prob+loc]
    return rows


def make_fig6_marginal(mu_grid: np.ndarray, W: np.ndarray, policy: np.ndarray, p: Params = P) -> None:
    egrid, comps = marginal_decomposition(p.mu0, mu_grid, W, p)

    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(egrid, comps[:,0], lw=1.8, color=COL["blue"], label="current payoff")
    ax.plot(egrid, comps[:,1], lw=1.8, color=COL["green"], label="success-probability")
    ax.plot(egrid, comps[:,2], lw=1.8, color=COL["red"], label="posterior-location")
    ax.plot(egrid, comps[:,3], lw=2.2, color=COL["dark"], label="total")
    add_zero(ax)
    ax.set_title("(a) Bellman-derivative decomposition at $\\mu_0$")
    panel_style(ax, "effort $e$", "marginal value")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig6a_marginal_decomposition")

    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(egrid, comps[:,2], lw=2.05, color=COL["red"])
    add_zero(ax)
    ax.set_title("(b) Posterior-location return")
    panel_style(ax, "effort $e$", "dynamic information return")
    save(fig, "fig6b_posterior_location")

    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(egrid, posterior_success(p.mu0, egrid, p), lw=2.0, color=COL["green"], label="$\\mu^S$")
    ax.plot(egrid, posterior_failure(p.mu0, egrid, p), lw=2.0, color=COL["orange"], label="$\\mu^F$")
    ax.axhline(p.mu0, color="black", lw=0.95, ls="--", label="prior")
    ax.set_title("(c) Posterior locations moved by effort")
    panel_style(ax, "effort $e$", "posterior belief")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig6c_posterior_states")

    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(egrid, comps[:,3], lw=2.1, color=COL["dark"], label="dynamic total")
    ax.plot(egrid, comps[:,0], lw=1.8, color=COL["blue"], ls="--", label="current-payoff derivative")
    add_zero(ax)
    ax.set_title("(d) Dynamic versus static marginal condition")
    panel_style(ax, "effort $e$", "marginal value")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig6d_total_marginal_value")

    rows = decomposition_at_policy(mu_grid, W, policy, p)
    fig, ax = plt.subplots(figsize=(4.65, 3.25))
    ax.plot(mu_grid, rows[:,0], lw=1.75, color=COL["blue"], label="current payoff")
    ax.plot(mu_grid, rows[:,1], lw=1.75, color=COL["green"], label="probability")
    ax.plot(mu_grid, rows[:,2], lw=1.75, color=COL["red"], label="posterior location")
    ax.plot(mu_grid, rows[:,3], lw=2.1, color=COL["dark"], label="total")
    add_zero(ax); ax.axvline(p.mu0, color="black", lw=0.95, ls="--")
    ax.set_title("(e) Decomposition along the optimal protocol")
    panel_style(ax, "belief $\\mu$", "marginal value at $e_M^*(\\mu)$")
    ax.legend(frameon=False, loc="best", ncol=2)
    save(fig, "fig6e_policy_decomposition")

    q = expected_success(p.mu0, egrid, p)
    s_gain = q - expected_success(p.mu0, 0.0, p)
    loc_penalty = -comps[:,2]
    fig, ax = plt.subplots(figsize=(4.55, 3.25))
    ax.plot(egrid, s_gain, lw=2.0, color=COL["green"], label="success gain from effort")
    ax2 = ax.twinx()
    ax2.plot(egrid, loc_penalty, lw=2.0, color=COL["red"], ls="--", label="diagnosticity penalty")
    ax.set_title("(f) Reliability gain versus diagnosticity cost")
    panel_style(ax, "effort $e$", "success-probability gain")
    ax2.set_ylabel("negative posterior-location return")
    lines, labs = ax.get_legend_handles_labels(); lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="best")
    save(fig, "fig6f_reliability_vs_diagnosticity")


def make_fig7_policy(p: Params = P) -> None:
    rows = policy_counterfactuals(p)
    labels = [str(r["scenario"]) for r in rows]
    x = np.arange(len(rows)); width = 0.36

    fig, ax = plt.subplots(figsize=(5.25, 3.55))
    ax.bar(x - width/2, [float(r["high_sep_margin"]) for r in rows], width, color=COL["green"], label="H accountability")
    ax.bar(x + width/2, [float(r["low_antimimic_margin"]) for r in rows], width, color=COL["orange"], label="L anti-mimicry")
    add_zero(ax)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_title("(a) Policy effects on separating IC margins")
    panel_style(ax, None, "IC margin")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig7a_policy_ic")

    fig, ax = plt.subplots(figsize=(5.25, 3.55))
    ax.bar(x - width/2, [float(r["market_success"]) for r in rows], width, color=COL["blue"], label="market success")
    ax2 = ax.twinx()
    ax2.plot(x + width/2, [float(r["pool_high_margin"]) for r in rows], marker="o", lw=1.9, color=COL["red"], label="pool H margin")
    ax2.plot(x + width/2, [float(r["pool_low_margin"]) for r in rows], marker="s", lw=1.8, color=COL["purple"], ls="--", label="pool L margin")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0.75, 1.0)
    ax.set_title("(b) Success performance and pooling pressure")
    panel_style(ax, None, "expected success")
    ax2.set_ylabel("pooling deviation margin")
    lines, labs = ax.get_legend_handles_labels(); lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="best")
    save(fig, "fig7b_policy_success_pooling")

    accuracies = np.linspace(0.51, 0.95, 140)
    favorable_mu = [audit_posterior(p.mu0, a, True) for a in accuracies]
    e_m = []; s_m = []
    for mu in favorable_mu:
        e, _ = static_opt(lambda ee, mm=mu: pi_m_public(mm, ee, p), p, n=500)
        e_m.append(e); s_m.append(float(expected_success(mu, e, p)))
    fig, ax = plt.subplots(figsize=(4.85, 3.45))
    ax.plot(accuracies, favorable_mu, lw=2.0, color=COL["purple"], label="favorable audit posterior")
    ax2 = ax.twinx()
    ax2.plot(accuracies, s_m, lw=1.95, color=COL["green"], ls="--", label="market success")
    ax.set_title("(c) Audit precision moves beliefs and operating success")
    panel_style(ax, "audit accuracy", "posterior $\\mu$ after favorable report")
    ax2.set_ylabel("expected success")
    lines, labs = ax.get_legend_handles_labels(); lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="lower right")
    save(fig, "fig7c_audit_precision")

    lams = np.linspace(0.55, 1.25, 150)
    H = []; Lm = []
    for lam in lams:
        margins = separating_ic_margins_fast(replace(p, lambdaR=float(lam)), n=420)
        H.append(margins["high_accountability_margin"])
        Lm.append(margins["low_antimimic_margin"])
    fig, ax = plt.subplots(figsize=(4.85, 3.45))
    ax.plot(lams, H, lw=2.0, color=COL["green"], label="H accountability")
    ax.plot(lams, Lm, lw=2.0, color=COL["orange"], label="L anti-mimicry")
    add_zero(ax); ax.axvline(p.lambdaR, color="black", lw=0.95, ls="--")
    ax.set_title("(d) Liability share creates a governance trade-off")
    panel_style(ax, "internalized-liability share $\\lambda_R$", "IC margin")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig7d_liability_share_tradeoff")


def make_fig8_robustness(p: Params = P) -> list[dict[str, float | str]]:
    base = separating_ic_margins(p)
    base_metric = base["low_antimimic_margin"]
    perturbations = [
        ("$\\theta_H$", "thetaH", max(0.76, p.thetaH - 0.06), min(0.98, p.thetaH + 0.04)),
        ("$\\theta_L$", "thetaL", max(0.35, p.thetaL - 0.12), min(0.85, p.thetaL + 0.12)),
        ("$\\gamma$", "gamma", p.gamma * 0.75, p.gamma * 1.25),
        ("$L$", "L", p.L * 0.60, p.L * 1.60),
        ("$\\kappa$", "kappa", p.kappa * 0.70, p.kappa * 1.30),
        ("$\\lambda_R$", "lambdaR", 0.68, 1.20),
    ]
    rows = []
    for label, attr, low, high in perturbations:
        plow = replace(p, **{attr: float(low)}); phigh = replace(p, **{attr: float(high)})
        low_val = separating_ic_margins_fast(plow, n=420)["low_antimimic_margin"]
        high_val = separating_ic_margins_fast(phigh, n=420)["low_antimimic_margin"]
        rows.append({
            "primitive": label,
            "low_value": low,
            "high_value": high,
            "low_antimimic_at_low": low_val,
            "low_antimimic_at_high": high_val,
            "range": abs(high_val - low_val),
        })
    rows_sorted = sorted(rows, key=lambda r: float(r["range"]), reverse=True)

    fig, ax = plt.subplots(figsize=(5.15, 3.75))
    y = np.arange(len(rows_sorted))
    for i, row in enumerate(rows_sorted):
        lo = min(float(row["low_antimimic_at_low"]), float(row["low_antimimic_at_high"])); hi = max(float(row["low_antimimic_at_low"]), float(row["low_antimimic_at_high"]))
        ax.plot([lo, hi], [i, i], lw=6.3, color=COL["blue"], solid_capstyle="round")
        ax.scatter([lo, hi], [i, i], s=18, color=COL["dark"], zorder=3)
    ax.axvline(base_metric, color=COL["red"], lw=1.15, ls="--", label="baseline")
    ax.axvline(0, color="black", lw=0.95)
    ax.set_yticks(y); ax.set_yticklabels([str(r["primitive"]) for r in rows_sorted])
    ax.set_title("(a) Tornado sensitivity of the anti-mimicry margin")
    panel_style(ax, "low-type anti-mimicry margin", None)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig8a_tornado")

    fig, ax = plt.subplots(figsize=(5.15, 3.55))
    response_specs = [
        ("$\\theta_L$", "thetaL", np.linspace(max(0.35, p.thetaL - 0.16), min(0.88, p.thetaL + 0.16), 120), COL["orange"]),
        ("$L$", "L", np.linspace(0.10, 0.90, 120), COL["blue"]),
        ("$\\lambda_R$", "lambdaR", np.linspace(0.55, 1.30, 120), COL["green"]),
    ]
    for label, attr, grid, color in response_specs:
        vals = [separating_ic_margins_fast(replace(p, **{attr: float(v)}), n=420)["low_antimimic_margin"] for v in grid]
        denom = getattr(p, attr)
        x = (grid - denom) / denom
        ax.plot(x, vals, lw=2.0, color=color, label=label)
    add_zero(ax); ax.axvline(0, color="black", lw=0.95, ls="--")
    ax.set_title("(b) Local nonlinear response around the baseline")
    panel_style(ax, "relative parameter change", "low-type anti-mimicry margin")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig8b_response_curves")

    write_csv(os.path.join(TAB, "table_robustness.csv"), rows_sorted)
    return rows_sorted



def simulate_validation_paths(theta: float, mu_grid: np.ndarray, policy: np.ndarray, p: Params = P,
                              reps: int = 1600, horizon: int = 60, seed: int = 20260519) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate public belief paths, visible outcomes, and on-path protocol effort."""
    rng = np.random.default_rng(seed + (1 if theta == p.thetaH else 17))
    beliefs = np.zeros((reps, horizon + 1))
    outcomes = np.zeros((reps, horizon), dtype=int)
    efforts = np.zeros((reps, horizon))
    beliefs[:, 0] = p.mu0
    for r in range(reps):
        mu = p.mu0
        for t in range(horizon):
            e = float(np.interp(mu, mu_grid, policy))
            efforts[r, t] = e
            y = rng.random() < float(success(theta, e, p))
            outcomes[r, t] = int(y)
            mu = float(posterior_success(mu, e, p) if y else posterior_failure(mu, e, p))
            mu = float(np.clip(mu, 0.001, 0.999))
            beliefs[r, t + 1] = mu
    return beliefs, outcomes, efforts


def dynamic_policy_foc_gap(mu_grid: np.ndarray, W: np.ndarray, policy: np.ndarray, p: Params = P,
                           h: float = 2e-4) -> np.ndarray:
    """Finite-difference derivative of the Bellman RHS evaluated at the grid policy."""
    gaps: list[float] = []
    for mu, e in zip(mu_grid, policy):
        if e <= h or e >= p.emax - h:
            gaps.append(np.nan)
            continue

        def rhs(ee: float) -> float:
            ps = float(np.clip(posterior_success(mu, ee, p), mu_grid[0], mu_grid[-1]))
            pf = float(np.clip(posterior_failure(mu, ee, p), mu_grid[0], mu_grid[-1]))
            q = float(expected_success(mu, ee, p))
            return float(pi_m_public(mu, ee, p)
                         + p.delta * (q * np.interp(ps, mu_grid, W)
                                      + (1.0 - q) * np.interp(pf, mu_grid, W)))

        gaps.append((rhs(float(e + h)) - rhs(float(e - h))) / (2.0 * h))
    return np.asarray(gaps, dtype=float)


def make_fig9_validation(mu_grid: np.ndarray, W: np.ndarray, policy: np.ndarray,
                         resW: np.ndarray, resH: np.ndarray, resL: np.ndarray,
                         p: Params = P) -> list[dict[str, float | str]]:
    """Monte Carlo validation and numerical certificates."""
    bH, oH, eH = simulate_validation_paths(p.thetaH, mu_grid, policy, p, seed=145)
    bL, oL, eL = simulate_validation_paths(p.thetaL, mu_grid, policy, p, seed=255)

    fig, ax = plt.subplots(figsize=(5.15, 3.35))
    bins = np.linspace(0.0, 1.0, 38)
    ax.hist(bH[:, -1], bins=bins, density=True, alpha=0.44, color=COL["green"], label="high type")
    ax.hist(bL[:, -1], bins=bins, density=True, alpha=0.44, color=COL["orange"], label="low type")
    ax.axvline(p.mu0, color=COL["gray"], ls="--", lw=1.0)
    ax.set_title("(a) Terminal beliefs after repeated public operation")
    panel_style(ax, r"terminal posterior belief $\mu_T$", "density")
    ax.legend(frameon=False)
    save(fig, "fig9a_terminal_beliefs")

    fig, ax = plt.subplots(figsize=(5.15, 3.35))
    T = oH.shape[1]
    x = np.arange(1, T + 1)
    for outcomes, color, label in [(oH, COL["green"], "high type"), (oL, COL["orange"], "low type")]:
        fail = 1 - outcomes
        mean = fail.mean(axis=0)
        se = np.sqrt(np.maximum(mean * (1.0 - mean), 1e-12) / fail.shape[0])
        lo = np.clip(mean - 1.96 * se, 0.0, 1.0)
        hi = np.clip(mean + 1.96 * se, 0.0, 1.0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0.0)
        ax.plot(x, mean, color=color, lw=1.95, label=label)
    ax.set_title("(b) Visible failure rates with 95% simulation bands")
    panel_style(ax, "period", "failure rate")
    ax.legend(frameon=False)
    save(fig, "fig9b_failure_rates")

    gap = dynamic_policy_foc_gap(mu_grid, W, policy, p)
    fig, ax = plt.subplots(figsize=(5.15, 3.35))
    ax.plot(mu_grid, np.log10(np.maximum(np.abs(gap), 1e-12)), color=COL["purple"], lw=1.85)
    ax.axhline(-4.0, color=COL["gray"], ls="--", lw=0.95)
    ax.set_title("(c) Finite-difference Bellman first-order gap")
    panel_style(ax, r"belief $\mu$", r"$\log_{10}\left|\partial \mathrm{RHS}/\partial e\right|$")
    save(fig, "fig9c_dynamic_foc_gap")

    fig, ax = plt.subplots(figsize=(5.15, 3.35))
    egrid = np.linspace(0.01, p.emax, 280)
    mus = [0.15, 0.35, 0.50, 0.70, 0.85]
    for mu in mus:
        ax.plot(egrid, bayes_martingale_error(mu, egrid, p), lw=1.45, label=rf"$\mu={mu:.2f}$")
    ax.axhline(0.0, color="black", lw=0.85)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set_title("(d) Bayesian martingale identity across efforts")
    panel_style(ax, r"effort $e$", "identity error")
    ax.legend(frameon=False, ncol=2)
    save(fig, "fig9d_martingale_grid")

    rows: list[dict[str, float | str]] = []
    for name, beliefs, outcomes, efforts in [("High type", bH, oH, eH), ("Low type", bL, oL, eL)]:
        rows.append({
            "type": name,
            "mean_terminal_belief": float(np.mean(beliefs[:, -1])),
            "p10_terminal_belief": float(np.quantile(beliefs[:, -1], 0.10)),
            "p90_terminal_belief": float(np.quantile(beliefs[:, -1], 0.90)),
            "mean_visible_failure_rate": float(np.mean(1 - outcomes)),
            "mean_protocol_effort": float(np.mean(efforts)),
            "final_period_failure_rate": float(np.mean(1 - outcomes[:, -1])),
        })
    write_csv(os.path.join(TAB, "table_simulation_validation.csv"), rows)
    return rows


def exact_frontier_thresholds(x: np.ndarray, y: np.ndarray, mode: str = "first_nonnegative") -> float:
    """Return a transparent threshold statistic without interpolation."""
    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)
    if mode == "first_nonnegative":
        idx = np.where(yy >= 0.0)[0]
        return float(xx[idx[0]]) if len(idx) else math.nan
    if mode == "last_nonnegative":
        idx = np.where(yy >= 0.0)[0]
        return float(xx[idx[-1]]) if len(idx) else math.nan
    raise ValueError(f"Unknown threshold mode: {mode}")


def static_pooling_margins_at(mu: float, p: Params = P, n: int = 420) -> tuple[float, float]:
    acc = accountability_values(p, n=n)
    _, pi_h = static_opt(lambda e: pi_m_type(mu, p.thetaH, e, p), p, n=n)
    _, pi_l = static_opt(lambda e: pi_m_type(mu, p.thetaL, e, p), p, n=n)
    pool_h = perpetual_value(pi_h, p) - acc["U_H_R"]
    pool_l = perpetual_value(pi_l, p) - acc["U_L_R_mimic"]
    return float(pool_h), float(pool_l)


def make_fig10_frontiers(mu_grid: np.ndarray, VH: np.ndarray, VL: np.ndarray, p: Params = P) -> list[dict[str, float | str]]:
    """Policy frontiers using exact sign tests rather than a stylized welfare index."""
    Ls = np.linspace(0.05, 1.40, 180)
    eH, eL, hmargin, lmargin, poolH_dyn, poolL_dyn = [], [], [], [], [], []
    mu0_idx = int(np.argmin(np.abs(mu_grid - p.mu0)))
    for L in Ls:
        pp = replace(p, L=float(L))
        acc = accountability_values(pp, n=420)
        eH.append(acc["e_H_R"])
        eL.append(acc["e_L_R_mimic"])
        sep = separating_ic_margins_fast(pp, n=420)
        hmargin.append(sep["high_accountability_margin"])
        lmargin.append(sep["low_antimimic_margin"])
        ph, pl = dynamic_pool_deviation_margins(mu_grid, VH, VL, pp)
        poolH_dyn.append(float(ph[mu0_idx]))
        poolL_dyn.append(float(pl[mu0_idx]))

    fig, ax = plt.subplots(figsize=(5.15, 3.35))
    ax.plot(Ls, eH, color=COL["green"], lw=1.95, label=r"$e_R^H$")
    ax.plot(Ls, eL, color=COL["orange"], lw=1.95, label=r"$e_R^L$ mimic")
    ax.axvline(p.L, color=COL["gray"], ls="--", lw=1.0)
    ax.set_title("(a) Liability exposure shifts internalized effort")
    panel_style(ax, r"failure exposure $L$", "optimal effort")
    ax.legend(frameon=False)
    save(fig, "fig10a_liability_effort_schedule")

    fig, ax = plt.subplots(figsize=(5.15, 3.35))
    ax.plot(Ls, hmargin, color=COL["green"], lw=1.85, label="high accountability")
    ax.plot(Ls, lmargin, color=COL["orange"], lw=1.85, label="low anti-mimicry")
    ax.plot(Ls, poolH_dyn, color=COL["blue"], lw=1.55, ls="--", label="dynamic pool-H")
    ax.plot(Ls, poolL_dyn, color=COL["purple"], lw=1.55, ls=":", label="dynamic pool-L")
    add_zero(ax)
    ax.axvline(p.L, color=COL["gray"], ls="--", lw=1.0)
    ax.set_title("(b) Liability separates types but can strain pooling")
    panel_style(ax, r"failure exposure $L$", "margin")
    ax.legend(frameon=False, ncol=2)
    save(fig, "fig10b_liability_margin_schedule")

    accs = np.linspace(0.52, 0.98, 150)
    pool_high, pool_low, succ = [], [], []
    for a in accs:
        mu = audit_posterior(p.mu0, float(a), True)
        e, _ = static_opt(lambda ee: pi_m_public(mu, ee, p), p, n=500)
        ph, pl = static_pooling_margins_at(mu, p, n=420)
        pool_high.append(ph)
        pool_low.append(pl)
        succ.append(float(expected_success(mu, e, p)))

    fig, ax = plt.subplots(figsize=(5.15, 3.35))
    ax.plot(accs, pool_high, color=COL["blue"], lw=1.85, label="pool-H margin")
    ax.plot(accs, pool_low, color=COL["orange"], lw=1.85, label="pool-L margin")
    add_zero(ax)
    ax2 = ax.twinx()
    ax2.plot(accs, succ, color=COL["green"], lw=1.70, ls="--", label="expected success")
    ax.set_title("(c) Audit-precision frontier")
    panel_style(ax, "audit accuracy", "pooling margin")
    ax2.set_ylabel("expected success")
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="best")
    save(fig, "fig10c_audit_precision_frontier")

    A = np.linspace(0.56, 0.96, 48)
    Lgrid = np.linspace(0.15, 1.35, 48)
    Z = np.zeros((len(Lgrid), len(A)), dtype=float)
    for i, L in enumerate(Lgrid):
        for j, a in enumerate(A):
            pp = replace(p, L=float(L), lambdaR=0.90)
            mu = audit_posterior(pp.mu0, float(a), True)
            sep = separating_ic_margins_fast(pp, n=280)
            pool_h, pool_l = static_pooling_margins_at(mu, pp, n=280)
            Z[i, j] = float(
                int(sep["high_accountability_margin"] >= 0.0)
                + int(sep["low_antimimic_margin"] >= 0.0)
                + int(pool_h >= 0.0)
                + int(pool_l >= 0.0)
            )
    fig, ax = plt.subplots(figsize=(5.15, 3.35))
    im = ax.imshow(
        Z, origin="lower", aspect="auto",
        extent=[A.min(), A.max(), Lgrid.min(), Lgrid.max()],
        cmap="viridis", vmin=0.0, vmax=4.0
    )
    # Contours only where classification changes.
    ax.contour(A, Lgrid, Z, levels=[1.5, 2.5, 3.5], colors="white", linewidths=0.8, alpha=0.85)
    ax.scatter([0.78], [p.L], color="white", edgecolors="black", s=28, marker="o", linewidths=0.7, zorder=5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("number of exact feasibility tests satisfied")
    cbar.set_ticks([0, 1, 2, 3, 4])
    ax.set_title("(d) Hybrid design frontier: exact sign-test count")
    panel_style(ax, "audit accuracy", r"failure exposure $L$")
    save(fig, "fig10d_hybrid_exact_frontier")

    rows = [
        {
            "frontier": "Minimum L for nonnegative low anti-mimicry",
            "threshold_value": exact_frontier_thresholds(Ls, lmargin, "first_nonnegative"),
            "criterion": r"$\Delta_L^{\mathrm{anti}}\geq 0$",
            "interpretation": "liability threshold that deters low-type mimicry",
        },
        {
            "frontier": "Largest plotted L with nonnegative high accountability",
            "threshold_value": exact_frontier_thresholds(Ls, hmargin, "last_nonnegative"),
            "criterion": r"$\Delta_H^{\mathrm{acct}}\geq 0$",
            "interpretation": "upper plotted liability exposure that preserves high-type participation",
        },
        {
            "frontier": "Minimum audit accuracy for nonnegative pool-H margin",
            "threshold_value": exact_frontier_thresholds(accs, pool_high, "first_nonnegative"),
            "criterion": r"$\Phi_H^M\geq 0$",
            "interpretation": "audit threshold that restores high-type pooling support",
        },
        {
            "frontier": "Minimum audit accuracy for nonnegative pool-L margin",
            "threshold_value": exact_frontier_thresholds(accs, pool_low, "first_nonnegative"),
            "criterion": r"$\Phi_L^M\geq 0$",
            "interpretation": "audit threshold that restores low-type pooling support",
        },
        {
            "frontier": "Maximum exact feasibility-test count on hybrid grid",
            "threshold_value": float(np.max(Z)),
            "criterion": r"$0\leq \#\{\mathrm{tests\ passed}\}\leq 4$",
            "interpretation": "diagnostic count; not a welfare index",
        },
        {
            "frontier": "Share of hybrid grid satisfying all four exact tests",
            "threshold_value": float(np.mean(Z >= 4.0)),
            "criterion": r"$\#\{\mathrm{tests\ passed}\}=4$",
            "interpretation": "design-space share with simultaneous sign-test support",
        },
    ]
    write_csv(os.path.join(TAB, "table_frontier_thresholds.csv"), rows)
    return rows


def tex_escape(x: object) -> str:
    s = str(x)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def tex_fmt(x: object) -> str:
    if x is None or str(x).strip() == "":
        return "not reached"
    try:
        v = float(x)
        if math.isnan(v):
            return "not reached"
        if abs(v) != 0.0 and (abs(v) < 1e-4 or abs(v) >= 1e4):
            return rf"${v:.4e}$"
        return f"{v:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return tex_escape(x)


def read_csv_rows(name: str) -> list[dict[str, str]]:
    path = os.path.join(TAB, name)
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_tex_tables() -> None:
    """Generate all manuscript-facing TeX tables directly from exported CSVs."""
    baseline = read_csv_rows("table_baseline_parameters.csv")
    body = "\n".join(
        f"{tex_escape(r['primitive'])} & {tex_fmt(r['value'])} & {tex_escape(r['role'])} \\\\"
        for r in baseline
    )
    with open(os.path.join(TAB, "table_baseline_parameters.tex"), "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[H]
\centering
\caption{Baseline numerical primitives used in the unified replication package.}
\label{tab:parameters}
\small
\begin{tabularx}{\textwidth}{l r X}
\toprule
Primitive & Value & Economic role \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table}
""")

    validation = read_csv_rows("table_validation_matrix.csv")
    body = "\n".join(
        f"{tex_escape(r['object'])} & {tex_escape(r['equation_or_claim'])} & {tex_escape(r['numerical_check'])} & {tex_fmt(r['value'])} \\\\"
        for r in validation
    )
    with open(os.path.join(TAB, "table_validation_matrix.tex"), "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[H]
\centering
\caption{Computational validation matrix. Every object is generated from the same equation-to-code chain.}
\label{tab:validation_matrix}
\small
\begin{tabularx}{\textwidth}{p{0.20\textwidth} p{0.27\textwidth} p{0.23\textwidth} X}
\toprule
Object & Analytical target & Numerical check & Value \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table}
""")

    diagnostics = read_csv_rows("table_numerical_diagnostics.csv")
    body = "\n".join(
        f"{tex_escape(r['diagnostic'])} & {tex_fmt(r['value'])} \\\\"
        for r in diagnostics
    )
    with open(os.path.join(TAB, "table_numerical_diagnostics.tex"), "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[H]
\centering
\caption{Numerical diagnostics and benchmark decision statistics.}
\label{tab:diagnostics}
\small
\begin{tabularx}{\textwidth}{X r}
\toprule
Diagnostic & Value \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table}
""")

    policy = read_csv_rows("table_policy_counterfactuals.csv")
    cols = [
        ("scenario", "Scenario"),
        ("mu_conditioned", r"$\hat\mu$"),
        ("L", r"$L$"),
        ("lambdaR", r"$\lambda_R$"),
        ("eM_static", r"$e_M$"),
        ("market_success", "Success"),
        ("high_sep_margin", r"$\Delta_H^{\mathrm{acct}}$"),
        ("low_antimimic_margin", r"$\Delta_L^{\mathrm{anti}}$"),
        ("pool_high_margin", r"$\Phi_H^M$"),
        ("pool_low_margin", r"$\Phi_L^M$"),
    ]
    body = "\n".join(
        " & ".join(tex_escape(r[k]) if k == "scenario" else tex_fmt(r[k]) for k, _ in cols) + r" \\"
        for r in policy
    )
    with open(os.path.join(TAB, "table_policy_counterfactuals.tex"), "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[H]
\centering
\caption{Policy counterfactuals generated by the replication code.}
\label{tab:policy}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
""" + " & ".join(h for _, h in cols) + r""" \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

    robust = read_csv_rows("table_robustness.csv")
    body = "\n".join(
        f"{r['primitive']} & {tex_fmt(r['low_value'])} & {tex_fmt(r['high_value'])} & {tex_fmt(r['low_antimimic_at_low'])} & {tex_fmt(r['low_antimimic_at_high'])} & {tex_fmt(r['range'])} \\\\"
        for r in robust
    )
    with open(os.path.join(TAB, "table_robustness.tex"), "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[H]
\centering
\caption{Robustness of the low-type anti-mimicry margin.}
\label{tab:robustness}
\small
\begin{tabular}{lrrrrr}
\toprule
Primitive & Lower & Upper & Margin at lower & Margin at upper & Range \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    sim = read_csv_rows("table_simulation_validation.csv")
    body = "\n".join(
        f"{tex_escape(r['type'])} & {tex_fmt(r['mean_terminal_belief'])} & {tex_fmt(r['p10_terminal_belief'])} & {tex_fmt(r['p90_terminal_belief'])} & {tex_fmt(r['mean_visible_failure_rate'])} & {tex_fmt(r['mean_protocol_effort'])} & {tex_fmt(r['final_period_failure_rate'])} \\\\"
        for r in sim
    )
    with open(os.path.join(TAB, "table_simulation_validation.tex"), "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[H]
\centering
\caption{Monte Carlo learning-path validation under the on-path marketplace protocol.}
\label{tab:simulation}
\small
\begin{tabular}{lrrrrrr}
\toprule
Type & Mean $\mu_T$ & P10 $\mu_T$ & P90 $\mu_T$ & Mean failure & Mean effort & Final failure \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    frontier = read_csv_rows("table_frontier_thresholds.csv")
    body = "\n".join(
        f"{tex_escape(r['frontier'])} & {tex_fmt(r['threshold_value'])} & {r['criterion']} & {tex_escape(r['interpretation'])} \\\\"
        for r in frontier
    )
    with open(os.path.join(TAB, "table_frontier_thresholds.tex"), "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[H]
\centering
\caption{Policy-frontier threshold diagnostics and exact sign-test frontier summaries.}
\label{tab:frontier}
\small
\begin{tabularx}{\textwidth}{X r p{0.18\textwidth} X}
\toprule
Frontier & Threshold & Criterion & Interpretation \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table}
""")


def write_tables(mu_grid: np.ndarray, W: np.ndarray, VH: np.ndarray, VL: np.ndarray, policy: np.ndarray, resW: np.ndarray, resH: np.ndarray, resL: np.ndarray, robustness_rows: list[dict[str, float | str]], p: Params = P) -> None:
    baseline_rows = [
        {"primitive": "theta_H", "value": p.thetaH, "role": "High-reliability type"},
        {"primitive": "theta_L", "value": p.thetaL, "role": "Low-reliability type"},
        {"primitive": "kappa", "value": p.kappa, "role": "Effort-to-reliability curvature"},
        {"primitive": "gamma", "value": p.gamma, "role": "Demand sensitivity to expected success"},
        {"primitive": "L", "value": p.L, "role": "Failure exposure scale"},
        {"primitive": "lambda_M", "value": p.lambdaM, "role": "Marketplace liability share"},
        {"primitive": "lambda_R", "value": p.lambdaR, "role": "Internalized liability share"},
        {"primitive": "tau", "value": p.tau, "role": "Convex safeguard cost"},
        {"primitive": "delta", "value": p.delta, "role": "Discount factor"},
        {"primitive": "mu_0", "value": p.mu0, "role": "Initial public belief"},
        {"primitive": "n_mu", "value": p.n_mu, "role": "Belief-grid resolution"},
        {"primitive": "n_e", "value": p.n_e, "role": "Effort-grid resolution"},
    ]
    write_csv(os.path.join(TAB, "table_baseline_parameters.csv"), baseline_rows)

    e_m, pi_m = static_opt(lambda e: pi_m_public(p.mu0, e, p), p)
    e_h_r, _ = static_opt(lambda e: pi_r_signal(1.0, p.thetaH, e, p), p)
    e_l_r, _ = static_opt(lambda e: pi_r_signal(1.0, p.thetaL, e, p), p)
    max_martingale = float(np.max(np.abs(bayes_martingale_error(p.mu0, np.linspace(0.01, p.emax, 500), p))))
    sep = separating_ic_margins(p)
    pool_H, pool_L = dynamic_pool_deviation_margins(mu_grid, VH, VL, p)
    mu0_idx = int(np.argmin(np.abs(mu_grid - p.mu0)))
    diag_rows = [
        {"diagnostic": "Public Bellman final residual", "value": float(resW[-1])},
        {"diagnostic": "High-type induced-value final residual", "value": float(resH[-1])},
        {"diagnostic": "Low-type induced-value final residual", "value": float(resL[-1])},
        {"diagnostic": "Public Bellman iterations", "value": len(resW)},
        {"diagnostic": "High-type induced-value iterations", "value": len(resH)},
        {"diagnostic": "Low-type induced-value iterations", "value": len(resL)},
        {"diagnostic": "Maximum Bayes martingale error", "value": max_martingale},
        {"diagnostic": "Static marketplace optimal effort at mu0", "value": e_m},
        {"diagnostic": "Static marketplace flow payoff at mu0", "value": pi_m},
        {"diagnostic": "Static high-type R optimal effort", "value": e_h_r},
        {"diagnostic": "Static low-type R mimic effort", "value": e_l_r},
        {"diagnostic": "High accountability margin", "value": sep["high_accountability_margin"]},
        {"diagnostic": "Low anti-mimicry margin", "value": sep["low_antimimic_margin"]},
        {"diagnostic": "Dynamic pooling H deviation margin at mu0", "value": float(pool_H[mu0_idx])},
        {"diagnostic": "Dynamic pooling L deviation margin at mu0", "value": float(pool_L[mu0_idx])},
        {"diagnostic": "Public value W_M at mu0", "value": float(np.interp(p.mu0, mu_grid, W))},
    ]
    write_csv(os.path.join(TAB, "table_numerical_diagnostics.csv"), diag_rows)

    policy_rows = policy_counterfactuals(p)
    write_csv(os.path.join(TAB, "table_policy_counterfactuals.csv"), policy_rows)

    validation_rows = [
        {"object": "Bayesian posteriors", "equation_or_claim": "Bayes rules and martingale identity", "numerical_check": "Maximum martingale error", "value": max_martingale},
        {"object": "Public Bellman value", "equation_or_claim": "Bellman fixed point", "numerical_check": "Final residual", "value": float(resW[-1])},
        {"object": "High-type induced value", "equation_or_claim": "Induced value recursion", "numerical_check": "Final residual", "value": float(resH[-1])},
        {"object": "Low-type induced value", "equation_or_claim": "Induced value recursion", "numerical_check": "Final residual", "value": float(resL[-1])},
        {"object": "Accountability separation", "equation_or_claim": "IC margins", "numerical_check": "Delta_H / Delta_L", "value": f"{sep['high_accountability_margin']:.6f} / {sep['low_antimimic_margin']:.6f}"},
    ]
    write_csv(os.path.join(TAB, "table_validation_matrix.csv"), validation_rows)
    write_csv(os.path.join(TAB, "table_robustness.csv"), robustness_rows)


def main() -> None:
    mu_grid, W, policy, resW = solve_pooling_bellman(P)
    VH, resH = induced_type_values(P.thetaH, mu_grid, policy, P)
    VL, resL = induced_type_values(P.thetaL, mu_grid, policy, P)

    # Clear old produced figures to avoid stale artifacts on reruns.
    for f in os.listdir(FIG):
        if f.startswith("fig") and (f.endswith(".pdf") or f.endswith(".png")):
            try:
                os.remove(os.path.join(FIG, f))
            except OSError:
                pass

    make_fig1_timing()
    make_fig2_static_incentives(mu_grid, policy, P)
    make_fig3_learning(mu_grid, policy, P)
    make_fig4_dynamic(mu_grid, W, VH, VL, policy, resW, resH, resL, P)
    make_fig5_governance(mu_grid, VH, VL, P)
    make_fig6_marginal(mu_grid, W, policy, P)
    make_fig7_policy(P)
    robustness_rows = make_fig8_robustness(P)
    write_tables(mu_grid, W, VH, VL, policy, resW, resH, resL, robustness_rows, P)
    simulation_rows = make_fig9_validation(mu_grid, W, policy, resW, resH, resL, P)
    frontier_rows = make_fig10_frontiers(mu_grid, VH, VL, P)
    write_tex_tables()

    summary = {
        "iterations_public": len(resW),
        "iterations_VH": len(resH),
        "iterations_VL": len(resL),
        "residual_public": float(resW[-1]),
        "residual_VH": float(resH[-1]),
        "residual_VL": float(resL[-1]),
        "figures_generated": len([f for f in os.listdir(FIG) if f.endswith('.pdf')]),
        "simulation_rows": len(simulation_rows),
        "frontier_rows": len(frontier_rows),
    }
    write_csv(os.path.join(TAB, "run_summary.csv"), [summary])
    np.savez(os.path.join(TAB, "dynamic_cache.npz"), mu_grid=mu_grid, W=W, policy=policy, resW=resW, VH=VH, resH=resH, VL=VL, resL=resL)
    print("Completed generation:", summary)


if __name__ == "__main__":
    main()
