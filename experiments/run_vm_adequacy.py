#!/usr/bin/env python
"""von Mises adequacy check for the DANCo angle statistic.

    python experiments/run_vm_adequacy.py
    python experiments/run_vm_adequacy.py --only gsm --gsm-sigma 0.25

For representative datasets (isotropic control, scale-mixture control,
MNIST digit 3, ImageNet koala; raw and contrast-normalized where relevant)
this script pools the per-point pairwise neighbor angles, fits a von Mises
law by maximum likelihood at the pooled level, and stores everything a
goodness-of-fit figure needs: histogram counts, fitted parameters, and
quantile pairs for a QQ plot. It also reports the KS-type maximum CDF
discrepancy (two-sided, max of D+ and D-) so the adequacy claim in the
paper has a number attached, plus the circular-model mass outside the
physical support [0, pi] -- the truncation the fitted-CDF comparison
ignores -- so the circular/truncated distinction can be quantified.

Because the angles of one center share its k neighbors, they are
dependent; the block bootstrap over centers (rows of the theta matrix)
supplies the honest uncertainty on the fitted parameters.

The scale-mixture control is the d = 70, N = 2500 GSM of
``run_angle_diagnostics.load_synthetic`` (same ``default_rng(0)`` stream) at
log-amplitude spread ``--gsm-sigma`` (default 0.25, the paper's focal
sigma_S); the value is stored in the npz as ``gsm_sigma``.
"""

import argparse

import numpy as np

from common import REPO_ROOT
from run_images import _load_category, load_mnist
from run_angle_diagnostics import load_synthetic, variant_contrast
from danco._kernels.angles import indnComb
from danco._kernels.vonmises import ml_vm
from danco.backends import get_backend

K = 10
N_BOOT = 200
OUT = REPO_ROOT / "results" / "diagnostics"


def vm_cdf(x, nu, tau, grid=4096):
    """CDF of VM(nu, tau) on [0, pi] via trapezoidal integration."""
    t = np.linspace(0, np.pi, grid)
    from scipy.special import i0e

    # exp(tau*(cos(t-nu)-1)) / (2*pi*i0e(tau)) equals the VM density with
    # overflow-safe scaling; renormalize on [0, pi] because the fitted
    # angles live there.
    dens = np.exp(tau * (np.cos(t - nu) - 1.0)) / i0e(tau)
    dens /= np.trapz(dens, t)
    cdf = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(t))])
    cdf /= cdf[-1]
    return np.interp(x, t, cdf)


def analyze(name, X, **extra):
    backend = get_backend("cpu")
    combs = indnComb(K)
    _, idx = backend.knn(X, K)
    thetas = backend.angle_stats(X, idx, combs, K)  # (N, C(k,2))

    pooled = thetas.ravel()
    # Pooled ML fit, same estimator applied to one long row.
    nu_p, tau_p = ml_vm(pooled[None, :])
    nu_p, tau_p = float(nu_p[0]), float(tau_p[0])

    # QQ pairs against the fitted law.
    q = np.linspace(0.005, 0.995, 199)
    emp_q = np.quantile(pooled, q)
    grid = np.linspace(0, np.pi, 4096)
    fit_cdf = vm_cdf(grid, nu_p, tau_p)
    fit_q = np.interp(q, fit_cdf, grid)
    sorted_cdf = vm_cdf(np.sort(pooled), nu_p, tau_p)
    n_pool = pooled.size
    d_plus = float(np.max(np.arange(1, n_pool + 1) / n_pool - sorted_cdf))
    d_minus = float(np.max(sorted_cdf - np.arange(0, n_pool) / n_pool))
    max_cdf_gap = max(d_plus, d_minus)

    # Circular-model mass outside [0, pi]: the truncation that the
    # [0, pi]-renormalized comparison ignores. Exponentially scaled to stay
    # finite at large tau.
    from scipy.special import i0e
    t_out = np.linspace(-np.pi, 0, 4096)
    dens_out = np.exp(tau_p * (np.cos(t_out - nu_p) - 1.0)) / i0e(tau_p)
    t_all = np.linspace(-np.pi, np.pi, 8192)
    dens_all = np.exp(tau_p * (np.cos(t_all - nu_p) - 1.0)) / i0e(tau_p)
    trunc_mass = float(np.trapz(dens_out, t_out) / np.trapz(dens_all, t_all))

    # Block bootstrap over centers for (nu, tau) uncertainty.
    rng = np.random.default_rng(0)
    boots = np.empty((N_BOOT, 2))
    for b in range(N_BOOT):
        rows = rng.integers(0, thetas.shape[0], thetas.shape[0])
        nb, tb = ml_vm(thetas[rows].ravel()[None, :])
        boots[b] = nb[0], tb[0]

    hist, edges = np.histogram(pooled, bins=120, range=(0, np.pi),
                               density=True)
    np.savez_compressed(
        OUT / f"vm_adequacy_{name}.npz",
        nu=nu_p, tau=tau_p, max_cdf_gap=max_cdf_gap,
        trunc_mass=trunc_mass,
        nu_boot_sd=boots[:, 0].std(), tau_boot_sd=boots[:, 1].std(),
        hist=hist, edges=edges, qq_emp=emp_q, qq_fit=fit_q, **extra)
    print(f"  {name}: nu={nu_p:.3f}±{boots[:,0].std():.3f} "
          f"tau={tau_p:.1f}±{boots[:,1].std():.1f} "
          f"max|F_emp-F_fit|={max_cdf_gap:.4f} "
          f"trunc_mass={trunc_mass:.2e}", flush=True)


ALL_NAMES = ("ball70", "gsm", "mnist3", "koala_raw", "koala_contrast")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gsm-sigma", type=float, default=0.25,
                        help="log-amplitude spread sigma_S of the GSM control")
    parser.add_argument("--only", default=",".join(ALL_NAMES),
                        help=f"comma-separated subset of {list(ALL_NAMES)}")
    args = parser.parse_args(argv)
    wanted = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = sorted(set(wanted) - set(ALL_NAMES))
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}")

    OUT.mkdir(parents=True, exist_ok=True)
    loaders = {
        "ball70": lambda: load_synthetic()["ball70"],
        "gsm": lambda: load_synthetic(args.gsm_sigma)[
            f"gsm_sigma{args.gsm_sigma:g}"],
        "mnist3": lambda: np.asarray(load_mnist()["3"], dtype=np.float64),
        "koala_raw": lambda: _load_category("n01882714"),
        "koala_contrast": lambda: variant_contrast(
            _load_category("n01882714")),
    }
    extras = {"gsm": {"gsm_sigma": args.gsm_sigma}}
    print("von Mises adequacy:")
    for name in wanted:
        analyze(name, loaders[name](), **extras.get(name, {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
