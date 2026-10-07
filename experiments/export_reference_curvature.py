"""Reference log-log slopes, the Fisher threshold, and the CNN mean-direction range.

Supplementary S4.2 compares, along the candidate axis, the curvature of the
distance term with that of the profiled angular term.  The distance term is
the more curved one whenever ``|kappa| < sqrt(F)``, where

* ``kappa`` is the least-squares slope of ``log tau_ref`` on ``log dhat_ref``
  over a candidate range ``m = m_lo..m_hi``;
* ``F = 1 + (k2 - k1 - 1) E[W^2 e^W / (e^W - 1)^2]`` with
  ``e^{-W} ~ Beta(k1, k2 - k1)`` is the Fisher information of ``log d``
  carried by one neighbor-distance ratio (Eq. eq:fisher-ratio); MiND with
  ``k = 10`` is ``(k1, k2) = (1, 11)`` and Gride is ``(5, 10)``.

Reference curves:

* image references -- ``results/diagnostics/image_reference_curves.csv``,
  written by ``export_reference_envelope.py`` with ``version="DANCo"``
  (MiND distance model, Monte Carlo calibration, k = 10), five seeds at each
  image sample size ``N in {547, 648, 1010, 1028, 5000}``;
* CNN surfaces -- the packaged spline pickles
  ``danco/assets/fastdanco_splines_400.pkl`` (MiND) and
  ``fastdanco_splines_gride_400.pkl`` (Gride), evaluated at their fitted
  sample-size grid ``N_range`` (450, 500, 580, 640, 700) over m = 6..400.

Three slope summaries are written.  ``direct`` is the global regression of
``log tau_ref`` on ``log dhat_ref`` (one fit per seed for the image curves,
one per N for the surface).  ``ratio_meancurve`` divides the global slope of
``log tau_ref`` on ``log m`` by the slope of ``log dhat_ref`` on ``log m``,
with the image curves first averaged over the five reference replicates at
each N.
``window_10`` and ``window_20`` (row_type ``local``) are the candidate-wise
diagnostics reported in Supplementary S4.2: windowed regressions of
``log tau_ref``, ``log dhat_ref`` and ``nu_ref`` on ``log m`` over m +- 10
(or 20) using the available candidates within the window, kappa_m as the
slope ratio, R_m = C(tau_m) kappa_m^2 / F for the Profiled angular term, and
the Full extra term A(tau_m) tau_m nu'_m^2 / F.  The reference sets are the
clean-benchmark Monte Carlo references behind Supplementary Table S3
(rebuilt at N = 2500 from calibration seeds 35..39, ``stream`` seeding, one
hyperball per candidate m = 1..100, cached in
``results/diagnostics/clean_reference_curves.csv``; MiND and Gride read the
same balls), the MiND image references (image experiments) and the Gride CNN
surface (CNN study).

Separately, Supplementary S3.7 quotes the range of the Gride surface's
reference mean direction over the CNN study's effective sample sizes
(N = 495..500 unique activation rows) and candidates m = 6..400.

Writes
  results/diagnostics/reference_curvature.csv
  results/diagnostics/clean_reference_curves.csv   (built once, then reused)
  table_reference_range_cnn_row.tex  -> --paper-dir/tables, else
                                        results/generated/tables/
"""
import argparse
import os
import tempfile
import time

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

from common import REPO_ROOT, add_paper_dir_argument, artifact_dirs
from danco.spline import default_spline_path, load_splines

CURVES = REPO_ROOT / "results" / "diagnostics" / "image_reference_curves.csv"
OUT_CSV = REPO_ROOT / "results" / "diagnostics" / "reference_curvature.csv"
OUT_ROW_NAME = "table_reference_range_cnn_row.tex"

FIRST = 6
IMAGE_RANGES = (100, 200)
SURFACE_RANGES = (100, 400)
CNN_N = tuple(range(495, 501))  # effective (unique-row) CNN sample sizes
CNN_M = (6, 400)
FISHER_PAIRS = {"MiND": (1, 11), "Gride": (5, 10)}
WINDOWS = (10, 20)  # local-slope windows: main diagnostic (+-10) and sensitivity (+-20)
CLEAN_N = 2500                 # clean benchmark sample size (Table S3)
CLEAN_SEEDS = tuple(range(35, 40))  # estimator random_state 35 + rep, five replicates
CLEAN_MAX_DIM = 100            # candidate bound of the clean benchmark
CLEAN_CURVES = REPO_ROOT / "results" / "diagnostics" / "clean_reference_curves.csv"


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def loglog_slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x, float)),
                            np.log(np.asarray(y, float)), 1)[0])


def ratio_slope(m, dhat, tau):
    return loglog_slope(m, tau) / loglog_slope(m, dhat)


def fisher_ratio(k1, k2):
    """F of Eq. (eq:fisher-ratio) by quadrature over v = e^{-W} ~ Beta(k1, k2-k1).

    With W = -log v the integrand W^2 e^W / (e^W - 1)^2 equals
    v log(v)^2 / (1 - v)^2, which is bounded on (0, 1).
    """
    law = stats.beta(k1, k2 - k1)

    def integrand(v):
        return v * np.log(v) ** 2 / (1.0 - v) ** 2 * law.pdf(v)

    expectation, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-13,
                                    epsrel=1e-12, limit=200)
    return 1.0 + (k2 - k1 - 1) * expectation


def bessel_ratio(tau):
    """A(tau) = I_1(tau) / I_0(tau) via exponentially scaled Bessel functions."""
    tau = np.asarray(tau, float)
    return special.i1e(tau) / special.i0e(tau)


def concentration_information(tau):
    """C(tau) = tau^2 A'(tau) with A' = 1 - A/tau - A^2, i.e. C = tau^2 - y - y^2, y = tau A."""
    tau = np.asarray(tau, float)
    y = tau * bessel_ratio(tau)
    return tau * tau - y - y * y


def windowed_slope(x, y, half_width):
    """Least-squares slope of y on x in a centered index window of +-half_width,
    truncated at the ends of the sequence."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out = np.full(x.size, np.nan)
    for i in range(x.size):
        lo, hi = max(0, i - half_width), min(x.size, i + half_width + 1)
        if hi - lo >= 3:
            out[i] = np.polyfit(x[lo:hi], y[lo:hi], 1)[0]
    return out


def local_curvature_ratios(m, dhat, tau, nu, fisher, half_width):
    """Local slopes along one reference curve and the resulting curvature ratios.

    kappa_m is the ratio of the windowed slopes of log tau and log dhat on
    log m; nu'_m the ratio for nu.  R_m = C(tau_m) kappa_m^2 / F is the
    Profiled angular-to-distance curvature ratio; the Full objective adds
    A(tau_m) tau_m nu'_m^2 / F from reference mean-direction motion.
    """
    log_m = np.log(np.asarray(m, float))
    s_d = windowed_slope(log_m, np.log(dhat), half_width)
    s_t = windowed_slope(log_m, np.log(tau), half_width)
    s_n = windowed_slope(log_m, nu, half_width)
    kappa = s_t / s_d
    nu_prime = s_n / s_d
    tau = np.asarray(tau, float)
    profiled = concentration_information(tau) * kappa ** 2 / fisher
    full_extra = bessel_ratio(tau) * tau * nu_prime ** 2 / fisher
    return dict(kappa_absmax=float(np.nanmax(np.abs(kappa))),
                R_max=float(np.nanmax(profiled)),
                full_extra_max=float(np.nanmax(full_extra)),
                full_total_max=float(np.nanmax(profiled + full_extra)),
                tau_min=float(tau.min()))


def local_row(model, reference_set, n_label, hi, half_width, per_curve):
    keys = ("kappa_absmax", "R_max", "full_extra_max", "full_total_max")
    row = dict(row_type="local", model=model, reference_set=reference_set,
               method=f"window_{half_width}", N_or_range=n_label, m_lo=FIRST,
               m_hi=hi, n_fits=len(per_curve))
    for key in keys:
        row[key] = max(item[key] for item in per_curve)
    row["tau_min"] = min(item["tau_min"] for item in per_curve)
    return row


def image_local_rows(curves, fisher):
    rows = []
    sizes = sorted(curves["N"].unique())
    n_label = f"{sizes[0]}-{sizes[-1]}"
    for hi in IMAGE_RANGES:
        window = curves[(curves["candidate_dim"] >= FIRST)
                        & (curves["candidate_dim"] <= hi)]
        mean = window.groupby(["N", "candidate_dim"])[
            ["dhat_ref", "tau_ref", "nu_ref"]].mean().reset_index()
        for half_width in WINDOWS:
            per_curve = [local_curvature_ratios(
                g["candidate_dim"], g["dhat_ref"], g["tau_ref"], g["nu_ref"],
                fisher, half_width) for _, g in mean.groupby("N")]
            rows.append(local_row("MiND", "image references", n_label, hi,
                                  half_width, per_curve))
    return rows


def surface_local_rows(model, splines, fisher):
    rows = []
    grid = [int(n) for n in splines["N_range"]]
    n_label = f"{min(grid)}-{max(grid)}"
    for hi in SURFACE_RANGES:
        m = np.arange(FIRST, hi + 1, dtype=float)
        for half_width in WINDOWS:
            per_curve = [local_curvature_ratios(
                m, splines["spline_dhat"](m, n)[:, 0],
                splines["spline_mu_tau"](m, n)[:, 0],
                splines["spline_mu_nu"](m, n)[:, 0], fisher, half_width)
                for n in grid]
            rows.append(local_row(model, "CNN surface", n_label, hi,
                                  half_width, per_curve))
    return rows


def build_clean_reference_curves():
    """Rebuild the clean-benchmark Monte Carlo references of Table S3.

    The benchmark calibrates with ``calibration_seeding="stream"`` and
    ``random_state = 35 + rep``; the growing calibration cache draws one
    dimension-m hyperball per candidate in order m = 1..100.  The same balls
    serve the MiND and the Gride estimator (identical RNG stream), so both
    distance fits are read from one ball while nu and tau coincide.
    """
    from danco import DANCo
    rows = []
    for seed in CLEAN_SEEDS:
        mind = DANCo(N=CLEAN_N, k=10, version="DANCo", random_state=seed,
                     fractal=True, verbose=False, calibration_seeding="stream")
        gride = DANCo(N=CLEAN_N, k=10, version="DANCo_gride", random_state=seed,
                      fractal=True, verbose=False, calibration_seeding="stream")
        started = time.perf_counter()
        for m in range(1, CLEAN_MAX_DIM + 1):
            ball = mind.hyperBall(CLEAN_N, m, center=[0] * m)
            ball_check = gride.hyperBall(CLEAN_N, m, center=[0] * m)
            if not np.array_equal(ball, ball_check):
                raise RuntimeError("MiND and Gride reference streams diverged")
            s_mind = mind.danco_statistic(ball, n_jobs=-1)
            s_gride = gride.danco_statistic(ball, n_jobs=-1)
            if not np.isclose(float(s_mind["mu_tau"]), float(s_gride["mu_tau"])):
                raise RuntimeError("angular statistics differ between versions")
            rows.append(dict(N=CLEAN_N, k=10, seed=seed, candidate_dim=m,
                             dhat_ref_mind=float(s_mind["dhat"]),
                             dhat_ref_gride=float(s_gride["dhat"]),
                             nu_ref=float(s_mind["mu_nu"]),
                             tau_ref=float(s_mind["mu_tau"]),
                             calibration_seeding="stream"))
        print(f"clean references seed={seed}: {CLEAN_MAX_DIM} candidates in "
              f"{time.perf_counter() - started:.1f}s", flush=True)
    return pd.DataFrame(rows)


def clean_local_rows(clean, fisher):
    rows = []
    window = clean[(clean["candidate_dim"] >= FIRST)
                   & (clean["candidate_dim"] <= CLEAN_MAX_DIM)]
    mean = window.groupby("candidate_dim")[
        ["dhat_ref_mind", "dhat_ref_gride", "nu_ref", "tau_ref"]].mean().reset_index()
    for model, column in (("MiND", "dhat_ref_mind"), ("Gride", "dhat_ref_gride")):
        for half_width in WINDOWS:
            per_curve = [local_curvature_ratios(
                mean["candidate_dim"], mean[column], mean["tau_ref"],
                mean["nu_ref"], fisher[model], half_width)]
            rows.append(local_row(model, "clean references", str(CLEAN_N),
                                  CLEAN_MAX_DIM, half_width, per_curve))
    return rows


def slope_row(model, reference_set, method, n_label, hi, values):
    values = np.asarray(values, float)
    return dict(row_type="kappa", model=model, reference_set=reference_set,
                method=method, N_or_range=n_label, m_lo=FIRST, m_hi=hi,
                n_fits=int(values.size), kappa_min=values.min(),
                kappa_max=values.max(), kappa_absmax=np.abs(values).max())


def image_rows(curves):
    rows = []
    sizes = sorted(curves["N"].unique())
    n_label = f"{sizes[0]}-{sizes[-1]}"
    for hi in IMAGE_RANGES:
        window = curves[(curves["candidate_dim"] >= FIRST)
                        & (curves["candidate_dim"] <= hi)]
        direct = [loglog_slope(g["dhat_ref"], g["tau_ref"])
                  for _, g in window.groupby(["N", "seed"])]
        mean = window.groupby(["N", "candidate_dim"])[
            ["dhat_ref", "tau_ref"]].mean().reset_index()
        ratio = [ratio_slope(g["candidate_dim"], g["dhat_ref"], g["tau_ref"])
                 for _, g in mean.groupby("N")]
        rows.append(slope_row("MiND", "image references", "direct",
                              n_label, hi, direct))
        rows.append(slope_row("MiND", "image references", "ratio_meancurve",
                              n_label, hi, ratio))
    return rows


def surface_rows(model, splines):
    rows = []
    grid = [int(n) for n in splines["N_range"]]
    n_label = f"{min(grid)}-{max(grid)}"
    for hi in SURFACE_RANGES:
        m = np.arange(FIRST, hi + 1, dtype=float)
        direct, ratio = [], []
        for n in grid:
            dhat = splines["spline_dhat"](m, n)[:, 0]
            tau = splines["spline_mu_tau"](m, n)[:, 0]
            direct.append(loglog_slope(dhat, tau))
            ratio.append(ratio_slope(m, dhat, tau))
        rows.append(slope_row(model, "CNN surface", "direct", n_label, hi,
                              direct))
        rows.append(slope_row(model, "CNN surface", "ratio_meancurve",
                              n_label, hi, ratio))
    return rows


def cnn_direction_row(splines):
    m = np.arange(CNN_M[0], CNN_M[1] + 1, dtype=float)
    surface = np.array([splines["spline_mu_nu"](m, n)[:, 0] for n in CNN_N])
    lo = np.unravel_index(surface.argmin(), surface.shape)
    hi = np.unravel_index(surface.argmax(), surface.shape)
    return dict(row_type="nu_range", model="Gride", reference_set="CNN surface",
                N_or_range=f"{CNN_N[0]}-{CNN_N[-1]}", m_lo=CNN_M[0],
                m_hi=CNN_M[1], nu_min=float(surface.min()),
                nu_max=float(surface.max()),
                nu_min_at=f"N={CNN_N[lo[0]]},m={int(m[lo[1]])}",
                nu_max_at=f"N={CNN_N[hi[0]]},m={int(m[hi[1]])}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0])
    add_paper_dir_argument(parser)
    args = parser.parse_args(argv)
    out_row = artifact_dirs(args.paper_dir, "tables")[0] / OUT_ROW_NAME
    start = time.perf_counter()
    curves = pd.read_csv(CURVES)
    if set(curves["k"].unique()) != {10}:
        raise ValueError("image reference curves are expected at k = 10")
    mind = load_splines(default_spline_path("mind"), variant="mind")
    gride = load_splines(default_spline_path("gride"), variant="gride")
    for name, surface in (("MiND", mind), ("Gride", gride)):
        if int(surface["k"]) != 10:
            raise ValueError(f"{name} surface was not fitted at k = 10")
    if [int(n) for n in mind["N_range"]] != [int(n) for n in gride["N_range"]]:
        raise ValueError("MiND and Gride surfaces use different N grids")

    rows = []
    fisher, sqrt_f = {}, {}
    for model, (k1, k2) in FISHER_PAIRS.items():
        value = fisher_ratio(k1, k2)
        fisher[model] = float(value)
        sqrt_f[model] = float(np.sqrt(value))
        rows.append(dict(row_type="fisher", model=model, k1=k1, k2=k2,
                         F=value, sqrt_F=sqrt_f[model]))

    rows += image_rows(curves)
    rows += surface_rows("Gride", gride)
    if CLEAN_CURVES.exists():
        clean = pd.read_csv(CLEAN_CURVES)
    else:
        clean = build_clean_reference_curves()
        atomic_write(CLEAN_CURVES, clean.to_csv(index=False))
    if sorted(clean["seed"].unique()) != list(CLEAN_SEEDS):
        raise ValueError("clean reference curves do not carry seeds 35..39")
    rows += clean_local_rows(clean, fisher)
    rows += image_local_rows(curves, fisher["MiND"])
    rows += surface_local_rows("Gride", gride, fisher["Gride"])

    rows.append(cnn_direction_row(gride))
    frame = pd.DataFrame(rows)
    for column in ("m_lo", "m_hi", "n_fits", "k1", "k2"):
        frame[column] = frame[column].astype("Int64")
    atomic_write(OUT_CSV, frame.to_csv(index=False))

    nu = rows[-1]
    # Cells of one tabular row, \input inside Table S1.  The row terminator
    # stays in the manuscript (``\input{...}\\``) because tokens that
    # \input emits at end of file would otherwise open a new row before
    # \bottomrule.  No comment line and no trailing newline for the same
    # reason.
    atomic_write(out_row, (
        f"{CNN_N[0]}--{CNN_N[-1]} & CNN surface (Gride) & {nu['nu_min']:.3f} & "
        f"{nu['nu_max']:.3f} & -- & --%"))

    kappa = frame[frame["row_type"] == "kappa"]
    print(kappa[["model", "reference_set", "method", "N_or_range", "m_lo",
                 "m_hi", "n_fits", "kappa_min", "kappa_max",
                 "kappa_absmax"]].round(4).to_string(index=False))
    local = frame[frame["row_type"] == "local"]
    print(local[["model", "reference_set", "method", "m_hi", "kappa_absmax",
                 "R_max", "full_extra_max", "full_total_max", "tau_min"]]
          .round(3).to_string(index=False))
    for model, (k1, k2) in FISHER_PAIRS.items():
        print(f"{model} (k1, k2) = ({k1}, {k2}): F = {sqrt_f[model] ** 2:.4f}, "
              f"sqrt(F) = {sqrt_f[model]:.4f}")
    print(f"Gride CNN reference mean direction, N={CNN_N[0]}..{CNN_N[-1]}, "
          f"m={CNN_M[0]}..{CNN_M[1]}: {nu['nu_min']:.6f} ({nu['nu_min_at']}) "
          f"to {nu['nu_max']:.6f} ({nu['nu_max_at']})")
    for path in (OUT_CSV, out_row):
        print(f"wrote {path}")
    print(f"elapsed {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
