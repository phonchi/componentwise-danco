"""Training routines for the FastDANCo calibration splines.

Used by ``experiments/train_splines.py``.  The caller injects the estimator
factory as the module-level name ``DANCo`` before calling
:func:`generate_calibration_data`.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
import pickle
from tqdm import tqdm
import argparse
import sys
from skdim.datasets import hyperBall
import os


def circular_mean(angles):
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def generate_calibration_data(k=10, d_range=range(1, 101),
                             N_range=None, n_replicas=35, random_state=42):
    if N_range is None:
        N_range = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]

    print("="*70)
    print("FastDANCo Pre-training: Generating Calibration Data")
    print("="*70)
    print(f"Parameters:")
    print(f"  k (neighbors): {k}")
    print(f"  d_range: [{min(d_range)}, {max(d_range)}] ({len(d_range)} values)")
    print(f"  N_range: {N_range} ({len(N_range)} values)")
    print(f"  n_replicas: {n_replicas}")
    print(f"  random_state: {random_state}")
    print(f"\nTotal iterations: {len(d_range) * len(N_range)} (N, d) pairs")
    print(f"Total hyperballs: {len(d_range) * len(N_range) * n_replicas}")
    print("="*70)

    # Initialize storage
    calibration_stats = {
        'dhat': np.zeros((len(d_range), len(N_range))),
        'mu_nu': np.zeros((len(d_range), len(N_range))),
        'mu_tau': np.zeros((len(d_range), len(N_range)))
    }

    total_iterations = len(d_range) * len(N_range)

    with tqdm(total=total_iterations, desc="Generating calibration data") as pbar:
        for i, d in enumerate(d_range):
            for j, N in enumerate(N_range):
                danco = DANCo(k=k, N=N)
                dhat_samples = []
                mu_nu_samples = []
                mu_tau_samples = []

                for replica in range(n_replicas):
                    # Generate hyperBall
                    Y = hyperBall(n=N, d=d, radius=1, center=[0]*d,
                                  random_state=random_state + replica)

                    # Compute statistics using DANCo's method
                    stats = danco.danco_statistic(Y)
                    dhat_samples.append(stats['dhat'])
                    mu_nu_samples.append(stats['mu_nu'])
                    mu_tau_samples.append(stats['mu_tau'])

                # Average over replicas
                calibration_stats['dhat'][i, j] = np.mean(dhat_samples)
                calibration_stats['mu_nu'][i, j] = circular_mean(mu_nu_samples)
                calibration_stats['mu_tau'][i, j] = np.mean(mu_tau_samples)

                pbar.update(1)
                pbar.set_postfix({
                    'd': d, 'N': N,
                    'dhat': f"{calibration_stats['dhat'][i, j]:.2f}"
                })

    print("\n✓ Calibration data generation complete")
    return calibration_stats


def fit_splines(calibration_stats, d_range, N_range, verbose=True):
    if verbose:
        print("\n" + "="*70)
        print("Fitting 2D Splines")
        print("="*70)

    d_list = list(d_range)

    # Determine spline degree based on number of points
    # Need at least degree+1 points for each dimension
    kx = min(3, len(d_list) - 1)  # cubic if possible, else lower
    ky = min(3, len(N_range) - 1)  # cubic if possible, else lower

    if verbose:
        degree_names = {1: 'linear', 2: 'quadratic', 3: 'cubic'}
        print(f"  d dimension: {degree_names.get(kx, str(kx))} (kx={kx}, {len(d_list)} points)")
        print(f"  N dimension: {degree_names.get(ky, str(ky))} (ky={ky}, {len(N_range)} points)")

    spline_dhat = RectBivariateSpline(
        x=d_list, y=N_range,
        z=calibration_stats['dhat'],
        kx=kx, ky=ky
    )
    if verbose:
        print("✓ Fitted spline_dhat")

    spline_mu_nu = RectBivariateSpline(
        x=d_list[:], y=N_range,
        z=calibration_stats['mu_nu'][:,:],
        kx=kx, ky=ky
    )
    if verbose:
        print("✓ Fitted spline_mu_nu")

    spline_mu_tau = RectBivariateSpline(
        x=d_list[:], y=N_range,
        z=calibration_stats['mu_tau'][:,:],
        kx=kx, ky=ky
    )
    if verbose:
        print("✓ Fitted spline_mu_tau")

    return {
        'spline_dhat': spline_dhat,
        'spline_mu_nu': spline_mu_nu,
        'spline_mu_tau': spline_mu_tau
    }


def validate_splines(splines, calibration_stats, d_range, N_range):
    print("\n" + "="*70)
    print("Validating Spline Quality")
    print("="*70)

    d_list = list(d_range)

    for stat_name in ['dhat', 'mu_nu', 'mu_tau']:
        spline = splines[f'spline_{stat_name}']
        true_values = calibration_stats[stat_name]

        # Evaluate on training points
        predicted = np.zeros_like(true_values)
        for i, d in enumerate(d_list):
            for j, N in enumerate(N_range):
                predicted[i, j] = spline(d, N)[0, 0]

        # Compute errors
        abs_error = np.abs(predicted - true_values)
        relative_error = abs_error / (np.abs(true_values) + 1e-10)

        mean_abs_error = np.mean(abs_error)
        mean_rel_error = np.mean(relative_error)
        max_abs_error = np.max(abs_error)
        max_rel_error = np.max(relative_error)

        print(f"\n{stat_name}:")
        print(f"  Mean absolute error: {mean_abs_error:.6f}")
        print(f"  Mean relative error: {mean_rel_error:.6f} ({mean_rel_error*100:.3f}%)")
        print(f"  Max absolute error: {max_abs_error:.6f}")
        print(f"  Max relative error: {max_rel_error:.6f} ({max_rel_error*100:.3f}%)")

        if mean_rel_error < 0.01:
            print(f"  ✓ Excellent fit (< 1% error)")
        elif mean_rel_error < 0.05:
            print(f"  ✓ Good fit (< 5% error)")
        else:
            print(f"  ⚠ Warning: High fitting error (>= 5%)")


def save_splines(splines, k, d_range, N_range, output_file=None):
    if output_file is None:
        output_file = f'fastdanco_splines_k{k}.pkl'

    fastdanco_splines = {
        'k': k,
        'd_range': list(d_range),
        'N_range': N_range,
        'spline_dhat': splines['spline_dhat'],
        'spline_mu_nu': splines['spline_mu_nu'],
        'spline_mu_tau': splines['spline_mu_tau']
    }

    with open(output_file, 'wb') as f:
        pickle.dump(fastdanco_splines, f)

    print(f"\n✓ Splines saved to: {output_file}")

    # Print file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")
