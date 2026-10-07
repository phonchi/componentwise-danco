# Reproduction ledger

Release 1.0.0 (2026-09-27). For every table, figure, and quoted number of the
manuscript this file names the archived result file (and, where needed, the
rows or columns) it comes from and the generated artifact that renders it.
Commands are in the repository [`README.md`](../README.md); script options,
seeds, and output files are in [`experiments/README.md`](../experiments/README.md).
Paths below are relative to `results/`; several files of one directory are
listed with the directory once.

## Main text

| Paper location | Quoted content | Result file and selector | Generated artifact |
| --- | --- | --- | --- |
| Table 1 | MiND--Full MPE `6.33%` and the nine reference estimators on 24 manifolds, 20 replicates; FisherS has 19 non-finite estimates (`NaN` in `dhat`, `status` stays `ok`) that count toward its error rate | `impl_comparison/impl_comparison.csv` (`impl == ours`; MPE = mean over replicates per manifold, then mean absolute relative error over the 24 manifolds); `baselines/baselines_benchmark.csv` | `table_baselines_main.tex` |
| Fig. 2(b) | reference mean direction on uniform balls by dimension, and on the GSM over `sigma_S in {0, .1, .15, .2, .25, .3, .35}`: `1.132, 1.055, 0.988, 0.906, 0.847, 0.763, 0.737` | `diagnostics/nu_toy_ball.csv`; `diagnostics/nu_toy_gsm.csv` (`nu` by `sigma_s`) | `fig_nu_mechanism_main.pdf` |
| Fig. 3 | MPE of the eight component objectives under neighborhood-relative noise, `eta in {0, .05, .1, .2, .4}`; at `eta = 0.4` Gride--Full `17.64%`, MiND--Full `27.74%` | `diagnostics/relative_noise_ablation_summary.csv` (`method in {danco, danco_gride}`, `noise_eta == 0.4`) | `fig_noise_curves.pdf` |
| Fig. 4 | GSM sweep at `d = 70` in `D = 100`, 30 replicates; at `sigma_S = 0.25`: MiND--Full `22.80 +/- 1.76`, MiND--Profiled `66.67 +/- 1.81`, MiND distance-only `71.20 +/- 1.75`; known-amplitude removal restores MiND--Full to `71.90 +/- 3.30`; shaded band = image-class CV range | `diagnostics/fusion_ablation_gsm_summary.csv` (`intervention == raw` / `latent_descaled`, `method in {danco, profiled_danco, mind_distance_only}`, `estimate_mean`, `estimate_sd`); `std_datasets/class_survey.csv` | `fig_fusion_ablation_main.pdf` |
| Table 2 | MiND/Gride Full and Profiled estimates, TWO-NN, `nu`, `tau` on six raw image classes (candidates to 200) | `tables/image_estimates_{mnist,cifar,imagenet}.csv`, `tables/image_statistics_{...}.csv` | `table_images.tex` |
| Fig. 5 | layer-wise Gride--Profiled, TWO-NN, and MLE profiles in four CNNs, three subset replicates, 672 complete cells with effective `N = 495..500` | `cnn/cnn_layer_ids_rep{0,1,2}.csv` | `fig_cnn_paired.pdf` |
| Sections 3--5 (runtime sentences) | matched CPU/GPU medians: observed-data statistics `0.039/0.404` s (`D=100`), `0.235/0.628` (`D=1000`), `4.282/3.569` (`D=10000`), `51.152/33.569` (`D=100000`); complete evaluation at `m_max = 400`: spline `0.053/0.225`, new Monte Carlo `8.142/15.428`, reused Monte Carlo `0.048/0.455`; at `D = 100000` peak RSS `3761` MB (CPU) and `2887` MB host + `3824` MB CUDA (GPU) | `runtime/runtime_data_side.csv`, `runtime/runtime_data_side_d100000.csv`, `runtime/runtime_calibration.csv` (one row per `repeat`; medians over `repeat` of `seconds`, `rss_mb`, `cuda_peak_mb` per `stage`, `backend`, and sweep value) | `fig_runtime.pdf` (Fig. S4) |

## Supplementary

| Paper location | Quoted content | Result file and selector | Generated artifact |
| --- | --- | --- | --- |
| Fig. S1 | candidate-wise angular profile of the GSM at `sigma_S = 0.25` | `diagnostics/nu_toy_gsm_profile.csv` | `fig_nu_mechanism.pdf` |
| Fig. S2, S1.4 | von Mises adequacy on the 70-ball and ImageNet koala; GSM at `sigma_S = 0.25`: maximum CDF gap `0.022`, `nu = 0.832 +/- 0.002`, `tau = 43.7 +/- 0.7` | `diagnostics/vm_adequacy_ball70.npz`, `diagnostics/vm_adequacy_koala_raw.npz` (figure); `diagnostics/vm_adequacy_gsm.npz` (text) | `fig_vm_adequacy.pdf` |
| Fig. S3 | at `N_ref = 500` the reference concentration first exceeds the overflow point `709.78` at candidate 278, where the unscaled Bessel evaluation departs from the scaled one; it turns negative at 284, while the scaled evaluation stays finite and positive through 400 | `diagnostics/overflow_candidate_curve.csv` (`calibration_n == 500`, columns `tau_ref`, `kl_legacy`, `kl_scaled`) | `fig_overflow.pdf` |
| Fig. S4 | runtime and memory figure (values above) | runtime files above | `fig_runtime.pdf` |
| Table S1 | image reference mean-direction range `[1.142, 1.495]` rad over 4,875 valid rows (candidates `6..200`, sizes `547, 648, 1010, 1028, 5000`, seeds `43..47`); CNN row: Gride surface, `N = 495..500`, `[1.115, 1.431]` over `m = 6..400` | `diagnostics/image_reference_curves.csv`, `diagnostics/image_reference_envelope.csv`; `diagnostics/reference_curvature.csv` (`row_type == nu_range`) | image rows entered manually in the manuscript; `table_reference_range_cnn_row.tex` |
| Table S2 | full clean benchmark (all estimators, all manifolds) | as Table 1 | `table_baselines_full.tex` |
| Fig. S5 | per-manifold error against true dimension | `baselines/baselines_benchmark.csv`, `impl_comparison/impl_comparison.csv` (`impl == ours`) | `fig_error_vs_d.pdf` |
| Table S3 | clean components on the 14 complete-case manifolds, five reference realizations: MiND--Full `8.62 +/- 0.67%`, MiND distance-only `10.12 +/- 0.24%`; signed errors distance-only `-1.6/-1.8%`, angle-only `+10.7/+6.2%`, combined within `2.6%` | `diagnostics/fusion_ablation_clean_summary.csv` (`method`, `mpe_mean`, `mpe_sd`); `diagnostics/fusion_ablation_signed_bias_by_replicate.csv` (`source == clean_confirmatory`, `signed_pe_mean`) | entered manually in the manuscript from the two files |
| S4.2 | clean-reference curvature ratios `R_m`: MiND `0.31`, Gride `0.38` (`N = 2500`, seeds `35..39`); image and CNN-surface log-log slopes | `diagnostics/reference_curvature.csv` (`row_type == local`, `reference_set == clean references`, `method == window_10`, column `R_max` by `model`; slopes in `row_type == kappa`); `diagnostics/clean_reference_curves.csv` | quoted in text |
| Table S4 | `k in {5, 10, 15, 20}` at `N = 2500` and `N in {625, 1250, 2500, 5000}` at `k = 10`, five replicates, four objectives plus TWO-NN | `baselines/sensitivity.csv` | `table_sensitivity.tex` |
| Table S5 | relative-noise MPE per objective and level (upper panel), with a restricted-subset row at `eta = 0.4` over the 17 manifolds that remain after excluding the seven with `true_d > 5` and `ambient_d <= true_d + 1`: `37.62/23.43/36.93/22.49` for MiND--Full, Gride--Full, MiND--Profiled, Gride--Profiled; signed errors (lower panel; monotone in `eta`, all eight positive at `eta = 0.4`) | `diagnostics/relative_noise_ablation_summary.csv`; `diagnostics/relative_noise_ablation.csv` (restricted row, `objective == joint`, `noise_eta == 0.4`); `diagnostics/fusion_ablation_signed_bias_by_replicate.csv` (`source == relative_noise`) | `table_relative_noise.tex`, `table_relative_noise_signed.tex` |
| Table S6 | GSM at the focal level `sigma_S = 0.25` with and without known-amplitude removal; `nu` moves from `0.832 +/- 0.014` to `1.137 +/- 0.003` rad | estimates as Fig. 4; `nu` from `diagnostics/fusion_ablation_gsm.csv` (`scale_sigma == 0.25`, one `mu_nu` per `intervention` and `rep`, mean and sample standard deviation over `rep`) | entered manually in the manuscript |
| Table S7 | GSM sweep over the seven `sigma_S` levels with the `nu` column; Profiled stays within `62.47--70.20` over the image-class range | `diagnostics/fusion_ablation_gsm_summary.csv` (estimates); `diagnostics/fusion_ablation_gsm.csv` (`mu_nu` per level, as for Table S6) | `table_gsm_sweep.tex` |
| S4.5 | angular-location penalty across candidates at the focal level, five replicates: rises from `1.28` to `14.76` nats over `m = 6..100`, increasing at `87.4/94` steps | `diagnostics/gsm_penalty_curve.csv` | quoted in text |
| Fig. S6 | KL curves of MNIST 3 (inside the reference range) and ImageNet koala (location mismatch) | `diagnostics/perpoint/mnist_3_raw.npz`, `diagnostics/perpoint/imagenet_koala_raw.npz` | `fig_kl_diagnosis.pdf` |
| Table S8 | angle statistics of the image classes under raw, sphere, and contrast preprocessing | `diagnostics/angle_diagnostics.csv` (non-synthetic rows) | `table_diagnostics.tex` |
| Fig. S7 | per-point scatter for ImageNet koala, raw against contrast-normalized | `diagnostics/perpoint/imagenet_koala_raw.npz`, `diagnostics/perpoint/imagenet_koala_contrast.npz` | `fig_nu_scatter.pdf` |
| Table S9 | 17-class centered-norm survey (10 CIFAR-10 classes, 7 ImageNet synsets); CV range `0.149--0.330` brackets the GSM `sigma_S` grid | `std_datasets/class_survey.csv` | `table_class_survey.tex` |
| Table S10 | CNN peak layers per architecture (Gride--Profiled `115--248` at the second downsampling stage, MLE `40--62`, TWO-NN `69--143`) and the endpoint panel (first-pooling/output `nu`, output Full/Profiled medians, median paired Full - Profiled differences `-1/-2/-4/-6`) | `cnn/cnn_layer_ids_rep{0,1,2}.csv` | `table_cnn_peak_summary.tex`, `table_cnn_endpoints.tex` |
| Graphical abstract | relative-noise MPE at `eta = 0.4` (MiND--Full, Gride--Full) and GSM estimates at `sigma_S = 0.25` (MiND--Full, MiND--Profiled) | `diagnostics/relative_noise_ablation_summary.csv`, `diagnostics/fusion_ablation_gsm_summary.csv` | `graphical_abstract.pdf`, `graphical_abstract.png` |

Method names inside the ablation and noise CSVs: `danco` = MiND--Full,
`profiled_danco` = MiND--Profiled, `danco_gride` = Gride--Full,
`profiled_danco_gride` = Gride--Profiled, `mind_distance_only` /
`gride_distance_only` = distance-only, `standard_angle_only` /
`profiled_angle_only` = angle-only. In the image and sensitivity CSVs the
estimator columns are `DANCo`/`aDANCo`/`accDANCo` (MiND--Full),
`DANCo_nustrip`/`DANCo_nus` (MiND--Profiled), `DANCo_Gride`/`DANCo_G`
(Gride--Full), `DANCo_Gride_nustrip`/`DANCo_G_nus` (Gride--Profiled), and
`2NN_ref` (reference TWO-NN).

## Archived files the paper does not use

Retained because a documented command produces them or for quality control.

- `diagnostics/gsm_selection_check.csv` (`gsm_selection_check.py`).
- `diagnostics/cnn_backend_parity.csv` (`audit_cnn_backend_parity.py`): 251
  of the 252 audited CNN cells agree exactly between CPU and GPU; the cell
  `(subset_rep 2, vgg16, n02279972, layer_idx 2)` differs by more than one
  dimension in the Profiled estimate (high-dimensional angle-reduction
  sensitivity) and is listed with its values in the script.
- `diagnostics/paper_method_contract_audit.csv` (`audit_paper_methods.py`).
- `diagnostics/fusion_ablation_clean_bootstrap.csv`,
  `fusion_ablation_gsm_bootstrap.csv`,
  `fusion_ablation_gsm_intervention_bootstrap.csv`: bootstrap
  intervals written by `fusion_ablation_summary.py`.
- `diagnostics/fusion_ablation_signed_bias.csv`: per-regime cells; the tables
  read the `_by_replicate` file.
- `diagnostics/overflow_curve.csv`: concentration sweep; the figure reads
  `overflow_candidate_curve.csv`.
- `diagnostics/vm_adequacy_mnist3.npz`, `vm_adequacy_koala_contrast.npz`.
- `diagnostics/angle_diagnostics.csv` rows with `dataset == synthetic` and the
  files `diagnostics/perpoint/synthetic_*.npz`; the other `perpoint/*.npz`
  files except `mnist_3_raw`, `imagenet_koala_raw`, and
  `imagenet_koala_contrast` are written by `run_angle_diagnostics.py` but read
  by no figure.
- `diagnostics/graphical_abstract_curves.csv`: optional single-sample KL
  diagnostic of `make_graphical_abstract.py --curves-only`.
- `diagnostics/clean_reference_curves.csv`: cache of
  `export_reference_curvature.py` (its `R_m` values enter S4.2 through
  `reference_curvature.csv`).
- `impl_comparison/impl_comparison.csv` rows with `impl != ours`:
  `skdim_stock` and `ours_legacy_bessel` comparison arms and the `cal_*`
  calibration-timing rows.
- `baselines/baselines_benchmark.csv` rows of the noise sweep, if present:
  the paper uses neighborhood-relative noise only.
