# Experiment scripts

One section per script: what it computes, the exact command used for the
paper, the options that matter, the seeds, and the files it writes. The
overview of which command produces which paper artifact is in the repository
[`README.md`](../README.md); the values the paper quotes from each output file
are in [`results/REPRODUCTION_STATUS.md`](../results/REPRODUCTION_STATUS.md).

## Conventions

- Run every command from the repository root in the `danco` environment. The
  scripts put `src/` on `sys.path` themselves; `common.py` (shared seeds,
  estimator registry, CLI helpers) and `paper_methods.py` (method identifiers
  and the reference TWO-NN) are imported by the others and are not run
  directly.
- Every script answers `--help`. `run_images.py` takes the shared options of
  `common.add_common_args`: `--output` (directory), `--seeds` (default 5),
  `--backend {cpu,gpu}`, `--n-jobs`, `--estimators`, `--subset` (dataset
  names, for smoke tests). The `--estimators` keys are `danco` (MiND--Full
  and, from the same evaluation, `nustrip` = MiND--Profiled and `acc` = the
  same MiND--Full values under the column name `accDANCo`), `gride`
  (Gride--Full and `gride_nus` = Gride--Profiled), `fast` (`DANCo_fast`, not
  used in the paper), `twonn` (reference TWO-NN, column `2NN_ref`), and
  `twonn_legacy` (untrimmed TWO-NN, column `2NN`, not used in the paper).
  `run_angle_diagnostics.py` has its own options (`--dataset`,
  `--no-evaluate`, `--seeds`, `--n-jobs`, `--output`).
- The archived results sit at the default output paths. `run_baselines.py`,
  `run_impl_comparison.py`, `run_fusion_ablation.py`, `run_sensitivity.py`,
  and `run_cnn_layers.py` resume from an existing output file and skip the
  rows it already contains, so a documented command rerun as is computes
  nothing. Pass `--no-resume` (not available for `run_sensitivity.py`) to
  recompute in place, which overwrites the archived file, or write to a new
  `--output`/`--outdir`. `run_relative_noise.py`, `run_images.py`, and the
  diagnostic scripts recompute and overwrite their outputs on every run.
- Side outputs that are not tracked: `results/tables/raw/` (per-seed image
  estimates), `results/logs/`, `results/splines/` (retrained surfaces), and
  `results/generated/` (tables and figures).
- Gride runs use the published generic-order likelihood score with its `N / d`
  term. The scripts call `DANCo.evaluate_all`, which returns the Full and the
  Profiled objective of one distance statistic from one pass
  (`stats["de_standard"]` and `stats["de_nustrip"]`), so both are compared on
  the same data sample and reference realization.

## Clean benchmark

### `run_baselines.py`

Nine reference estimators (`MLE, TwoNN, lPCA, ESS, MADA, TLE, CorrInt, MiND_ML,
FisherS`, all from scikit-dimension; MiND_ML with `--mind-dmax 100`) on the
24-manifold benchmark of scikit-dimension, `N=2500` points per manifold.

```bash
python experiments/run_baselines.py --reps 20
```

- Replicate `r` generates the manifolds with data seed `1000 + r`.
  `--manifolds` takes short names that match by prefix (`M1` selects
  `M1_Sphere`; `M5` selects both helices `M5a_Helix1d` and `M5b_Helix2d`).
- `--timeout 300` seconds per estimator call. A non-finite estimate stays in
  the CSV as `NaN` in `dhat` (its `status` remains `ok`) and counts as an
  error in the summaries; FisherS has 19 such rows in the archived file.
- The full run is about 14 CPU-hours of estimator time (sum of the recorded
  `seconds` column).
- `--noise-sweep` runs a fixed-amplitude Gaussian noise sweep instead; the paper
  does not use it.
- Output: `results/baselines/baselines_benchmark.csv` (long format: manifold,
  estimator, replicate, estimate, status, seconds). `--summarize` prints the
  aggregate tables from an existing CSV.

### `run_impl_comparison.py`

MiND--Full on the same 20 data replicates (`--impls ours`), plus two comparison
arms the paper does not report: `skdim_stock` (scikit-dimension's DANCo) and
`ours_legacy_bessel` (unscaled Bessel functions in the angular KL).

```bash
python experiments/run_impl_comparison.py --impls ours --reps 20
```

- Same data seeds as `run_baselines.py`; the calibration `random_state` is the
  same `1000 + r`, through one shared calibration pool per replicate. Rows
  with `impl == cal_*` record the wall time of building that pool and carry
  no estimate.
- Output: `results/impl_comparison/impl_comparison.csv`.

### `run_sensitivity.py`

Neighborhood-size and sample-size sensitivity of the four calibrated objectives
and TWO-NN on the benchmark: `k in {5, 10, 15, 20}` at `N=2500` and
`N in {625, 1250, 2500, 5000}` at `k=10`, five data replicates per setting.

```bash
python experiments/run_sensitivity.py --mode both
```

- Output: `results/baselines/sensitivity.csv`.

## Component ablations

### `run_fusion_ablation.py`

Distance-only, angle-only, and joint objectives for MiND and Gride, each with
the full and the profiled angular KL, on synthetic suites selected with
`--suites` (`clean`, `noise`, `gsm`, `crossed`; the command-line defaults are
exploratory and do not reproduce the paper). Two paper runs:

```bash
# clean components (Table S3): 24 manifolds, five shared reference realizations
python experiments/run_fusion_ablation.py --suites clean --reps 5 \
    --output results/diagnostics/fusion_ablation_clean.csv
# Gaussian scale mixture (Fig. 4, Tables S6-S7): d = 70 zero-padded into D = 100
python experiments/run_fusion_ablation.py --suites gsm --reps 30 \
    --scale-sigmas 0,0.1,0.15,0.2,0.25,0.3,0.35 \
    --latent-descale-sigmas 0.25 --gsm-ambient-d 100 \
    --output results/diagnostics/fusion_ablation_gsm.csv
```

- Clean suite: one benchmark realization (`BenchmarkManifolds(random_state=71)`)
  evaluated under five reference realizations, calibration seed `35 + rep`;
  the summary keeps the 14 manifolds for which every objective is finite
  (complete cases).
- GSM suite: `X_i = S_i Z_i` with `Z_i` a 70-dimensional Gaussian and
  `log S_i ~ N(0, sigma_S^2)`; `--scale-sigmas` is the list of `sigma_S`
  (unrelated to the Gride orders). `--gsm-ambient-d 100` zero-pads `d=70` into
  `D=100` (an isometry) so that the candidate search covers `1..100`; without
  it the search is capped at the ambient dimension 70. Data seed `rep`,
  calibration seed `700 + rep`, shared across `sigma_S` levels.
  `--latent-descale-sigmas 0.25` adds, at that level, the known-amplitude
  control that evaluates `X_i / S_i = Z_i` on the same 30 latent samples and
  reference realizations.
- Output: the `--output` CSV (one row per dataset, replicate, objective, level,
  intervention).

### `fusion_ablation_summary.py`

```bash
python experiments/fusion_ablation_summary.py <source> <summary> <bootstrap> \
    [--intervention-bootstrap <file>]
```

Complete-case, dataset-first aggregation of a `run_fusion_ablation.py` CSV into
means and standard deviations (`<summary>`), plus matched-replicate bootstrap
intervals (`<bootstrap>`, and `--intervention-bootstrap` for the descaling
control). The paper quotes the summaries; the bootstrap files are not used.
Paper invocations: the clean file to
`results/diagnostics/fusion_ablation_clean_{summary,bootstrap}.csv`, and the
GSM file to `results/diagnostics/fusion_ablation_gsm_{summary,bootstrap,intervention_bootstrap}.csv`.

### `fusion_ablation_signed_bias.py`

```bash
python experiments/fusion_ablation_signed_bias.py
```

Same aggregation as the summary script but with the sign of the percentage
error retained. Reads `fusion_ablation_clean.csv` and
`relative_noise_ablation.csv` and writes
`results/diagnostics/fusion_ablation_signed_bias.csv` (per-regime cells) and
`fusion_ablation_signed_bias_by_replicate.csv` (replicate-first aggregation;
the source of the signed panel of Table S5 and of the signed values in Table
S3). `--ablation-source <csv>` optionally appends the rows of another
`run_fusion_ablation.py` output with `source == ablation`; the paper does not
use this.

### `run_relative_noise.py`

Component ablation under neighborhood-relative Gaussian noise on the 24
manifolds: for clean data `X` the coordinate noise standard deviation is
`sigma_eps = eta * median_i(r_{i,10}) / sqrt(2 D)`, so that the root-mean-square
norm of the difference between the noise vectors of two points is `eta` times
the median tenth-neighbor distance.

```bash
python experiments/run_relative_noise.py --reps 10
```

- `--etas 0,0.05,0.1,0.2,0.4` (default). Clean-data seed `12000 + rep`,
  calibration seed `2000 + rep`, and a deterministic noise seed per level and
  manifold; all recorded per row.
- Output: `results/diagnostics/relative_noise_ablation.csv` and
  `relative_noise_ablation_summary.csv`.

### `gsm_penalty_curve.py`

Candidate-wise angular-location penalty (full minus profiled angular KL) at the
focal GSM level `sigma_S = 0.25`, five replicates (Supplementary S4.5). Runs in
minutes. Output: `results/diagnostics/gsm_penalty_curve.csv`.

### `gsm_selection_check.py`

Single-sample check of which neighbors the GSM selects (low-amplitude neighbors
first). Output: `results/diagnostics/gsm_selection_check.csv`. Not quoted in
the paper.

## Images

### `run_images.py`

MiND--Full, Gride--Full, MiND--Profiled, Gride--Profiled, and reference TWO-NN
on six raw image classes (Table 2): MNIST test digits 3 and 7, CIFAR-10 train
classes 2 and 3 (first 5000 each), ImageNet koala (`n01882714`) and butterfly
(`n02279972`). Raw pixels as float64, `k=10`, candidates `1..200`,
`fractal=True`. ImageNet images are filtered (grayscale files and images
smaller than 224 pixels on either side are dropped) and center-cropped to
224 x 224, leaving 547 koala and 648 butterfly images.

```bash
python experiments/run_images.py --dataset all
```

- Five calibration seeds, `random_state = 43 + seed`; `N` is the class size.
  MNIST and ImageNet build a fresh estimator per class; CIFAR builds one
  estimator per seed and evaluates both classes with it.
- Output: `results/tables/image_estimates_{mnist,cifar,imagenet}.csv`
  (seed-averaged estimates per class),
  `results/tables/image_statistics_{...}.csv` (`nu`, `tau`, `dhat` from the
  first seed), and the untracked per-seed table `results/tables/raw/`.

### `run_angle_diagnostics.py`

Angle-channel diagnostics on the same classes plus two synthetic controls (a
70-ball and a GSM sample), under three preprocessings: `raw`, `sphere`
(centered and projected to the unit sphere), and `contrast` (per-image
standardization).

```bash
python experiments/run_angle_diagnostics.py --dataset all --seeds 5
```

- Output: `results/diagnostics/angle_diagnostics.csv` (Table S8) and per-point
  arrays `results/diagnostics/perpoint/<dataset>_<class>_<variant>.npz`.
  Figures read three of them: `fig_nu_scatter` reads
  `imagenet_koala_raw.npz` and `imagenet_koala_contrast.npz`;
  `fig_kl_diagnosis` reads `mnist_3_raw.npz` and `imagenet_koala_raw.npz`. The
  synthetic rows and files are not used in the paper.

### `run_class_survey.py`

Centered-norm coefficient of variation and angular statistics for 17 image
classes: the 10 CIFAR-10 training classes (5000 images each) and the 7 ImageNet
synsets (the images of each `<synset>/0/` folder after the same filtering and
cropping as `run_images.py`). Table S9; the range brackets the `sigma_S` grid
of the GSM experiment.

```bash
python experiments/run_class_survey.py --source all
```

- Output: `results/std_datasets/class_survey.csv`.

### `export_reference_envelope.py`

Reference mean-direction curves at the image sample sizes
(`547, 648, 1010, 1028, 5000`), seeds `43..47`, `k=10`, candidates `1..200`,
and their pooled range over the candidates with valid angular fits (`6..200`),
Table S1.

```bash
python experiments/export_reference_envelope.py --max-dim 200
```

- Output: `results/diagnostics/image_reference_curves.csv` and
  `image_reference_envelope.csv`. `--reuse-curves` recomputes the envelope from
  an existing curves file.

### `export_reference_curvature.py`

Log-log slopes and local curvature diagnostics of the reference curves
(Supplementary S4.2) and the CNN row of Table S1 (reference mean direction of
the Gride surface over `N = 495..500` and `m = 6..400`).

```bash
python experiments/export_reference_curvature.py [--paper-dir <dir>]
```

- Reads `image_reference_curves.csv` and the two packaged spline surfaces.
  The clean-reference curves (`N=2500`, seeds `35..39`) are built on first run
  and cached in `results/diagnostics/clean_reference_curves.csv`; delete the
  cache to rebuild it.
- Output: `results/diagnostics/reference_curvature.csv`, rewritten in place on
  every run (a rerun reproduces the archived file up to floating-point
  summation order, differences below 1e-11), and
  `table_reference_range_cnn_row.tex` (to `--paper-dir`/tables, or
  `results/generated/tables/`).

## CNN layer profiles

### `run_cnn_layers.py`

Layer-wise ID in AlexNet, VGG-16, ResNet-18, and ResNet-34 (torchvision
`DEFAULT` weights) on 500-image subsets of the seven ImageNet synsets, at the
checkpoints defined in `danco.hooks` (Fig. 5, Table S10).

```bash
for r in 0 1 2; do
  python experiments/run_cnn_layers.py --backend gpu --diagnostics \
      --subset-rep "$r" --output "results/cnn/cnn_layer_ids_rep${r}.csv"
done
```

- Without CUDA use `--backend cpu` (the GPU backend raises rather than
  falling back). One replicate takes about 25 minutes on one GPU
  (`wall_s` in the summary files).
- Image subset: `numpy.random.default_rng(seed).permutation(len(dataset))[:500]`
  with `seed = 1000 + 100 * arch_index + category_index + 100000 * rep`, where
  `arch_index` is the position in `alexnet, vgg16, resnet18, resnet34` and
  `category_index` the position in the script's `CATEGORY_TAGS` order
  `n02086240, n01882714, n02087394, n02094433, n02100583, n02100735,
  n02279972`, and `dataset` is `torchvision.datasets.ImageFolder(<synset>/)`
  (see the data section of the repository README for the folders this
  indexes).
- Exact-duplicate activation rows are removed once per checkpoint; the common
  matrix (effective `N` = number of unique rows, 495..500) is evaluated with
  Gride--Profiled (`--estimator gride`: `version="DANCo_gride_fast"` on the
  packaged Gride surface through `evaluate_all`, reading `stats["de_nustrip"]`,
  the value `DANCo_gride_nustrip_fast` returns directly; `fractal=False`,
  candidates `1..400`), with the reference TWO-NN of Facco et al. (scikit-dimension's
  discarded-tail fit), and, with `--diagnostics`, with scikit-dimension's
  `MLE(K=10)` plus per-layer centered-norm and neighbor-distance columns.
  `--estimator mind` evaluates MiND--Profiled on the MiND surface (not used in
  the paper).
- The GPU backend runs with its near-duplicate filter disabled
  (`dup_threshold=None`) so that CPU and GPU see the same deduplicated matrix.
- Output: the `--output` CSV (one row per architecture, synset, checkpoint) and
  a `.summary.json` beside it with `archs`, `categories`, `nsamples`, `maxdim`,
  `k`, `estimator`, `version`, `backend`, `device`, `backend_options`,
  `duplicate_policy`, `method_ids`, `pretrained`, `completed`, `failures`,
  `wall_s`, `subset_rep`, and the `software` versions. A full replicate takes
  hours on one GPU; `--dry-run` lists the checkpoints and activation shapes
  without estimating.

### `audit_paper_methods.py`

Writes `results/diagnostics/paper_method_contract_audit.csv`, the
machine-readable list of which estimator (method identifier, spline surface,
deduplication policy) stands behind every manuscript table and figure, as
registered in `paper_methods.py`. Quality control; not quoted in the paper.

### `audit_cnn_backend_parity.py`

Recomputes the input, aggregate-peak, output, and maximum-duplicate checkpoints
of the CNN profiles on the CPU backend and compares them with the archived GPU
values (`--tolerance 1`: at most one dimension of drift). One cell,
`(subset_rep 2, vgg16, n02279972, layer_idx 2)`, differs by more than that
because of high-dimensional angle-reduction sensitivity; its exact CPU and GPU
values are listed in the script (`DOCUMENTED_EXCEPTIONS`) and in the output
file. Requires the GPU run's inputs. Output: `results/diagnostics/cnn_backend_parity.csv`.
Quality control; not quoted in the paper.

## Mechanism and numerical diagnostics

### `run_nu_toy_sweep.py`

Reference mean direction of the angular statistic on uniform balls (by
dimension) and on the GSM over `sigma_S in {0, 0.1, 0.15, 0.2, 0.25, 0.3,
0.35}` (Fig. 2(b)), and the mean per-point angular location by centered-norm
decile at `sigma_S = 0.25` (Fig. S1). Output:
`results/diagnostics/nu_toy_{ball,gsm,gsm_profile}.csv`.

### `run_vm_adequacy.py`

Descriptive adequacy of the von Mises fit to the pairwise-angle distribution on
`ball70`, `gsm` (`--gsm-sigma 0.25`), `mnist3`, `koala_raw`, and
`koala_contrast`. Fig. S2 plots `ball70` and `koala_raw`; Supplementary S1.4
quotes the `gsm` file; `mnist3` and `koala_contrast` are not used. Output:
`results/diagnostics/vm_adequacy_<name>.npz`.

### `run_overflow_curve.py`

Ordinary versus exponentially scaled Bessel evaluation of the angular KL along
the candidate search at the CNN reference sizes (Fig. S3):

```bash
python experiments/run_overflow_curve.py --calibration-n 450,500,580,640,700
```

Output: `results/diagnostics/overflow_candidate_curve.csv` (read by the
figure) and `overflow_curve.csv` (a concentration sweep, not used).

## Runtime

### `run_runtime.py`

Matched CPU/GPU wall time and peak memory. Each cell is one untimed warm-up
followed by `--repeats` measured runs in isolated subprocesses; when CUDA is
available, CPU and GPU rows are written for the same stage and `--dtype`
(`float64` in the paper). Monte Carlo reference construction stays on the CPU
for both backends.

```bash
python experiments/run_runtime.py --repeats 3 --sweeps D --values 100,1000,10000 \
    --stages data_side --output results/runtime/runtime_data_side.csv
python experiments/run_runtime.py --repeats 3 --sweeps D --values 100000 \
    --stages data_side --output results/runtime/runtime_data_side_d100000.csv
python experiments/run_runtime.py --repeats 3 --sweeps maxDim \
    --stages spline_evaluate,full_mc_evaluate,cached_mc_evaluate \
    --output results/runtime/runtime_calibration.csv
```

- Stages: `data_side` (observed neighbor search and distance/angular
  statistics only), `spline_evaluate` (complete evaluation with precomputed
  spline references), `full_mc_evaluate` (complete evaluation with newly
  generated Monte Carlo references), `cached_mc_evaluate` (complete evaluation
  with reused Monte Carlo references).
- Sweeps: `N` (625..10000), `D` (100..100000), `k` (5..20), `maxDim`
  (50..400); `--values` overrides the grid. `--max-load` skips measurement when
  the machine is busy (`--force` overrides).
- Output: the `--output` CSV with one row per measured repeat (`seconds`,
  per-process peak RSS `rss_mb`, peak CUDA memory `cuda_peak_mb`); the
  reported values are medians over the repeats (Fig. S4).

## References and surfaces

### `train_splines.py`

Trains a spline reference surface for one distance statistic (`--variant
mind` or `gride`): for every cell of the grid `--d-min`..`--d-max` by
`--n-range`, it draws `--replicas` uniform hyperballs (`skdim.datasets.hyperBall`,
replica `r` with `random_state = --seed + r`), computes `danco_statistic`
(`dhat`, `mu_nu`, `mu_tau`) on each, averages them per cell (circular mean for
`mu_nu`), fits three `RectBivariateSpline` surfaces over `(d, N)`, prints a
validation report at the training points (`--no-validate` skips it), and
writes the pickle stamped with `variant` and `seed`. Without `--yes` the
script only reports the size of the job. The repository README section
"Spline surfaces" describes the packaged surfaces and how to use a trained one.

```bash
# the packaged Gride surface: d = 1..400, N in {450,500,580,640,700}, k = 10,
# 35 replicas, seed 42 (70,000 calibration runs, several hours)
python experiments/train_splines.py --variant gride --yes
# a small demonstration surface (about ten seconds)
python experiments/train_splines.py --variant gride --d-max 30 --n-range 450,500     --replicas 5 --seed 1 --output results/splines/demo_gride.pkl --yes
```

- Defaults: `--d-min 1`, `--d-max 400`, `--n-range 450,500,580,640,700`,
  `--k 10`, `--replicas 35`, `--seed 42`, `--output
  results/splines/fastdanco_splines[_gride]_400.pkl` (untracked).
- Use: `DANCo(N=..., k=10, version="<any _fast version of that variant>").evaluate(X, maxDim=..., spline_file=<pickle>)`
  (or `evaluate_all`). Loading refuses a surface whose `variant` or `k` does
  not match the estimator; `N` must lie inside the surface's `N` range and
  `maxDim` within its `d` range unless `check_spline_range=False`.
- The packaged Gride surface was trained with exactly the defaults; the
  packaged MiND surface has the same grid but predates the seed stamp.

## Tables, figures, graphical abstract

```bash
python experiments/make_tables.py            [--paper-dir <dir>]
python experiments/make_figures.py           [--paper-dir <dir>] [--figures a,b]
python experiments/make_graphical_abstract.py [--paper-dir <dir>] [--design-dir <dir>]
```

`make_tables.py` writes eleven manuscript tables (`table_images`,
`table_diagnostics`, `table_gsm_sweep`, `table_cnn_peak_summary`,
`table_cnn_endpoints`, `table_sensitivity`, `table_baselines_main`,
`table_baselines_full`, `table_class_survey`, `table_relative_noise`,
`table_relative_noise_signed`) and `table_impl.tex` (implementation
comparison, not in the manuscript). `make_figures.py` writes eleven
manuscript figures (`fig_nu_mechanism_main`, `fig_noise_curves`,
`fig_fusion_ablation_main`, `fig_cnn_paired`, `fig_nu_mechanism`,
`fig_vm_adequacy`, `fig_overflow`, `fig_runtime`, `fig_error_vs_d`,
`fig_kl_diagnosis`, `fig_nu_scatter`) and `fig_kl_diagnosis_main.pdf` (not in
the manuscript); `--figures` selects a subset by the keys listed in `--help`.
`make_graphical_abstract.py` draws the graphical abstract from the recorded
relative-noise and GSM summaries (PDF and PNG) and writes the editable SVG,
an HTML preview, and a layout check to `--design-dir`
(default `results/generated/graphical_abstract/`); `--curves-only --rep R`
recomputes the optional single-sample KL diagnostic
`results/diagnostics/graphical_abstract_curves.csv` instead of drawing.

Without `--paper-dir`, everything goes to `results/generated/` (`tables/`,
`figures/`, and `graphical_abstract/` for the editable SVG, preview, and
checks). With `--paper-dir <dir>`, the manuscript tables and figures go to
`<dir>/tables` and `<dir>/figures` and the rest stays in `results/generated/`.
A table or figure whose source file is missing is skipped with a message; a
source with unexpected content (replicate count, columns) raises.

## Seed and metric invariants

- Clean benchmark: 20 independent data replicates, data seed `1000 + rep`,
  `N=2500`, `k=10`, 24 manifolds; MPE first averages the estimates over
  replicates within each manifold, takes the absolute relative error against
  the true dimension, and averages over the 24 manifolds. Non-finite estimates
  stay in the CSV (`NaN` in `dhat`) and count as errors.
- Component ablations: clean suite on benchmark seed 71 with calibration
  seeds `35 + rep`; GSM suite with data seed `rep` and calibration seed
  `700 + rep`. The GSM summary is computed per replicate before aggregation,
  and the `latent_descaled` rows share the latent samples and calibration
  seeds of the raw rows.
- Relative noise: clean-data seed `12000 + rep`, calibration seed
  `2000 + rep`, deterministic noise seeds per level and manifold; the paper uses
  neighborhood-relative noise only.
- Images: `N` = class size, calibration seeds `43..47`, candidates `1..200`;
  the image reference range excludes candidates with invalid angular fits.
- CNN profiles: three matched image-subset replicates, `N=500` before
  deduplication, `k=10`, candidate bound 400, `DANCo_gride_fast` on the Gride
  surface; each spline surface serves exactly one distance statistic (checked
  at load).
