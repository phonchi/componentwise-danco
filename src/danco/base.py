"""Shared DANCo skeleton.

One ``evaluate`` for every variant.  What changes between variants is injected:

* the **distance strategy** (:mod:`danco.variants`) supplies the dimension
  estimate and the distance KL;
* the **backend** (:mod:`danco.backends`) supplies neighbour search and the
  angle reduction;
* the **calibration source** is either Monte-Carlo hyperballs or the pre-trained
  splines.

The angle KL is *always* computed both ways -- the standard form and the
nu-restricted ("strip") form -- because they share every expensive input and
differ only in a ``cos(nu1 - nu2)`` factor.  A single run therefore reports both
estimates, so an experiment never has to pay for calibration twice.

Calibration state:

* **RNG stream.**  ``random_state_`` is created once in ``__init__`` and, in the
  default ``calibration_seeding='stream'`` mode, each calibration dimension
  consumes exactly one ``normal()`` draw.  Reordering, adding or dropping a
  draw changes every downstream number.
* **Growing calibration cache.**  ``calibration_statistic`` persists on the
  instance and only ever *extends*.  Candidate ``m`` always consumes the
  ``m``-th draw of the estimator's RNG, so a reference replicate is a
  deterministic function of ``(random_state, N)`` and does not depend on which
  datasets were evaluated first.  ``calibration_seeding='derived'``
  additionally makes each dimension's draw independent of the others.
* The ``dhat <= 5`` shortcut returns *before* the calibration grows, so
  low-dimensional datasets leave the cache untouched.

Monte Carlo calibration therefore supports stream mode (default, shared RNG)
and derived mode (per-dimension seeds); spline calibration loads precomputed
references.  The Gride distance strategy uses the published generic-order
likelihood score with ``N / d``.  The unscaled Bessel angle KL remains
available through the explicit ``use_legacy_bessel`` switch.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.utils.validation import check_array, check_random_state

from ._kernels.angles import indnComb
from ._kernels.vonmises import (
    circular_mean_resultant,
    kl_vonmises,
    kl_vonmises_strip,
    ml_vm,
)
from .backends import get_backend
from .calibration import empty_calibration, hyper_ball
from .spline import grow_spline_calibration
from .variants import GrideVariant, MiNDVariant

__all__ = ["DANCoBase", "VERSIONS", "OBJECTIVES"]


# version -> (distance strategy, spline reference?, return the Profiled estimate?)
#
# Four basic versions cover the paper's four objectives with Monte Carlo
# references; the ``_fast`` suffix switches each of them to the packaged
# spline reference surface of its distance strategy.
VERSIONS = {
    # MiND distance statistic
    "DANCo": (MiNDVariant, False, False),               # MiND--Full
    "DANCo_nustrip": (MiNDVariant, False, True),        # MiND--Profiled
    "DANCo_fast": (MiNDVariant, True, False),           # MiND--Full, spline
    "DANCo_nustrip_fast": (MiNDVariant, True, True),    # MiND--Profiled, spline
    # Gride distance statistic
    "DANCo_gride": (GrideVariant, False, False),        # Gride--Full
    "DANCo_gride_nustrip": (GrideVariant, False, True),  # Gride--Profiled
    "DANCo_gride_fast": (GrideVariant, True, False),    # Gride--Full, spline
    "DANCo_gride_nustrip_fast": (GrideVariant, True, True),  # Gride--Profiled, spline
}

OBJECTIVES = ("full", "profiled")


_REQUIRED_STATS = ("dhat", "mu_nu", "mu_tau")


def _validate_calibration(calibration_data):
    """Reject a calibration cache that cannot serve both angle KLs.

    Worth checking up front because a cache built by an estimator that fitted
    only ``tau`` carries no ``mu_nu``.  Since this class always computes
    the standard angle KL alongside the strip one, such a cache would otherwise
    blow up with a bare ``KeyError`` deep inside the KL loop, after the run had
    already spent its time.
    """
    if not isinstance(calibration_data, dict):
        raise TypeError(
            "calibration_data should be a dict "
            "{'statistics': [...], 'maxdim': int}, got "
            f"{type(calibration_data).__name__}"
        )
    missing_top = {"statistics", "maxdim"} - set(calibration_data)
    if missing_top:
        raise ValueError(
            f"calibration_data is missing key(s) {sorted(missing_top)}; expected "
            "{'statistics': [...], 'maxdim': int}"
        )
    stats = calibration_data["statistics"]
    if len(stats) != calibration_data["maxdim"]:
        raise ValueError(
            f"calibration_data claims maxdim={calibration_data['maxdim']} but "
            f"carries {len(stats)} statistics; the two must agree, since "
            "dimension d is looked up at index d-1"
        )
    for d, entry in enumerate(stats, start=1):
        missing = [key for key in _REQUIRED_STATS if key not in entry]
        if missing:
            raise ValueError(
                f"calibration_data['statistics'][{d - 1}] (dimension {d}) is "
                f"missing {missing}. A cache without 'mu_nu' (built by an "
                "estimator that never fitted nu) cannot be reused here, since "
                "this estimator reports the standard and the nu-restricted "
                "estimate from one pass."
            )


class DANCoBase:
    """Calibrated intrinsic-dimension estimator.

    Parameters
    ----------
    N : int
        Number of points every dataset passed to :meth:`evaluate` must have, and
        the size of each calibration hyperball.
    k : int
        Neighbourhood size.
    calibration_data : dict or None
        Pre-computed cache ``{'statistics': [...], 'maxdim': int}``.  Reused and
        extended in place.
    fractal : bool
        Refine the integer argmin by interpolating the KL curve.
    random_state : int, RandomState or None
        Seeds the calibration sampler.  ``calibration_seeding='derived'``
        requires an integer.
    version : str
        One of :data:`VERSIONS`: the objective returned by :meth:`evaluate`
        (``'DANCo'`` MiND--Full, ``'DANCo_nustrip'`` MiND--Profiled,
        ``'DANCo_gride'`` Gride--Full, ``'DANCo_gride_nustrip'``
        Gride--Profiled), optionally with the ``_fast`` suffix to read the
        packaged spline reference surface instead of running Monte Carlo
        calibration.  :meth:`evaluate_all` returns both objectives of the
        chosen distance statistic from one pass.
    backend : str
        ``'cpu'`` or ``'gpu'``.  Applies to the *data* statistic only: the
        Monte-Carlo calibration hyperballs always run on the CPU backend.  They
        are small (``N`` points in ``d <= maxDim`` dimensions) and dominated by
        transfer, so a device calibration would not pay off.
    backend_options : dict or None
        Forwarded to the backend constructor.  ``'cpu'`` takes none;
        ``'gpu'`` takes ``query_batch``, ``angle_batch``, ``dup_threshold``,
        ``extra_neighbors``, ``device`` and ``dtype`` (see
        :class:`danco.backends.gpu.GpuBackend`).
    use_legacy_bessel : bool
        Reproduce the earlier implementation's angle-KL expression exactly,
        including the unscaled ``i0``/``i1`` with a ``float_info.max`` clamp
        that breaks down at large concentration.  It affects the *standard*
        angle KL only: the profiled form is written with the scaled Bessel
        functions and needs no switch.

        This is a compatibility switch, not a supported mode.  It does not
        revert the modern Gride likelihood score to the legacy finite-sample
        adjustment.  Turning it on can inject NaN into the Full angular curve
        once a concentration passes ~709.78, which happens from candidate 278
        at N = 500; the Profiled curve is unaffected.
    check_spline_range : bool
        For the ``_fast`` versions, refuse to extrapolate outside the grid the
        splines were fitted on instead of doing it silently.
    calibration_seeding : {'stream', 'derived'}
        How Monte Carlo calibration hyperballs are seeded.  ``'stream'``
        (default) draws every dimension from the shared ``random_state_`` in
        order.  ``'derived'`` requires an integer ``random_state`` and seeds
        each reference independently from ``(random_state, N, dimension)``.
        The ``_fast`` versions do not sample and ignore it (the integer-seed
        requirement of ``'derived'`` is still checked at construction).

    Notes
    -----
    ``N``, ``k``, ``version`` and ``backend`` are resolved once, in
    ``__init__``: the distance strategy and the neighbour-pair index are built
    there.  Reassigning those attributes afterwards will not take effect --
    construct a new estimator instead.
    """

    def __init__(self, N=2500, k=10, calibration_data=None, fractal=True,
                 verbose=False, random_state=None, version="DANCo",
                 backend="cpu", backend_options=None, use_legacy_bessel=False,
                 check_spline_range=True, calibration_seeding="stream"):
        self.N = N
        self.k = k
        self.verbose = verbose
        self.fractal = fractal
        self.random_state_ = check_random_state(random_state)
        if calibration_seeding not in ("stream", "derived"):
            raise ValueError(
                "calibration_seeding should be 'stream' or 'derived', "
                f"got {calibration_seeding!r}."
            )
        # 'stream': candidate m always consumes the m-th draw of the shared
        # random_state_, so a reference is a deterministic function of
        # (random_state, N).  'derived' seeds each calibration dimension
        # independently from (seed, N, dim), so each dimension's draw is also
        # independent of the others.
        if calibration_seeding == "derived" and (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, (int, np.integer))
        ):
            raise ValueError(
                "calibration_seeding='derived' requires an integer "
                f"random_state, got {type(random_state).__name__}."
            )
        self.calibration_seeding = calibration_seeding
        self._calibration_seed_base = (
            int(random_state) if calibration_seeding == "derived" else 0
        )
        if calibration_data is None:
            self.calibration_statistic = empty_calibration()
        else:
            _validate_calibration(calibration_data)
            self.calibration_statistic = calibration_data
        if version not in VERSIONS:
            raise ValueError(
                f"version should be one of {sorted(VERSIONS)}, got {version!r}."
            )
        self.version = version
        variant_cls, self._use_spline, self._report_strip = VERSIONS[version]
        self.variant = variant_cls(k=self.k, N=self.N)
        self.backend = backend
        self.backend_options = dict(backend_options or {})
        self._backend = get_backend(backend, **self.backend_options)
        # Calibration hyperballs are small and never worth a device round trip.
        self._calibration_backend = (
            self._backend if backend == "cpu" else get_backend("cpu")
        )
        self.use_legacy_bessel = use_legacy_bessel
        self.check_spline_range = check_spline_range
        self.combs = indnComb(self.k)  # shape (2, C(k,2))

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    def danco_statistic(self, X, n_jobs=1, backend=None):
        """The three-number summary DANCo compares against calibration.

        ``mu_nu``/``mu_tau`` are skipped, and reported as ``0``, whenever the
        distance estimate is below 5.  For an observed dataset the shortcut in
        :meth:`evaluate` (``dhat <= 5``) then returns before they are used.
        For a reference hyperball of dimension ``m <= 5`` the zeros are kept
        and enter the angular KL as ``tau = nu = 0`` references, so the search
        still runs over every candidate.  ``angle_valid`` and
        ``angle_status`` distinguish that shortcut from an undefined circular
        mean caused by an exactly zero resultant.

        ``backend`` overrides the estimator's; the calibration loop uses it to
        keep its hyperballs on the CPU.
        """
        if backend is None:
            backend = self._backend
        D = X.shape[1]
        dists, idx = backend.knn(X, self.k, n_jobs=n_jobs)
        dhat = self.variant.estimate_dimension(dists, D)
        if self.verbose:
            print("dhat=", dhat, end=" ")

        if not (dhat < 5):
            thetas = backend.angle_stats(X, idx, self.combs, self.k)
            nu, tau, center_resultant = ml_vm(
                thetas, return_resultant=True
            )
            if np.any(center_resultant == 0):
                mu_nu = np.nan
                mu_tau = np.nan
                angle_valid = False
                angle_status = "zero_resultant"
            else:
                mu_nu, sample_resultant = circular_mean_resultant(nu)
                if sample_resultant == 0:
                    mu_nu = np.nan
                    mu_tau = np.nan
                    angle_valid = False
                    angle_status = "zero_resultant"
                else:
                    mu_tau = np.mean(tau)
                    angle_valid = True
                    angle_status = "ok"
        else:
            mu_nu = 0
            mu_tau = 0
            angle_valid = False
            angle_status = "distance_below_threshold"
        if self.verbose:
            print("nu=", mu_nu, " tau=", mu_tau)
        return dict(
            dhat=dhat,
            mu_nu=mu_nu,
            mu_tau=mu_tau,
            angle_valid=angle_valid,
            angle_status=angle_status,
        )

    def hyperBall(self, n, d, radius=1.0, center=[]):
        """Uniform sample from the ``d``-ball, consuming this estimator's RNG."""
        return hyper_ball(self.random_state_, n, d, radius=radius, center=center)

    # ------------------------------------------------------------------
    # divergences
    # ------------------------------------------------------------------
    def _kl_angle_full(self, data, caldata):
        """Full von Mises KL between observed and reference angle statistics."""
        return kl_vonmises(
            data["mu_nu"], data["mu_tau"],
            caldata["mu_nu"], caldata["mu_tau"],
            use_legacy_bessel=self.use_legacy_bessel,
        )

    def _kl_angle_profiled(self, data, caldata):
        """Mean-direction-profiled von Mises KL (concentrations only)."""
        return kl_vonmises_strip(data["mu_tau"], caldata["mu_tau"])

    _ANGLE_KL = {"full": _kl_angle_full, "profiled": _kl_angle_profiled}

    @property
    def objective(self):
        """``'full'`` or ``'profiled'``: the objective :meth:`evaluate` returns."""
        return "profiled" if self._report_strip else "full"

    # ------------------------------------------------------------------
    # main entry points
    # ------------------------------------------------------------------
    def _check_input(self, X, maxDim):
        X = check_array(X, ensure_min_samples=self.k + 1, ensure_min_features=2, dtype='float')
        if len(X) != self.N:
            raise ValueError(
                f"X must have exactly N={self.N} rows (got {len(X)}); "
                "construct DANCo(N=len(X), ...) for this dataset"
            )
        return X, min(maxDim, X.shape[1])

    def _grow_calibration(self, maxDim, n_jobs, spline_file):
        if self.verbose:
            print("Collecting calibration data.")
        if self._use_spline:
            grow_spline_calibration(
                self.calibration_statistic, maxDim, self.N,
                spline_file=spline_file, check_range=self.check_spline_range,
                variant=self.variant.name, k=self.k,
            )
            return
        while self.calibration_statistic["maxdim"] < maxDim:
            new_dim = self.calibration_statistic["maxdim"] + 1
            if self.verbose:
                print(new_dim, ":", end="")
            if self.calibration_seeding == "derived":
                rng = np.random.default_rng(
                    (self._calibration_seed_base, self.N, new_dim))
                ball = hyper_ball(rng, self.N, new_dim,
                                  center=[0] * new_dim)
            else:
                ball = self.hyperBall(self.N, new_dim,
                                      center=[0] * new_dim)
            calibration_entry = self.danco_statistic(
                ball,
                n_jobs=n_jobs,
                backend=self._calibration_backend,
            )
            # Validity describes the current observed dataset, not the
            # reusable calibration reference.  Keep generated cache entries
            # on the established three-field schema so existing and new caches
            # remain interchangeable.
            calibration_entry.pop("angle_valid", None)
            calibration_entry.pop("angle_status", None)
            self.calibration_statistic["statistics"].append(
                calibration_entry
            )
            self.calibration_statistic["maxdim"] = new_dim

    def _evaluate_core(self, X, maxDim, n_jobs, spline_file, objectives):
        """Shared pipeline behind :meth:`evaluate` and :meth:`evaluate_all`.

        Returns ``(data_statistic, kl_distance, kl_angle, estimates)``.  On the
        two shortcuts (distance estimate at most five, or no mean direction)
        ``kl_distance`` and ``kl_angle`` are ``None`` and every requested
        objective's estimate is the distance estimate ``dhat``; the calibration
        cache is not grown.  Otherwise ``kl_distance`` has shape ``(maxDim,)``,
        ``kl_angle[objective]`` has shape ``(maxDim,)`` for each requested
        objective, and ``estimates[objective]`` is the (refined) argmin of
        their sum.
        """
        X, maxDim = self._check_input(X, maxDim)
        if self.verbose:
            print("Calculating input data statistic.")
        data_statistic = self.danco_statistic(X, n_jobs=n_jobs)

        # Below 5 the calibration adds nothing -- and, importantly, is *not*
        # grown, so a low-dimensional dataset leaves the cache untouched for
        # whatever is evaluated next.
        if data_statistic["dhat"] <= 5:
            data_statistic["angle_valid"] = False
            data_statistic["angle_status"] = "distance_below_threshold"
            dhat = data_statistic["dhat"]
            return data_statistic, None, None, {o: dhat for o in objectives}

        # A zero circular resultant has no mean direction.  Returning the
        # distance estimate here avoids comparing an undefined angular
        # statistic against calibration, and preserves the no-cache-growth
        # behaviour of the low-distance shortcut above.
        if not data_statistic["angle_valid"]:
            dhat = data_statistic["dhat"]
            return data_statistic, None, None, {o: dhat for o in objectives}

        self._grow_calibration(maxDim, n_jobs, spline_file)

        if self.verbose:
            print("Calculating kl-divergence.")
        references = self.calibration_statistic["statistics"]
        kl_distance = np.zeros(maxDim)
        kl_angle = {o: np.zeros(maxDim) for o in objectives}
        for d in range(maxDim):
            kl_distance[d] = self.variant.kl_distance(
                data_statistic["dhat"], references[d]["dhat"])
            for o in objectives:
                kl_angle[o][d] = self._ANGLE_KL[o](self, data_statistic,
                                                   references[d])

        estimates = {}
        for o in objectives:
            curve = kl_distance + kl_angle[o]
            de = np.argmin(curve) + 1
            if self.fractal and maxDim >= 2:
                # The interpolation order is limited by the number of
                # candidates.  Refinement is restricted to [1, maxDim]; a
                # minimum at the boundary is returned there.  No fill_value,
                # so the interpolant returns NaN outside the candidate range.
                kind = "cubic" if maxDim > 3 else ("quadratic" if maxDim == 3 else "linear")
                f = interp1d(np.arange(1, maxDim + 1), curve,
                             kind=kind, bounds_error=False)
                de = minimize(f, de, bounds=[(1, maxDim)], tol=1e-3)["x"][0]
            estimates[o] = de
        data_statistic["calibration_statistic"] = self.calibration_statistic
        return data_statistic, kl_distance, kl_angle, estimates

    def evaluate(self, X, maxDim=100, n_jobs=1, spline_file=None):
        """Estimate the intrinsic dimension of ``X`` with this version's objective.

        Returns ``(dimension, stats)``.  ``stats`` carries the observed
        statistics ``dhat``, ``mu_nu``, ``mu_tau``, ``angle_valid`` and
        ``angle_status`` (``'ok'``, ``'distance_below_threshold'`` or
        ``'zero_resultant'``), the name of the objective in ``objective``
        (``'full'`` or ``'profiled'``), the returned value again as
        ``estimate``, and the two discrepancy curves over the candidates
        ``1..maxDim``: ``kl_distance`` and ``kl_angle`` (this objective's
        angular curve).  On the shortcuts the curves are ``None`` and the
        estimate is ``dhat``.  Only the selected objective is computed; use
        :meth:`evaluate_all` to obtain Full and Profiled together.

        ``stats['calibration_statistic']`` (present when calibration ran) is
        the estimator's **live** cache, not a copy, so that it can be handed to
        another estimator.  Mutating it changes how this one grows next time.
        """
        objective = self.objective
        stats, kl_distance, kl_angle, estimates = self._evaluate_core(
            X, maxDim, n_jobs, spline_file, (objective,))
        stats["objective"] = objective
        stats["estimate"] = estimates[objective]
        stats["kl_distance"] = kl_distance
        stats["kl_angle"] = None if kl_angle is None else kl_angle[objective]
        return stats["estimate"], stats

    def evaluate_all(self, X, maxDim=100, n_jobs=1, spline_file=None):
        """Estimate both objectives of this distance statistic in one pass.

        Returns ``stats`` with the observed statistics of :meth:`evaluate`
        plus ``de_standard`` (Full estimate), ``de_nustrip`` (Profiled
        estimate), ``kl_divergence`` of shape ``(maxDim, 2)`` (distance and
        Full angular curves) and ``kl_ang_strip`` of shape ``(maxDim,)`` (the
        Profiled angular curve).  On the shortcuts the curves are ``None`` and
        both estimates equal ``dhat``.  The experiments use this method so that
        Full and Profiled are compared on one data sample and one reference
        realization; the Profiled curve costs about two percent of one
        evaluation.
        """
        stats, kl_distance, kl_angle, estimates = self._evaluate_core(
            X, maxDim, n_jobs, spline_file, OBJECTIVES)
        stats["de_standard"] = estimates["full"]
        stats["de_nustrip"] = estimates["profiled"]
        if kl_distance is None:
            stats["kl_divergence"] = None
            stats["kl_ang_strip"] = None
        else:
            stats["kl_divergence"] = np.column_stack(
                [kl_distance, kl_angle["full"]])
            stats["kl_ang_strip"] = kl_angle["profiled"]
        return stats
