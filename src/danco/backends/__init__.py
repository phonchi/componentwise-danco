"""Compute backends.  Each exposes ``knn(X, k, n_jobs)`` and ``angle_stats``.

A backend is whatever object carries those two names, so a stateless one can
simply *be* its module (``cpu``) while a configurable one is an instance
(``gpu``).  The distinction is hidden behind :func:`get_backend`: a module that
defines ``make_backend`` is treated as a factory, everything else is used as-is.
"""

__all__ = ["get_backend"]

_MODULES = {}


def _load(name):
    if name in _MODULES:
        return _MODULES[name]
    if name == "cpu":
        from . import cpu as module
    elif name == "gpu":
        from . import gpu as module
    else:
        raise ValueError(
            f"unknown backend {name!r}; expected 'cpu' or 'gpu'"
        )
    _MODULES[name] = module
    return module


def get_backend(name, **options):
    """Return the backend registered under ``name`` (``'cpu'``/``'gpu'``).

    ``options`` are forwarded to the backend's ``make_backend`` factory; a
    backend without one takes no options and passing any is an error rather
    than a silent no-op.
    """
    module = _load(name)
    factory = getattr(module, "make_backend", None)
    if factory is None:
        if options:
            raise TypeError(
                f"backend {name!r} takes no options, got "
                f"{sorted(options)}"
            )
        return module
    return factory(**options)
