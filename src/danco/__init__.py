"""DANCo intrinsic-dimension estimators.

>>> from danco import DANCo

``two_nn_legacy`` is the untrimmed full-sample TWO-NN estimator of the
earlier implementation (derived from scikit-dimension's ``skdim.id.DANCo``
code base); the paper's TWO-NN values use the reference (discarded-tail) fit
in ``experiments/paper_methods.py``.
"""

from .estimator import DANCo
from .twonn import two_nn_legacy

__version__ = "1.0.0"

__all__ = ["DANCo", "two_nn_legacy", "__version__"]
