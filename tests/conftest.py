"""Make the package and the experiment scripts importable from the tests.

Tests import ``danco`` from ``src/`` and the experiment modules either as
``experiments.<name>`` or directly by module name (``from common import ...``),
so both the repository root and ``experiments/`` are put on ``sys.path``.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT / "src", _ROOT, _ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
