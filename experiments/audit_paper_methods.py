#!/usr/bin/env python
"""Write the machine-readable manuscript estimator-contract audit."""

import argparse
import csv
from pathlib import Path

from paper_methods import method_contract_rows


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "results" / "diagnostics" / \
    "paper_method_contract_audit.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    rows = method_contract_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.output)
    print(f"wrote {args.output} ({len(rows)} artifact-method contracts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
