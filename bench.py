"""Scale benchmark for ``lagstats.compare``, including real peak memory.

Run with::

    python3 bench.py                 # sizes up to the 20000 x 20000 limit
    python3 bench.py --json          # machine-readable results on stdout

Each measurement runs in a fresh child process so that peak RSS is specific
to the measured call rather than to the whole interpreter session.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time

SIZES = [(1000, 1000), (5000, 5000), (10000, 10000), (20000, 20000),
         (20000, 1000), (1000, 20000)]


def _make_series(n, rng, missing=0.2):
    return [None if rng.random() < missing else rng.randint(-1000, 1000)
            for _ in range(n)]


def _worker(n, m, seed, result_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lagstats

    rng = random.Random(seed)
    left = _make_series(n, rng)
    right = _make_series(m, rng)

    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    rows = lagstats.compare(left, right)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    stats = {
        "n": n,
        "m": m,
        "lags": len(rows),
        "seconds": round(elapsed, 4),
        "peak_rss_kib": peak,
        "delta_rss_kib": peak - base,
        "middle_row": rows[len(rows) // 2],
        "all_int": all(type(field) is int for row in rows for field in row),
    }
    with open(result_path, "w") as handle:
        json.dump(stats, handle)


def measure(n, m, seed=0):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "result.json")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--worker", str(n), str(m), str(seed), path],
            check=True,
        )
        with open(path) as handle:
            return json.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", nargs=4, metavar=("N", "M", "SEED", "PATH"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        _worker(int(args.worker[0]), int(args.worker[1]),
                int(args.worker[2]), args.worker[3])
        return 0

    results = [measure(n, m) for n, m in SIZES]
    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"{'n x m':<14}{'lags':>8}{'seconds':>10}{'us/lag':>9}"
          f"{'peak MB':>10}{'delta MB':>10}  mid-row")
    for r in results:
        print(f"{r['n']}x{r['m']:<12}{r['lags']:>8}{r['seconds']:>10.3f}"
              f"{r['seconds'] / r['lags'] * 1e6:>9.1f}"
              f"{r['peak_rss_kib'] / 1024:>10.1f}"
              f"{r['delta_rss_kib'] / 1024:>10.1f}  {r['middle_row']}")
    print("\nAll rows are exact ints:", all(r["all_int"] for r in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
