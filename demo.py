"""Demonstration of ``lagstats.compare`` on small, readable examples.

Run with::

    python3 demo.py

Everything printed is derived from exact integer outputs; the decimal
correlations are only a convenience rendering, computed on the fly with
``decimal`` and never fed back into any statistic.
"""

from __future__ import annotations

import random
import time
from decimal import Decimal, getcontext
from fractions import Fraction

import lagstats

getcontext().prec = 28


def corr(row):
    """Exact r = c / sqrt(vx*vy) where it is defined, else None.

    Returned as (numerator, denominator, sqrt_denominator) so callers can
    print an exact fraction and a decimal rendering.
    """
    _, p, c, vx, vy = row
    if p == 0 or vx == 0 or vy == 0:
        return None
    # c / sqrt(vx*vy): keep exact integers, render decimal separately.
    value = Decimal(c) / (Decimal(vx) * Decimal(vy)).sqrt()
    return Fraction(c, 1), value


def show(title, left, right):
    print(title)
    print(f"  left  = {left}")
    print(f"  right = {right}")
    rows = lagstats.compare(left, right)
    for k, p, c, vx, vy in rows:
        r = corr([k, p, c, vx, vy])
        r_text = "n/a" if r is None else f"{r[1]:+.6f}"
        print(f"  k={k:>3}  p={p:<2}  c={c:>6}  vx={vx:>7}  vy={vy:>7}  r={r_text}")
    print()


def section_missing_data():
    print("=" * 72)
    print("1. Missing observations are skipped, never treated as zero")
    print("=" * 72)
    show("example from the brief: [1,3] vs [2,4]",
         [1, 3], [2, 4])

    left = [1, None, 3, 4]
    right = [2, 3, None, 5]
    show("missing entries reduce p and are excluded from every sum",
         left, right)

    rows = lagstats.compare(left, right)
    k0 = rows[3]  # k = 0
    imputed = lagstats.compare([1, 0, 3, 4], [2, 3, 0, 5])[3]
    print("  k=0 with pairwise deletion :", k0)
    print("  k=0 if None were zeros     :", imputed)
    print("  -> different p, different sums; deletion is the requested behavior.")
    print()


def section_per_lag_means():
    print("=" * 72)
    print("2. Each lag gets its own mean; nothing is shared between lags")
    print("=" * 72)
    left = [10, 0, -10, 0, 10]
    right = [-5, 5, 0]
    show("c is shift-invariant, so a per-lag mean cannot leak between rows",
         left, right)
    base = lagstats.compare(left, right)
    shifted = lagstats.compare([v + 30 if v is not None else None for v in left],
                               [v - 30 if v is not None else None for v in right])
    same = all(a[1:] == b[1:] for a, b in zip(base, shifted))
    print(f"  all statistics unchanged after shifting each series by a constant: {same}")
    print()


def section_direction():
    print("=" * 72)
    print("3. Direction and strength follow from c against vx, vy")
    print("=" * 72)
    left = [1, 2, 3, 4, 5, 6]
    for label, right in (
        ("positively related", [2, 4, 6, 8, 10, 12]),
        ("negatively related", [12, 10, 8, 6, 4, 2]),
        ("unrelated", [5, 1, 6, 2, 5, 3]),
    ):
        row = lagstats.compare(left, right)[5]  # k = 0
        _, p, c, vx, vy = row
        r = corr(row)
        print(f"  {label:<20} k=0 row={row}  r={r[1]:+.6f}")
    print()


def section_scale():
    print("=" * 72)
    print("4. Scale behaviour (exact integers throughout)")
    print("=" * 72)
    rng = random.Random(2026)
    print(f"  {'n x m':<14}{'lags':>8}{'seconds':>10}{'us/lag':>10}")

    for n, m in ((100, 100), (1000, 1000), (5000, 5000), (20000, 20000)):
        left = [None if rng.random() < 0.2 else rng.randint(-1000, 1000)
                for _ in range(n)]
        right = [None if rng.random() < 0.2 else rng.randint(-1000, 1000)
                 for _ in range(m)]
        start = time.perf_counter()
        rows = lagstats.compare(left, right)
        elapsed = time.perf_counter() - start
        print(f"  {n}x{m:<12}{len(rows):>8}{elapsed:>10.3f}"
              f"{elapsed / len(rows) * 1e6:>10.1f}")
    print("  (lags = n + m - 1; time grows like (n+m) log(n+m), not n*m)")
    print("  (peak RSS at 20000x20000 is reported by the benchmark script)")
    print()


def main():
    print("lagstats.compare -- exact lagged statistics demo\n")
    section_missing_data()
    section_per_lag_means()
    section_direction()
    section_scale()
    print("demo complete")


if __name__ == "__main__":
    main()
