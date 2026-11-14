"""Independent verification of ``lagstats.compare`` against direct pairing.

This script deliberately does *not* reuse any of the transform machinery: the
reference implementation below is a literal transcription of the task
definition, pairing every ``(left[i], right[i + k])`` for every lag.  It is
quadratic on purpose and is only applied to sizes where that is cheap, plus a
handful of sampled lags at full scale.

Run with::

    python3 verify.py            # full verification suite
    python3 verify.py --quick    # skip the maximum-size sampling check
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import lagstats


def reference(left, right):
    """Literal O(n*m) reference: build every lag's pair list directly."""
    n, m = len(left), len(right)
    rows = []
    for k in range(-(n - 1), m):
        xs, ys = [], []
        for i in range(n):
            j = i + k
            if 0 <= j < m and left[i] is not None and right[j] is not None:
                xs.append(left[i])
                ys.append(right[j])
        p = len(xs)
        if p == 0:
            rows.append([k, 0, 0, 0, 0])
            continue
        sx, sy = sum(xs), sum(ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sxx = sum(x * x for x in xs)
        syy = sum(y * y for y in ys)
        rows.append([k, p, p * sxy - sx * sy, p * sxx - sx * sx, p * syy - sy * sy])
    return rows


def reference_lag(left, right, k):
    """Reference row for a single lag (used for full-scale spot checks)."""
    n, m = len(left), len(right)
    xs, ys = [], []
    for i in range(n):
        j = i + k
        if 0 <= j < m and left[i] is not None and right[j] is not None:
            xs.append(left[i])
            ys.append(right[j])
    p = len(xs)
    if p == 0:
        return [k, 0, 0, 0, 0]
    sx, sy = sum(xs), sum(ys)
    return [
        k,
        p,
        p * sum(x * y for x, y in zip(xs, ys)) - sx * sy,
        p * sum(x * x for x in xs) - sx * sx,
        p * sum(y * y for y in ys) - sy * sy,
    ]


def random_series(rng, n, missing, lo=-1000, hi=1000):
    return [
        None if rng.random() < missing else rng.randint(lo, hi) for _ in range(n)
    ]


def check_exact_types(rows):
    for row in rows:
        assert len(row) == 5
        for field in row:
            assert type(field) is int, f"non-int field {field!r} ({type(field)})"


def check_shape(left, right, rows):
    n, m = len(left), len(right)
    assert [r[0] for r in rows] == list(range(-(n - 1), m)), "lag order/coverage"
    for k, p, c, vx, vy in rows:
        assert 0 <= p <= min(n, m)
        if p == 0:
            assert c == 0 and vx == 0 and vy == 0, "p==0 row must be zeroed"
            continue
        assert vx >= 0 and vy >= 0, "variance numerators must be non-negative"
        if p == 1:
            assert c == 0 and vx == 0 and vy == 0, "single pair has no spread"
            continue
        assert c * c <= vx * vy, "Cauchy-Schwarz violated (corr outside [-1,1])"
        if vx == 0 or vy == 0:
            assert c == 0, "constant series must have zero covariance"


def case(name, left, right):
    left_copy, right_copy = list(left), list(right)
    rows = lagstats.compare(left, right)
    assert left == left_copy and right == right_copy, "input was modified"
    expected = reference(left, right)
    assert rows == expected, (
        f"{name}: mismatch\n got={rows}\nwant={expected}"
    )
    check_exact_types(rows)
    check_shape(left, right, rows)
    print(f"  ok  {name}  (len {len(left)}x{len(right)}, {len(rows)} lags)")


def run_quick_suite():
    rng = random.Random(20261114)

    print("hand-built cases")
    case("example", [1, 3], [2, 4])
    assert lagstats.compare([1, 3], [2, 4])[1] == [0, 2, 4, 4, 4]
    case("singletons", [5], [7])
    case("all missing left", [None] * 7, [3, 4, 5])
    case("all missing both", [None] * 4, [None] * 6)
    case("zero is observed", [0, None, 0], [None, 0, 0])
    case("extremes", [1000, -1000] * 12, [-1000, 1000] * 9)
    case("constant", [1000] * 25, [-1000] * 31)
    case("no overlap", [None] * 10, [None] * 10)

    print("randomised exact comparisons against direct pairing")
    for trial in range(120):
        n = rng.randint(1, 45)
        m = rng.randint(1, 45)
        missing = rng.choice([0.0, 0.15, 0.5, 0.85, 0.98])
        lo, hi = rng.choice([(-1000, 1000), (-3, 3), (0, 0), (990, 1000)])
        left = random_series(rng, n, missing, lo, hi)
        right = random_series(rng, m, missing, lo, hi)
        rows = lagstats.compare(left, right)
        expected = reference(left, right)
        assert rows == expected, f"random trial {trial} mismatch"
        check_exact_types(rows)
        check_shape(left, right, rows)
    print("  ok  120 randomised cases")

    print("lag symmetry: compare(a,b)[k] mirrors compare(b,a)[-k]")
    for trial in range(40):
        n, m = rng.randint(1, 50), rng.randint(1, 50)
        left = random_series(rng, n, 0.4)
        right = random_series(rng, m, 0.4)
        forward = lagstats.compare(left, right)
        backward = lagstats.compare(right, left)
        for k, p, c, vx, vy in forward:
            mirror = backward[-k + (m - 1)]
            assert mirror[1:] == [p, c, vy, vx], "symmetry broken"
    print("  ok  40 symmetry cases")

    print("validation errors")
    for bad, exc in (([], ValueError), ([1] * 20001, ValueError),
                     ([1001], ValueError), ([-1001], ValueError)):
        try:
            lagstats.compare(bad, [1])
        except exc:
            pass
        else:
            raise AssertionError(f"expected {exc.__name__} for {bad!r}")
    for bad in ([1.5], ["x"], [None, object()]):
        try:
            lagstats.compare(bad, [1])
        except TypeError:
            pass
        else:
            raise AssertionError(f"expected TypeError for {bad!r}")
    print("  ok  invalid inputs rejected")


def run_scale_check():
    rng = random.Random(7)
    n = m = 20000
    left = random_series(rng, n, 0.25)
    right = random_series(rng, m, 0.25)
    start = time.perf_counter()
    rows = lagstats.compare(left, right)
    elapsed = time.perf_counter() - start
    check_exact_types(rows)
    check_shape(left, right, rows)
    for k in (-(n - 1), -(n - 1) + 1, -12345, -1, 0, 1, 7777, m - 2, m - 1):
        got = rows[k + (n - 1)]
        want = reference_lag(left, right, k)
        assert got == want, f"scale spot check failed at k={k}: {got} != {want}"
    print(f"  ok  20000x20000: {len(rows)} lags in {elapsed:.2f}s; "
          f"9 sampled lags match direct pairing exactly")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the 20000x20000 sampled-lag check")
    args = parser.parse_args(argv)

    print("verifying lagstats.compare")
    run_quick_suite()
    if not args.quick:
        print("full-scale spot checks")
        run_scale_check()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
