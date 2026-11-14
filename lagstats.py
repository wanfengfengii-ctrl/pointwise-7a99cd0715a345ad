"""Exact lagged covariance/correlation statistics for two sampled series.

Public entry point
------------------
``compare(left, right)`` returns, for every integer lag ``k`` from
``-(len(left) - 1)`` through ``len(right) - 1``, one row
``[k, p, c, vx, vy]`` with the exact integer sufficient statistics of the
pairs ``(left[i], right[i + k])`` whose two entries are both observed.

Only the Python standard library is used.  All six required
cross-correlations are evaluated in one batch with a Number Theoretic
Transform (NTT) under a single 36-bit NTT-friendly prime, then each signed
coefficient is recovered exactly by centring the residue modulo that prime.
The whole batch costs O((n + m) log(n + m)) modular arithmetic steps and
O(n + m) auxiliary words.  No lag is processed by scanning its overlapping
samples, and no floating point value is ever produced.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import List, Sequence

__all__ = ["compare"]

# A prime with a large power of two in p - 1, and 11 as a primitive root:
#   p = 45000818689 = 343329 * 2**17 + 1 = 2**17 * 3 * 7 * 16349 + 1
# Transform lengths up to 2**17 are supported, i.e. n + m - 1 <= 131072.
_MOD = 45000818689
_ROOT = 11
_HALF_MOD = _MOD // 2

_MAX_LEN = 20000
_MAX_ABS_VALUE = 1000

# With n, m <= _MAX_LEN and |value| <= _MAX_ABS_VALUE, every convolution
# coefficient handled here is bounded by
#   p, sum(x), sum(y)                <= 2.0e7      (<= _MAX_LEN * _MAX_ABS_VALUE)
#   sum(x*x), sum(y*y), |sum(x*y)|   <= 2.0e10     (<= _MAX_LEN * _MAX_ABS_VALUE**2)
# Both are far below _HALF_MOD (~2.25e10), so a single residue determines the
# signed coefficient uniquely.  The two output quantities
#   c = p*sum(x*y) - sum(x)*sum(y), etc.
# are formed afterwards in ordinary (unbounded) Python integers.
_SINGLE_BOUND = _MAX_LEN * _MAX_ABS_VALUE
_COEFF_BOUND = _MAX_LEN * _MAX_ABS_VALUE * _MAX_ABS_VALUE

assert _COEFF_BOUND < _HALF_MOD


def _validate(name: str, seq: Sequence) -> None:
    try:
        length = len(seq)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sized sequence") from exc
    if not 1 <= length <= _MAX_LEN:
        raise ValueError(f"{name} length must be in 1..{_MAX_LEN}, got {length}")
    for i, value in enumerate(seq):
        if value is None:
            continue
        try:
            value = operator.index(value)      # int and integer subclasses
        except TypeError:
            raise TypeError(
                f"{name}[{i}] must be None or int, got {type(value).__name__}"
            ) from None
        if not -_MAX_ABS_VALUE <= value <= _MAX_ABS_VALUE:
            raise ValueError(
                f"{name}[{i}] has |value| = {abs(value)} > {_MAX_ABS_VALUE}"
            )


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


@lru_cache(maxsize=2)
def _plan(size: int, invert: bool):
    """Per-stage twiddle tables (roots[j] == w**j) for one transform size.

    Cached (bounded to one size's forward/inverse pair) so repeated calls at
    the same size do not rebuild them; retained auxiliary memory is O(size),
    i.e. O(n + m) for the current call.
    """
    stages = []
    length = 2
    while length <= size:
        half = length >> 1
        wlen = pow(_ROOT, (_MOD - 1) // length, _MOD)
        if invert:
            wlen = pow(wlen, _MOD - 2, _MOD)
        roots = [1] * half
        for i in range(1, half):
            roots[i] = roots[i - 1] * wlen % _MOD
        stages.append((length, half, roots))
        length <<= 1
    return stages


def _ntt_forward(a: List[int], stages) -> None:
    """Decimation-in-frequency NTT.

    Natural-order input, bit-reversed output.  Elements stay in [0, _MOD).
    """
    n = len(a)
    mod = _MOD
    for length, half, roots in reversed(stages):
        for start in range(0, n, length):
            upper = start + half
            lo = a[start:upper]
            for j in range(half):
                u = lo[j]
                v = a[upper + j]
                x = u + v
                if x >= mod:
                    x -= mod
                lo[j] = x
                a[upper + j] = (u - v) * roots[j] % mod
            a[start:upper] = lo


def _ntt_inverse(a: List[int], stages) -> None:
    """Decimation-in-time inverse NTT: bit-reversed input, natural output."""
    n = len(a)
    mod = _MOD
    for length, half, roots in stages:
        for start in range(0, n, length):
            upper = start + half
            lo = a[start:upper]
            for j in range(half):
                u = lo[j]
                v = a[upper + j] * roots[j] % mod
                x = u + v
                if x >= mod:
                    x -= mod
                y = u - v
                if y < 0:
                    y += mod
                lo[j] = x
                a[upper + j] = y
            a[start:upper] = lo
    inv_n = pow(n, mod - 2, mod)
    for i in range(n):
        a[i] = a[i] * inv_n % mod


def _prepare(left: Sequence, right: Sequence, size: int):
    """Build the six NTT operands.

    The left series is stored reversed, so ordinary convolution puts the
    pairs with ``j - i == k`` at index ``(len(left) - 1) + k``.  A missing
    entry contributes 0 to the value arrays and 0 to the presence masks, so
    it is excluded from every sum and from the pair count -- it is never
    treated as a zero observation.
    """
    n = len(left)
    li = [0] * size       # reversed left presence mask
    lv = [0] * size       # reversed left values
    ls = [0] * size       # reversed left squared values
    for i, value in enumerate(left):
        if value is not None:
            value = operator.index(value)
            pos = n - 1 - i
            li[pos] = 1
            lv[pos] = value % _MOD
            ls[pos] = value * value % _MOD

    ri = [0] * size       # right presence mask
    rv = [0] * size       # right values
    rs = [0] * size       # right squared values
    for j, value in enumerate(right):
        if value is not None:
            value = operator.index(value)
            ri[j] = 1
            rv[j] = value % _MOD
            rs[j] = value * value % _MOD
    return [li, lv, ls, ri, rv, rs]


def _convolve(left, right, size: int, forward, inverse):
    """Return the six convolution coefficient arrays modulo _MOD.

    Result order: ``[p, sum(x), sum(y), sum(x*x), sum(y*y), sum(x*y)]``.
    """
    operands = _prepare(left, right, size)
    for arr in operands:
        _ntt_forward(arr, forward)
    li, lv, ls, ri, rv, rs = operands
    mod = _MOD
    out = [
        [x * y % mod for x, y in zip(li, ri)],   # p
        [x * y % mod for x, y in zip(lv, ri)],   # sum(x)
        [x * y % mod for x, y in zip(li, rv)],   # sum(y)
        [x * y % mod for x, y in zip(ls, ri)],   # sum(x*x)
        [x * y % mod for x, y in zip(li, rs)],   # sum(y*y)
        [x * y % mod for x, y in zip(lv, rv)],   # sum(x*y)
    ]
    for arr in out:
        _ntt_inverse(arr, inverse)
    return out


def _signed(residue: int) -> int:
    """Recover the signed coefficient from its residue modulo _MOD."""
    return residue - _MOD if residue > _HALF_MOD else residue


def compare(left: Sequence, right: Sequence) -> List[List[int]]:
    """Exact lag statistics for two series with missing observations.

    Parameters
    ----------
    left, right:
        Sized sequences of length 1..20000 whose elements are ``None`` or
        integers with absolute value at most 1000.  Inputs are only read,
        never modified.

    Returns
    -------
    list of ``[k, p, c, vx, vy]``, one row per integer lag in increasing
    order from ``-(len(left) - 1)`` to ``len(right) - 1``.
        ``p`` counts the pairs ``(left[i], right[i + k])`` in which both
        values are non-``None``.  With the sums taken over exactly that pair
        set::

            c  = p*sum(x*y) - sum(x)*sum(y)
            vx = p*sum(x*x) - sum(x)**2
            vy = p*sum(y*y) - sum(y)**2

        When ``p == 0`` the fields ``c, vx, vy`` are all 0.  Every field is
        an exact Python ``int`` (never a float or ``None``).

    Notes
    -----
    Missing observations are excluded pair by pair through presence masks;
    they are never replaced by zero and no mean is imputed or shared between
    lags.  Each lag carries its own ``p`` and its own sums, so the rows are
    mutually independent.  ``c`` and ``vx``/``vy`` are the numerators of the
    lag-wise Pearson covariance/correlation: sign and relative magnitude can
    be read from ``c`` against ``sqrt(vx*vy)`` computed exactly elsewhere.
    """
    _validate("left", left)
    _validate("right", right)

    n = len(left)
    m = len(right)
    size = _next_power_of_two(n + m - 1)
    forward = _plan(size, False)
    inverse = _plan(size, True)

    arrs = _convolve(left, right, size, forward, inverse)
    p_arr, sx_arr, sy_arr, sxx_arr, syy_arr, sxy_arr = arrs

    offset = n - 1
    result: List[List[int]] = []
    append = result.append
    for k in range(-(n - 1), m):
        index = offset + k
        p = p_arr[index]
        if p == 0:
            append([k, 0, 0, 0, 0])
            continue

        sx = _signed(sx_arr[index])
        sy = _signed(sy_arr[index])
        sxx = _signed(sxx_arr[index])
        syy = _signed(syy_arr[index])
        sxy = _signed(sxy_arr[index])

        # Defensive exactness gate: the input limits guarantee these bounds,
        # so a violation can only mean an internal arithmetic error.
        if not (
            abs(sx) <= _SINGLE_BOUND
            and abs(sy) <= _SINGLE_BOUND
            and abs(sxx) <= _COEFF_BOUND
            and abs(syy) <= _COEFF_BOUND
            and abs(sxy) <= _COEFF_BOUND
        ):
            raise ArithmeticError("coefficient recovery out of bounds")

        append([
            k,
            p,
            p * sxy - sx * sy,
            p * sxx - sx * sx,
            p * syy - sy * sy,
        ])
    return result
