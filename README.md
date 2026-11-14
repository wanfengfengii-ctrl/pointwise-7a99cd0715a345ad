# lagstats — exact lagged statistics with missing data

`lagstats.compare(left, right)` computes, for **every** integer lag
`k = -(len(left)-1) … len(right)-1`, the exact integer sufficient statistics of
all pairs `(left[i], right[i+k])` in which both observed values exist.

Each returned row is `[k, p, c, vx, vy]`:

| field | definition |
|-------|------------|
| `p`   | number of both-present pairs at this lag |
| `c`   | `p*Σxy − Σx·Σy` |
| `vx`  | `p*Σx² − (Σx)²` |
| `vy`  | `p*Σy² − (Σy)²` |
| zero case | `p == 0` ⇒ `c = vx = vy = 0` |

All sums run over the valid pairs of **that lag only**. Every field is a Python
`int`; no float is ever produced or consumed. `c` and `vx`/`vy` are the
numerators of the lag-wise Pearson covariance / correlation
(`r = c / sqrt(vx·vy)`), so sign and strength can be read off directly.

```python
>>> import lagstats
>>> lagstats.compare([1, 3], [2, 4])
[[-1, 1, 0, 0, 0], [0, 2, 4, 4, 4], [1, 1, 0, 0, 0]]
```

## Analysis core

The batch is evaluated as one set of six polynomial convolutions:

1. Store the **left** series reversed and the **right** series forwards.
2. Build three arrays per side: presence mask (`1` where observed), value, and
   squared value (with `0` wherever the entry is `None`).
3. Ordinary convolution of `reversed(left)` with `right` places lag `k` at
   index `(len(left)-1) + k`. Six products are needed:
   `mask·mask → p`, `value·mask → Σx`, `mask·value → Σy`,
   `square·mask → Σx²`, `mask·square → Σy²`, `value·value → Σxy`.
4. The six operand transforms and six inverse transforms of one length `N`
   (the next power of two ≥ `n+m−1`, at most 65536) are done in a single
   Number Theoretic Transform batch. Work is `O(N log N) = O((n+m) log(n+m))`; auxiliary arrays
   are `N` words each, i.e. `O(n+m)`. No lag is visited by scanning its
   overlapping samples — if it were, the total would be `O(n·m)`.

The transform modulus is the 36-bit prime

```
p = 45000818689 = 343329·2^17 + 1 = 2^17 · 3 · 7 · 16349 + 1,   root = 11
```

chosen so that a **single** residue determines each signed convolution
coefficient:

| coefficient | bound at the input limits | fits in `± p/2 ≈ ±2.25·10^10` |
|---|---|---|
| `p`, `Σx`, `Σy` | `n·1000 = 2.0·10^7` | yes |
| `Σx²`, `Σy²`, `Σxy` | `n·1000² = 2.0·10^10` | yes |

Recovery is `value = residue` if `residue ≤ p/2` else `residue − p`. The
final `c, vx, vy` are then formed in unbounded Python integers, so they are
exact even though they can be much larger than the modulus (e.g.
`c ≈ 10^12`). A defensive bounds check raises `ArithmeticError` if an
internally corrupted residue would ever fall outside the proven bounds; it
cannot trigger for valid inputs.

### Missing data handling

`None` is **not** zero and is **not** mean-imputed:

* The presence masks are zero there, so those pairs are excluded from `p` and
  from every sum.
* A valid observed `0` is retained with mask `1`; it contributes `0` to the
  value sums (as it should) but still increments `p`.
* Each lag has its own `p`, `Σx`, `Σy`, `Σx²`, `Σy²`, `Σxy`. No mean or
  normalization is shared between lags, so `c` is invariant to shifting either
  series by a constant and lags cannot leak into one another.

### Numerical recovery

There is no floating point in the pipeline, hence no tolerance, no rounding
mode, and no platform-dependent `math.fsum` behaviour. The only reconstruction
step is the bounded residue-to-signed-integer map above. `c² ≤ vx·vy`
(Cauchy–Schwarz) holds exactly for `p ≥ 2`; `verify.py` checks this invariant,
as well as the mirror symmetry `compare(a,b)[k] = [p, c, vy, vx]` at `-k` in
`compare(b,a)`.

### 缺测与数值恢复（中文摘要）

* **缺测**：`None` 只让该位置的存在掩码为 0，对应的 `(i, j)` 配对不进入 `p`，
  也不进入任何和式；观测到的 `0` 正常参与并计入 `p`。两列长度不同、缺测率任意
  都不影响其余配对，也不会用 0 填充或用某一套均值插补。每个时差各自拥有独立的
  `p, Σx, Σy, Σx², Σy², Σxy`，所以 `c` 对整体平移不变，时差之间互不串扰。
* **数值恢复**：全过程只用整数。卷积在 36 位素数
  `45000818689 = 343329·2^17 + 1`（原根 11）下用 NTT 完成；在给定输入上限内
  每个带符号卷积系数绝对值不超过 `2.0·10^10`，而 `p/2 ≈ 2.25·10^10`，因此单个
  余数即可唯一恢复该整数（取模后居中到 `(-p/2, p/2]`）。`c, vx, vy` 随后用
  Python 任意精度整数直接相乘相减，故 `c` 即使达到 `10^14` 量级仍精确。
  代码保留了越界自检，输入合法时不可能触发。浮点从未参与，所以没有容差、
  舍入模式或平台差异。

## Files

| file | purpose |
|---|---|
| `lagstats.py` | the deliverable: `compare(left, right)`, stdlib only |
| `verify.py` | independent direct-pairing reference + full test suite |
| `demo.py` | readable examples of missing-data, per-lag means, direction, scale |
| `bench.py` | scale timing and true peak RSS, each size in a fresh process |
| `.pointwise-runtime.json` | machine-readable runtime manifest |

## Verification

`python3 verify.py` implements a literal O(n·m) transcription of the task
definition (for each lag, build the list of valid pairs and sum) and compares
it row for row with `lagstats.compare`. It covers:

* the brief's example (`[1,3]` vs `[2,4]` → `[0,2,4,4,4]` at `k=0`);
* singleton, all-missing, observed-zero, extreme-magnitude and constant series;
* 120 randomised cases with 0–98 % missingness and narrow/wide value ranges;
* 40 lag-symmetry cases (swap the two series, mirror the lag);
* exact output types, lag order/coverage, `p==0` zeroing, `vx,vy ≥ 0`,
  single-pair degeneracy, and `c² ≤ vx·vy`;
* input immutability and rejection of out-of-range / non-integer inputs;
* 9 sampled lags at 20000×20000 against the direct pairing.

```
$ python3 verify.py
…
  ok  20000x20000: 39999 lags in ~2.0s; 9 sampled lags match direct pairing exactly
ALL CHECKS PASSED
```

## Measured scale

`python3 bench.py` (Python 3.11, one fresh process per row, 20 % missing):

```
n x m             lags   seconds   us/lag  peak MB  delta MB
1000x1000         1999     0.042     20.8     12.2       1.4
5000x5000         9999     0.444     44.4     21.1      10.0
10000x10000      19999     0.958     47.9     31.3      19.9
20000x20000      39999     2.077     51.9     51.6      39.6
20000x1000       20999     0.872     41.5     31.3      19.9
1000x20000       20999     0.823     39.2     31.3      19.9
```

Per-lag cost rises by only ~2.5× between 1 000 and 20 000 points (transform
log factor), whereas a per-lag overlap scan would rise ~20×. Peak RSS at the
maximum size is ~52 MB, consistent with `O(n+m)` arrays of size 65536.

## Runtime manifest

`.pointwise-runtime.json` records the interpreter requirements, entry points,
commands, input limits, complexity and observed timings, and the expected
verify/demo exit status. It is plain JSON and contains no secrets.
