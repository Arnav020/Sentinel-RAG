"""
Uncertainty reporting.

Every proportion in the old README was a bare point estimate. "Guardrail
precision 1.000" over three positives has a 95% Wilson interval of roughly
[0.44, 1.00] - a number that looks decisive and carries almost no information.

Wilson intervals are used for proportions (they behave correctly at 0 and 1,
where the normal approximation does not), and a bootstrap for the mean of a
continuous score such as a RAGAS metric.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

Z95 = 1.959963984540054


@dataclass(slots=True)
class Interval:
    point: float
    low: float
    high: float
    n: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] n={self.n}"

    def as_dict(self) -> dict:
        return {
            "value": round(self.point, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
            "n": self.n,
        }


def wilson(successes: int, n: int, z: float = Z95) -> Interval:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return Interval(0.0, 0.0, 1.0, 0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin), n)


def bootstrap_mean(values: list[float], iterations: int = 5000, seed: int = 12345) -> Interval:
    """Percentile bootstrap CI for a mean. Seeded so a run is reproducible."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return Interval(0.0, 0.0, 0.0, 0)
    if len(clean) == 1:
        return Interval(clean[0], clean[0], clean[0], 1)

    rng = random.Random(seed)
    n = len(clean)
    means = []
    for _ in range(iterations):
        means.append(sum(rng.choice(clean) for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * iterations)]
    hi = means[int(0.975 * iterations) - 1]
    return Interval(sum(clean) / n, lo, hi, n)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank; None means the target was never retrieved."""
    if not ranks:
        return 0.0
    return sum(0.0 if r is None else 1.0 / r for r in ranks) / len(ranks)


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """
    Binary-relevance nDCG@k for one query.

    `relevances` is the retrieved list in rank order, 1 where the item is
    relevant. Ideal ordering places every relevant item first.
    """
    rel = relevances[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def summarise_binary(name: str, successes: int, n: int) -> dict:
    return {"metric": name, **wilson(successes, n).as_dict()}


def summarise_scores(name: str, values: list[float]) -> dict:
    return {"metric": name, **bootstrap_mean(values).as_dict()}
