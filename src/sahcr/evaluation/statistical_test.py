"""Statistical significance testing for comparative retrieval evaluation."""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats
from pydantic import BaseModel


class SignificanceResult(BaseModel):
    """Result of statistical significance testing between two retrieval systems."""
    system_a: str
    system_b: str
    metric: str
    mean_a: float
    mean_b: float
    delta_percentage: float
    wilcoxon_stat: float
    wilcoxon_p_value: float
    ttest_stat: float
    ttest_p_value: float
    is_significant_05: bool
    is_significant_01: bool


def compare_paired_systems(
    system_a_name: str,
    system_b_name: str,
    metric_name: str,
    scores_a: List[float],
    scores_b: List[float]
) -> SignificanceResult:
    """
    Executes paired Wilcoxon signed-rank test and paired t-test between two paired query result lists.
    Compares system B (proposed) against system A (baseline).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have equal length")

    a = np.array(scores_a)
    b = np.array(scores_b)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    delta_pct = ((mean_b - mean_a) / max(1e-9, mean_a)) * 100.0

    # Paired Student's t-test
    if np.allclose(a, b):
        t_stat, t_pval = 0.0, 1.0
        w_stat, w_pval = 0.0, 1.0
    else:
        t_res = stats.ttest_rel(b, a)
        t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)

        # Wilcoxon Signed-Rank Test
        try:
            diff = b - a
            # Exclude zero differences for standard wilcoxon
            nonzero_diff = diff[diff != 0]
            if len(nonzero_diff) == 0:
                w_stat, w_pval = 0.0, 1.0
            else:
                w_res = stats.wilcoxon(b, a, zero_method="wilcox")
                w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_pval = 0.0, 1.0

    return SignificanceResult(
        system_a=system_a_name,
        system_b=system_b_name,
        metric=metric_name,
        mean_a=mean_a,
        mean_b=mean_b,
        delta_percentage=delta_pct,
        wilcoxon_stat=w_stat,
        wilcoxon_p_value=w_pval,
        ttest_stat=t_stat,
        ttest_p_value=t_pval,
        is_significant_05=(w_pval < 0.05),
        is_significant_01=(w_pval < 0.01),
    )

