"""Paired statistical analysis for pneumonia classification models.

The package implements the statistical-rigour layer of the dissertation:
Wilson intervals, exact McNemar tests, DeLong's paired AUC test, a paired
bootstrap and Holm-Bonferroni correction.
"""

from .core import (
    DEFAULT_ALPHA,
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    auc_midrank,
    clopper_pearson,
    confusion_counts,
    delong_paired,
    delong_single,
    holm_adjust,
    mcnemar_exact,
    model_summary,
    paired_bootstrap,
    proportion_summary,
    wilson_interval,
)
from .io import (
    ALL_COLUMNS,
    PROBABILITY_COLUMN,
    REQUIRED_COLUMNS,
    PredictionFileError,
    align_predictions,
    load_predictions,
)

__version__ = "1.0.0"

__all__ = [
    "ALL_COLUMNS",
    "DEFAULT_ALPHA",
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_SEED",
    "PROBABILITY_COLUMN",
    "PredictionFileError",
    "REQUIRED_COLUMNS",
    "__version__",
    "align_predictions",
    "auc_midrank",
    "clopper_pearson",
    "confusion_counts",
    "delong_paired",
    "delong_single",
    "holm_adjust",
    "load_predictions",
    "mcnemar_exact",
    "model_summary",
    "paired_bootstrap",
    "proportion_summary",
    "wilson_interval",
]
