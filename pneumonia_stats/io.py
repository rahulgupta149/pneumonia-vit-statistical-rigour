"""Loading and pairing of per-image prediction files.

A prediction CSV must contain one row per test image with the columns::

    image_path, y_true, y_pred, prob_normal, prob_pneumonia

``prob_normal`` and ``prob_pneumonia`` are optional for accuracy-only
analyses but required for AUC comparisons. Labels are ``0 = NORMAL`` and
``1 = PNEUMONIA``.

Paired tests are only valid when both files cover exactly the same images,
so :func:`align_predictions` refuses to merge anything else.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("image_path", "y_true", "y_pred")
PROBABILITY_COLUMN = "prob_pneumonia"
ALL_COLUMNS = ("image_path", "y_true", "y_pred", "prob_normal", "prob_pneumonia")

__all__ = [
    "ALL_COLUMNS",
    "PROBABILITY_COLUMN",
    "PredictionFileError",
    "REQUIRED_COLUMNS",
    "align_predictions",
    "load_predictions",
]


class PredictionFileError(ValueError):
    """Raised when a prediction CSV is missing or incompatible."""


def load_predictions(path: str, require_probabilities: bool = False) -> pd.DataFrame:
    """Load a prediction CSV and validate its schema.

    Raises
    ------
    PredictionFileError
        If the file is missing, empty, has duplicate image paths, or lacks
        required columns.
    """
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise PredictionFileError(f"file not found: {path}") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise PredictionFileError(f"could not parse {path}: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PredictionFileError(
            f"{path} is missing required column(s): {', '.join(missing)}. "
            "Expected: image_path, y_true, y_pred, prob_normal, prob_pneumonia."
        )
    if require_probabilities and PROBABILITY_COLUMN not in frame.columns:
        raise PredictionFileError(
            f"{path} has no '{PROBABILITY_COLUMN}' column, which AUC analyses "
            "require."
        )
    if frame.empty:
        raise PredictionFileError(f"{path} contains no rows")
    if frame["image_path"].duplicated().any():
        raise PredictionFileError(f"{path} contains duplicate image_path entries")

    return frame


def align_predictions(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    label_a: str = "A",
    label_b: str = "B",
) -> pd.DataFrame:
    """Merge two prediction frames after verifying they cover the same images.

    The returned frame is sorted by ``image_path`` and uses the suffixes
    ``_a`` and ``_b`` on every shared column.

    Raises
    ------
    PredictionFileError
        If the image sets differ or the ground-truth labels disagree.
    """
    set_a = set(frame_a["image_path"])
    set_b = set(frame_b["image_path"])
    if set_a != set_b:
        only_a = len(set_a - set_b)
        only_b = len(set_b - set_a)
        raise PredictionFileError(
            "the two prediction files cover different test sets, so paired "
            "tests are not valid.\n"
            f"  {label_a}: {len(set_a)} images ({only_a} not in B)\n"
            f"  {label_b}: {len(set_b)} images ({only_b} not in A)\n"
            "Re-evaluate both models on the same (cleaned) test set."
        )

    merged = frame_a.merge(frame_b, on="image_path", suffixes=("_a", "_b"))
    if not (merged["y_true_a"] == merged["y_true_b"]).all():
        raise PredictionFileError(
            "ground-truth labels disagree between the two files; they must "
            "come from the same test set."
        )
    return merged.sort_values("image_path").reset_index(drop=True)
