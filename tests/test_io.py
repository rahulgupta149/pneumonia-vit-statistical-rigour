"""Tests for prediction-file loading and pairing."""

import pandas as pd
import pytest

from pneumonia_stats.io import (
    PredictionFileError,
    align_predictions,
    load_predictions,
)


def _frame(paths, labels=None, predictions=None, probabilities=True):
    labels = labels if labels is not None else [0] * len(paths)
    predictions = predictions if predictions is not None else labels
    data = {"image_path": paths, "y_true": labels, "y_pred": predictions}
    if probabilities:
        data["prob_normal"] = [0.2] * len(paths)
        data["prob_pneumonia"] = [0.8] * len(paths)
    return pd.DataFrame(data)


def test_load_predictions_round_trip(tmp_path):
    path = tmp_path / "predictions.csv"
    _frame(["a.jpeg", "b.jpeg"]).to_csv(path, index=False)
    loaded = load_predictions(str(path))
    assert list(loaded.columns) == [
        "image_path",
        "y_true",
        "y_pred",
        "prob_normal",
        "prob_pneumonia",
    ]
    assert len(loaded) == 2


def test_load_predictions_missing_file(tmp_path):
    with pytest.raises(PredictionFileError, match="file not found"):
        load_predictions(str(tmp_path / "absent.csv"))


def test_load_predictions_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"image_path": ["a.jpeg"], "y_true": [0]}).to_csv(
        path, index=False
    )
    with pytest.raises(PredictionFileError, match="missing required column"):
        load_predictions(str(path))


def test_load_predictions_requires_probabilities(tmp_path):
    path = tmp_path / "no_probs.csv"
    _frame(["a.jpeg"], probabilities=False).to_csv(path, index=False)
    with pytest.raises(PredictionFileError, match="prob_pneumonia"):
        load_predictions(str(path), require_probabilities=True)


def test_load_predictions_rejects_duplicates(tmp_path):
    path = tmp_path / "dup.csv"
    _frame(["a.jpeg", "a.jpeg"]).to_csv(path, index=False)
    with pytest.raises(PredictionFileError, match="duplicate"):
        load_predictions(str(path))


def test_load_predictions_rejects_empty(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(PredictionFileError, match="could not parse"):
        load_predictions(str(path))


def test_align_predictions_happy_path_sorts_by_image():
    # labels are per image, so both files must agree after reordering
    frame_a = _frame(["c.jpeg", "a.jpeg", "b.jpeg"], [1, 0, 1])
    frame_b = _frame(["b.jpeg", "c.jpeg", "a.jpeg"], [1, 1, 0])
    merged = align_predictions(frame_a, frame_b)
    assert list(merged["image_path"]) == ["a.jpeg", "b.jpeg", "c.jpeg"]
    assert {"y_true_a", "y_true_b", "prob_pneumonia_a", "prob_pneumonia_b"} <= set(
        merged.columns
    )


def test_align_predictions_rejects_different_test_sets():
    frame_a = _frame(["a.jpeg", "b.jpeg"])
    frame_b = _frame(["a.jpeg", "c.jpeg"])
    with pytest.raises(PredictionFileError, match="different test sets"):
        align_predictions(frame_a, frame_b, "baseline.csv", "vit.csv")


def test_align_predictions_rejects_label_mismatch():
    frame_a = _frame(["a.jpeg", "b.jpeg"], [0, 1])
    frame_b = _frame(["a.jpeg", "b.jpeg"], [0, 0])
    with pytest.raises(PredictionFileError, match="ground-truth labels"):
        align_predictions(frame_a, frame_b)
