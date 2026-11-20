"""Command-line interface for the paired statistical analyses."""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .core import (
    DEFAULT_ALPHA,
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    delong_paired,
    delong_single,
    holm_adjust,
    mcnemar_exact,
    model_summary,
    paired_bootstrap,
)
from .io import PredictionFileError, align_predictions, load_predictions

CLASS_NAMES = ("NORMAL", "PNEUMONIA")

METHODS_NOTE = (
    "Methods note: exact binomial McNemar on paired correctness "
    "(Dietterich, 1998), with Edwards' continuity-corrected chi-square shown "
    "for comparison; the odds ratio uses an exact Clopper-Pearson interval. "
    "DeLong's paired AUC test uses the fast mid-rank formulation "
    "(DeLong et al., 1988; Sun & Xu, 2014). Bootstrap: percentile intervals "
    "over paired resamples (Efron & Tibshirani, 1993); the two-sided "
    "bootstrap p is approximate. Wilson score intervals follow Newcombe "
    "(1998). Holm-Bonferroni step-down correction (Holm, 1979) controls the "
    "family-wise error rate across baselines. Labels: 0 = NORMAL, "
    "1 = PNEUMONIA."
)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    return "< 1e-12" if p < 1e-12 else f"{p:.2e}"


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f}%"


def fmt_or(x: float) -> str:
    return "inf" if not math.isfinite(x) else f"{x:.2f}"


def jsonable(value):
    """Recursively convert numpy scalars for JSON; non-finite floats -> None."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _ci_text(summary: dict) -> str:
    return f"[{fmt_pct(summary['ci_low'])}, {fmt_pct(summary['ci_high'])}]"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse_pair(
    merged: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run every paired test for an aligned (merged) prediction frame."""
    y_true = merged["y_true_a"].to_numpy()
    pred_a = merged["y_pred_a"].to_numpy()
    pred_b = merged["y_pred_b"].to_numpy()

    summary_a = model_summary(y_true, pred_a)
    summary_b = model_summary(y_true, pred_b)
    mcnemar = mcnemar_exact(
        merged["y_true_a"] == merged["y_pred_a"],
        merged["y_true_b"] == merged["y_pred_b"],
        alpha,
    )
    bootstrap_accuracy = paired_bootstrap(
        y_true, pred_a, pred_b, "accuracy", n_resamples, seed, alpha
    )

    delong = None
    bootstrap_auc = None
    if {"prob_pneumonia_a", "prob_pneumonia_b"} <= set(merged.columns):
        probs_a = merged["prob_pneumonia_a"].to_numpy(dtype=float)
        probs_b = merged["prob_pneumonia_b"].to_numpy(dtype=float)
        single_a = delong_single(y_true, probs_a, alpha)
        single_b = delong_single(y_true, probs_b, alpha)
        summary_a["auc"] = single_a["auc"]
        summary_a["auc_ci"] = [single_a["ci_low"], single_a["ci_high"]]
        summary_b["auc"] = single_b["auc"]
        summary_b["auc_ci"] = [single_b["ci_low"], single_b["ci_high"]]
        delong = delong_paired(y_true, probs_a, probs_b, alpha)
        bootstrap_auc = paired_bootstrap(
            y_true, probs_a, probs_b, "auc", n_resamples, seed, alpha
        )

    return {
        "n": int(len(merged)),
        "model_a": summary_a,
        "model_b": summary_b,
        "mcnemar": mcnemar,
        "bootstrap_accuracy": bootstrap_accuracy,
        "delong": delong,
        "bootstrap_auc": bootstrap_auc,
    }


def _load(path: str, require_probabilities: bool = False) -> pd.DataFrame:
    try:
        return load_predictions(path, require_probabilities)
    except PredictionFileError as exc:
        raise SystemExit(f"error: {exc}") from exc


def _compare(
    baseline_paths: Sequence[str],
    challenger_path: str,
    alpha: float,
    n_resamples: int,
    seed: int,
) -> tuple[list[dict], list[tuple[str, str]]]:
    df_b = _load(challenger_path)
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for path in baseline_paths:
        try:
            df_a = load_predictions(path)
            merged = align_predictions(df_a, df_b, path, challenger_path)
        except PredictionFileError as exc:
            failures.append((path, str(exc).splitlines()[0]))
            continue
        result = analyse_pair(merged, alpha, n_resamples, seed)
        result["path"] = path
        rows.append(result)

    if not rows:
        raise SystemExit("error: no baseline could be paired with the challenger")

    for row, adjusted in zip(
        rows, holm_adjust([row["mcnemar"]["p_exact"] for row in rows])
    ):
        row["mcnemar"]["p_holm"] = float(adjusted)

    with_auc = [row for row in rows if row["delong"] is not None]
    if with_auc:
        adjusted_values = holm_adjust(
            [row["delong"]["p_value"] for row in with_auc]
        )
        for row, adjusted in zip(with_auc, adjusted_values):
            row["delong"]["p_holm"] = float(adjusted)

    return rows, failures


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_model_summary(label: str, summary: dict, path: str | None = None) -> None:
    header = f"  {label}: {path}" if path else f"  {label}"
    print(header)
    print(
        f"    accuracy    {fmt_pct(summary['accuracy']['estimate'])}  "
        f"95% CI {_ci_text(summary['accuracy'])}"
    )
    if "sensitivity" in summary:
        item = summary["sensitivity"]
        print(
            f"    sensitivity {fmt_pct(item['estimate'])}  "
            f"95% CI {_ci_text(item)}  (PNEUMONIA recall)"
        )
    if "specificity" in summary:
        item = summary["specificity"]
        print(
            f"    specificity {fmt_pct(item['estimate'])}  "
            f"95% CI {_ci_text(item)}  (NORMAL recall)"
        )
    if "auc" in summary:
        print(
            f"    AUC         {summary['auc']:.3f}  "
            f"95% CI [{summary['auc_ci'][0]:.3f}, {summary['auc_ci'][1]:.3f}]"
        )
    counts = summary["confusion"]
    print(
        f"    confusion   TN {counts['tn']} | FP {counts['fp']} | "
        f"FN {counts['fn']} | TP {counts['tp']}"
    )


def print_mcnemar(result: dict, alpha: float) -> None:
    print("McNemar's test (paired per-image correctness)")
    print("-" * 62)
    print(
        f"    discordant pairs: {result['discordant']}  "
        f"(A wrong/B right: {result['n01_baseline_wrong_challenger_right']} "
        f"favours B | A right/B wrong: "
        f"{result['n10_baseline_right_challenger_wrong']} favours A)"
    )
    if result["discordant"] == 0:
        print("    the models are correct on exactly the same images.")
        return
    print(
        f"    exact binomial p (two-sided): {fmt_p(result['p_exact'])}"
    )
    print(
        f"    Edwards-corrected chi-square: {result['chi2_edwards']:.3f} "
        f"(p = {fmt_p(result['p_chi2'])})"
    )
    low, high = result["odds_ratio_ci"]
    print(
        f"    effect size: odds ratio n01/n10 = {fmt_or(result['odds_ratio'])} "
        f"(>1 favours B)  95% CI [{fmt_or(low)}, {fmt_or(high)}]; "
        f"risk difference {100 * result['risk_difference']:+.2f} pp "
        f"(95% CI {100 * result['risk_difference_ci'][0]:+.2f} to "
        f"{100 * result['risk_difference_ci'][1]:+.2f})"
    )
    if result["significant"]:
        better = "B" if result["n01_baseline_wrong_challenger_right"] >= \
            result["n10_baseline_right_challenger_wrong"] else "A"
        print(f"    => significant at alpha = {alpha} (favours {better}).")
    else:
        print(f"    => no significant difference at alpha = {alpha}.")


def print_delong(result: dict, alpha: float) -> None:
    print("DeLong's test (paired AUC comparison)")
    print("-" * 62)
    print(
        f"    AUC A: {result['auc_a']:.4f}  "
        f"95% CI [{result['auc_a_ci'][0]:.4f}, {result['auc_a_ci'][1]:.4f}]"
    )
    print(
        f"    AUC B: {result['auc_b']:.4f}  "
        f"95% CI [{result['auc_b_ci'][0]:.4f}, {result['auc_b_ci'][1]:.4f}]"
    )
    print(
        f"    difference (B - A): {result['auc_diff']:+.4f}  "
        f"z = {result['z']:.3f}"
    )
    print(f"    two-sided p: {fmt_p(result['p_value'])}")
    if result["significant"]:
        better = "B" if result["auc_diff"] > 0 else "A"
        print(f"    => significant at alpha = {alpha} (favours {better}).")
    else:
        print(f"    => no significant AUC difference at alpha = {alpha}.")


def print_bootstrap(
    label: str, result: dict, alpha: float, as_percentage: bool = False
) -> None:
    print(f"{label} (paired bootstrap, B - A)")
    print("-" * 62)
    if as_percentage:
        print(f"    point estimate: {100 * result['point_estimate']:+.2f} pp")
        print(
            f"    {int((1 - alpha) * 100)}% percentile CI: "
            f"[{100 * result['ci_low']:+.2f} pp, "
            f"{100 * result['ci_high']:+.2f} pp]"
        )
    else:
        print(f"    point estimate: {result['point_estimate']:+.4f}")
        print(
            f"    {int((1 - alpha) * 100)}% percentile CI: "
            f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
        )
    if result["n_skipped"]:
        print(f"    resamples skipped (single class): {result['n_skipped']}")
    print(
        f"    approximate two-sided bootstrap p: {fmt_p(result['p_value'])}"
    )
    if result["significant"]:
        better = "B" if result["point_estimate"] > 0 else "A"
        print(
            f"    => CI excludes 0: significant at alpha = {alpha} "
            f"(favours {better})."
        )
    else:
        print(f"    => CI includes 0: not significant at alpha = {alpha}.")


def print_detail(row: dict, baseline: str, challenger: str, alpha: float) -> None:
    print(f"\nViT vs {os.path.basename(baseline)}")
    print(f"   baseline: {baseline}")
    print(f"   challenger: {challenger}")
    print_bootstrap(
        "Accuracy difference", row["bootstrap_accuracy"], alpha, True
    )
    print()
    print_mcnemar(row["mcnemar"], alpha)
    if row["delong"] is not None:
        print()
        print_delong(row["delong"], alpha)
        print()
        print_bootstrap("AUC difference", row["bootstrap_auc"], alpha)
    else:
        print("\nDeLong: not applicable (constant scores, so AUC is undefined)")


def print_table(rows: Iterable[dict], alpha: float) -> None:
    rows = list(rows)
    print(
        f"\nFamily of {len(rows)} comparison(s); Holm-Bonferroni across "
        f"baselines (alpha = {alpha})"
    )
    print("-" * 104)
    print(
        f"{'Baseline':40s} {'dAcc(pp)':>9s} {'McNemar p':>10s} "
        f"{'Holm p':>10s} {'dAUC':>8s} {'DeLong p':>9s} {'Holm p':>9s}  Verdict"
    )
    print("-" * 104)
    for row in rows:
        mcnemar = row["mcnemar"]
        delong = row["delong"]
        delong_p = fmt_p(delong["p_value"]) if delong else "n/a"
        delong_holm = fmt_p(delong["p_holm"]) if delong else "n/a"
        auc_diff = f"{delong['auc_diff']:+.3f}" if delong else "n/a"
        accuracy_ok = mcnemar["p_holm"] < alpha
        auc_ok = (delong is None) or (delong["p_holm"] < alpha)
        verdict = (
            "YES"
            if (accuracy_ok and auc_ok)
            else ("partial" if (accuracy_ok or auc_ok) else "no")
        )
        print(
            f"{os.path.basename(row['path'])[:40]:40s} "
            f"{100 * row['bootstrap_accuracy']['point_estimate']:>+8.2f} "
            f"{fmt_p(mcnemar['p_exact']):>10s} "
            f"{fmt_p(mcnemar['p_holm']):>10s} "
            f"{auc_diff:>8s} {delong_p:>9s} {delong_holm:>9s}  {verdict}"
        )
    print("-" * 104)
    print(
        "Verdict = significant for accuracy AND (where available) AUC after "
        "Holm correction."
    )
    print()
    print(METHODS_NOTE)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_wilson(args: argparse.Namespace) -> None:
    frame = _load(args.csv)
    summary = model_summary(frame["y_true"], frame["y_pred"])
    if "prob_pneumonia" in frame.columns:
        auc = delong_single(
            frame["y_true"].to_numpy(), frame["prob_pneumonia"].to_numpy(float)
        )
        summary["auc"] = auc["auc"]
        summary["auc_ci"] = [auc["ci_low"], auc["ci_high"]]
    print(f"Wilson score 95% intervals for {args.csv} ({len(frame)} images)\n")
    print_model_summary("model", summary, args.csv)


def cmd_check(args: argparse.Namespace) -> None:
    frame_a = _load(args.a)
    frame_b = _load(args.b)
    merged = align_predictions(frame_a, frame_b, args.a, args.b)
    print(f"compatible: both files cover the same {len(merged)} test images")
    print(f"  A: {args.a}")
    print(f"  B: {args.b}")
    labels = np.unique(merged["y_true_a"])
    print(f"  classes present: {labels.tolist()} (0 = NORMAL, 1 = PNEUMONIA)")


def cmd_mcnemar(args: argparse.Namespace) -> None:
    frame_a = _load(args.a)
    frame_b = _load(args.b)
    merged = align_predictions(frame_a, frame_b, args.a, args.b)
    result = mcnemar_exact(
        merged["y_true_a"] == merged["y_pred_a"],
        merged["y_true_b"] == merged["y_pred_b"],
        args.alpha,
    )
    print(f"Paired comparison on {len(merged)} shared test images\n")
    print_mcnemar(result, args.alpha)


def cmd_delong(args: argparse.Namespace) -> None:
    frame_a = _load(args.a, require_probabilities=True)
    frame_b = _load(args.b, require_probabilities=True)
    merged = align_predictions(frame_a, frame_b, args.a, args.b)
    result = delong_paired(
        merged["y_true_a"].to_numpy(),
        merged["prob_pneumonia_a"].to_numpy(float),
        merged["prob_pneumonia_b"].to_numpy(float),
        args.alpha,
    )
    print(f"Paired comparison on {len(merged)} shared test images\n")
    print_delong(result, args.alpha)


def cmd_bootstrap(args: argparse.Namespace) -> None:
    require_probs = args.metric == "auc"
    frame_a = _load(args.a, require_probabilities=require_probs)
    frame_b = _load(args.b, require_probabilities=require_probs)
    merged = align_predictions(frame_a, frame_b, args.a, args.b)

    if args.metric == "accuracy":
        values_a = merged["y_pred_a"].to_numpy()
        values_b = merged["y_pred_b"].to_numpy()
        label, as_percentage = "Accuracy difference", True
    else:
        values_a = merged["prob_pneumonia_a"].to_numpy(float)
        values_b = merged["prob_pneumonia_b"].to_numpy(float)
        label, as_percentage = "AUC difference", False

    result = paired_bootstrap(
        merged["y_true_a"].to_numpy(),
        values_a,
        values_b,
        args.metric,
        args.n_resamples,
        args.seed,
        args.alpha,
    )
    print(f"Paired comparison on {len(merged)} shared test images\n")
    print_bootstrap(label, result, args.alpha, as_percentage)


def cmd_report(args: argparse.Namespace) -> None:
    frame_a = _load(args.a)
    frame_b = _load(args.b)
    merged = align_predictions(frame_a, frame_b, args.a, args.b)
    result = analyse_pair(merged, args.alpha, args.n_resamples, args.seed)
    result["mcnemar"]["p_holm"] = result["mcnemar"]["p_exact"]
    if result["delong"] is not None:
        result["delong"]["p_holm"] = result["delong"]["p_value"]

    print(f"Statistical comparison report - {len(merged)} shared test images")
    print(f"  A: {args.a}")
    print(f"  B: {args.b}\n")
    print("Per-model Wilson score 95% intervals")
    print("-" * 62)
    print_model_summary("A", result["model_a"], args.a)
    print_model_summary("B", result["model_b"], args.b)
    print()
    print_bootstrap("Accuracy difference", result["bootstrap_accuracy"],
                    args.alpha, True)
    print()
    print_mcnemar(result["mcnemar"], args.alpha)
    if result["delong"] is not None:
        print()
        print_delong(result["delong"], args.alpha)
        print()
        print_bootstrap("AUC difference", result["bootstrap_auc"], args.alpha)
    print()
    print(METHODS_NOTE)

    if args.json:
        payload = {
            "settings": {
                "model_a": args.a,
                "model_b": args.b,
                "n_shared_images": len(merged),
                "n_resamples": args.n_resamples,
                "seed": args.seed,
                "alpha": args.alpha,
            },
            **result,
            "methods_note": METHODS_NOTE,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, indent=2)
        print(f"\nSaved JSON results to {args.json}")


def cmd_compare(args: argparse.Namespace) -> None:
    rows, failures = _compare(
        args.a, args.b, args.alpha, args.n_resamples, args.seed
    )
    print(
        f"Family-wise comparison: {os.path.basename(args.b)} vs "
        f"{len(rows)} baseline(s) on {rows[0]['n']} shared test images"
    )
    print_table(rows, args.alpha)
    for path, message in failures:
        print(f"  skipped {os.path.basename(path)}: {message}")


def cmd_demo(args: argparse.Namespace) -> None:
    outdir = os.path.abspath(os.path.expanduser(args.outdir))
    os.makedirs(outdir, exist_ok=True)
    paths = _write_demo_files(outdir, args.seed)

    print("DEMO - synthetic paired predictions resembling the dissertation")
    print(f"(380-image cleaned test set, seed {args.seed})\n")
    for name, path in paths.items():
        print(f"  wrote {name}: {path}")

    rows, failures = _compare(
        [paths["resnet50"], paths["vgg16_svm"]],
        paths["vit"],
        args.alpha,
        args.n_resamples,
        args.seed,
    )
    print()
    for row in rows:
        print_detail(row, row["path"], paths["vit"], args.alpha)
        print()
    print_table(rows, args.alpha)
    for path, message in failures:
        print(f"  skipped {os.path.basename(path)}: {message}")
    print(
        f"\nReplace these demo files with real exports "
        f"(test_predictions.csv from the notebook) to analyse your models."
    )


def _write_demo_files(outdir: str, seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    n, n_normal = 380, 99
    y_true = np.array([0] * n_normal + [1] * (n - n_normal))
    rng.shuffle(y_true)
    image_paths = [
        f"demo://chest_xray/test/{CLASS_NAMES[label].lower()}/{index:04d}.jpeg"
        for index, label in enumerate(y_true)
    ]

    def simulate(recall_normal: float, recall_pneumonia: float) -> pd.DataFrame:
        p_correct = np.where(y_true == 0, recall_normal, recall_pneumonia)
        correct = rng.random(n) < p_correct
        y_pred = np.where(correct, y_true, 1 - y_true)
        confidence = 0.5 + 0.49 * rng.beta(2.0, 2.0, n)
        prob_pneumonia = np.where(y_pred == 1, confidence, 1.0 - confidence)
        return pd.DataFrame(
            {
                "image_path": image_paths,
                "y_true": y_true,
                "y_pred": y_pred,
                "prob_normal": 1.0 - prob_pneumonia,
                "prob_pneumonia": prob_pneumonia,
            }
        )

    specs = {
        "vit": (0.73, 0.98),
        "resnet50": (0.88, 0.70),
        "vgg16_svm": (0.38, 0.99),
    }
    paths = {}
    for name, (recall_normal, recall_pneumonia) in specs.items():
        path = os.path.join(outdir, f"demo_{name}.csv")
        simulate(recall_normal, recall_pneumonia).to_csv(path, index=False)
        paths[name] = path
    return paths


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="two-sided significance level (default 0.05)")
    common.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES,
                        dest="n_resamples",
                        help="bootstrap resamples (default 10000)")
    common.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="bootstrap random seed (default 42)")

    parser = argparse.ArgumentParser(
        prog="pneumonia-stats",
        description=(
            "Paired statistical comparison of pneumonia classifiers: exact "
            "McNemar, DeLong AUC, paired bootstrap, Wilson intervals and "
            "Holm-Bonferroni correction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m pneumonia_stats demo\n"
            "  python -m pneumonia_stats report --a resnet50_predictions.csv "
            "--b test_predictions.csv\n"
            "  python -m pneumonia_stats compare --a resnet50_predictions.csv "
            "vgg16_predictions.csv --b test_predictions.csv\n"
            "  python -m pneumonia_stats wilson --csv test_predictions.csv\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser(
        "demo", parents=[common],
        help="run every analysis on synthetic data (no files needed)",
    )
    demo.add_argument("--outdir", default="demo_output",
                      help="directory for the generated CSV files")
    demo.set_defaults(func=cmd_demo)

    report = sub.add_parser(
        "report", parents=[common],
        help="all tests for one baseline vs one challenger",
    )
    report.add_argument("--a", required=True, help="baseline predictions CSV")
    report.add_argument("--b", required=True, help="challenger predictions CSV")
    report.add_argument("--json", help="write the results to this JSON file")
    report.set_defaults(func=cmd_report)

    compare = sub.add_parser(
        "compare", parents=[common],
        help="challenger vs several baselines with Holm correction",
    )
    compare.add_argument("--a", required=True, nargs="+",
                         help="one or more baseline prediction CSVs")
    compare.add_argument("--b", required=True,
                         help="challenger predictions CSV")
    compare.set_defaults(func=cmd_compare)

    for name, function, helptext in (
        ("mcnemar", cmd_mcnemar, "exact McNemar test and effect sizes"),
        ("delong", cmd_delong, "paired DeLong AUC test"),
    ):
        item = sub.add_parser(name, parents=[common], help=helptext)
        item.add_argument("--a", required=True, help="baseline predictions CSV")
        item.add_argument("--b", required=True,
                          help="challenger predictions CSV")
        item.set_defaults(func=function)

    bootstrap = sub.add_parser(
        "bootstrap", parents=[common],
        help="paired bootstrap CI for the accuracy or AUC difference",
    )
    bootstrap.add_argument("--a", required=True, help="baseline predictions CSV")
    bootstrap.add_argument("--b", required=True,
                           help="challenger predictions CSV")
    bootstrap.add_argument("--metric", choices=("accuracy", "auc"),
                           default="accuracy")
    bootstrap.set_defaults(func=cmd_bootstrap)

    wilson = sub.add_parser(
        "wilson", parents=[common],
        help="Wilson score 95%% intervals for one prediction CSV",
    )
    wilson.add_argument("--csv", required=True)
    wilson.set_defaults(func=cmd_wilson)

    check = sub.add_parser(
        "check", parents=[common],
        help="verify that two prediction files cover the same test set",
    )
    check.add_argument("--a", required=True)
    check.add_argument("--b", required=True)
    check.set_defaults(func=cmd_check)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
