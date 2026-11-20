# Pneumonia Detection with Vision Transformers — Statistical Rigour Edition

Vision Transformer (ViT-Base/16) for paediatric chest X-ray pneumonia
classification, with a **paired statistical comparison layer** that tests whether
architectural differences are real rather than a lucky test-set draw.

This repository accompanies the MSc dissertation **COM726** at Southampton Solent
University and is intended to be reusable: the statistics package works on any
pair of per-image prediction files, with or without the original models.

---

## Contents

| Path | What it is |
|---|---|
| `notebooks/pneumonia_vit_statistical_rigour.ipynb` | End-to-end Kaggle notebook: quality control, training, evaluation, prediction export and statistical analysis |
| `pneumonia_stats/` | The statistical methods as an importable, tested package with a command-line interface |
| `docs/METHODS.md` | Method definitions, assumptions and full references |
| `tests/` | Unit tests checked against independent reference implementations |

---

## Reported results

From the dissertation run on the cleaned test set:

| Metric | Value |
|---|---|
| Accuracy | **91.58%** |
| AUC | **0.965** |
| Sensitivity (PNEUMONIA recall) | 98% |
| Specificity (NORMAL recall) | 73% |
| Test set | 380 images (99 NORMAL / 281 PNEUMONIA) |

Re-running the notebook can produce slightly different values, because training is
stochastic and the training loop now restores the best-validation checkpoint
correctly (earlier versions used a shallow copy that was overwritten each epoch).
Report the numbers from your own run, not these.

---

## Quick start (statistics only — no GPU, no dataset)

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m pneumonia_stats demo
pytest -q
```

`demo` simulates the 380-image cleaned test set, writes three prediction CSVs to
`demo_output/`, and runs every analysis on them. No data files are needed.

---

## Analysing your own models

Each model must be evaluated on **exactly the same test images**. Export one CSV
per model with these columns:

| Column | Required | Meaning |
|---|---|---|
| `image_path` | yes | Unique image identifier; must match across files |
| `y_true` | yes | Ground truth: `0` = NORMAL, `1` = PNEUMONIA |
| `y_pred` | yes | Predicted label: `0` or `1` |
| `prob_normal` | no | P(NORMAL) |
| `prob_pneumonia` | for AUC | P(PNEUMONIA) |

The notebook writes the ViT file as `test_predictions.csv`.

```bash
# 1. Confirm the two files cover the same test set
python -m pneumonia_stats check \
    --a resnet50_predictions.csv --b test_predictions.csv

# 2. Full report for one baseline vs the ViT, saved as JSON
python -m pneumonia_stats report \
    --a resnet50_predictions.csv --b test_predictions.csv \
    --json significance.json

# 3. Several baselines, with Holm-Bonferroni correction across the family
python -m pneumonia_stats compare \
    --a resnet50_predictions.csv vgg16_predictions.csv \
    --b test_predictions.csv
```

Individual tests are also available:

```bash
python -m pneumonia_stats mcnemar   --a A.csv --b B.csv
python -m pneumonia_stats delong    --a A.csv --b B.csv
python -m pneumonia_stats bootstrap --a A.csv --b B.csv --metric auc
python -m pneumonia_stats wilson    --csv test_predictions.csv
```

> Paired tests are invalid if the two files were evaluated on different images,
> so `check` / `report` / `compare` refuse mismatched test sets. This includes the
> common mistake of pairing results from the original 624-image split with the
> cleaned 380-image split.

---

## Reproducing the full pipeline on Kaggle

1. **Create → Notebook**, then **File → Upload Notebook** and select
   `notebooks/pneumonia_vit_statistical_rigour.ipynb`.
2. **Add Data →** search for **Chest X-Ray Images (Pneumonia)** and add it.
   The notebook locates the dataset automatically, wherever Kaggle mounts it.
3. **Settings → Accelerator → GPU T4 x2**.
4. **Settings → Internet → On** (required: `timm` downloads the ImageNet weights).
5. **Run → Run All Cells**.

Expected runtime is roughly 30–45 minutes, dominated by training. The notebook
writes:

| File | Contents |
|---|---|
| `test_predictions.csv` | Per-image ground truth, prediction and probabilities |
| `statistical_results.json` | Every statistic, ready for an appendix |
| `pneumonia_vit_model.pth` | Model weights |
| `pneumonia_vit_complete.pth` | Full serialised model |
| `training_history.npy`, `evaluation_results.npy`, `model_info.npy` | Training artefacts |
| `training_summary.txt` | Human-readable summary |

If you hit CUDA out-of-memory, reduce `BATCH_SIZE` from 32 to 16 in the
configuration cell.

---

## Statistical methods

| Method | Purpose | Reference |
|---|---|---|
| Wilson score interval | 95% CI for accuracy, sensitivity, specificity | Newcombe (1998) |
| Exact McNemar test | Paired difference in correctness | Dietterich (1998) |
| Edwards-corrected χ² | Large-sample McNemar comparison | Edwards (1948) |
| Conditional odds ratio | Effect size for discordant pairs | — |
| DeLong paired AUC test | Difference between correlated ROC AUCs | DeLong et al. (1988); Sun & Xu (2014) |
| Paired bootstrap (10,000) | Distribution-free CI for Δaccuracy / ΔAUC | Efron & Tibshirani (1993) |
| Holm–Bonferroni | Family-wise error control across baselines | Holm (1979) |

Full definitions and assumptions are in [`docs/METHODS.md`](docs/METHODS.md).

---

## Repository layout

```
.
├── .github/workflows/ci.yml        # tests on Python 3.10 and 3.11
├── CITATION.cff
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt                # analysis package
├── requirements-dev.txt            # + pytest
├── requirements-notebook.txt       # + torch, timm, jupyter
├── docs/
│   └── METHODS.md
├── notebooks/
│   └── pneumonia_vit_statistical_rigour.ipynb
├── pneumonia_stats/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── core.py
│   └── io.py
└── tests/
    ├── test_core.py
    └── test_io.py
```

---

## Tests and continuous integration

```bash
pytest -q
```

The suite cross-checks the implementation against independent references:

* DeLong against a brute-force O(mn) implementation, and AUC against
  `sklearn.metrics.roc_auc_score` (ties included);
* Wilson intervals against published values;
* Holm against the textbook example;
* McNemar effect sizes hand-computed on a small table;
* pairing logic, including rejection of mismatched test sets.

`.github/workflows/ci.yml` runs the same suite on every push and pull request.

---

## Known limitations

* **Small validation split.** The cleaned validation set contains only 13 images
  (the source dataset provides 16), so checkpoint selection is noisy.
* **Quality-control thresholds** are estimated across all splits. For a strictly
  leakage-free protocol they should be fitted on the training split alone and
  applied unchanged to validation and test. This does not affect the paired
  tests (which condition on the fixed test set) but should be stated.
* **Single-centre test set.** No external validation cohort is used.
* **Bootstrap p-values** are approximate; McNemar and DeLong are the confirmatory
  tests.

---

## Citation

If you use this code, please cite the repository (see `CITATION.cff`) or the
associated dissertation.

## License

MIT — see [`LICENSE`](LICENSE).

## Disclaimer

Research and educational use only. This is **not** a clinical diagnostic device
and must not be used to inform patient care.
