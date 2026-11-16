# Forgetting the Pipeline Too: PLU-GGN

**ES 678 (Machine Learning) course project, IIT Gandhinagar. Ojas Gupta (26250031).**

When patients withdraw consent, a model should behave as if they were never in the training data.
Unlearning methods usually update only the weights, but tabular pipelines also *learn* their preprocessing
(imputation means, scaling, target encodings). This project measures what a frozen preprocessor costs on
30-day readmission prediction (UCI Diabetes 130-US Hospitals), and proposes **PLU-GGN**: an exact
O(|D_f|) downdate of the preprocessor plus a pipeline-shift-corrected Gauss–Newton step.

- **Paper (4 pages, NeurIPS style):** [`paper/main.pdf`](paper/main.pdf), source in [`paper/main.tex`](paper/main.tex)
- **Website with in-browser replication (Pyodide):** https://ojas1024.github.io/plu-ggn-unlearning/ (source in [`docs/`](docs/))
- **Original Kaggle notebook:** [`notebooks/plu_ggn_original.ipynb`](notebooks/plu_ggn_original.ipynb)

## Main result (delete the 3% highest-utilization training encounters)

| Method | Weight error to retrain | Test KL to retrain | Pipeline leak | Speedup |
|---|---|---|---|---|
| No unlearning | 1.25e-1 | 3.7e-4 | 100% | – |
| Gradient ascent (frozen φ) | 1.71e-1 | 1.2e-4 | 100% | ~10³× |
| Standard influence (frozen φ) | 1.55e-1 | 7.7e-5 | 100% | 63× |
| **PLU-GGN 1-step** | **5.98e-3** | **4.4e-6** | **0%** | 5.5× |
| **PLU-GGN 2-step** | **2.42e-5** | **7.3e-11** | **0%** | 5.1× |

Standard influence ends *further* from retraining than doing nothing in weight space (5/5 seeds), but is
still closer in prediction space. A feature-norm "membership audit" we first used turned out to measure how
extreme a patient is, not membership (AUC drops from 0.98 to 0.56 against matched controls); see §4 of the paper.

## Reproduce

```bash
pip install -r requirements.txt
python experiments/run_all.py   # every number/figure -> results/results.json, paper/figures/ (~20 s)
python src/plu_ggn.py           # just the main 3% comparison
scripts/build_site.sh           # copy code/data/results/paper into docs/ for the website
```

- **Data:** `data/diabetes_25k.csv` is a fixed random sample (`random_state=42`) of 25,000 encounters from
  [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296) (OpenML 4541), CC BY 4.0, public and
  de-identified. If the file is deleted, the loader downloads it again from OpenML.
- **Split:** `GroupShuffleSplit(test_size=0.2, random_state=42)` grouped by `patient_nbr` (0 shared patients).
- **Missingness:** 12% of numeric cells masked completely at random (`RandomState(42)`), since the released counts are complete.
- **Fixed hyperparameters:** λ = 10 (intercept 1e-4), target-encoding smoothing m = 10; nothing tuned on test data.
- Timings depend on the machine; accuracy numbers are deterministic.

## Layout

```
src/plu_ggn.py            core library (data, preprocessor, model, unlearning methods, metrics)
experiments/run_all.py    all experiments + figure
experiments/ablation_original.py   the notebook's original ablation script
results/                  results.json, figure PNG
paper/                    main.tex, neurips_2025.sty, figures, main.pdf
docs/                     GitHub Pages site (static HTML/JS + Pyodide)
notebooks/                original notebook
```

## AI use

Claude (Anthropic) helped with code refactoring, re-running and auditing the experiments (including the
matched-control membership audit and the oracle-precision check), building the website and editing the paper.
All reported results were checked by the author.
