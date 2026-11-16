"""
PLU-GGN: Pipeline-Leakage-Aware Unlearning via Generalized Gauss-Newton.

Core library for the ES 678 course project
"Forgetting the Pipeline Too: Fast Tabular Machine Unlearning under
Preprocessing and Imputation Shifts".

This single file holds everything the experiments need (data loading,
downdatable preprocessor, logistic model, unlearning methods, metrics), so the
same code runs locally and in the browser through Pyodide (see docs/).

Dependencies: numpy, pandas, scipy, scikit-learn.
"""

import os
import time
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
RANDOM_SEED = 42

NUM_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
CAT_COLS = ["admission_type_id", "discharge_disposition_id", "medical_specialty", "A1Cresult"]
DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "diabetes_25k.csv")


# =====================================================================
# 1. DATA
# =====================================================================
def fetch_raw_sample(n_samples: int = 25000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Download UCI Diabetes 130-US Hospitals (OpenML 4541) and draw the fixed 25k-encounter sample."""
    from sklearn.datasets import fetch_openml

    df = fetch_openml(data_id=4541, as_frame=True, parser="auto").frame.copy()
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    return df[NUM_COLS + CAT_COLS + ["readmitted", "patient_nbr"]]


def load_clinical_readmission_data(
    csv_path: Optional[str] = DEFAULT_CSV,
    n_samples: int = 25000,
    missing_rate: float = 0.12,
    mask_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns (X_df, y, patient_ids) for 30-day readmission prediction.

    Reads the cached 25k sample from `csv_path` if it exists, otherwise downloads it from OpenML.
    The numeric columns of this dataset have no missing values, so `missing_rate` of numeric
    cells are masked completely at random (MCAR) to exercise the imputer. With
    mask_seed=42 this reproduces the original notebook exactly.
    """
    if csv_path is not None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={c: str for c in CAT_COLS}, keep_default_na=False)
    else:
        df = fetch_raw_sample(n_samples)

    y = (df["readmitted"].astype(str) == "<30").astype(np.float64).values
    patient_ids = df["patient_nbr"].values

    X_df = df[NUM_COLS + CAT_COLS].copy()
    for c in NUM_COLS:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce").astype(np.float64)
    mask = np.random.RandomState(mask_seed).rand(len(X_df), len(NUM_COLS)) < missing_rate
    X_df[NUM_COLS] = X_df[NUM_COLS].mask(mask)
    for c in CAT_COLS:
        X_df[c] = X_df[c].astype(str).replace({"?": "Missing", "nan": "Missing", "None": "Missing", "": "Missing"}).astype(object)
    return X_df, y, patient_ids


def utilization_score(X_df: pd.DataFrame) -> np.ndarray:
    """Ranking used to pick the high-utilization forget cohort."""
    return X_df["number_inpatient"].fillna(0).values * 3.0 + X_df["num_lab_procedures"].fillna(0).values * 0.1


# =====================================================================
# 2. DOWNDATABLE PREPROCESSOR
# =====================================================================
class DowndatablePreprocessor:
    """
    Mean imputation + standardization for numeric columns and Empirical-Bayes
    smoothed target encoding for categoricals, all stored as additive
    sufficient statistics so that removing rows costs O(|D_f|):
        numeric:     (n_obs, sum x, sum x^2) per column
        categorical: (count, sum y) per level, plus total sum y
    Note: target encoding is fitted in-sample on the training set (not out-of-fold).
    """

    def __init__(self, num_cols: list, cat_cols: list, smoothing: float = 10.0):
        self.num_cols, self.cat_cols, self.smoothing = num_cols, cat_cols, smoothing
        self.n_obs_, self.sum_x_, self.sum_x2_ = {}, {}, {}
        self.total_rows_ = 0
        self.sum_y_total_ = 0.0
        self.cat_counts_, self.cat_y_sums_ = {}, {}
        self.means_, self.stds_, self.cat_maps_ = {}, {}, {}
        self.global_prior_ = 0.0

    def _compute_derived_params(self):
        for c in self.num_cols:
            n_c = max(self.n_obs_[c], 1)
            mu = self.sum_x_[c] / n_c
            # After mean imputation, sum over all rows of (x - mu)^2 = sum_obs x^2 - n_obs mu^2
            var_num = max(self.sum_x2_[c] - n_c * (mu ** 2), 1e-8)
            self.means_[c] = mu
            self.stds_[c] = max(np.sqrt(var_num / max(self.total_rows_, 1)), 1e-6)
        self.global_prior_ = self.sum_y_total_ / max(self.total_rows_, 1)
        for c in self.cat_cols:
            mapping = {}
            for val, cnt in self.cat_counts_[c].items():
                if cnt > 0:
                    mapping[val] = (self.cat_y_sums_[c][val] + self.smoothing * self.global_prior_) / (cnt + self.smoothing)
                else:
                    mapping[val] = self.global_prior_
            self.cat_maps_[c] = mapping

    def fit(self, X_df: pd.DataFrame, y: np.ndarray) -> "DowndatablePreprocessor":
        self.total_rows_ = len(X_df)
        self.sum_y_total_ = float(np.sum(y))
        for c in self.num_cols:
            vals = X_df[c].values.astype(np.float64)
            obs = vals[~np.isnan(vals)]
            self.n_obs_[c], self.sum_x_[c], self.sum_x2_[c] = int(obs.size), float(np.sum(obs)), float(np.sum(obs ** 2))
        for c in self.cat_cols:
            self.cat_counts_[c], self.cat_y_sums_[c] = {}, {}
            vals = X_df[c].values.astype(str)
            for v in np.unique(vals):
                m = vals == v
                self.cat_counts_[c][v] = int(np.sum(m))
                self.cat_y_sums_[c][v] = float(np.sum(y[m]))
        self._compute_derived_params()
        return self

    def downdate(self, X_forget_df: pd.DataFrame, y_forget: np.ndarray) -> "DowndatablePreprocessor":
        """Exact statistics of D_r = D minus D_f, computed from D_f alone in O(|D_f|)."""
        new = DowndatablePreprocessor(self.num_cols, self.cat_cols, self.smoothing)
        new.total_rows_ = self.total_rows_ - len(X_forget_df)
        new.sum_y_total_ = self.sum_y_total_ - float(np.sum(y_forget))
        for c in self.num_cols:
            vals = X_forget_df[c].values.astype(np.float64)
            obs = vals[~np.isnan(vals)]
            new.n_obs_[c] = self.n_obs_[c] - int(obs.size)
            new.sum_x_[c] = self.sum_x_[c] - float(np.sum(obs))
            new.sum_x2_[c] = self.sum_x2_[c] - float(np.sum(obs ** 2))
        for c in self.cat_cols:
            new.cat_counts_[c] = self.cat_counts_[c].copy()
            new.cat_y_sums_[c] = self.cat_y_sums_[c].copy()
            vals = X_forget_df[c].values.astype(str)
            for v in np.unique(vals):
                m = vals == v
                new.cat_counts_[c][v] = new.cat_counts_[c].get(v, 0) - int(np.sum(m))
                new.cat_y_sums_[c][v] = new.cat_y_sums_[c].get(v, 0.0) - float(np.sum(y_forget[m]))
        new._compute_derived_params()
        return new

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        """Design matrix: [standardized numerics | 10*(target encoding - prior) | intercept]."""
        feats = []
        for c in self.num_cols:
            vals = X_df[c].values.astype(np.float64).copy()
            vals[np.isnan(vals)] = self.means_[c]
            feats.append((vals - self.means_[c]) / self.stds_[c])
        for c in self.cat_cols:
            enc = np.array([self.cat_maps_[c].get(v, self.global_prior_) for v in X_df[c].values.astype(str)])
            feats.append((enc - self.global_prior_) * 10.0)
        feats.append(np.ones(len(X_df)))
        return np.column_stack(feats)


def fast_affine_matrix_downdate(X_old, nan_mask_num, prep_old, prep_new, X_raw_cat_df) -> np.ndarray:
    """
    Re-express an already standardized matrix under downdated (mu, sigma) without re-parsing raw data:
        z_new = (sigma_old / sigma_new) * z_old + (mu_old - mu_new) / sigma_new,
    with imputed entries pinned to 0 and categorical columns re-mapped.
    """
    X_new = X_old.copy()
    for j, c in enumerate(prep_old.num_cols):
        mu_o, s_o, mu_n, s_n = prep_old.means_[c], prep_old.stds_[c], prep_new.means_[c], prep_new.stds_[c]
        X_new[:, j] = X_old[:, j] * (s_o / s_n) + (mu_o - mu_n) / s_n
        X_new[nan_mask_num[:, j], j] = 0.0
    off = len(prep_old.num_cols)
    for k, c in enumerate(prep_old.cat_cols):
        mapped = X_raw_cat_df[c].astype(str).map(prep_new.cat_maps_[c]).astype(float).fillna(prep_new.global_prior_).values
        X_new[:, off + k] = (mapped - prep_new.global_prior_) * 10.0
    return X_new


# =====================================================================
# 3. MODEL AND UNLEARNING METHODS
# =====================================================================
class RegularizedLogisticModel:
    """L2-regularized logistic regression (intercept penalized at 1e-4), fitted with L-BFGS-B."""

    def __init__(self, l2_reg: float = 10.0):
        self.l2_reg = l2_reg
        self.theta: Optional[np.ndarray] = None

    def _reg(self, d):
        r = np.full(d, self.l2_reg)
        r[-1] = 1e-4
        return r

    def loss_and_grad(self, theta, X, y):
        p = expit(X @ theta)
        bce = -np.sum(y * np.log(np.clip(p, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-12, 1.0)))
        r = self._reg(len(theta))
        return bce + 0.5 * np.sum(r * theta ** 2), X.T @ (p - y) + r * theta

    def fit(self, X, y, init_theta=None):
        x0 = np.zeros(X.shape[1]) if init_theta is None else init_theta.copy()
        res = minimize(lambda w: self.loss_and_grad(w, X, y), x0=x0, jac=True, method="L-BFGS-B",
                       options={"gtol": 1e-10, "ftol": 1e-12, "maxiter": 500})
        self.theta = res.x
        return self

    def predict_proba(self, X):
        return expit(X @ self.theta)

    def compute_ggn_hessian(self, X, theta=None):
        """G = X^T diag(p(1-p)) X + Lambda (exact Hessian for logistic loss)."""
        w = self.theta if theta is None else theta
        p = expit(X @ w)
        return (X.T * (p * (1.0 - p))) @ X + np.diag(self._reg(X.shape[1]))


def unlearn_naive_gradient_ascent(model, X_f_old, y_f, step_size=1e-4, steps=5):
    """Baseline: a few gradient-ascent steps on the forget-set loss, preprocessor frozen."""
    th = model.theta.copy()
    for _ in range(steps):
        th += step_size * (X_f_old.T @ (expit(X_f_old @ th) - y_f))
    return th


def unlearn_standard_influence_ggn(model, X_r_old, X_f_old, y_f):
    """Baseline: one influence/Newton removal step theta + G(D_r, phi_D)^-1 g_rem, preprocessor frozen."""
    th = model.theta
    g_rem = X_f_old.T @ (expit(X_f_old @ th) - y_f)
    return th + np.linalg.solve(model.compute_ggn_hessian(X_r_old, th), g_rem)


def unlearn_plu_ggn(model, X_r_old, X_r_new, X_f_old, y_r, y_f, n_steps=1):
    """
    PLU-GGN: theta + G_new^-1 (g_rem - g_shift), where g_shift is the change in the retain-set
    gradient caused by moving from phi_D to the downdated phi_{D_r}. Each extra step is a
    full (undamped) Newton step on L(theta; D_r, phi_{D_r}).
    """
    th = model.theta.copy()
    g_rem = X_f_old.T @ (expit(X_f_old @ th) - y_f)
    g_shift = X_r_new.T @ (expit(X_r_new @ th) - y_r) - X_r_old.T @ (expit(X_r_old @ th) - y_r)
    th = th + np.linalg.solve(model.compute_ggn_hessian(X_r_new, th), g_rem - g_shift)
    for _ in range(n_steps - 1):
        _, g = model.loss_and_grad(th, X_r_new, y_r)
        th = th - np.linalg.solve(model.compute_ggn_hessian(X_r_new, th), g)
    return th


# =====================================================================
# 4. METRICS AND AUDITS
# =====================================================================
def kl_divergence_bernoulli(p_true, p_approx) -> float:
    """Mean Bernoulli KL(p_true || p_approx) over rows."""
    p = np.clip(p_true, 1e-12, 1 - 1e-12)
    q = np.clip(p_approx, 1e-12, 1 - 1e-12)
    return float(np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))))


def audit_pipeline_leakage_score(prep_eval, prep_oracle, prep_full) -> float:
    """Residual memory of numeric (mu, sigma): 0 = matches retrained preprocessor, 1 = unchanged from full."""
    a = b = 0.0
    for c in prep_oracle.num_cols:
        a += abs(prep_eval.means_[c] - prep_oracle.means_[c]) + abs(prep_eval.stds_[c] - prep_oracle.stds_[c])
        b += abs(prep_full.means_[c] - prep_oracle.means_[c]) + abs(prep_full.stds_[c] - prep_oracle.stds_[c])
    return float(a / max(b, 1e-12))


def evaluate_pipeline_mia_auc(prep_eval, prep_oracle, X_forget_df, X_control_df) -> float:
    """
    Oracle-reference audit from the original notebook. Score per record:
        ||phi_oracle(x)|| - ||phi_eval(x)||   (numeric + categorical features, no intercept)
    AUC of forget (label 1) vs control (label 0), folded to [0.5, 1].
    Caveats: needs the oracle preprocessor, and it is identically 0.5 whenever
    prep_eval == prep_oracle. Its value depends strongly on how extreme the control set is.
    """
    sf = np.linalg.norm(prep_oracle.transform(X_forget_df)[:, :-1], axis=1) - np.linalg.norm(prep_eval.transform(X_forget_df)[:, :-1], axis=1)
    sc = np.linalg.norm(prep_oracle.transform(X_control_df)[:, :-1], axis=1) - np.linalg.norm(prep_eval.transform(X_control_df)[:, :-1], axis=1)
    scores = np.concatenate([sf, sc])
    if np.allclose(scores, 0.0):
        return 0.5
    auc = roc_auc_score(np.r_[np.ones(len(sf)), np.zeros(len(sc))], scores)
    return float(max(auc, 1 - auc))


# =====================================================================
# 5. ONE UNLEARNING EXPERIMENT (shared by scripts and the website)
# =====================================================================
def split_train_test(X_df, y, patient_ids, seed=RANDOM_SEED, test_size=0.20):
    """Patient-grouped 80/20 split: no patient appears in both train and test."""
    from sklearn.model_selection import GroupShuffleSplit

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X_df, y, groups=patient_ids))
    return (X_df.iloc[tr].reset_index(drop=True), y[tr], X_df.iloc[te].reset_index(drop=True), y[te])


def choose_forget_set(X_train_df, frac, mode="top_utilization", seed=0):
    n_f = max(int(frac * len(X_train_df)), 1)
    if mode == "top_utilization":
        return np.argsort(utilization_score(X_train_df))[-n_f:]
    return np.random.RandomState(seed).choice(len(X_train_df), size=n_f, replace=False)


def run_unlearning(X_train_df, y_train, X_test_df, y_test, forget_idx, l2_reg=10.0, fitted=None, timing_repeats=1):
    """
    Runs oracle retraining and every unlearning method for one deletion request.
    Returns a dict of per-method metrics plus raw objects for further analysis.
    `fitted` = (prep_full, X_train_full, model_full) lets callers reuse the original fit.
    """
    num_cols, cat_cols = NUM_COLS, CAT_COLS
    if fitted is None:
        prep_full = DowndatablePreprocessor(num_cols, cat_cols).fit(X_train_df, y_train)
        X_train_full = prep_full.transform(X_train_df)
        model_full = RegularizedLogisticModel(l2_reg).fit(X_train_full, y_train)
    else:
        prep_full, X_train_full, model_full = fitted

    r_mask = np.ones(len(X_train_df), dtype=bool)
    r_mask[forget_idx] = False
    X_r_df, y_r = X_train_df.iloc[r_mask].reset_index(drop=True), y_train[r_mask]
    X_f_df, y_f = X_train_df.iloc[forget_idx].reset_index(drop=True), y_train[forget_idx]
    X_r_old, X_f_old = X_train_full[r_mask], X_train_full[forget_idx]
    nan_mask_r = X_r_df[num_cols].isna().values
    X_test_old = prep_full.transform(X_test_df)

    def timed(fn):
        ts, out = [], None
        for _ in range(timing_repeats):
            t0 = time.perf_counter()
            out = fn()
            ts.append(time.perf_counter() - t0)
        return out, float(np.median(ts))

    def oracle():
        p = DowndatablePreprocessor(num_cols, cat_cols).fit(X_r_df, y_r)
        return p, RegularizedLogisticModel(l2_reg).fit(p.transform(X_r_df), y_r)

    (prep_orac, model_orac), t_orac = timed(oracle)
    # Polish the reference to machine precision (not timed): L-BFGS alone stops ~1e-5 from the optimum,
    # which would otherwise dominate the error of the most accurate methods.
    X_r_orac = prep_orac.transform(X_r_df)
    for _ in range(3):
        _, g = model_orac.loss_and_grad(model_orac.theta, X_r_orac, y_r)
        model_orac.theta = model_orac.theta - np.linalg.solve(model_orac.compute_ggn_hessian(X_r_orac), g)
    th_ga, t_ga = timed(lambda: unlearn_naive_gradient_ascent(model_full, X_f_old, y_f))
    th_std, t_std = timed(lambda: unlearn_standard_influence_ggn(model_full, X_r_old, X_f_old, y_f))

    def plu(n):
        def f():
            p = prep_full.downdate(X_f_df, y_f)
            Xn = fast_affine_matrix_downdate(X_r_old, nan_mask_r, prep_full, p, X_r_df[cat_cols])
            return p, unlearn_plu_ggn(model_full, X_r_old, Xn, X_f_old, y_r, y_f, n_steps=n)
        return f

    (prep_plu, th_plu1), t_plu1 = timed(plu(1))
    (_, th_plu2), t_plu2 = timed(plu(2))
    X_test_plu = prep_plu.transform(X_test_df)
    X_test_orac = prep_orac.transform(X_test_df)
    p_orac = model_orac.predict_proba(X_test_orac)

    from sklearn.metrics import average_precision_score, brier_score_loss

    methods = {
        "oracle": (model_orac.theta, X_test_orac, prep_orac, t_orac),
        "before": (model_full.theta, X_test_old, prep_full, None),
        "grad_ascent": (th_ga, X_test_old, prep_full, t_ga),
        "std_influence": (th_std, X_test_old, prep_full, t_std),
        "plu_1": (th_plu1, X_test_plu, prep_plu, t_plu1),
        "plu_2": (th_plu2, X_test_plu, prep_plu, t_plu2),
    }
    out = {"n_forget": int(len(forget_idx)), "n_retain": int(r_mask.sum()), "n_test": int(len(y_test)),
           "test_prevalence": float(np.mean(y_test)), "methods": {}}
    for k, (th, Xt, prep, t) in methods.items():
        p = expit(Xt @ th)
        out["methods"][k] = {
            "l2_err": float(np.linalg.norm(th - model_orac.theta)),
            "test_kl": kl_divergence_bernoulli(p_orac, p),
            "max_abs_dp": float(np.max(np.abs(p - p_orac))),
            "pipe_leak": audit_pipeline_leakage_score(prep, prep_orac, prep_full),
            "pr_auc": float(average_precision_score(y_test, p)),
            "roc_auc": float(roc_auc_score(y_test, p)),
            "brier": float(brier_score_loss(y_test, p)),
            "seconds": t,
            "speedup": (t_orac / t) if t else None,
        }
    out["_objects"] = dict(prep_full=prep_full, prep_orac=prep_orac, prep_plu=prep_plu, model_full=model_full,
                           model_orac=model_orac, theta_std=th_std, theta_plu1=th_plu1, theta_plu2=th_plu2,
                           X_f_df=X_f_df, X_r_df=X_r_df, p_orac=p_orac, X_test_plu=X_test_plu, X_test_old=X_test_old)
    return out


if __name__ == "__main__":
    X_df, y, pid = load_clinical_readmission_data()
    Xtr, ytr, Xte, yte = split_train_test(X_df, y, pid)
    res = run_unlearning(Xtr, ytr, Xte, yte, choose_forget_set(Xtr, 0.03))
    print(f"|D_f|={res['n_forget']} |D_r|={res['n_retain']} |D_test|={res['n_test']}")
    for k, m in res["methods"].items():
        sp = f"{m['speedup']:.1f}x" if m["speedup"] else "-"
        print(f"{k:<14} L2={m['l2_err']:.4e} KL={m['test_kl']:.3e} leak={m['pipe_leak']*100:5.1f}% PR-AUC={m['pr_auc']:.4f} speedup={sp}")
