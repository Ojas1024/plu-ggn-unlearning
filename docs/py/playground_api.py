"""
JSON-in / JSON-out glue between the website playground (docs/playground.html, run in Pyodide)
and plu_ggn.py. Every public function takes/returns JSON strings so JavaScript can call it directly.

State lives in module globals: the raw table, the configured train/test split, and the fitted
full model, so repeated deletion requests do not refit everything.
"""

import json

import numpy as np
import pandas as pd

import plu_ggn as P

MAX_ROWS = 60000
_S = {}  # session state


def _ok(**kw):
    return json.dumps({"ok": True, **kw}, default=_np)


def _err(msg):
    return json.dumps({"ok": False, "error": str(msg)})


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------- 1. load a table
def load_csv(path, sample_seed=42):
    """Read a CSV as strings, sample down to MAX_ROWS, and describe each column for the UI."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[], low_memory=False)
    except Exception as e:  # noqa: BLE001
        return _err(f"Could not read the CSV: {e}")
    if df.shape[1] < 2:
        return _err("The table needs at least two columns (features and a target).")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        return _err("Column names must be unique.")
    n_orig = len(df)
    if n_orig > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=sample_seed).reset_index(drop=True)
    if len(df) < 50:
        return _err("Need at least 50 rows.")
    _S.clear()
    _S["raw"] = df
    cols = []
    for c in df.columns:
        v = df[c]
        blank = v.str.strip().isin(["", "?", "nan", "NaN", "NA", "None", "null"])
        num = pd.to_numeric(v[~blank], errors="coerce")
        frac_num = float(num.notna().mean()) if (~blank).any() else 0.0
        nun = int(v.nunique())
        guess = "numeric" if frac_num > 0.95 and nun > 10 else "categorical"
        if nun == len(df) and guess == "categorical":
            guess = "id"
        top = v.value_counts().head(30)
        cols.append({"name": c, "guess": guess, "n_unique": nun, "missing": float(blank.mean()),
                     "numeric_frac": frac_num, "top_values": [[str(k), int(n)] for k, n in top.items()],
                     "examples": [str(x) for x in v.head(3)]})
    return _ok(n_rows=len(df), n_rows_original=n_orig, columns=cols)


# ---------------------------------------------------------------- 2. configure + fit
def configure(cfg_json):
    """
    cfg = {target, positive, num_cols, cat_cols, group_col|null, missing_rate, seed, test_size, l2}
    Builds features, makes the (grouped) split, fits the full pipeline and model.
    """
    try:
        cfg = json.loads(cfg_json)
        df = _S["raw"]
        num, cat = list(cfg.get("num_cols", [])), list(cfg.get("cat_cols", []))
        if cfg["target"] in num + cat:
            return _err("The target cannot also be a feature.")
        if not num and not cat:
            return _err("Pick at least one feature column.")
        X_df, y = P.prepare_dataset(df, cfg["target"], cfg["positive"], num, cat,
                                    float(cfg.get("missing_rate", 0.0)), int(cfg.get("seed", 42)))
        for c in num:
            if X_df[c].isna().all():
                return _err(f"Column '{c}' has no numeric values; mark it categorical or drop it.")
        if y.sum() < 10 or (1 - y).sum() < 10:
            return _err(f"Need at least 10 rows of each class; found {int(y.sum())} positive / {int((1 - y).sum())} negative.")
        g = cfg.get("group_col")
        if g:
            gnum = pd.to_numeric(df[g], errors="coerce")
            groups = gnum.values if gnum.notna().all() else df[g].astype(str).values  # numeric ids sort numerically
        else:
            groups = np.arange(len(df))
        from sklearn.model_selection import GroupShuffleSplit

        tr, te = next(GroupShuffleSplit(1, test_size=float(cfg.get("test_size", 0.2)),
                                        random_state=int(cfg.get("seed", 42))).split(X_df, y, groups))
        Xtr, Xte = X_df.iloc[tr].reset_index(drop=True), X_df.iloc[te].reset_index(drop=True)
        ytr, yte = y[tr], y[te]
        if len(np.unique(yte)) < 2:
            return _err("The test split has only one class; use a different seed or a larger table.")
        l2 = float(cfg.get("l2", 10.0))
        prep = P.DowndatablePreprocessor(num, cat).fit(Xtr, ytr)
        Xf = prep.transform(Xtr)
        model = P.RegularizedLogisticModel(l2).fit(Xf, ytr)
        _S.update(cfg=cfg, num=num, cat=cat, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, l2=l2,
                  raw_tr=df.iloc[tr].reset_index(drop=True), fitted=(prep, Xf, model))
        from sklearn.metrics import average_precision_score, roc_auc_score

        p = model.predict_proba(prep.transform(Xte))
        overlap = len(set(groups[tr]) & set(groups[te])) if g else 0
        return _ok(n_train=len(tr), n_test=len(te), prevalence=float(y.mean()), group_overlap=overlap,
                   test_pr_auc=float(average_precision_score(yte, p)), test_roc_auc=float(roc_auc_score(yte, p)))
    except KeyError as e:
        return _err(f"Missing setting or column: {e}")
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------- 3. forget-set selection
_OPS = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, ">": lambda a, b: a > b,
        "<": lambda a, b: a < b, ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b}


def _score_column(col):
    if col == "__utilization__":
        return P.utilization_score(_S["Xtr"])
    return pd.to_numeric(_S["raw_tr"][col], errors="coerce").fillna(-np.inf).values


def _forget_indices(spec):
    n = len(_S["Xtr"])
    mode = spec["mode"]
    if mode == "top":
        k = max(int(float(spec["frac"]) * n), 1)
        s = _score_column(spec["column"])
        order = np.argsort(s)  # same tie-breaking as the paper's choose_forget_set
        if spec.get("direction", "high") == "low":
            s2 = np.where(np.isfinite(s), s, np.inf)
            order = np.argsort(-s2, kind="stable")
        return order[-k:]
    if mode == "random":
        k = max(int(float(spec["frac"]) * n), 1)
        return np.random.RandomState(int(spec.get("seed", 0))).choice(n, size=k, replace=False)
    if mode == "filter":
        raw = _S["raw_tr"]
        m = np.ones(n, dtype=bool)
        for c in spec["conditions"]:
            col, op, val = c["col"], c["op"], str(c["value"])
            if op in ("==", "!="):
                m &= _OPS[op](raw[col].astype(str).str.strip().values, val.strip())
            else:
                a = pd.to_numeric(raw[col], errors="coerce").values
                try:
                    b = float(val)
                except ValueError:
                    raise ValueError(f"'{val}' is not a number (needed for {op} on {col}).")
                m &= np.nan_to_num(_OPS[op](a, b), nan=False).astype(bool)
        return np.where(m)[0]
    if mode == "pick":
        idx = np.unique(np.asarray(spec["rows"], dtype=int))
        return idx[(idx >= 0) & (idx < n)]
    raise ValueError(f"unknown mode {mode}")


def _check_forget(idx):
    n = len(_S["Xtr"])
    if len(idx) == 0:
        return "The deletion request matches no training rows."
    if n - len(idx) < 20:
        return "The deletion request removes almost all training rows."
    keep = np.ones(n, bool)
    keep[idx] = False
    yr = _S["ytr"][keep]
    if yr.sum() < 2 or (1 - yr).sum() < 2:
        return "After deletion the retained data has (almost) only one class."
    return None


def preview_forget(spec_json):
    """How many rows a request selects, and how the deleted rows differ from the retained ones."""
    try:
        idx = _forget_indices(json.loads(spec_json))
        msg = _check_forget(idx)
        keep = np.ones(len(_S["Xtr"]), bool)
        keep[idx] = False
        Xtr = _S["Xtr"]
        comp = []
        for c in _S["num"]:
            comp.append({"feature": c, "kind": "mean", "forget": float(np.nanmean(Xtr[c].values[idx])) if len(idx) else None,
                         "retain": float(np.nanmean(Xtr[c].values[keep]))})
        for c in _S["cat"]:
            top = Xtr[c][keep].value_counts().index[0]
            comp.append({"feature": c, "kind": f"share '{top}'",
                         "forget": float((Xtr[c].values[idx] == top).mean()) if len(idx) else None,
                         "retain": float((Xtr[c].values[keep] == top).mean())})
        return _ok(n_forget=int(len(idx)), n_train=int(len(Xtr)), warning=msg,
                   forget_positive_rate=float(_S["ytr"][idx].mean()) if len(idx) else None,
                   retain_positive_rate=float(_S["ytr"][keep].mean()), comparison=comp)
    except Exception as e:  # noqa: BLE001
        return _err(e)


def rows_page(req_json):
    """Paged, sortable, searchable view of training rows for hand-picking. req = {page, size, sort, desc, query}"""
    try:
        r = json.loads(req_json)
        raw = _S["raw_tr"]
        cols = [_S["cfg"]["target"]] + _S["num"] + _S["cat"]
        view = raw[cols].copy()
        view.insert(0, "#", np.arange(len(raw)))
        q = str(r.get("query", "")).strip().lower()
        if q:
            hit = np.zeros(len(view), bool)
            for c in cols:
                hit |= view[c].astype(str).str.lower().str.contains(q, regex=False).values
            view = view[hit]
        s = r.get("sort")
        if s and s in view.columns:
            key = pd.to_numeric(view[s], errors="coerce")
            key = key if key.notna().mean() > 0.9 else view[s].astype(str)
            view = view.iloc[np.argsort(key.values, kind="stable")[::-1] if r.get("desc") else np.argsort(key.values, kind="stable")]
        size, page = int(r.get("size", 20)), int(r.get("page", 0))
        part = view.iloc[page * size:(page + 1) * size]
        return _ok(columns=["#"] + cols, total=int(len(view)),
                   rows=[[int(x[0])] + [str(v) for v in x[1:]] for x in part.itertuples(index=False)])
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------- 4. run unlearning
def run(spec_json):
    try:
        spec = json.loads(spec_json)
        idx = _forget_indices(spec)
        msg = _check_forget(idx)
        if msg:
            return _err(msg)
        res = P.run_unlearning(_S["Xtr"], _S["ytr"], _S["Xte"], _S["yte"], idx, l2_reg=_S["l2"], fitted=_S["fitted"],
                               timing_repeats=3, num_cols=_S["num"], cat_cols=_S["cat"])
        o = res.pop("_objects")
        names = _S["num"] + _S["cat"] + ["intercept"]
        res["coef"] = {"names": names, "oracle": o["model_orac"].theta, "before": o["model_full"].theta,
                       "std_influence": o["theta_std"], "plu_1": o["theta_plu1"], "plu_2": o["theta_plu2"]}
        pf, po = o["prep_full"], o["prep_orac"]
        res["pipeline"] = [{"feature": c, "mu_full": pf.means_[c], "mu_retain": po.means_[c],
                            "sd_full": pf.stds_[c], "sd_retain": po.stds_[c]} for c in _S["num"]]
        res["prior"] = {"full": pf.global_prior_, "retain": po.global_prior_}
        return _ok(result=res)
    except Exception as e:  # noqa: BLE001
        return _err(e)


def sweep(spec_json):
    """Re-run a 'top' or 'random' request over several deletion fractions."""
    try:
        spec = json.loads(spec_json)
        out = []
        for f in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]:
            s = dict(spec, frac=f)
            idx = _forget_indices(s)
            if _check_forget(idx):
                continue
            r = P.run_unlearning(_S["Xtr"], _S["ytr"], _S["Xte"], _S["yte"], idx, l2_reg=_S["l2"], fitted=_S["fitted"],
                                 num_cols=_S["num"], cat_cols=_S["cat"])
            out.append({"frac": f * 100, "n_forget": int(len(idx)),
                        **{m: {"l2_err": r["methods"][m]["l2_err"], "test_kl": r["methods"][m]["test_kl"]}
                           for m in ["before", "std_influence", "plu_1", "plu_2"]}})
        return _ok(sweep=out)
    except Exception as e:  # noqa: BLE001
        return _err(e)


DIABETES_PRESET = {
    "target": "readmitted", "positive": "<30", "num_cols": P.NUM_COLS, "cat_cols": P.CAT_COLS,
    "group_col": "patient_nbr", "missing_rate": 0.12, "seed": 42, "test_size": 0.2, "l2": 10.0,
}


def preset():
    return json.dumps(DIABETES_PRESET)
