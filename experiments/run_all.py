"""
Reproduces every number and figure in the paper and on the website.

    python experiments/run_all.py            # ~2-4 min on a laptop CPU

Writes:
    results/results.json           all metrics
    docs/data/results.json         same file, read by the website
    paper/figures/fig_ablation.pdf main figure
"""

import json
import os
import platform
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from plu_ggn import (  # noqa: E402
    CAT_COLS, NUM_COLS, DowndatablePreprocessor, RegularizedLogisticModel, choose_forget_set,
    evaluate_pipeline_mia_auc, load_clinical_readmission_data, run_unlearning, split_train_test,
    utilization_score,
)

FRACTIONS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["before", "grad_ascent", "std_influence", "plu_1", "plu_2"]


def strip(res):
    return {k: v for k, v in res.items() if k != "_objects"}


def fit_full(Xtr, ytr):
    prep = DowndatablePreprocessor(NUM_COLS, CAT_COLS).fit(Xtr, ytr)
    Xf = prep.transform(Xtr)
    return prep, Xf, RegularizedLogisticModel(10.0).fit(Xf, ytr)


def matched_controls(X_f_df, X_test_df, min_n=30):
    """Held-out test encounters at least as extreme (utilization) as the least extreme deleted one."""
    thr = utilization_score(X_f_df).min()
    u = utilization_score(X_test_df)
    idx = np.where(u >= thr)[0]
    if len(idx) < min_n:
        idx = np.argsort(u)[-min_n:]
    return X_test_df.iloc[idx].reset_index(drop=True)


def patient_overlap(X_df, y, pid, seed=42):
    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X_df, y, groups=pid))
    return int(len(set(pid[tr]) & set(pid[te])))


def main():
    out = {"machine": f"{platform.processor() or platform.machine()} / Python {platform.python_version()}"}
    X_df, y, pid = load_clinical_readmission_data()
    Xtr, ytr, Xte, yte = split_train_test(X_df, y, pid, seed=42)
    fitted = fit_full(Xtr, ytr)
    prep_full = fitted[0]

    # ---------------- data summary ----------------
    out["data"] = {
        "n_encounters": int(len(X_df)), "n_patients": int(len(np.unique(pid))),
        "prevalence": float(y.mean()), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "train_prevalence": float(ytr.mean()), "test_prevalence": float(yte.mean()),
        "patient_overlap": patient_overlap(X_df, y, pid),
        "numeric_summary": {
            c: {"mean": float(np.nanmean(Xtr[c])), "median": float(np.nanmedian(Xtr[c])),
                "p99": float(np.nanpercentile(Xtr[c], 99)), "max": float(np.nanmax(Xtr[c])),
                "missing": float(Xtr[c].isna().mean())}
            for c in NUM_COLS
        },
        "medical_specialty_missing": float((X_df["medical_specialty"] == "Missing").mean()),
    }

    # ---------------- simple predictive baselines on the held-out test set ----------------
    u_te = utilization_score(Xte)
    out["predictive_baselines"] = {
        "prevalence_pr_auc": float(yte.mean()),
        "number_inpatient_only": {
            "pr_auc": float(average_precision_score(yte, Xte["number_inpatient"].fillna(0))),
            "roc_auc": float(roc_auc_score(yte, Xte["number_inpatient"].fillna(0))),
        },
        "utilization_score_only": {
            "pr_auc": float(average_precision_score(yte, u_te)), "roc_auc": float(roc_auc_score(yte, u_te)),
        },
    }

    # ---------------- main table: 3% top-utilization deletion ----------------
    f_idx = choose_forget_set(Xtr, 0.03)
    main_res = run_unlearning(Xtr, ytr, Xte, yte, f_idx, fitted=fitted, timing_repeats=7)
    out["main"] = strip(main_res)
    o = main_res["_objects"]

    # coefficient-level view: which coefficients move most (coordinate-frame story)
    names = NUM_COLS + CAT_COLS + ["intercept"]
    th_o, th_b, th_s, th_p = o["model_orac"].theta, o["model_full"].theta, o["theta_std"], o["theta_plu1"]
    out["coefficients"] = [
        {"feature": n, "oracle": float(a), "before": float(b), "std_influence": float(s), "plu_1": float(p)}
        for n, a, b, s, p in zip(names, th_o, th_b, th_s, th_p)
    ]
    out["preprocessor_shift"] = {
        c: {"mu_full": prep_full.means_[c], "mu_retain": o["prep_orac"].means_[c],
            "sd_full": prep_full.stds_[c], "sd_retain": o["prep_orac"].stds_[c]}
        for c in NUM_COLS
    }
    # differencing attack: if phi_D and phi_{D_r} statistics are both exposed, forget-cohort mean is exact
    c = "number_inpatient"
    n_f_obs = prep_full.n_obs_[c] - o["prep_orac"].n_obs_[c]
    out["differencing"] = {
        "feature": c,
        "recovered_forget_mean": (prep_full.sum_x_[c] - o["prep_orac"].sum_x_[c]) / n_f_obs,
        "true_forget_mean": float(o["X_f_df"][c].mean()),
        "retain_mean": float(o["X_r_df"][c].mean()),
    }
    # representative prediction errors of PLU-1 vs oracle on the test set
    p_plu1 = 1 / (1 + np.exp(-(o["X_test_plu"] @ th_p)))
    p_std = 1 / (1 + np.exp(-(o["X_test_old"] @ th_s)))
    d1, ds = np.abs(p_plu1 - o["p_orac"]), np.abs(p_std - o["p_orac"])
    top = np.argsort(d1)[-5:][::-1]
    out["errors"] = {
        "plu1_abs_dp_p50_p99_max": [float(np.median(d1)), float(np.percentile(d1, 99)), float(d1.max())],
        "std_abs_dp_p50_p99_max": [float(np.median(ds)), float(np.percentile(ds, 99)), float(ds.max())],
        "spearman_dp_vs_utilization_plu1": float(
            np.corrcoef(np.argsort(np.argsort(d1)), np.argsort(np.argsort(u_te)))[0, 1]),
        "top5_plu1": [
            {"abs_dp": float(d1[i]), "p_oracle": float(o["p_orac"][i]), "p_plu1": float(p_plu1[i]),
             "number_inpatient": None if np.isnan(Xte["number_inpatient"].iloc[i]) else float(Xte["number_inpatient"].iloc[i]),
             "num_lab_procedures": None if np.isnan(Xte["num_lab_procedures"].iloc[i]) else float(Xte["num_lab_procedures"].iloc[i]),
             "utilization_pct": float((u_te < u_te[i]).mean() * 100)}
            for i in top
        ],
    }

    # ---------------- ablation over deletion fraction, outlier vs random cohorts ----------------
    out["ablation"] = {"top_utilization": [], "random": []}
    for mode in ["top_utilization", "random"]:
        for frac in FRACTIONS:
            idx = choose_forget_set(Xtr, frac, mode=mode, seed=0)
            r = run_unlearning(Xtr, ytr, Xte, yte, idx, fitted=fitted, timing_repeats=5)
            rec = {"frac": frac * 100, "n_forget": r["n_forget"]}
            for m in METHODS:
                rec[m] = {k: r["methods"][m][k] for k in ["l2_err", "test_kl", "speedup", "pipe_leak"]}
            ob = r["_objects"]
            ctrl = matched_controls(ob["X_f_df"], Xte)
            rec["mia"] = {
                "std_vs_all_test": evaluate_pipeline_mia_auc(prep_full, ob["prep_orac"], ob["X_f_df"], Xte),
                "std_vs_matched": evaluate_pipeline_mia_auc(prep_full, ob["prep_orac"], ob["X_f_df"], ctrl),
                "plu_vs_all_test": evaluate_pipeline_mia_auc(ob["prep_plu"], ob["prep_orac"], ob["X_f_df"], Xte),
                "n_matched": int(len(ctrl)),
                "utilization_only_vs_all_test": float(roc_auc_score(
                    np.r_[np.ones(len(ob["X_f_df"])), np.zeros(len(Xte))],
                    np.r_[utilization_score(ob["X_f_df"]), u_te])),
            }
            out["ablation"][mode].append(rec)
            print(f"{mode:<16} {frac*100:4.1f}%  StdInf L2={rec['std_influence']['l2_err']:.3e} "
                  f"PLU1={rec['plu_1']['l2_err']:.3e} PLU2={rec['plu_2']['l2_err']:.3e} "
                  f"MIA all={rec['mia']['std_vs_all_test']:.3f} matched={rec['mia']['std_vs_matched']:.3f} "
                  f"(n={rec['mia']['n_matched']})")

    # ---------------- robustness over patient-grouped split / missingness seeds ----------------
    seeds = []
    for s in SEEDS:
        Xd, yy, pp = load_clinical_readmission_data(mask_seed=s)
        a, b, c_, d = split_train_test(Xd, yy, pp, seed=s)
        r = run_unlearning(a, b, c_, d, choose_forget_set(a, 0.03), timing_repeats=3)
        seeds.append({"seed": s, **{m: {k: r["methods"][m][k] for k in ["l2_err", "test_kl", "pr_auc", "roc_auc", "speedup"]}
                                      for m in ["oracle"] + METHODS}})
        print(f"seed {s}: StdInf L2={seeds[-1]['std_influence']['l2_err']:.3e} before={seeds[-1]['before']['l2_err']:.3e} "
              f"PLU1={seeds[-1]['plu_1']['l2_err']:.3e} PR-AUC={seeds[-1]['oracle']['pr_auc']:.4f}")

    def agg(m, k):
        v = np.array([x[m][k] for x in seeds], dtype=float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}

    out["seeds"] = {"runs": seeds, "summary": {m: {k: agg(m, k) for k in ["l2_err", "test_kl", "pr_auc", "roc_auc"]}
                                                 for m in ["oracle"] + METHODS}}
    out["seeds"]["summary"]["plu_1_speedup"] = agg("plu_1", "speedup")

    for p in [os.path.join(ROOT, "results"), os.path.join(ROOT, "docs", "data")]:
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "results.json"), "w") as f:
            json.dump(out, f, indent=1, default=float)
    make_figure(out)
    print("wrote results/results.json, docs/data/results.json, paper/figures/fig_ablation.pdf")


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    col = {"before": "#555555", "std_influence": "#d95f02", "plu_1": "#1b9e77", "plu_2": "#7570b3"}
    lab = {"before": "No unlearning", "std_influence": r"Std. influence (frozen $\phi$)",
           "plu_1": "PLU-GGN 1-step", "plu_2": "PLU-GGN 2-step"}
    A = out["ablation"]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.45))
    x = [r["frac"] for r in A["top_utilization"]]
    for m in ["before", "std_influence", "plu_1", "plu_2"]:
        ax[0].plot(x, [r[m]["l2_err"] for r in A["top_utilization"]], marker="o", ms=3, color=col[m], label=lab[m],
                   ls="--" if m == "before" else "-")
        ax[0].plot(x, [r[m]["l2_err"] for r in A["random"]], marker="o", ms=2, color=col[m], alpha=0.35, ls=":")
        ax[1].plot(x, [r[m]["test_kl"] for r in A["top_utilization"]], marker="o", ms=3, color=col[m],
                   ls="--" if m == "before" else "-")
    for a in ax[:2]:
        a.set_yscale("log")
        a.set_xlabel("deleted cohort (% of train)")
    ax[0].set_ylabel(r"$\|\theta-\theta_r^*\|_2$")
    ax[0].set_title("(a) weight error (faint: random cohort)", fontsize=8)
    ax[1].set_ylabel("mean test KL to oracle")
    ax[1].set_title("(b) function-space error", fontsize=8)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    ax[2].plot(x, [r["mia"]["std_vs_all_test"] for r in A["top_utilization"]], marker="s", ms=3, color=col["std_influence"],
               label="vs. random test")
    ax[2].plot(x, [r["mia"]["std_vs_matched"] for r in A["top_utilization"]], marker="s", ms=3, color=col["std_influence"],
               ls=":", label="vs. matched test")
    ax[2].plot(x, [r["mia"]["utilization_only_vs_all_test"] for r in A["top_utilization"]], marker="x", ms=3, color="#888",
               label="utilization alone")
    ax[2].axhline(0.5, color="#bbb", lw=0.8)
    ax[2].set_ylim(0.45, 1.02)
    ax[2].set_xlabel("deleted cohort (% of train)")
    ax[2].set_ylabel("audit AUC")
    ax[2].set_title("(c) frozen-$\\phi$ audit AUC", fontsize=8)
    ax[2].legend(fontsize=6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.68))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    os.makedirs(os.path.join(ROOT, "paper", "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "paper", "figures", "fig_ablation.pdf"))
    fig.savefig(os.path.join(ROOT, "results", "fig_ablation.png"), dpi=200)


if __name__ == "__main__":
    main()
