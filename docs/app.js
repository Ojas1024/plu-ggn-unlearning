/* Front-end for the PLU-GGN project page. Precomputed results come from data/results.json
   (written by experiments/run_all.py); the live section runs py/plu_ggn.py in Pyodide. */
(() => {
  "use strict";
  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const C = () => ({
    before: css("--grey"), grad_ascent: "#b8860b", std_influence: css("--warn"),
    plu_1: css("--accent"), plu_2: css("--purple"), ink: css("--ink"), muted: css("--muted"), line: css("--line"),
  });
  const NAMES = {
    oracle: "Oracle retrain", before: "No unlearning", grad_ascent: "Gradient ascent (frozen φ)",
    std_influence: "Standard influence (frozen φ)", plu_1: "PLU-GGN 1-step (ours)", plu_2: "PLU-GGN 2-step (ours)",
  };
  const ORDER = ["oracle", "before", "grad_ascent", "std_influence", "plu_1", "plu_2"];

  const sci = (x, d = 2) => {
    if (x === null || x === undefined) return "–";
    if (x === 0) return "0";
    const e = Math.floor(Math.log10(Math.abs(x)));
    if (e >= -2 && e < 3) return String(Number(x.toPrecision(d + 1)));
    return `${(x / 10 ** e).toFixed(d)}×10<sup>${e}</sup>`;
  };
  const fmt = {
    pct: (x) => `${(x * 100).toFixed(0)}%`, f2: (x) => x.toFixed(2), f3: (x) => x.toFixed(3),
    sci: (x) => sci(x), int: (x) => x.toLocaleString("en-US"),
  };
  const get = (o, path) => path.split(".").reduce((a, k) => (a == null ? a : a[k]), o);

  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  const baseOpts = () => {
    const c = C();
    Chart.defaults.color = c.muted;
    Chart.defaults.borderColor = c.line;
    return { responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
      plugins: { legend: { labels: { boxWidth: 12, usePointStyle: true } } } };
  };

  // ---------------- precomputed results ----------------
  let R = null;
  const charts = {};

  function fillSpans() {
    document.querySelectorAll("[data-k]").forEach((el) => {
      const v = get(R, el.dataset.k);
      if (v === undefined || v === null) return;
      el.innerHTML = el.dataset.fmt ? fmt[el.dataset.fmt](v) : fmt.int(v);
    });
  }

  function hero() {
    const m = R.main.methods;
    const items = [
      [`${(m.std_influence.l2_err / m.plu_1.l2_err).toFixed(0)}×`, "lower weight error than standard influence (PLU-GGN, 1 step, 3% deletion)"],
      [sci(m.plu_2.l2_err, 1), "weight error vs. full retraining after 2 steps"],
      [`${R.preprocessor_shift.number_inpatient.sd_full.toFixed(2)} → ${R.preprocessor_shift.number_inpatient.sd_retain.toFixed(2)}`, "σ of number_inpatient: what a frozen preprocessor keeps vs. retraining"],
      [`${m.plu_1.speedup.toFixed(1)}×`, "faster than retraining pipeline + model (this CPU)"],
    ];
    document.getElementById("hero-stats").innerHTML = items.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  }

  function tailsChart() {
    const ns = R.data.numeric_summary, cols = Object.keys(ns), c = C();
    charts.tails = new Chart(document.getElementById("c-tails"), {
      type: "bar",
      data: { labels: cols.map((s) => s.replace("number_", "n_").replace("num_", "n_")),
        datasets: [
          { label: "median", data: cols.map((k) => (ns[k].median > 0 ? ns[k].median : null)), backgroundColor: c.muted },
          { label: "99th pct", data: cols.map((k) => ns[k].p99), backgroundColor: c.plu_1 },
          { label: "max", data: cols.map((k) => ns[k].max), backgroundColor: c.std_influence },
        ] },
      options: { ...baseOpts(), indexAxis: "y", scales: { x: { type: "logarithmic", title: { display: true, text: "value (log scale)" } } } },
    });
  }

  function mainTable() {
    const m = R.main.methods, s = R.seeds.summary;
    const head = `<thead><tr><th>Method</th><th>Weight error ↓</th><th>Test KL ↓</th><th>max |Δp|</th><th>Pipe leak</th><th>PR-AUC</th><th>Speedup</th><th>Weight error, 5 seeds</th></tr></thead>`;
    const rows = ORDER.map((k) => {
      const r = m[k], ss = s[k];
      const seed = ss ? (ss.l2_err.mean === 0 ? "0" : `${sci(ss.l2_err.mean, 2)} ± ${sci(ss.l2_err.sd, 1)}`) : "–";
      const sp = k === "oracle" ? `1× (${(r.seconds * 1000).toFixed(0)} ms)` : r.speedup ? `${r.speedup >= 100 ? r.speedup.toFixed(0) : r.speedup.toFixed(1)}×` : "–";
      return `<tr class="${k.startsWith("plu") ? "ours" : ""}"><td>${NAMES[k]}</td><td>${sci(r.l2_err)}</td><td>${sci(r.test_kl, 1)}</td><td>${sci(r.max_abs_dp, 2)}</td><td>${(r.pipe_leak * 100).toFixed(0)}%</td><td>${r.pr_auc.toFixed(3)}</td><td>${sp}</td><td>${seed}</td></tr>`;
    }).join("");
    document.getElementById("main-table").innerHTML = head + `<tbody>${rows}</tbody>`;
  }

  const sweep = { mode: "top_utilization", metric: "l2_err" };
  function sweepChart() {
    const A = R.ablation[sweep.mode], c = C();
    const ds = ["before", "std_influence", "plu_1", "plu_2"].map((k) => ({
      label: NAMES[k], data: A.map((r) => Math.max(r[k][sweep.metric], 1e-16)),
      borderColor: c[k], backgroundColor: c[k], borderDash: k === "before" ? [5, 4] : [], tension: 0.2, pointRadius: 3,
    }));
    const opts = { ...baseOpts(), scales: {
      x: { title: { display: true, text: "deleted cohort (% of training encounters)" } },
      y: { type: "logarithmic", title: { display: true, text: sweep.metric === "l2_err" ? "‖θ − θ*ᵣ‖₂ (log)" : "mean test KL to oracle (log)" },
        ticks: { callback: (v) => { const e = Math.log10(v); return Math.abs(e - Math.round(e)) < 1e-9 ? `1e${Math.round(e)}` : ""; } } } },
      plugins: { ...baseOpts().plugins, tooltip: { callbacks: { label: (t) => `${t.dataset.label}: ${t.parsed.y.toExponential(2)}` } } } };
    if (charts.sweep) charts.sweep.destroy();
    charts.sweep = new Chart(document.getElementById("c-sweep"), { type: "line", data: { labels: A.map((r) => `${r.frac}%`), datasets: ds }, options: opts });
    const worse = A.filter((r) => r.std_influence.l2_err > r.before.l2_err).map((r) => `${r.frac}%`);
    document.getElementById("sweep-note").textContent = sweep.metric === "l2_err"
      ? (worse.length ? `Standard influence ends further from retraining than doing nothing at: ${worse.join(", ")}.` : "Standard influence beats doing nothing at every size here.")
      : "In prediction space every method improves on doing nothing; PLU-GGN is several orders of magnitude closer to retraining.";
  }

  let auditKey = "std_vs_all_test";
  function auditChart() {
    const A = R.ablation.top_utilization, c = C();
    const data = {
      labels: A.map((r) => `${r.frac}%`),
      datasets: [
        { type: "bar", label: auditKey === "std_vs_all_test" ? "frozen φ audit vs. random test patients" : "frozen φ audit vs. equally extreme test patients",
          data: A.map((r) => r.mia[auditKey]), backgroundColor: c.std_influence },
        { type: "line", label: "utilization score alone (no model at all)", data: A.map((r) => r.mia.utilization_only_vs_all_test),
          borderColor: c.muted, backgroundColor: c.muted, pointRadius: 3, hidden: auditKey !== "std_vs_all_test" },
        { type: "line", label: "chance", data: A.map(() => 0.5), borderColor: c.line, pointRadius: 0, borderDash: [4, 4] },
      ],
    };
    if (charts.audit) charts.audit.destroy();
    charts.audit = new Chart(document.getElementById("c-audit"), { data, options: { ...baseOpts(),
      scales: { y: { min: 0.4, max: 1, title: { display: true, text: "audit AUC" } }, x: { title: { display: true, text: "deleted cohort size" } } } } });
    document.getElementById("audit-note").textContent = auditKey === "std_vs_all_test"
      ? "Looks like near-perfect leakage, but the grey line shows the utilization score alone gets the same AUC."
      : `Against test patients at least as extreme (n = ${A.map((r) => r.mia.n_matched).join(", ")}), the audit is close to chance for deletions ≥ 2%. At 0.5–1% only 30–41 controls exist and they are slightly less extreme.`;
  }

  function errTable() {
    const rows = R.errors.top5_plu1.map((e) => `<tr><td>top ${(100 - e.utilization_pct).toFixed(2)}% utilization</td><td>${e.number_inpatient ?? "missing"}</td><td>${e.num_lab_procedures ?? "missing"}</td><td>${e.p_oracle.toFixed(3)}</td><td>${e.p_plu1.toFixed(3)}</td><td>${e.abs_dp.toFixed(4)}</td></tr>`).join("");
    document.getElementById("err-table").innerHTML = `<thead><tr><th>Held-out patient</th><th>prior inpatient</th><th>lab procedures</th><th>p (oracle)</th><th>p (PLU-1)</th><th>|Δp|</th></tr></thead><tbody>${rows}</tbody>`;
  }

  function wireSegs() {
    document.querySelectorAll(".seg").forEach((seg) => seg.addEventListener("click", (ev) => {
      const b = ev.target.closest("button");
      if (!b) return;
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      if (b.dataset.mode) { sweep.mode = b.dataset.mode; sweepChart(); }
      if (b.dataset.metric) { sweep.metric = b.dataset.metric; sweepChart(); }
      if (b.dataset.ctrl) { auditKey = b.dataset.ctrl; auditChart(); }
    }));
  }

  function renderAll() { fillSpans(); hero(); tailsChart(); mainTable(); sweepChart(); auditChart(); errTable(); }

  fetch("data/results.json").then((r) => r.json()).then((j) => { R = j; renderAll(); wireSegs(); })
    .catch((e) => { document.getElementById("hero-stats").textContent = "Could not load results.json: " + e; });

  // repository link: derive from *.github.io/<repo>/ when hosted on GitHub Pages
  (() => {
    const m = location.hostname.match(/^([^.]+)\.github\.io$/);
    const repo = location.pathname.split("/").filter(Boolean)[0];
    if (m && repo) {
      const url = `https://github.com/${m[1]}/${repo}`;
      document.getElementById("repo-link").href = url;
      document.querySelectorAll(".repo-url").forEach((el) => (el.textContent = url));
    }
  })();

  // ---------------- live Pyodide run ----------------
  const status = document.getElementById("live-status");
  const frac = document.getElementById("i-frac");
  frac.addEventListener("input", () => (document.getElementById("o-frac").textContent = `${Number(frac.value).toFixed(1)}%`));
  let pyReady = null;

  function loadScript(src) {
    return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error("failed to load " + src)); document.head.appendChild(s); });
  }
  async function initPy() {
    status.textContent = "Downloading Python runtime (Pyodide)…";
    await loadScript(PYODIDE_URL);
    const py = await loadPyodide();
    status.textContent = "Installing numpy, pandas, scipy, scikit-learn…";
    await py.loadPackage(["numpy", "pandas", "scipy", "scikit-learn"]);
    status.textContent = "Loading data and project code…";
    const [code, csv] = await Promise.all([fetch("py/plu_ggn.py").then((r) => r.text()), fetch("data/diabetes_25k.csv").then((r) => r.text())]);
    py.FS.writeFile("/home/pyodide/plu_ggn.py", code);
    py.FS.writeFile("/home/pyodide/diabetes_25k.csv", csv);
    await py.runPythonAsync("import sys; sys.path.insert(0, '/home/pyodide'); import plu_ggn");
    return py;
  }

  const RUN_PY = `
import json, numpy as np
from plu_ggn import *
_X, _y, _pid = load_clinical_readmission_data(csv_path="/home/pyodide/diabetes_25k.csv", mask_seed=SEED)
_Xtr, _ytr, _Xte, _yte = split_train_test(_X, _y, _pid, seed=SEED)
_idx = choose_forget_set(_Xtr, FRAC, mode=MODE, seed=0)
_res = run_unlearning(_Xtr, _ytr, _Xte, _yte, _idx, l2_reg=LAM, timing_repeats=3)
_o = _res.pop("_objects")
_names = NUM_COLS + CAT_COLS + ["intercept"]
_res["coef"] = {"names": _names, "oracle": _o["model_orac"].theta.tolist(), "before": _o["model_full"].theta.tolist(),
                "std_influence": _o["theta_std"].tolist(), "plu_1": _o["theta_plu1"].tolist(), "plu_2": _o["theta_plu2"].tolist()}
_res["sigma"] = {c: [_o["prep_full"].stds_[c], _o["prep_orac"].stds_[c], _o["prep_plu"].stds_[c]] for c in NUM_COLS}
json.dumps(_res)
`;

  document.getElementById("live-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = document.getElementById("run-btn");
    btn.disabled = true;
    status.classList.remove("err");
    try {
      if (!pyReady) pyReady = initPy();
      const py = await pyReady;
      const f = Number(frac.value) / 100, mode = document.getElementById("i-mode").value;
      const lam = Number(document.getElementById("i-lam").value), seed = Number(document.getElementById("i-seed").value);
      status.textContent = `Running: split, fit, oracle retrain and 5 unlearning methods for a ${(f * 100).toFixed(1)}% ${mode === "random" ? "random" : "high-utilization"} cohort…`;
      await new Promise((r) => setTimeout(r, 30));
      const t0 = performance.now();
      const params = `SEED = ${seed}\nFRAC = ${f}\nMODE = ${JSON.stringify(mode)}\nLAM = ${lam}\n`;
      const out = JSON.parse(await py.runPythonAsync(params + RUN_PY));
      renderLive(out);
      status.textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s in your browser. |D_f| = ${out.n_forget}, |D_r| = ${out.n_retain}, test = ${out.n_test}.`;
    } catch (e) {
      console.error(e);
      status.classList.add("err");
      status.textContent = "Run failed: " + (e.message || e).toString().split("\n").slice(-2).join(" ");
      pyReady = null;
    } finally { btn.disabled = false; }
  });

  function renderLive(o) {
    document.getElementById("live-out").hidden = false;
    const m = o.methods;
    const head = `<thead><tr><th>Method</th><th>Weight error ↓</th><th>Test KL ↓</th><th>Pipe leak</th><th>PR-AUC</th><th>Speedup (in browser)</th></tr></thead>`;
    const rows = ORDER.map((k) => {
      const r = m[k];
      const sp = k === "oracle" ? `1× (${(r.seconds * 1000).toFixed(0)} ms)` : r.speedup ? `${r.speedup >= 100 ? r.speedup.toFixed(0) : r.speedup.toFixed(1)}×` : (r.seconds === 0 ? "&gt;100× (below timer resolution)" : "–");
      const flag = k === "std_influence" && r.l2_err > m.before.l2_err ? ' <span title="further from retraining than doing nothing">⚠</span>' : "";
      return `<tr class="${k.startsWith("plu") ? "ours" : ""}"><td>${NAMES[k]}${flag}</td><td>${sci(r.l2_err)}</td><td>${sci(r.test_kl, 1)}</td><td>${(r.pipe_leak * 100).toFixed(0)}%</td><td>${r.pr_auc.toFixed(3)}</td><td>${sp}</td></tr>`;
    }).join("");
    document.getElementById("live-table").innerHTML = head + `<tbody>${rows}</tbody>`;

    const c = C(), names = o.coef.names.map((s) => s.replace("number_", "n_").replace("num_", "n_").replace("_disposition_id", "_disp").replace("medical_", "med_"));
    const diff = (k) => o.coef[k].map((v, i) => v - o.coef.oracle[i]);
    if (charts.coef) charts.coef.destroy();
    charts.coef = new Chart(document.getElementById("c-coef"), { type: "bar",
      data: { labels: names, datasets: ["before", "std_influence", "plu_1"].map((k) => ({ label: NAMES[k], data: diff(k), backgroundColor: c[k] })) },
      options: { ...baseOpts(), indexAxis: "y", scales: { x: { title: { display: true, text: "coefficient − oracle coefficient" } } } } });

    const cols = Object.keys(o.sigma);
    if (charts.sigma) charts.sigma.destroy();
    charts.sigma = new Chart(document.getElementById("c-sigma"), { type: "bar",
      data: { labels: cols.map((s) => s.replace("number_", "n_").replace("num_", "n_")),
        datasets: [
          { label: "frozen φ (before deletion)", data: cols.map((k) => o.sigma[k][0] / o.sigma[k][1]), backgroundColor: c.std_influence },
          { label: "PLU-GGN downdate", data: cols.map((k) => o.sigma[k][2] / o.sigma[k][1]), backgroundColor: c.plu_1 },
        ] },
      options: { ...baseOpts(), indexAxis: "y", scales: { x: { title: { display: true, text: "σ relative to retrained σ (1 = exact)" } } } } });
  }

  // re-theme charts if the OS theme changes
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (R) { Object.values(charts).forEach((ch) => ch.destroy()); renderAll(); } });
})();
