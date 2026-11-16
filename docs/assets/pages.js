/* Precomputed-results widgets. Each renderer runs only if its element exists on the current page. */
(() => {
  "use strict";
  const { colors, NAMES, sci, chartOpts, logTicks, methodsTable, loadResults, fillSpans } = window.PLU;
  const $ = (id) => document.getElementById(id);
  const charts = {};
  let R = null;

  function heroStats() {
    const el = $("hero-stats");
    if (!el) return;
    const m = R.main.methods, ps = R.preprocessor_shift.number_inpatient;
    el.innerHTML = [
      [`${(m.std_influence.l2_err / m.plu_1.l2_err).toFixed(0)}×`, "lower weight error than standard influence (PLU-GGN 1-step, 3% deletion)"],
      [sci(m.plu_2.l2_err, 1), "weight error vs. full retraining after 2 steps"],
      [`${ps.sd_full.toFixed(2)} → ${ps.sd_retain.toFixed(2)}`, "σ of number_inpatient that a frozen preprocessor keeps vs. retraining"],
      [`${m.plu_1.speedup.toFixed(1)}×`, "faster than retraining the pipeline + model (this CPU)"],
    ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  }

  function tails() {
    if (!$("c-tails")) return;
    const ns = R.data.numeric_summary, cols = Object.keys(ns), c = colors();
    charts.tails = new Chart($("c-tails"), { type: "bar",
      data: { labels: cols.map((s) => s.replace("number_", "n_").replace("num_", "n_")), datasets: [
        { label: "median", data: cols.map((k) => (ns[k].median > 0 ? ns[k].median : null)), backgroundColor: c.muted },
        { label: "99th pct", data: cols.map((k) => ns[k].p99), backgroundColor: c.plu_1 },
        { label: "max", data: cols.map((k) => ns[k].max), backgroundColor: c.std_influence }] },
      options: chartOpts({ indexAxis: "y", scales: { x: { type: "logarithmic", title: { display: true, text: "value (log scale; medians of 0 not drawn)" } } } }) });
  }

  function mainTable() {
    if ($("main-table")) methodsTable($("main-table"), R.main.methods, { seeds: R.seeds.summary });
  }

  const sweep = { mode: "top_utilization", metric: "l2_err" };
  function sweepChart() {
    if (!$("c-sweep")) return;
    const A = R.ablation[sweep.mode], c = colors();
    const ds = ["before", "std_influence", "plu_1", "plu_2"].map((k) => ({
      label: NAMES[k], data: A.map((r) => Math.max(r[k][sweep.metric], 1e-16)), borderColor: c[k], backgroundColor: c[k],
      borderDash: k === "before" ? [5, 4] : [], tension: 0.2, pointRadius: 3 }));
    charts.sweep?.destroy();
    charts.sweep = new Chart($("c-sweep"), { type: "line", data: { labels: A.map((r) => `${r.frac}%`), datasets: ds },
      options: chartOpts({ scales: { x: { title: { display: true, text: "deleted cohort (% of training encounters)" } },
        y: { type: "logarithmic", ticks: logTicks, title: { display: true, text: sweep.metric === "l2_err" ? "‖θ − θ*ᵣ‖₂ (log)" : "mean test KL to oracle (log)" } } },
        plugins: { tooltip: { callbacks: { label: (t) => `${t.dataset.label}: ${t.parsed.y.toExponential(2)}` } } } }) });
    const worse = A.filter((r) => r.std_influence.l2_err > r.before.l2_err).map((r) => `${r.frac}%`);
    $("sweep-note").textContent = sweep.metric === "l2_err"
      ? (worse.length ? `Standard influence ends further from retraining than doing nothing at: ${worse.join(", ")}.` : "Standard influence beats doing nothing at every size here.")
      : "In prediction space every method improves on doing nothing; PLU-GGN is orders of magnitude closer to retraining.";
  }

  let auditKey = "std_vs_all_test";
  function auditChart() {
    if (!$("c-audit")) return;
    const A = R.ablation.top_utilization, c = colors();
    charts.audit?.destroy();
    charts.audit = new Chart($("c-audit"), { data: { labels: A.map((r) => `${r.frac}%`), datasets: [
      { type: "bar", label: auditKey === "std_vs_all_test" ? "frozen-φ audit vs. random test patients" : "frozen-φ audit vs. equally extreme test patients",
        data: A.map((r) => r.mia[auditKey]), backgroundColor: c.std_influence },
      { type: "line", label: "utilization score alone (no model)", data: A.map((r) => r.mia.utilization_only_vs_all_test),
        borderColor: c.muted, backgroundColor: c.muted, pointRadius: 3, hidden: auditKey !== "std_vs_all_test" },
      { type: "line", label: "chance", data: A.map(() => 0.5), borderColor: c.line, pointRadius: 0, borderDash: [4, 4] }] },
      options: chartOpts({ scales: { y: { min: 0.4, max: 1, title: { display: true, text: "audit AUC" } }, x: { title: { display: true, text: "deleted cohort size" } } } }) });
    $("audit-note").textContent = auditKey === "std_vs_all_test"
      ? "Looks like near-perfect leakage, but the grey line shows that the utilization score alone gets the same AUC."
      : `Against test patients at least as extreme (n = ${A.map((r) => r.mia.n_matched).join(", ")}), the audit is close to chance for deletions ≥ 2%. At 0.5–1% only 30–41 controls exist and they are slightly less extreme.`;
  }

  function errTable() {
    if (!$("err-table")) return;
    const rows = R.errors.top5_plu1.map((e) => `<tr><td>top ${(100 - e.utilization_pct).toFixed(2)}% utilization</td><td>${e.number_inpatient ?? "missing"}</td><td>${e.num_lab_procedures ?? "missing"}</td><td>${e.p_oracle.toFixed(3)}</td><td>${e.p_plu1.toFixed(3)}</td><td>${e.abs_dp.toFixed(4)}</td></tr>`).join("");
    $("err-table").innerHTML = `<thead><tr><th>Held-out patient</th><th>prior inpatient</th><th>lab procedures</th><th>p (oracle)</th><th>p (PLU-1)</th><th>|Δp|</th></tr></thead><tbody>${rows}</tbody>`;
  }

  function coefTable() {
    if (!$("coef-table")) return;
    const rows = R.coefficients.map((c) => `<tr><td>${c.feature}</td><td>${c.oracle.toFixed(3)}</td><td>${c.before.toFixed(3)}</td><td>${c.std_influence.toFixed(3)}</td><td>${c.plu_1.toFixed(3)}</td></tr>`).join("");
    $("coef-table").innerHTML = `<thead><tr><th>Coefficient</th><th>Oracle</th><th>No unlearning</th><th>Std. influence</th><th>PLU-GGN 1</th></tr></thead><tbody>${rows}</tbody>`;
  }

  function wire() {
    document.querySelectorAll(".seg").forEach((seg) => seg.addEventListener("click", (ev) => {
      const b = ev.target.closest("button");
      if (!b) return;
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      if (b.dataset.mode) { sweep.mode = b.dataset.mode; sweepChart(); }
      if (b.dataset.metric) { sweep.metric = b.dataset.metric; sweepChart(); }
      if (b.dataset.ctrl) { auditKey = b.dataset.ctrl; auditChart(); }
    }));
  }

  function renderAll() { fillSpans(R); heroStats(); tails(); mainTable(); sweepChart(); auditChart(); errTable(); coefTable(); }
  loadResults().then((j) => { R = j; renderAll(); wire(); })
    .catch((e) => console.error("results.json", e));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (R) { Object.values(charts).forEach((ch) => ch.destroy()); renderAll(); }
  });
})();
