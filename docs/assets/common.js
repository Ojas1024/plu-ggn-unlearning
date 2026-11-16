/* Shared helpers for every page of the PLU-GGN site. */
window.PLU = (() => {
  "use strict";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const colors = () => ({
    before: css("--grey"), grad_ascent: "#b8860b", std_influence: css("--warn"),
    plu_1: css("--accent"), plu_2: css("--purple"), ink: css("--ink"), muted: css("--muted"), line: css("--line"),
  });
  const NAMES = {
    oracle: "Oracle retrain", before: "No unlearning", grad_ascent: "Gradient ascent (frozen φ)",
    std_influence: "Standard influence (frozen φ)", plu_1: "PLU-GGN 1-step (ours)", plu_2: "PLU-GGN 2-step (ours)",
  };
  const ORDER = ["oracle", "before", "grad_ascent", "std_influence", "plu_1", "plu_2"];

  const sci = (x, d = 2) => {
    if (x === null || x === undefined || Number.isNaN(x)) return "–";
    if (x === 0) return "0";
    const e = Math.floor(Math.log10(Math.abs(x)));
    if (e >= -2 && e < 3) return String(Number(x.toPrecision(d + 1)));
    return `${(x / 10 ** e).toFixed(d)}×10<sup>${e}</sup>`;
  };
  const fmt = {
    pct: (x) => `${(x * 100).toFixed(0)}%`, pct1: (x) => `${(x * 100).toFixed(1)}%`, f2: (x) => x.toFixed(2),
    f3: (x) => x.toFixed(3), sci: (x) => sci(x), int: (x) => Number(x).toLocaleString("en-US"),
  };
  const get = (o, path) => path.split(".").reduce((a, k) => (a == null ? a : a[k]), o);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function chartOpts(extra = {}) {
    const c = colors();
    if (window.Chart) {
      Chart.defaults.color = c.muted;
      Chart.defaults.borderColor = c.line;
      Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    }
    return { responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
      plugins: { legend: { labels: { boxWidth: 12, usePointStyle: true } } }, ...extra };
  }
  const logTicks = { callback: (v) => { const e = Math.log10(v); return Math.abs(e - Math.round(e)) < 1e-9 ? `1e${Math.round(e)}` : ""; } };

  function speedCell(k, r) {
    if (k === "oracle") return `1× (${(r.seconds * 1000).toFixed(0)} ms)`;
    if (r.speedup) return `${r.speedup >= 100 ? r.speedup.toFixed(0) : r.speedup.toFixed(1)}×`;
    return r.seconds === 0 ? "&gt;100×" : "–";
  }

  /** Comparison table for one deletion request. opts.seeds = seed summary (optional). */
  function methodsTable(el, methods, opts = {}) {
    const cols = [["Weight error ↓", (k, r) => sci(r.l2_err)], ["Test KL ↓", (k, r) => sci(r.test_kl, 1)],
      ["max |Δp|", (k, r) => sci(r.max_abs_dp, 2)], ["Pipe leak", (k, r) => `${(r.pipe_leak * 100).toFixed(0)}%`],
      ["PR-AUC", (k, r) => r.pr_auc.toFixed(3)], [opts.speedLabel || "Speedup", speedCell]];
    if (opts.seeds) cols.push(["Weight error, 5 seeds", (k) => { const s = opts.seeds[k]; return !s ? "–" : s.l2_err.mean === 0 ? "0" : `${sci(s.l2_err.mean, 2)} ± ${sci(s.l2_err.sd, 1)}`; }]);
    const head = `<thead><tr><th>Method</th>${cols.map((c) => `<th>${c[0]}</th>`).join("")}</tr></thead>`;
    const rows = ORDER.map((k) => {
      const r = methods[k];
      const flag = k === "std_influence" && r.l2_err > methods.before.l2_err ? ' <span title="further from retraining than doing nothing" class="pill warn">worse than nothing</span>' : "";
      return `<tr class="${k.startsWith("plu") ? "ours" : ""}"><td>${NAMES[k]}${flag}</td>${cols.map((c) => `<td>${c[1](k, r)}</td>`).join("")}</tr>`;
    }).join("");
    el.innerHTML = head + `<tbody>${rows}</tbody>`;
  }

  let resultsPromise = null;
  const loadResults = () => (resultsPromise ||= fetch("data/results.json", { cache: "no-cache" }).then((r) => r.json()));

  function fillSpans(R) {
    document.querySelectorAll("[data-k]").forEach((el) => {
      const v = get(R, el.dataset.k);
      if (v === undefined || v === null) return;
      el.innerHTML = el.dataset.fmt ? fmt[el.dataset.fmt](v) : fmt.int(v);
    });
  }

  // repository link: derive from <user>.github.io/<repo>/
  document.addEventListener("DOMContentLoaded", () => {
    const m = location.hostname.match(/^([^.]+)\.github\.io$/);
    const repo = location.pathname.split("/").filter(Boolean)[0];
    if (m && repo) document.querySelectorAll("[data-repo]").forEach((a) => (a.href = `https://github.com/${m[1]}/${repo}`));
  });

  return { colors, NAMES, ORDER, sci, fmt, get, esc, chartOpts, logTicks, methodsTable, loadResults, fillSpans };
})();
