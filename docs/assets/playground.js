/* Playground: drives src/playground_api.py inside Pyodide. */
(() => {
  "use strict";
  const { colors, NAMES, chartOpts, logTicks, methodsTable, esc, sci } = window.PLU;
  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
  const $ = (id) => document.getElementById(id);
  const tick = () => new Promise((r) => setTimeout(r, 30));
  const S = { py: null, api: null, cols: [], isDefault: false, fitted: false, tab: "top", picked: new Set(),
    pick: { page: 0, size: 15, sort: null, desc: true, query: "", total: 0 }, charts: {} };

  const setStatus = (id, msg, err = false) => { const el = $(id); el.innerHTML = msg; el.classList.toggle("err", err); };
  const lock = (id, on) => $(id).classList.toggle("locked", on);
  function call(fn, arg) {
    const out = JSON.parse(arg === undefined ? S.api[fn]() : S.api[fn](typeof arg === "string" ? arg : JSON.stringify(arg)));
    if (out.ok === false) throw new Error(out.error);
    return out;
  }

  // ------------------------------------------------------------ boot Python
  function loadScript(src) {
    return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error("could not load " + src)); document.head.appendChild(s); });
  }
  async function boot() {
    try {
      setStatus("py-status", "Downloading the Python runtime (first visit ≈ 30 MB, cached afterwards)…");
      await loadScript(PYODIDE_URL);
      const py = await loadPyodide();
      setStatus("py-status", "Installing numpy, pandas, scipy, scikit-learn…");
      await py.loadPackage(["numpy", "pandas", "scipy", "scikit-learn"]);
      // Always revalidate the Python sources: a stale cached plu_ggn.py next to a newer playground_api.py breaks imports.
      const src = (f) => fetch(`py/${f}`, { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(`could not load ${f}`); return r.text(); });
      const [a, b] = await Promise.all([src("plu_ggn.py"), src("playground_api.py")]);
      py.FS.writeFile("/home/pyodide/plu_ggn.py", a);
      py.FS.writeFile("/home/pyodide/playground_api.py", b);
      py.runPython("import sys, importlib; sys.path.insert(0, '/home/pyodide'); importlib.invalidate_caches()");
      if (!py.runPython("import plu_ggn; hasattr(plu_ggn, 'prepare_dataset')")) throw new Error("the site's Python files are out of sync; reload the page");
      S.py = py;
      S.api = py.pyimport("playground_api");
      setStatus("py-status", "Python is ready. Everything below runs locally in this tab.");
      $("use-default").disabled = false;
      $("file").disabled = false;
    } catch (e) {
      setStatus("py-status", "Could not start Python in this browser: " + esc(e.message), true);
    }
  }

  // ------------------------------------------------------------ step 1: data
  async function loadTable(text, isDefault, label) {
    try {
      setStatus("data-status", `Reading ${esc(label)}…`);
      await tick();
      S.py.FS.writeFile("/home/pyodide/table.csv", text);
      const r = call("load_csv", "/home/pyodide/table.csv");
      S.cols = r.columns;
      S.isDefault = isDefault;
      S.fitted = false;
      S.picked.clear();
      const sampled = r.n_rows < r.n_rows_original ? ` (randomly sampled from ${r.n_rows_original.toLocaleString()})` : "";
      setStatus("data-status", `Loaded <strong>${esc(label)}</strong>: ${r.n_rows.toLocaleString()} rows${sampled}, ${r.columns.length} columns.`);
      buildSetup();
      lock("step2", false); lock("step3", true); lock("step4", true);
      $("fit-kv").innerHTML = "";
      if (isDefault) await fit();
      else $("step2").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { setStatus("data-status", esc(e.message), true); }
  }
  $("use-default").addEventListener("click", async () => {
    $("use-default").disabled = true;
    const text = await fetch("data/diabetes_25k.csv", { cache: "no-cache" }).then((r) => r.text());
    await loadTable(text, true, "Diabetes readmission sample");
    $("use-default").disabled = false;
  });
  function handleFile(f) {
    if (!f) return;
    if (f.size > 20 * 1024 * 1024) return setStatus("data-status", "That file is larger than 20 MB.", true);
    f.text().then((t) => loadTable(t, false, f.name));
  }
  $("file").addEventListener("change", (e) => handleFile(e.target.files[0]));
  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => { if (!$("file").disabled) handleFile(e.dataTransfer.files[0]); });

  // ------------------------------------------------------------ step 2: setup
  const PRESET = { target: "readmitted", positive: "<30", group: "patient_nbr", miss: 0.12, l2: "10", seed: 42,
    num: ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"],
    cat: ["admission_type_id", "discharge_disposition_id", "medical_specialty", "A1Cresult"] };
  const col = (name) => S.cols.find((c) => c.name === name);
  const opt = (v, label, sel) => `<option value="${esc(v)}"${sel ? " selected" : ""}>${esc(label)}</option>`;

  function defaultRole(c) { return c.guess === "id" ? "drop" : c.guess; }
  function buildSetup() {
    let target, positive, group = "";
    if (S.isDefault) {
      target = PRESET.target; positive = PRESET.positive; group = PRESET.group;
      S.cols.forEach((c) => (c.role = PRESET.num.includes(c.name) ? "numeric" : PRESET.cat.includes(c.name) ? "categorical" : "drop"));
      $("miss").value = PRESET.miss; $("l2").value = PRESET.l2; $("seed").value = PRESET.seed;
    } else {
      const binary = S.cols.filter((c) => c.n_unique === 2);
      target = (binary.length ? binary[binary.length - 1] : S.cols[S.cols.length - 1]).name;
      S.cols.forEach((c) => (c.role = defaultRole(c)));
      $("miss").value = 0;
    }
    $("o-miss").textContent = `${Math.round($("miss").value * 100)}%`;
    $("target").innerHTML = S.cols.map((c) => opt(c.name, c.name, c.name === target)).join("");
    $("group").innerHTML = opt("", "none (row-level split)", !group) + S.cols.map((c) => opt(c.name, c.name, c.name === group)).join("");
    fillPositive(positive);
    renderColTable();
  }
  function fillPositive(pref) {
    const c = col($("target").value);
    const vals = c.top_values;
    // default: the rarer of the two most common values (usually the event of interest)
    const def = pref ?? (vals.length >= 2 ? (vals[1][1] <= vals[0][1] ? vals[1][0] : vals[0][0]) : vals[0][0]);
    $("positive").innerHTML = vals.map(([v, n]) => opt(v, `${v}  (${n.toLocaleString()} rows)`, v === def)).join("");
    if (c.n_unique > vals.length) $("positive").insertAdjacentHTML("beforeend", "<option disabled>… more values (positive = one value, all others negative)</option>");
  }
  function renderColTable() {
    const t = $("target").value;
    const rows = S.cols.map((c, i) => {
      const isT = c.name === t;
      const role = isT ? "<em>target</em>" : `<select data-i="${i}">${["numeric", "categorical", "drop"].map((r) => opt(r, r === "drop" ? "drop (not a feature)" : r, c.role === r)).join("")}</select>`;
      return `<tr class="${isT ? "is-target" : ""}"><td><strong>${esc(c.name)}</strong></td><td>${role}</td><td>${c.guess}</td><td>${c.n_unique.toLocaleString()}</td><td>${(c.missing * 100).toFixed(1)}%</td><td class="muted">${c.examples.map(esc).join(", ")}</td></tr>`;
    }).join("");
    $("coltable").innerHTML = `<thead><tr><th>Column</th><th>Use as</th><th>Detected</th><th>Unique</th><th>Blank</th><th>Examples</th></tr></thead><tbody>${rows}</tbody>`;
  }
  $("coltable").addEventListener("change", (e) => { const s = e.target.closest("select[data-i]"); if (s) { S.cols[+s.dataset.i].role = s.value; invalidate(); } });
  $("target").addEventListener("change", () => { fillPositive(); renderColTable(); invalidate(); });
  ["positive", "group", "l2", "test-size", "seed"].forEach((id) => $(id).addEventListener("change", invalidate));
  $("miss").addEventListener("input", () => { $("o-miss").textContent = `${Math.round($("miss").value * 100)}%`; invalidate(); });
  document.querySelectorAll("[data-bulk]").forEach((b) => b.addEventListener("click", () => {
    S.cols.forEach((c) => (c.role = b.dataset.bulk === "drop" ? "drop" : defaultRole(c)));
    renderColTable(); invalidate();
  }));
  function invalidate() {
    if (!S.fitted) return;
    S.fitted = false;
    lock("step3", true); lock("step4", true);
    setStatus("fit-status", "Settings changed. Fit again to continue.");
  }

  async function fit() {
    const t = $("target").value;
    const cfg = { target: t, positive: $("positive").value, group_col: $("group").value || null,
      num_cols: S.cols.filter((c) => c.name !== t && c.role === "numeric").map((c) => c.name),
      cat_cols: S.cols.filter((c) => c.name !== t && c.role === "categorical").map((c) => c.name),
      missing_rate: +$("miss").value, seed: +$("seed").value, test_size: +$("test-size").value, l2: +$("l2").value };
    try {
      $("fit-btn").disabled = true;
      setStatus("fit-status", "Splitting, fitting the preprocessor and the model…");
      await tick();
      const r = call("configure", cfg);
      S.cfg = cfg;
      S.fitted = true;
      setStatus("fit-status", "Model fitted.");
      $("fit-kv").innerHTML = [
        [r.n_train.toLocaleString(), "training rows"], [r.n_test.toLocaleString(), "test rows"],
        [`${(r.prevalence * 100).toFixed(1)}%`, `positive (${esc(t)} = ${esc(cfg.positive)})`],
        [cfg.group_col ? String(r.group_overlap) : "n/a", "groups in both train and test"],
        [r.test_pr_auc.toFixed(3), `test PR-AUC (prevalence ${r.prevalence.toFixed(3)})`], [r.test_roc_auc.toFixed(3), "test ROC-AUC"],
        [String(cfg.num_cols.length + cfg.cat_cols.length), `features (${cfg.num_cols.length} numeric, ${cfg.cat_cols.length} categorical)`],
      ].map(([b, s]) => `<div><b>${b}</b><span>${s}</span></div>`).join("");
      buildForgetControls();
      lock("step3", false); lock("step4", false);
      S.picked.clear();
      updatePickCount();
      await preview();
    } catch (e) { setStatus("fit-status", esc(e.message), true); }
    finally { $("fit-btn").disabled = false; }
  }
  $("fit-btn").addEventListener("click", fit);

  // ------------------------------------------------------------ step 3: forget set
  function buildForgetControls() {
    const t = S.cfg.target;
    const numericish = S.cols.filter((c) => c.name !== t && (c.role === "numeric" || c.guess === "numeric"));
    $("top-col").innerHTML = (S.isDefault ? opt("__utilization__", "Utilization score (as in the paper)", true) : "") +
      numericish.map((c, i) => opt(c.name, c.name + (c.role === "drop" ? " (not a feature)" : ""), !S.isDefault && i === 0)).join("");
    $("conds").innerHTML = "";
    addCond(S.isDefault ? { col: "number_inpatient", op: ">=", value: "5" } : undefined);
    S.pick = { ...S.pick, page: 0, sort: null, query: "" };
    $("pick-q").value = "";
  }
  function addCond(pre) {
    const d = document.createElement("div");
    d.className = "cond";
    const cols = S.cols.map((c) => c.name);
    const c0 = pre?.col ?? cols[0];
    d.innerHTML = `<select class="c-col">${cols.map((c) => opt(c, c, c === c0)).join("")}</select>
      <select class="c-op">${["==", "!=", ">", ">=", "<", "<="].map((o) => opt(o, o, o === (pre?.op ?? "=="))).join("")}</select>
      <input type="text" class="c-val" list="dl-${c0}" value="${esc(pre?.value ?? col(c0).top_values[0]?.[0] ?? "")}" style="width:150px">
      <button class="icon-btn c-del" type="button" aria-label="remove condition">✕</button>`;
    $("conds").appendChild(d);
    ensureDatalist(c0);
  }
  function ensureDatalist(name) {
    if (document.getElementById(`dl-${name}`)) return;
    const dl = document.createElement("datalist");
    dl.id = `dl-${name}`;
    dl.innerHTML = col(name).top_values.map(([v]) => `<option value="${esc(v)}">`).join("");
    document.body.appendChild(dl);
  }
  $("add-cond").addEventListener("click", () => { addCond(); schedulePreview(); });
  $("conds").addEventListener("click", (e) => { if (e.target.closest(".c-del")) { e.target.closest(".cond").remove(); schedulePreview(); } });
  $("conds").addEventListener("change", (e) => {
    if (e.target.classList.contains("c-col")) {
      const v = e.target.closest(".cond").querySelector(".c-val");
      ensureDatalist(e.target.value);
      v.setAttribute("list", `dl-${e.target.value}`);
      v.value = col(e.target.value).top_values[0]?.[0] ?? "";
    }
    schedulePreview();
  });
  $("conds").addEventListener("input", schedulePreview);

  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    S.tab = b.dataset.tab;
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.dataset.panel !== S.tab));
    $("sweep-btn").disabled = !["top", "random"].includes(S.tab);
    if (S.tab === "pick") loadPickPage();
    schedulePreview();
  }));
  [["top-frac", "o-top"], ["rand-frac", "o-rand"]].forEach(([i, o]) => $(i).addEventListener("input", () => { $(o).textContent = `${(+$(i).value).toFixed(1)}%`; schedulePreview(); }));
  ["top-col", "top-dir", "rand-seed"].forEach((id) => $(id).addEventListener("change", schedulePreview));

  function spec() {
    if (S.tab === "top") return { mode: "top", column: $("top-col").value, direction: $("top-dir").value, frac: +$("top-frac").value / 100 };
    if (S.tab === "random") return { mode: "random", frac: +$("rand-frac").value / 100, seed: +$("rand-seed").value };
    if (S.tab === "filter") return { mode: "filter", conditions: [...document.querySelectorAll(".cond")].map((d) => ({ col: d.querySelector(".c-col").value, op: d.querySelector(".c-op").value, value: d.querySelector(".c-val").value })) };
    return { mode: "pick", rows: [...S.picked] };
  }

  let pvTimer = null;
  function schedulePreview() { clearTimeout(pvTimer); pvTimer = setTimeout(preview, 250); }
  async function preview() {
    if (!S.fitted) return;
    try {
      const r = call("preview_forget", spec());
      $("pv-n").textContent = `${r.n_forget.toLocaleString()} of ${r.n_train.toLocaleString()} training rows (${(100 * r.n_forget / r.n_train).toFixed(1)}%)`;
      $("pv-n").classList.toggle("warn", !!r.warning);
      $("pv-note").innerHTML = r.warning ? `<span style="color:var(--warn)">${esc(r.warning)}</span>`
        : `Positive rate: ${r.forget_positive_rate == null ? "–" : (r.forget_positive_rate * 100).toFixed(1) + "%"} in the deleted rows vs. ${(r.retain_positive_rate * 100).toFixed(1)}% in the retained rows. The more the deleted rows differ from the rest, the more the preprocessor shifts.`;
      const rows = r.comparison.map((c) => `<tr><td>${esc(c.feature)}</td><td>${c.kind}</td><td>${c.forget == null ? "–" : sci(c.forget, 2)}</td><td>${sci(c.retain, 2)}</td></tr>`).join("");
      $("pv-table").innerHTML = `<thead><tr><th>Feature</th><th>Statistic</th><th>Deleted rows</th><th>Retained rows</th></tr></thead><tbody>${rows}</tbody>`;
      $("run-btn").disabled = !!r.warning;
    } catch (e) {
      $("pv-n").textContent = "invalid request"; $("pv-n").classList.add("warn");
      $("pv-note").innerHTML = `<span style="color:var(--warn)">${esc(e.message)}</span>`;
      $("pv-table").innerHTML = ""; $("run-btn").disabled = true;
    }
  }

  // hand-picking
  function updatePickCount() { $("pick-count").textContent = `${S.picked.size.toLocaleString()} selected`; }
  function loadPickPage() {
    if (!S.fitted) return;
    try {
      const p = S.pick;
      const r = call("rows_page", { page: p.page, size: p.size, sort: p.sort, desc: p.desc, query: p.query });
      p.total = r.total;
      const head = `<thead><tr><th></th>${r.columns.map((c) => `<th data-sort="${esc(c)}">${esc(c)}${p.sort === c ? (p.desc ? " ↓" : " ↑") : ""}</th>`).join("")}</tr></thead>`;
      const body = r.rows.map((row) => `<tr class="${S.picked.has(row[0]) ? "picked" : ""}"><td><input type="checkbox" data-row="${row[0]}" ${S.picked.has(row[0]) ? "checked" : ""} aria-label="forget row ${row[0]}"></td>${row.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("");
      $("pick-table").innerHTML = head + `<tbody>${body}</tbody>`;
      const pages = Math.max(1, Math.ceil(r.total / p.size));
      $("pick-pageinfo").textContent = `page ${p.page + 1} of ${pages.toLocaleString()} · ${r.total.toLocaleString()} rows`;
      $("pick-prev").disabled = p.page === 0; $("pick-next").disabled = p.page >= pages - 1;
    } catch (e) { $("pick-table").innerHTML = `<tr><td>${esc(e.message)}</td></tr>`; }
  }
  $("pick-table").addEventListener("change", (e) => {
    const cb = e.target.closest("input[data-row]");
    if (!cb) return;
    const i = +cb.dataset.row;
    cb.checked ? S.picked.add(i) : S.picked.delete(i);
    cb.closest("tr").classList.toggle("picked", cb.checked);
    updatePickCount(); schedulePreview();
  });
  $("pick-table").addEventListener("click", (e) => {
    const th = e.target.closest("th[data-sort]");
    if (!th) return;
    const c = th.dataset.sort;
    S.pick.desc = S.pick.sort === c ? !S.pick.desc : true;
    S.pick.sort = c; S.pick.page = 0; loadPickPage();
  });
  let qTimer = null;
  $("pick-q").addEventListener("input", () => { clearTimeout(qTimer); qTimer = setTimeout(() => { S.pick.query = $("pick-q").value; S.pick.page = 0; loadPickPage(); }, 300); });
  $("pick-prev").addEventListener("click", () => { S.pick.page--; loadPickPage(); });
  $("pick-next").addEventListener("click", () => { S.pick.page++; loadPickPage(); });
  $("pick-clear").addEventListener("click", () => { S.picked.clear(); updatePickCount(); loadPickPage(); schedulePreview(); });
  $("pick-page-all").addEventListener("click", () => {
    document.querySelectorAll("#pick-table input[data-row]").forEach((cb) => S.picked.add(+cb.dataset.row));
    updatePickCount(); loadPickPage(); schedulePreview();
  });

  // ------------------------------------------------------------ step 4: run
  $("run-btn").addEventListener("click", async () => {
    try {
      $("run-btn").disabled = true;
      setStatus("run-status", "Retraining from scratch and running every unlearning method…");
      await tick();
      const t0 = performance.now();
      const r = call("run", spec()).result;
      renderRun(r);
      setStatus("run-status", `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s. Forgot ${r.n_forget.toLocaleString()} rows; ${r.n_retain.toLocaleString()} retained; ${r.n_test.toLocaleString()} test rows.`);
    } catch (e) { setStatus("run-status", esc(e.message), true); }
    finally { $("run-btn").disabled = false; }
  });

  function renderRun(o) {
    $("run-out").hidden = false;
    methodsTable($("run-table"), o.methods, { speedLabel: "Speedup (in browser)" });
    const c = colors(), short = (s) => (s.length > 22 ? s.slice(0, 21) + "…" : s);
    const diff = (k) => o.coef[k].map((v, i) => v - o.coef.oracle[i]);
    $("coef-box").style.height = `${Math.max(300, o.coef.names.length * 26 + 80)}px`;
    S.charts.coef?.destroy();
    S.charts.coef = new Chart($("c-coef"), { type: "bar",
      data: { labels: o.coef.names.map(short), datasets: ["before", "std_influence", "plu_1"].map((k) => ({ label: NAMES[k], data: diff(k), backgroundColor: c[k] })) },
      options: chartOpts({ indexAxis: "y", scales: { x: { title: { display: true, text: "coefficient − retrained coefficient" } } } }) });
    const rows = o.pipeline.map((p) => {
      const ratio = p.sd_full / p.sd_retain;
      const flag = Math.abs(ratio - 1) > 0.05 ? ' <span class="pill warn">shifted</span>' : "";
      return `<tr><td>${esc(p.feature)}${flag}</td><td>${sci(p.mu_full, 3)}</td><td>${sci(p.mu_retain, 3)}</td><td>${sci(p.sd_full, 3)}</td><td>${sci(p.sd_retain, 3)}</td></tr>`;
    }).join("");
    $("pipe-table").innerHTML = o.pipeline.length
      ? `<thead><tr><th>Feature</th><th>μ frozen</th><th>μ retrained</th><th>σ frozen</th><th>σ retrained</th></tr></thead><tbody>${rows}</tbody>`
      : `<tbody><tr><td>No numeric features. Target-encoding prior: ${sci(o.prior.full, 3)} frozen vs. ${sci(o.prior.retain, 3)} retrained.</td></tr></tbody>`;
  }

  $("sweep-btn").addEventListener("click", async () => {
    try {
      $("sweep-btn").disabled = true;
      setStatus("run-status", "Running six deletion sizes (0.5% to 8%)…");
      await tick();
      const sp = spec(); delete sp.frac;
      const r = call("sweep", sp).sweep;
      if (!r.length) throw new Error("No valid deletion size for this setup.");
      const c = colors();
      $("sweep-out").hidden = false;
      S.charts.sweep?.destroy();
      S.charts.sweep = new Chart($("c-psweep"), { type: "line",
        data: { labels: r.map((x) => `${x.frac}% (${x.n_forget})`), datasets: ["before", "std_influence", "plu_1", "plu_2"].map((k) => ({
          label: NAMES[k], data: r.map((x) => Math.max(x[k].l2_err, 1e-16)), borderColor: c[k], backgroundColor: c[k], borderDash: k === "before" ? [5, 4] : [], pointRadius: 3 })) },
        options: chartOpts({ scales: { y: { type: "logarithmic", ticks: logTicks, title: { display: true, text: "weight error to retraining (log)" } },
          x: { title: { display: true, text: "deleted share of training rows (count)" } } } }) });
      setStatus("run-status", "Sweep finished. Where the orange line is above the dashed grey line, standard influence is worse than not unlearning at all.");
    } catch (e) { setStatus("run-status", esc(e.message), true); }
    finally { $("sweep-btn").disabled = !["top", "random"].includes(S.tab); }
  });

  window.PLU_PLAYGROUND = { loadTable }; // used by automated tests
  boot();
})();
