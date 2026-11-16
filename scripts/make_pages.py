"""Generates the static pages in docs/ (shared header, nav and footer). Run: python scripts/make_pages.py docs"""
import os, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "docs")
REPO = "https://github.com/Ojas1024/plu-ggn-unlearning"
NAV = [("index.html", "Overview"), ("problem.html", "Problem &amp; data"), ("method.html", "Method"),
       ("results.html", "Results"), ("playground.html", "Playground"), ("limitations.html", "Limitations &amp; reproduce")]

def page(fname, title, desc, body, scripts=("pages.js",), chart=True):
    nav = "".join(f'<a href="{f}"{" aria-current=\"page\"" if f == fname else ""}>{t}</a>' for f, t in NAV)
    i = [f for f, _ in NAV].index(fname)
    prev = f'<a href="{NAV[i-1][0]}">← {NAV[i-1][1]}</a>' if i > 0 else "<span></span>"
    nxt = f'<a href="{NAV[i+1][0]}">{NAV[i+1][1]} →</a>' if i < len(NAV) - 1 else "<span></span>"
    js = "".join(f'<script src="assets/{s}"></script>' for s in scripts)
    cdn = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>\n' if chart else ""
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="stylesheet" href="assets/style.css">
{cdn}</head>
<body>
<header class="site-head"><div class="wrap"><a class="brand" href="index.html">PLU-GGN</a><nav class="site-nav" aria-label="Pages">{nav}</nav></div></header>
{body}
<div class="wrap"><nav class="pager" aria-label="Previous and next page">{prev}{nxt}</nav></div>
<footer><div class="wrap muted small">Ojas Gupta · ES 678, IIT Gandhinagar · Data: UCI Diabetes 130-US Hospitals (CC BY 4.0) · <a data-repo href="{REPO}">Source code</a> · <a href="paper.pdf">Paper (PDF)</a></div></footer>
<script src="assets/common.js"></script>{js}
</body>
</html>
"""
    open(os.path.join(OUT, fname), "w").write(html)

# ------------------------------------------------------------------ overview
page("index.html", "Forgetting the Pipeline Too",
     "PLU-GGN: tabular machine unlearning that also unlearns the preprocessing pipeline. ES 678 course project, IIT Gandhinagar.", f"""
<header class="hero">
  <div class="wrap">
    <p class="kicker">ES 678 · Machine Learning · IIT Gandhinagar · Ojas Gupta</p>
    <h1>Forgetting the Pipeline Too</h1>
    <p class="lede">When a patient withdraws consent, a hospital model should behave as if that patient was never in the training data.
      Most unlearning methods update only the model weights, but tabular models also learn their <em>preprocessing</em>
      (imputation means, scaling, target encodings) from the data. This project measures what that costs, and fixes it.</p>
    <div class="links">
      <a class="btn" href="paper.pdf">Read the paper (PDF, 4 pages)</a>
      <a class="btn ghost" href="playground.html">Try it on your own data</a>
      <a class="btn ghost" data-repo href="{REPO}">Source code</a>
    </div>
    <div class="stats" id="hero-stats"></div>
  </div>
</header>
<main class="wrap">
<section>
  <h2>The short version</h2>
  <div class="findings">
    <div><strong>A frozen preprocessor remembers the deleted patients.</strong> Deleting the 3% heaviest hospital users shrinks the scale of <code>number_inpatient</code> from 1.20 to 0.78, but a frozen pipeline keeps 1.20.</div>
    <div class="warn"><strong>Standard unlearning moves the weights the wrong way.</strong> A standard influence (Newton) update ends <em>further</em> from the retrained weights than doing nothing (5 out of 5 seeds), even though its predictions improve.</div>
    <div><strong>PLU-GGN fixes both.</strong> An exact O(|D<sub>f</sub>|) downdate of the preprocessor, plus a gradient term for the changed coordinates. Weight error drops 26× with one step and to ~10<sup>−5</sup> with two, about 5× faster than retraining.</div>
    <div class="warn"><strong>A tempting privacy result didn't hold up.</strong> A membership audit that looked like 0.98 AUC was really measuring how extreme a patient is. Against equally extreme controls it is ~0.56.</div>
  </div>
</section>
<section>
  <h2>Explore</h2>
  <div class="cards">
    <a class="card" href="problem.html"><h3>Problem &amp; data</h3><p>The question, what one row is, the target, data quality and preprocessing.</p></a>
    <a class="card" href="method.html"><h3>Method</h3><p>Baselines, why a frozen pipeline hurts, PLU-GGN, and how we evaluate.</p></a>
    <a class="card" href="results.html"><h3>Results</h3><p>Main comparison, deletion-size sweep, the privacy audit pitfall, representative errors.</p></a>
    <a class="card accent" href="playground.html"><h3>Playground</h3><p>Run everything in your browser. Use our data or upload a CSV, choose features, and decide which rows to forget.</p></a>
    <a class="card" href="limitations.html"><h3>Limitations &amp; reproduce</h3><p>What the evidence does not show, conclusions, and how to rerun every number.</p></a>
  </div>
</section>
</main>""")

# ------------------------------------------------------------------ problem
page("problem.html", "Problem and Data · PLU-GGN", "Question, dataset, data quality and preprocessing for the PLU-GGN project.", """
<main class="wrap">
<div class="page-head"><h1>Problem &amp; data</h1><p>What we ask, what one row means, and how the data is prepared.</p></div>
<section id="question">
  <h2>Question</h2>
  <p><strong>How closely do cheap unlearning updates match full retraining when the preprocessing is also learned from the data, and what does a frozen preprocessor leak?</strong></p>
  <p>The gold standard is <em>oracle retraining</em>: delete the patients, then refit the whole pipeline and the model. Cheaper methods try to approximate it.
    The question is about <em>unlearning fidelity</em>, not about building the best readmission model.</p>
  <dl class="facts">
    <div><dt>One row</dt><dd>One inpatient encounter of a diabetic patient</dd></div>
    <div><dt>Target</dt><dd>Readmitted within 30 days (11.2% positive)</dd></div>
    <div><dt>Prediction time</dt><dd>At discharge: only pre-discharge counts and categories are used</dd></div>
    <div><dt>Deletion request</dt><dd>Top <var>q</var>% of training encounters by utilization (3·inpatient visits + 0.1·lab procedures); default 3% = 599 encounters. Random cohorts serve as a control.</dd></div>
  </dl>
</section>
<section id="data">
  <h2>Data</h2>
  <p>UCI <a href="https://archive.ics.uci.edu/dataset/296">Diabetes 130-US Hospitals</a> (1999–2008, Strack et al. 2014; CC BY 4.0), via OpenML id 4541.
    We use a fixed random sample of <span data-k="data.n_encounters"></span> of the 101,766 encounters, from <span data-k="data.n_patients"></span> distinct patients.
    The data is public and de-identified, so the sample ships with this site.</p>
  <p>Features: 8 counts (<code>time_in_hospital</code>, <code>num_lab_procedures</code>, <code>num_procedures</code>, <code>num_medications</code>,
    prior-year <code>number_outpatient/emergency/inpatient</code>, <code>number_diagnoses</code>) and 4 categoricals (admission type, discharge disposition, medical specialty, A1C result).</p>
  <div class="grid2">
    <div class="card"><h3>Heavy right tails</h3><p class="muted small">Training set: median, 99th percentile and maximum of each count. The few extreme patients set the scaling.</p><div class="chart-box"><canvas id="c-tails"></canvas></div></div>
    <div class="card"><h3>Quality issues we handle</h3>
      <ul>
        <li><strong>Repeat patients.</strong> Some patients have several encounters, so we split <em>by patient</em>: <span data-k="data.patient_overlap"></span> patients appear in both train and test.</li>
        <li><strong>Missing specialty.</strong> <span data-k="data.medical_specialty_missing" data-fmt="pct"></span> of encounters have no medical specialty; kept as its own level.</li>
        <li><strong>Simulated missing counts.</strong> The released numeric columns are complete, so 12% of numeric cells are masked completely at random to exercise the imputer. This is a simulation; real missingness is rarely random.</li>
        <li><strong>Imbalance.</strong> 11.2% positives, so we check the model with PR-AUC, not accuracy.</li>
      </ul></div>
  </div>
</section>
<section id="preprocessing">
  <h2>Preprocessing (fitted on training rows only)</h2>
  <ul>
    <li>Numeric: mean imputation, then standardization <code>z = (x − μ)/σ</code>.</li>
    <li>Categorical: Empirical-Bayes target encoding <code>(Σy + 10·ȳ)/(n + 10)</code>, centred and scaled by 10.</li>
    <li>Everything is stored as <em>additive sufficient statistics</em> (counts, sums, sums of squares), so removing rows means subtracting their statistics. That costs O(|D<sub>f</sub>|) and never touches the retained data.</li>
  </ul>
</section>
</main>""")

# ------------------------------------------------------------------ method
page("method.html", "Method · PLU-GGN", "Baselines, PLU-GGN algorithm and evaluation design.", """
<main class="wrap">
<div class="page-head"><h1>Method</h1><p>The baselines, why freezing the preprocessor fails, what PLU-GGN does, and how everything is evaluated.</p></div>
<section>
  <h2>Baselines and methods</h2>
  <p>Model: L2-regularized logistic regression (λ = 10, fixed in advance and not tuned on test data). For each deletion request every method is compared against oracle retraining:</p>
  <div class="methods">
    <div class="card"><h3>No unlearning</h3><p>Keep the old weights and the old preprocessor. The trivial baseline.</p></div>
    <div class="card"><h3>Gradient ascent</h3><p>5 small steps uphill on the deleted patients' loss. Preprocessor frozen.</p></div>
    <div class="card"><h3>Standard influence (GGN)</h3><p>One Newton step that removes the deleted rows' gradient (Koh &amp; Liang 2017; Guo et al. 2020). Preprocessor frozen.</p></div>
    <div class="card accent"><h3>PLU-GGN (ours)</h3><p>(1) Exactly downdate the preprocessor. (2) Add the gradient change caused by the new coordinates, <code>g<sub>shift</sub></code>. (3) Take a Gauss–Newton step in the new coordinates. Optional 2nd Newton step.</p></div>
  </div>
</section>
<section>
  <h2>Why a frozen preprocessor hurts</h2>
  <p>The total change from the original weights to the retrained ones has two parts:</p>
  <p class="note"><strong>θ*<sub>r</sub> − θ̂ = [sample removal] + [coordinate change]</strong><br>
    Standard influence estimates only the first part. Removing the heaviest hospital users shrinks σ for <code>number_inpatient</code> from 1.20 to 0.78, so the same real-world effect needs a <em>smaller</em> coefficient per standardized unit.
    Removing those patients' gradient pushes the same coefficient up. When the two parts point in opposite directions, standard influence can land further from retraining than doing nothing.</p>
  <p class="muted small">This is an identity plus an argument about direction, not a theorem; the Results page checks it.</p>
  <div class="card"><h3>The coefficients at 3% deletion</h3><p class="muted small"><code>number_inpatient</code>: oracle 0.253, full model 0.328, standard influence 0.390. The removal step moves it the wrong way.</p>
    <div class="table-wrap"><table id="coef-table"></table></div></div>
</section>
<section>
  <h2>PLU-GGN in three steps</h2>
  <ol>
    <li><strong>Exact preprocessor downdate.</strong> Subtract the deleted rows' (count, Σx, Σx²) and (category count, Σy) and recompute μ, σ and encodings: O(|D<sub>f</sub>|). The stored design matrix is re-expressed by the affine map <code>z' = (σ/σ')·z + (μ − μ')/σ'</code>. Imputed cells stay 0 and categorical columns are re-mapped.</li>
    <li><strong>Pipeline-shift gradient.</strong> <code>g<sub>shift</sub> = Σ<sub>retained</sub> [∇ℓ(new features) − ∇ℓ(old features)]</code>, evaluated at the old weights.</li>
    <li><strong>Gauss–Newton step</strong> in the new coordinates: <code>θ = θ̂ + G<sub>new</sub><sup>−1</sup>(g<sub>rem</sub> − g<sub>shift</sub>)</code>. PLU-GGN-2 adds one more Newton step.</li>
  </ol>
  <p class="muted small">Cost: only step 1 is O(|D<sub>f</sub>|). Steps 2–3 make one pass over the retained data (O(|D<sub>r</sub>|d² + d³), d = 13) but need no iterative optimization, so PLU-GGN is cheaper than retraining but not free.</p>
</section>
<section>
  <h2>Evaluation design</h2>
  <ul>
    <li><strong>Split:</strong> patient-grouped 80/20 (19,995 train / 5,005 test), fixed before any comparison. Test data is used only for reporting.</li>
    <li><strong>Weight error</strong> ‖θ − θ*<sub>r</sub>‖: the standard unlearning metric. Frozen-φ methods express θ in different coordinates, so we also report</li>
    <li><strong>Function-space error:</strong> mean KL divergence between the method's and the oracle's test predictions, which doesn't depend on coordinates.</li>
    <li><strong>Pipeline leak:</strong> how far the deployed μ, σ are from the retrained ones (100% = unchanged from before deletion).</li>
    <li><strong>Sanity check:</strong> test PR-AUC (prevalence baseline <span data-k="predictive_baselines.prevalence_pr_auc" data-fmt="f3"></span>; <code>number_inpatient</code> alone <span data-k="predictive_baselines.number_inpatient_only.pr_auc" data-fmt="f3"></span>).</li>
    <li><strong>Exact reference:</strong> L-BFGS stops about 10<sup>−5</sup> from the optimum, so the oracle is polished with Newton steps. Only the ordinary L-BFGS refit is timed.</li>
    <li><strong>Robustness:</strong> 5 seeds that change both the patient split and the missingness mask; a random-deletion control; deletion sizes from 0.5% to 8%.</li>
  </ul>
</section>
</main>""")

# ------------------------------------------------------------------ results
page("results.html", "Results · PLU-GGN", "Main comparison, deletion sweep, privacy audit and representative errors.", """
<main class="wrap">
<div class="page-head"><h1>Results</h1><p>All numbers are produced by <code>experiments/run_all.py</code>. To rerun them yourself, use the <a href="playground.html">Playground</a>.</p></div>
<section>
  <h2>Main comparison: deleting the 3% highest-utilization encounters</h2>
  <div class="table-wrap"><table id="main-table"></table></div>
  <p class="muted small">Speedups measured on one CPU (median of 7 runs). Retraining itself takes only ~0.09 s here, so the speedup matters for larger pipelines. "5 seeds" = mean ± s.d.</p>
</section>
<section>
  <h2>How it changes with deletion size</h2>
  <div class="controls">
    <div class="seg" role="group" aria-label="cohort type"><button class="on" data-mode="top_utilization">High-utilization cohort</button><button data-mode="random">Random cohort</button></div>
    <div class="seg" role="group" aria-label="metric"><button class="on" data-metric="l2_err">Weight error</button><button data-metric="test_kl">Function-space error</button></div>
  </div>
  <div class="card"><div class="chart-box tall"><canvas id="c-sweep"></canvas></div><p class="muted small" id="sweep-note"></p></div>
  <div class="takeaways">
    <div><strong>Standard influence goes the wrong way in weights.</strong> For high-utilization deletions it ends further from retraining than doing nothing (all 5 seeds), though its predictions still improve.</div>
    <div><strong>PLU-GGN closes the gap.</strong> One step: 26× lower weight error than standard influence at 3%. Two steps: ~10<sup>−5</sup>.</div>
    <div><strong>Random deletions are easy.</strong> The pipeline barely moves, so the failure is specific to high-leverage patients.</div>
    <div><strong>Predictive quality is unchanged.</strong> PR-AUC differs by &lt;0.002 across methods, far less than the ±0.02 between seeds.</div>
  </div>
</section>
<section id="audit">
  <h2>Depth analysis: what a "membership audit" really measured</h2>
  <p>Our first privacy test scored each patient by how differently the deployed and the retrained preprocessor standardize them. It reported
    AUC 0.95–0.999 for "deleted vs. test patient" under a frozen preprocessor, and exactly 0.5 for PLU-GGN. Both numbers turned out to be misleading:</p>
  <ul>
    <li>The score depends only on a patient's features, not on whether they were in training, so it cannot separate a member from a non-member with the same features.</li>
    <li>It is identically zero when the deployed preprocessor equals the retrained one, so PLU-GGN's 0.5 holds by construction.</li>
    <li>The deleted patients are the most extreme ones. Compared with ordinary test patients, extremeness alone gives a high AUC.</li>
  </ul>
  <div class="controls"><div class="seg" role="group" aria-label="control set"><button class="on" data-ctrl="std_vs_all_test">Compare with random test patients</button><button data-ctrl="std_vs_matched">Compare with equally extreme test patients</button></div></div>
  <div class="card"><div class="chart-box"><canvas id="c-audit"></canvas></div><p class="muted small" id="audit-note"></p></div>
  <p>What <em>is</em> real is aggregate leakage: the frozen preprocessor still contains the deleted cohort's statistics. If old and new statistics are both exposed, subtracting them recovers the deleted cohort's mean
    <code>number_inpatient</code> exactly (<span data-k="differencing.recovered_forget_mean" data-fmt="f2"></span> vs. <span data-k="differencing.retain_mean" data-fmt="f2"></span> for retained patients). PLU-GGN removes the first problem, not the second; that would need added noise.</p>
</section>
<section id="errors">
  <h2>Representative errors</h2>
  <p>PLU-GGN (1 step) disagrees most with retraining on the most extreme held-out patients, where the curvature approximation is weakest. Median |Δp| is
    <span data-k="errors.plu1_abs_dp_p50_p99_max.0" data-fmt="sci"></span>. The largest cases:</p>
  <div class="table-wrap"><table id="err-table"></table></div>
  <p class="muted small">Rank correlation between |Δp| and utilization: <span data-k="errors.spearman_dp_vs_utilization_plu1" data-fmt="f2"></span>. A second Newton step removes these errors (max |Δp| ≈ 5·10<sup>−5</sup>).</p>
</section>
</main>""")

# ------------------------------------------------------------------ limitations
page("limitations.html", "Limitations and Reproduction · PLU-GGN", "Limitations, conclusion, reproduction steps and AI-use statement.", f"""
<main class="wrap">
<div class="page-head"><h1>Limitations &amp; reproduce</h1><p>What the evidence does and does not show, and how to get every number back.</p></div>
<section>
  <h2>Limitations</h2>
  <ul>
    <li>Exact O(|D<sub>f</sub>|) downdates exist only for additive statistics; median or k-NN imputation would need approximations.</li>
    <li>Only the preprocessor downdate is O(|D<sub>f</sub>|). The weight update makes one pass over the retained data: cheaper than retraining, but not free.</li>
    <li>Missingness is simulated (completely at random).</li>
    <li>One convex model with 13 parameters. For neural networks the Gauss–Newton step is only a local approximation.</li>
    <li>One deletion rule in the paper. Other request patterns can be tried in the <a href="playground.html">Playground</a> but are not part of the reported evidence.</li>
    <li>No per-patient privacy claim (see <a href="results.html#audit">the audit analysis</a>). Associations in observational hospital data are not causal effects.</li>
  </ul>
</section>
<section>
  <h2>Conclusion</h2>
  <p>When preprocessing is learned, unlearning has to update it too. For additive statistics that is cheap and exact. Adding the pipeline-shift gradient brings a one- or two-step update
    to within 10<sup>−2</sup>–10<sup>−5</sup> of retraining. Without it, standard influence can move weights the wrong way, on exactly the high-leverage patients where deletion matters most.</p>
</section>
<section>
  <h2>Reproduce</h2>
<pre><code>git clone {REPO}
cd plu-ggn-unlearning
pip install -r requirements.txt
python experiments/run_all.py     # all tables, figures, results.json (~20 s)
python src/plu_ggn.py             # just the main 3% comparison</code></pre>
  <ul>
    <li><strong>Data:</strong> <code>data/diabetes_25k.csv</code> is cached in the repo. Delete it and the loader re-downloads it from OpenML (id 4541) with the same seed.</li>
    <li><strong>Split:</strong> <code>GroupShuffleSplit(test_size=0.2, random_state=42)</code> on <code>patient_nbr</code>.</li>
    <li><strong>Missingness:</strong> 12% of numeric cells masked with <code>RandomState(42)</code>.</li>
    <li><strong>Fixed hyperparameters:</strong> λ = 10 (intercept 10<sup>−4</sup>), target-encoding smoothing m = 10.</li>
    <li>Timings depend on the machine; accuracy numbers are deterministic. The original Kaggle notebook is in <code>notebooks/</code>.</li>
  </ul>
</section>
<section>
  <h2>AI use</h2>
  <p class="muted">Claude (Anthropic) helped with code refactoring, re-running and auditing experiments, this website and text editing. All results were checked by the author.</p>
</section>
</main>""", scripts=(), chart=False)

# ------------------------------------------------------------------ playground
page("playground.html", "Unlearning Playground · PLU-GGN",
     "Run PLU-GGN in your browser on the default diabetes data or your own CSV; choose features and which rows to forget.", """
<main class="wrap">
<div class="page-head"><h1>Unlearning playground</h1>
  <p>Everything runs in your browser with the project's own Python code (via <a href="https://pyodide.org">Pyodide</a>). Uploaded files never leave your computer.
  Pick a dataset, decide how it is modelled, choose which rows to forget, and compare every unlearning method against full retraining.</p></div>
<p id="py-status" class="status" aria-live="polite">Starting Python…</p>

<section class="step" id="step1">
  <h2 class="step-title"><span class="step-num">1</span> Choose a dataset</h2>
  <div class="dataset-choice">
    <div class="card accent"><h3>Diabetes readmission (default)</h3>
      <p class="muted small">The 25,000-encounter sample used in the paper, with the paper's settings preloaded.</p>
      <button class="btn small" id="use-default" disabled>Use this dataset</button></div>
    <label class="dropzone card" id="dropzone"><h3>Upload your own CSV</h3>
      <p class="muted small">Any table with a header row and a binary (or binarizable) target column. Up to 20 MB; larger tables are sampled to 60,000 rows.</p>
      <span class="btn small ghost">Choose file…</span> <span class="muted small">or drop it here</span>
      <input type="file" id="file" accept=".csv,text/csv" disabled></label>
  </div>
  <p id="data-status" class="status"></p>
</section>

<section class="step locked" id="step2">
  <h2 class="step-title"><span class="step-num">2</span> Set up the model</h2>
  <p class="muted small">Choose the target, how each column is used, and how to split. Any column you set to <em>drop</em> is not used as a feature (you can still use it to choose rows to forget).</p>
  <div class="row">
    <label class="field">Target column <select id="target"></select></label>
    <label class="field">Positive class (target = …) <select id="positive"></select></label>
    <label class="field">Group column for the split <select id="group"></select></label>
    <label class="field">Simulated missing numeric cells <output id="o-miss">0%</output><input type="range" id="miss" min="0" max="0.4" step="0.02" value="0"></label>
    <label class="field">Regularization λ <select id="l2"><option>0.1</option><option>1</option><option selected>10</option><option>100</option></select></label>
    <label class="field">Test share <select id="test-size"><option value="0.2" selected>20%</option><option value="0.3">30%</option></select></label>
    <label class="field">Seed <input type="number" id="seed" value="42" min="0" max="9999" style="width:90px"></label>
  </div>
  <div class="row"><button class="icon-btn" type="button" data-bulk="detected">reset to detected types</button><button class="icon-btn" type="button" data-bulk="drop">drop all features</button></div>
  <div class="table-wrap"><table class="coltable" id="coltable"></table></div>
  <div class="row"><button class="btn" id="fit-btn">Fit model</button><span id="fit-status" class="status"></span></div>
  <div class="kv" id="fit-kv"></div>
</section>

<section class="step locked" id="step3">
  <h2 class="step-title"><span class="step-num">3</span> Choose what to forget</h2>
  <p class="muted small">Rows are chosen from the training split only. Try a rule that singles out unusual rows and compare it with a random cohort.</p>
  <div class="tabs" role="tablist">
    <button class="on" data-tab="top" role="tab">Extreme rows</button><button data-tab="random" role="tab">Random rows</button>
    <button data-tab="filter" role="tab">Rule-based</button><button data-tab="pick" role="tab">Pick by hand</button>
  </div>
  <div class="tab-panel" data-panel="top">
    <div class="row">
      <label class="field">Rank by <select id="top-col"></select></label>
      <label class="field">Forget the <select id="top-dir"><option value="high">highest</option><option value="low">lowest</option></select></label>
      <label class="field">Share of training rows <output id="o-top">3.0%</output><input type="range" id="top-frac" min="0.5" max="20" step="0.5" value="3"></label>
    </div>
  </div>
  <div class="tab-panel" data-panel="random" hidden>
    <div class="row">
      <label class="field">Share of training rows <output id="o-rand">3.0%</output><input type="range" id="rand-frac" min="0.5" max="20" step="0.5" value="3"></label>
      <label class="field">Random seed <input type="number" id="rand-seed" value="0" min="0" style="width:90px"></label>
    </div>
  </div>
  <div class="tab-panel" data-panel="filter" hidden>
    <p class="muted small">Forget every training row that matches <strong>all</strong> conditions.</p>
    <div id="conds"></div>
    <button class="icon-btn" type="button" id="add-cond">+ add condition</button>
  </div>
  <div class="tab-panel" data-panel="pick" hidden>
    <div class="row">
      <label class="field">Search <input type="search" id="pick-q" placeholder="any value…"></label>
      <span class="pill" id="pick-count">0 selected</span>
      <button class="icon-btn" type="button" id="pick-page-all">select this page</button>
      <button class="icon-btn" type="button" id="pick-clear">clear</button>
    </div>
    <p class="muted small">Click a column header to sort. Tick rows to forget them.</p>
    <div class="table-wrap"><table class="picktable" id="pick-table"></table></div>
    <div class="row"><button class="icon-btn" id="pick-prev">← prev</button><span class="muted small" id="pick-pageinfo"></span><button class="icon-btn" id="pick-next">next →</button></div>
  </div>
  <div class="card" id="preview">
    <h3>Deletion request <span class="pill" id="pv-n">–</span></h3>
    <p class="muted small" id="pv-note"></p>
    <div class="table-wrap"><table id="pv-table"></table></div>
  </div>
</section>

<section class="step locked" id="step4">
  <h2 class="step-title"><span class="step-num">4</span> Unlearn and compare</h2>
  <div class="row"><button class="btn" id="run-btn">Run unlearning</button><button class="btn ghost" id="sweep-btn" title="Repeat for 0.5–8% (extreme or random rows only)">Sweep deletion sizes</button><span id="run-status" class="status"></span></div>
  <div id="run-out" hidden>
    <div class="table-wrap"><table id="run-table"></table></div>
    <p class="muted small">Weight error = distance to the retrained weights. Test KL = how differently the method predicts on the test split. Pipe leak = how much of the deleted rows' effect on μ and σ is still in the deployed preprocessor.</p>
    <div class="grid2">
      <div class="card"><h3>Coefficients vs. retraining</h3><p class="muted small">Each method's coefficient minus the retrained one. Bars near zero = faithful unlearning.</p><div class="chart-box tall" id="coef-box"><canvas id="c-coef"></canvas></div></div>
      <div class="card"><h3>What the frozen preprocessor remembers</h3><p class="muted small">μ and σ before deletion (what a frozen pipeline keeps) and after retraining (what PLU-GGN's exact downdate gives).</p><div class="table-wrap"><table id="pipe-table"></table></div></div>
    </div>
  </div>
  <div class="card" id="sweep-out" hidden><h3>Deletion-size sweep</h3><div class="chart-box tall"><canvas id="c-psweep"></canvas></div></div>
</section>
</main>""", scripts=("playground.js",))

print("pages written")
