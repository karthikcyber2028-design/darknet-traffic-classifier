/* global document, fetch */
"use strict";

const $ = (id) => document.getElementById(id);

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

const statusData = { targets: [] };

function targetOf(name) {
  return statusData.targets.find((t) => t.name === name);
}

function populateSelect(el, items, includeModelHelp) {
  el.innerHTML = "";
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = item;
    opt.textContent = includeModelHelp
      ? `${item} — ${statusData.model_help[item] || item}`
      : item;
    el.appendChild(opt);
  }
}

function number(n) {
  if (n === "" || n === null || n === undefined) return null;
  const v = parseFloat(n);
  return Number.isFinite(v) ? v : null;
}

/* ---------------- tabs ---------------- */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "compare") loadCompare();
  });
});

/* ---------------- status ---------------- */
async function init() {
  try {
    const st = await fetchJson("/api/status");
    statusData.targets = st.targets;
    statusData.model_help = st.model_help;

    $("dataset-label").textContent =
      `Dataset: ${st.dataset_source}${st.dataset_source === "synthetic" ? " (demo)" : ""} · ${st.synth_rows} rows available`;
    const badge = $("data-badge");
    badge.textContent = st.dataset_source === "real" ? "Real dataset" : "Synthetic demo data";
    badge.className = "badge " + (st.dataset_source === "real" ? "badge-good" : "badge-warn");

    const names = st.targets.map((t) => t.name);
    populateSelect($("target"), names);
    populateSelect($("csv-target"), names);
    populateSelect($("an-target"), names);
    populateSelect($("model"), st.models, true);
    populateSelect($("csv-model"), st.models, true);
    populateSelect($("an-model"), st.models, true);

    $("target").addEventListener("change", () => { syncModels("model", "target"); onTargetChange(); });
    $("csv-target").addEventListener("change", () => syncModels("csv-model", "csv-target"));
    $("an-target").addEventListener("change", () => syncModels("an-model", "an-target"));
    $("an-model").addEventListener("change", loadAnalytics);

    syncModels("model", "target");
    syncModels("csv-model", "csv-target");
    syncModels("an-model", "an-target");
    onTargetChange();
    loadAnalytics();
  } catch (err) {
    $("data-badge").textContent = "offline";
    $("feature-form").innerHTML = `<p class="muted">${err.message}</p>`;
  }
}

function syncModels(modelSelId, targetSelId) {
  const t = targetOf($(targetSelId).value);
  populateSelect($(modelSelId), t && t.models.length ? t.models : statusData.models, true);
}

function onTargetChange() {
  const t = targetOf($("target").value);
  $("target-desc").textContent = t ? t.description : "";
  loadFeatures();
}

/* ---------------- feature form ---------------- */
let featureDefaults = {};

async function loadFeatures() {
  const target = $("target").value;
  const model = $("model").value;
  $("feature-form").innerHTML = '<p class="muted">Loading features…</p>';
  $("btn-predict").disabled = true;
  try {
    const data = await fetchJson(`/api/features?target=${encodeURIComponent(target)}&model=${encodeURIComponent(model)}`);
    featureDefaults = {};
    const form = $("feature-form");
    form.innerHTML = "";
    for (const f of data.features) {
      featureDefaults[f.name] = f.default;
      const lbl = document.createElement("label");
      lbl.textContent = f.name;
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.value = f.default;
      input.dataset.feat = f.name;
      lbl.appendChild(input);
      form.appendChild(lbl);
    }
    $("btn-predict").disabled = false;
    $("result-box").innerHTML = '<p class="muted">Ready. Click <b>Predict</b>.</p>';
  } catch (err) {
    $("feature-form").innerHTML = `<p class="muted">${err.message}</p>`;
  }
}

function collectFeatures() {
  const values = {};
  document.querySelectorAll("#feature-form input[data-feat]").forEach((inp) => {
    values[inp.dataset.feat] = number(inp.value);
  });
  return values;
}

function fillForm(features) {
  document.querySelectorAll("#feature-form input[data-feat]").forEach((inp) => {
    const v = features[inp.dataset.feat];
    if (v !== undefined && v !== null) inp.value = v;
  });
}

$("btn-reset").addEventListener("click", () => fillForm(featureDefaults));
$("btn-load-sample").addEventListener("click", async () => {
  const target = $("target").value, model = $("model").value;
  try {
    const s = await fetchJson(`/api/sample?target=${encodeURIComponent(target)}&model=${encodeURIComponent(model)}`);
    fillForm(s.features);
    $("sample-truth").textContent = s.truth_type
      ? `sample truth: ${s.truth_label} / ${s.truth_type}`
      : `sample truth: ${s.truth_label}`;
  } catch (err) {
    $("sample-truth").textContent = err.message;
  }
});

$("btn-predict").addEventListener("click", async () => {
  const target = $("target").value, model = $("model").value;
  $("btn-predict").disabled = true;
  try {
    const r = await fetchJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target, model, values: collectFeatures() }),
    });
    renderPrediction(r);
  } catch (err) {
    $("result-box").innerHTML = `<p class="muted">${err.message}</p>`;
  } finally {
    $("btn-predict").disabled = false;
  }
});

function renderPrediction(r) {
  const box = $("result-box");
  let probHtml = "";
  if (r.probabilities) {
    probHtml = Object.entries(r.probabilities)
      .map(([k, v]) => `
        <div class="prob-row">
          <span class="lbl">${k}</span>
          <div class="prob-bar"><div class="prob-fill" style="width:${(v * 100).toFixed(1)}%"></div></div>
          <span class="prob-pct">${(v * 100).toFixed(1)}%</span>
        </div>`).join("");
  }
  const isTor = r.prediction.toLowerCase().includes("tor") || r.prediction.toLowerCase() === "darknet";
  box.innerHTML = `
    <div class="big-pred ${isTor ? "pill-tor" : "pill-good"}">${r.prediction}</div>
    <div style="color:var(--muted);font-size:12px;text-align:center;margin-bottom:10px">
      target: ${r.target} · model: ${r.model}
    </div>
    ${probHtml}`;
}

/* ---------------- batch CSV ---------------- */
$("csv-file").addEventListener("change", () => {
  $("btn-csv-run").disabled = !$("csv-file").files.length;
});

$("btn-csv-run").addEventListener("click", async () => {
  const file = $("csv-file").files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("target", $("csv-target").value);
  fd.append("model", $("csv-model").value);
  fd.append("file", file);
  $("btn-csv-run").disabled = true;
  try {
    const r = await fetchJson("/api/predict-csv", { method: "POST", body: fd });
    const sum = $("csv-summary");
    const rows = Object.entries(r.summary)
      .map(([k, v]) => `<div class="prob-row"><span class="lbl">${k}</span><span class="prob-pct">${v} (${((v / r.n_rows) * 100).toFixed(1)}%)</span></div>`)
      .join("");
    sum.innerHTML = `<div style="font-size:16px;font-weight:700;margin-bottom:6px">${r.n_rows} flows classified</div>${rows}`;

    const wrap = $("csv-table-wrap");
    wrap.classList.remove("hidden");
    const head = r.columns.map((c) => `<th>${c}</th>`).join("");
    const body = r.rows.map((row) => `<tr>${row.map((v) => `<td>${v}</td>`).join("")}</tr>`).join("");
    wrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  } catch (err) {
    $("csv-summary").innerHTML = `<p class="muted">${err.message}</p>`;
  } finally {
    $("btn-csv-run").disabled = false;
  }
});

/* ---------------- analytics ---------------- */
async function loadAnalytics() {
  const target = $("an-target").value;
  const model = $("an-model").value;
  const cont = $("an-content");
  cont.innerHTML = '<p class="muted">Loading…</p>';
  try {
    const a = await fetchJson(`/api/analytics?target=${encodeURIComponent(target)}&model=${encodeURIComponent(model)}`);
    const m = a.metrics;
    const metricKeys = [
      ["accuracy", "Accuracy"], ["f1", "F1"], ["f1_macro", "F1 (macro)"],
      ["roc_auc", "ROC-AUC"], ["precision", "Precision"], ["recall", "Recall"],
      ["average_precision", "Avg Precision"], ["precision_macro", "Precision (macro)"],
      ["recall_macro", "Recall (macro)"], ["f1_weighted", "F1 (weighted)"],
      ["n_features", "Features"], ["n_test_samples", "Test samples"],
    ];
    const cards = metricKeys
      .filter(([k]) => m[k] !== undefined)
      .map(([k, lbl]) => `<div class="metric"><div class="v">${typeof m[k] === "number" ? (Number.isInteger(m[k]) ? m[k] : m[k].toFixed(4)) : m[k]}</div><div class="k">${lbl}</div></div>`)
      .join("");

    const imgs = a.images.map((img) =>
      `<figure><img src="${img.url}" alt="${img.name}"><figcaption>${img.name.replace(/_/g, " ")}</figcaption></figure>`).join("");

    cont.innerHTML = `
      <div class="card">
        <h2>${a.target} · ${a.model} <span class="muted">(${a.description})</span></h2>
        <div class="metrics-grid">${cards}</div>
        <pre class="report">${escapeHtml(a.report_text || "")}</pre>
      </div>
      ${imgs ? `<div class="card"><h2>Plots</h2><div class="imgs">${imgs}</div></div>` : ""}`;
  } catch (err) {
    cont.innerHTML = `<p class="muted">${err.message}</p>`;
  }
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/* ---------------- compare ---------------- */
async function loadCompare() {
  const wrap = $("compare-wrap");
  if (!wrap.classList.contains("compare-done")) {
    try {
      const r = await fetchJson("/api/compare");
      wrap.classList.add("compare-done");
      if (r.error) { wrap.innerHTML = `<p class="muted">${r.error}</p>`; return; }
      const head = r.columns.map((c) => `<th>${c}</th>`).join("");
      const body = r.rows.map((row) => `<tr>${row.map((v) => `<td class="num">${v}</td>`).join("")}</tr>`).join("");
      wrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    } catch (err) {
      wrap.innerHTML = `<p class="muted">${err.message}</p>`;
    }
  }
}

init();
