/* Loss Prevention Agent — production UI logic */
(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    customers: [],
    interventions: [],
    architecture: null,
    current: null,
    busy: false,
  };

  const PIPELINE = [
    ["01", "Customer"],
    ["02", "Risk model"],
    ["03", "Uplift / CATE"],
    ["04", "Rank + constraints"],
    ["05", "Explain"],
  ];

  function toast(msg, isError = false) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.toggle("error", isError);
    el.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => el.classList.remove("show"), 3200);
  }

  function setBusy(busy, label = "Working…") {
    state.busy = busy;
    document.querySelectorAll("[data-busy-disable]").forEach((btn) => {
      btn.disabled = busy;
    });
    const status = $("statusPill");
    if (!status) return;
    status.innerHTML = busy
      ? `<span class="spinner" aria-hidden="true"></span><span>${label}</span>`
      : `<span class="dot" aria-hidden="true"></span><span>Ready</span>`;
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { Accept: "application/json", ...(opts.headers || {}) },
      ...opts,
    });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const body = await res.json();
        detail = body.detail || JSON.stringify(body);
      } catch (_) { /* ignore */ }
      throw new Error(detail);
    }
    if (res.status === 204) return null;
    return res.json();
  }

  function fmtPct(x) {
    return `${((Number(x) || 0) * 100).toFixed(1)}%`;
  }
  function fmtMoney(x) {
    return `$${Math.round(Number(x) || 0).toLocaleString()}`;
  }
  function nice(name) {
    return String(name || "").replaceAll("_", " ");
  }
  function segmentBadge(seg) {
    const map = { low: "ok", moderate: "warn", high: "danger", critical: "danger" };
    return `<span class="badge badge-${map[seg] || "neutral"}">${seg || "n/a"}</span>`;
  }

  function setPipelineStep(idx) {
    document.querySelectorAll(".pipe-step").forEach((el, i) => {
      el.classList.toggle("active", i === idx);
      el.classList.toggle("done", i < idx);
    });
  }

  function activateTab(tabId) {
    document.querySelectorAll(".side-nav button").forEach((b) => {
      const on = b.dataset.tab === tabId;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    document.querySelectorAll(".panel").forEach((p) => {
      p.classList.toggle("active", p.id === tabId);
    });
    const titles = {
      overview: "Risk Overview",
      factors: "Risk Factors",
      recs: "Recommended Interventions",
      impact: "Expected Impact",
      evidence: "Evidence & Agent Trace",
      cf: "Counterfactual Simulator",
      methods: "Model Comparison",
      research: "Research Lab",
      architecture: "Architecture",
    };
    $("pageTitle").textContent = titles[tabId] || tabId;
  }

  function metrics(items) {
    return `<div class="metrics" role="group" aria-label="Key metrics">${items
      .map(
        ([k, v, s]) =>
          `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div>${
            s ? `<div class="s">${s}</div>` : ""
          }</div>`
      )
      .join("")}</div>`;
  }

  function riskGauge(p) {
    const pct = Math.max(0, Math.min(1, p));
    const angle = -90 + pct * 180;
    const color = pct >= 0.45 ? "#8f3d32" : pct >= 0.28 ? "#8a6a20" : "#1f6b4a";
    return `
      <div class="gauge-wrap" aria-label="Loss probability gauge ${fmtPct(p)}">
        <svg width="220" height="130" viewBox="0 0 220 130" class="chart">
          <path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="#e5ebe7" stroke-width="14" stroke-linecap="round"/>
          <path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="${color}" stroke-width="14"
            stroke-linecap="round" stroke-dasharray="${pct * 283} 283"/>
          <line x1="110" y1="110" x2="${110 + 70 * Math.cos((angle * Math.PI) / 180)}"
            y2="${110 + 70 * Math.sin((angle * Math.PI) / 180)}" stroke="${color}" stroke-width="3"/>
          <circle cx="110" cy="110" r="5" fill="${color}"/>
          <text x="110" y="95" text-anchor="middle" font-size="22" font-family="Fraunces, serif" fill="#14241f">${fmtPct(p)}</text>
          <text x="110" y="118" text-anchor="middle" font-size="11" fill="#5d7268">P(future loss)</text>
        </svg>
      </div>`;
  }

  function barChart(rows, valueKey = "value", labelKey = "label") {
    const max = Math.max(...rows.map((r) => Math.abs(r[valueKey])), 0.001);
    return rows
      .map((r) => {
        const v = r[valueKey];
        const w = (Math.abs(v) / max) * 100;
        const cls = v < 0 ? "danger" : v < 0.03 ? "warn" : "";
        return `<div class="bar-row">
          <div class="mono">${r[labelKey]}</div>
          <div class="bar-track"><div class="bar-fill ${cls}" style="width:${w}%"></div></div>
          <div class="mono">${(v * 100).toFixed(1)}%</div>
        </div>`;
      })
      .join("");
  }

  function upliftCompareSvg(items) {
    const w = 520, h = 220, pad = 36;
    const max = Math.max(...items.map((i) => Math.abs(i.cate)), 0.01);
    const bw = (w - pad * 2) / items.length - 12;
    const bars = items
      .map((it, idx) => {
        const x = pad + idx * ((w - pad * 2) / items.length) + 6;
        const bh = (Math.abs(it.cate) / max) * (h - pad * 2);
        const y = it.cate >= 0 ? h - pad - bh : h - pad;
        const fill = it.cate >= 0 ? "#2f5d4c" : "#8f3d32";
        return `<rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="6" fill="${fill}" opacity="0.9"/>
          <text x="${x + bw / 2}" y="${h - 10}" text-anchor="middle" font-size="10" fill="#5d7268">${it.label}</text>
          <text x="${x + bw / 2}" y="${y - 6}" text-anchor="middle" font-size="11" fill="#14241f">${(it.cate * 100).toFixed(1)}%</text>`;
      })
      .join("");
    return `<div class="chart-box"><svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="CATE comparison chart">${bars}
      <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#c9d4ce"/>
    </svg></div>`;
  }

  function renderOverview(rec) {
    const top = rec.rank_list.find((s) => s.eligible) || rec.rank_list[0];
    const cateRows = rec.rank_list
      .filter((s) => s.eligible)
      .slice(0, 6)
      .map((s) => ({ label: nice(s.intervention).split(" ").slice(0, 2).join(" "), cate: s.uplift?.cate || 0 }));
    $("overview").innerHTML = `
      ${metrics([
        ["P(future loss)", fmtPct(rec.risk.p_loss), segmentBadge(rec.risk.risk_segment)],
        ["Recommended", nice(rec.recommended_intervention), `rank #${top.rank}`],
        ["Est. CATE", fmtPct(top.uplift?.cate), top.uplift?.method || ""],
        ["Expected benefit", fmtMoney(top.expected_benefit), `confidence ${(top.confidence || 0).toFixed(2)}`],
      ])}
      <div class="grid-2">
        <div class="card">
          <h3>Risk vs treatment effect</h3>
          ${riskGauge(rec.risk.p_loss)}
          <div class="alert alert-info">
            <strong>Prediction ≠ uplift.</strong> High predicted risk does not guarantee a large intervention effect.
            This customer’s recommended action maximizes constrained utility over estimated CATE.
          </div>
        </div>
        <div class="card">
          <h3>Eligible CATE ranking</h3>
          ${upliftCompareSvg(cateRows)}
          <p style="margin-top:.8rem">${rec.explanation}</p>
        </div>
      </div>`;
  }

  function renderFactors(rec) {
    const factors = rec.risk.top_risk_factors || [];
    if (!factors.length) {
      $("factors").innerHTML = `<div class="empty">No risk factors available for this customer.</div>`;
      return;
    }
    const maxImp = Math.max(...factors.map((f) => f.importance), 0.001);
    $("factors").innerHTML = `
      <p>Calibrated predictive model <span class="mono">${rec.risk.model_version}</span>
      ${rec.risk.calibrated ? '<span class="badge badge-ok">calibrated</span>' : ""}</p>
      ${factors
        .map((f) => {
          const w = (f.importance / maxImp) * 100;
          const cls = f.direction === "elevates" ? "danger" : "";
          return `<div class="bar-row">
            <div><strong>${f.feature}</strong><div class="mono">${Number(f.value).toFixed(3)} · ${f.direction}</div></div>
            <div class="bar-track"><div class="bar-fill ${cls}" style="width:${w}%"></div></div>
            <div class="mono">${f.importance.toFixed(3)}</div>
          </div>`;
        })
        .join("")}
      <div class="alert alert-warn" style="margin-top:1rem">
        These factors explain <em>baseline risk</em>, not treatment effect heterogeneity.
      </div>`;
  }

  function renderRecs(rec) {
    $("recs").innerHTML = `
      <div class="alert alert-info">Hard constraints cannot be overridden by the LLM/agent. Ineligible rows stay visible for transparency.</div>
      <div class="chart-box" style="overflow:auto">
      <table aria-label="Ranked interventions">
        <thead><tr>
          <th>Rank</th><th>Intervention</th><th>Status</th><th>CATE</th><th>Benefit</th>
          <th>Conf</th><th>Cost</th><th>Burden</th><th>Feas.</th><th>Utility</th>
        </tr></thead>
        <tbody>${rec.rank_list
          .map((s, idx) => {
            const selected = s.intervention === rec.recommended_intervention;
            return `<tr class="${selected ? "selected" : ""}">
              <td>${s.rank}</td>
              <td>${nice(s.intervention)}${selected ? ' <span class="badge badge-gold">selected</span>' : ""}</td>
              <td>${
                s.eligible
                  ? '<span class="badge badge-ok">eligible</span>'
                  : `<span class="badge badge-danger" title="${(s.ineligibility_reasons || []).join("; ")}">blocked</span>`
              }</td>
              <td>${fmtPct(s.uplift?.cate)}</td>
              <td>${fmtMoney(s.expected_benefit)}</td>
              <td>${(s.confidence || 0).toFixed(2)}</td>
              <td>${fmtMoney(s.cost)}</td>
              <td>${(s.customer_burden || 0).toFixed(2)}</td>
              <td>${(s.feasibility || 0).toFixed(2)}</td>
              <td>${s.eligible ? (s.utility || 0).toFixed(3) : "—"}</td>
            </tr>`;
          })
          .join("")}</tbody>
      </table></div>
      <h3 style="margin-top:1rem">Trade-offs</h3>
      <ul>${(rec.tradeoffs || []).map((t) => `<li>${t}</li>`).join("") || "<li>No material trade-offs vs runner-up.</li>"}</ul>
      ${(rec.constraints_applied || []).length
        ? `<h3>Constraints applied</h3><ul>${rec.constraints_applied.map((c) => `<li>${c}</li>`).join("")}</ul>`
        : ""}`;
  }

  function renderImpact(rec) {
    const cf = rec.counterfactual || {};
    $("impact").innerHTML = `
      ${metrics([
        ["Uplift P0 (control)", fmtPct(cf.p_loss_baseline)],
        ["Uplift P1 (treated)", fmtPct(cf.p_loss_with_intervention)],
        ["Risk reduction", fmtPct(cf.expected_risk_reduction)],
        ["Net utility", (cf.net_utility || 0).toFixed(1)],
      ])}
      <div class="grid-2">
        <div class="card">
          <h3>Impact decomposition</h3>
          ${barChart([
            { label: "risk ↓", value: cf.expected_risk_reduction || 0 },
            { label: "benefit $ /5k", value: (cf.expected_severity_reduction || 0) / 5000 },
            { label: "cost /250", value: -((cf.cost || 0) / 250) },
            { label: "burden", value: -(cf.burden || 0) },
          ])}
        </div>
        <div class="card">
          <h3>Caveats</h3>
          <ul>${(cf.caveats || []).map((c) => `<li>${c}</li>`).join("")}</ul>
          <p class="mono">method=${cf.method || "n/a"} · confidence=${(cf.confidence || 0).toFixed(2)}</p>
        </div>
      </div>`;
  }

  function renderEvidence(rec) {
    const trace = rec.agent_trace || [];
    $("evidence").innerHTML = `
      <div class="grid-2">
        <div class="card">
          <h3>Grounded explanation bullets</h3>
          <ul>${(rec.explanation_bullets || []).map((b) => `<li>${b}</li>`).join("")}</ul>
          <div class="alert alert-ok">Numeric claims are model-derived. The agent cannot invent treatment effects.</div>
        </div>
        <div class="card">
          <h3>Agent tool timeline</h3>
          <div class="timeline">
            ${
              trace.length
                ? trace
                    .map(
                      (t) => `<div class="tl-item">
                        <div class="tl-dot"></div>
                        <div>
                          <div class="tool">${t.tool}</div>
                          <div class="meta">${JSON.stringify(t.args)} → ${JSON.stringify(t.preview).slice(0, 120)}</div>
                        </div>
                      </div>`
                    )
                    .join("")
                : `<div class="empty">No agent trace yet. Load a recommendation.</div>`
            }
          </div>
        </div>
      </div>
      <h3 style="margin-top:1rem">Structured evidence</h3>
      <pre class="block">${JSON.stringify(rec.evidence || [], null, 2)}</pre>`;
  }

  async function loadRecommendation() {
    const id = $("customer").value;
    const method = $("method").value;
    if (!id) {
      toast("Select a customer first", true);
      return;
    }
    setBusy(true, "Running agent + models…");
    setPipelineStep(0);
    try {
      setPipelineStep(1);
      const rec = await api(`/customers/${id}/recommend?method=${encodeURIComponent(method)}`);
      setPipelineStep(3);
      state.current = rec;
      renderOverview(rec);
      renderFactors(rec);
      renderRecs(rec);
      renderImpact(rec);
      renderEvidence(rec);
      setPipelineStep(4);
      toast(`Recommended: ${nice(rec.recommended_intervention)}`);
      activateTab("overview");
    } catch (err) {
      toast(String(err.message || err), true);
      $("overview").innerHTML = `<div class="alert alert-danger">Failed to load recommendation: ${err.message}</div>`;
    } finally {
      setBusy(false);
    }
  }

  async function runCounterfactual() {
    const body = {
      customer_id: $("customer").value,
      intervention: $("cfIntervention").value,
    };
    const method = $("method").value;
    setBusy(true, "Simulating counterfactual…");
    try {
      const r = await api(`/counterfactual?method=${encodeURIComponent(method)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      $("cfOut").innerHTML = `
        ${metrics([
          ["Uplift P(loss|control)", fmtPct(r.p_loss_baseline)],
          ["Uplift P(loss|treated)", fmtPct(r.p_loss_with_intervention)],
          ["Δ risk (CATE)", fmtPct(r.expected_risk_reduction)],
          ["Net utility", (r.net_utility || 0).toFixed(1)],
        ])}
        <div class="card">
          <p>Benefit ${fmtMoney(r.expected_severity_reduction)} · cost ${fmtMoney(r.cost)} ·
          burden ${(r.burden || 0).toFixed(2)} · confidence ${(r.confidence || 0).toFixed(2)}
          ${
            r.p_loss_predictive != null
              ? ` · predictive P(loss)=${fmtPct(r.p_loss_predictive)}`
              : ""
          }</p>
          <ul>${(r.caveats || []).map((c) => `<li>${c}</li>`).join("")}</ul>
        </div>`;
      toast("Counterfactual ready");
    } catch (err) {
      $("cfOut").innerHTML = `<div class="alert alert-danger">${err.message}</div>`;
      toast(err.message, true);
    } finally {
      setBusy(false);
    }
  }

  async function loadMethodCompare() {
    const id = $("customer").value;
    const intervention = $("methodIntervention").value;
    setBusy(true, "Comparing meta-learners…");
    try {
      const data = await api(
        `/customers/${id}/method-compare?intervention=${encodeURIComponent(intervention)}`
      );
      const rows = Object.entries(data.methods).map(([m, v]) => ({
        label: m.replace("_learner", "").replace("_", " "),
        cate: v.cate,
      }));
      $("methodsOut").innerHTML = `
        <p>Intervention: <strong>${nice(intervention)}</strong></p>
        ${upliftCompareSvg(rows)}
        <table style="margin-top:1rem"><thead><tr><th>Method</th><th>CATE</th><th>P0</th><th>P1</th><th>Conf</th></tr></thead>
        <tbody>${Object.entries(data.methods)
          .map(
            ([m, v]) => `<tr>
              <td class="mono">${m}</td>
              <td>${fmtPct(v.cate)}</td>
              <td>${fmtPct(v.p_loss_control)}</td>
              <td>${fmtPct(v.p_loss_treated)}</td>
              <td>${v.confidence == null ? "—" : v.confidence.toFixed(2)}</td>
            </tr>`
          )
          .join("")}</tbody></table>
        <div class="alert alert-info">S/T/X learners and causal forest estimate the same causal quantity with different inductive biases.</div>`;
    } catch (err) {
      $("methodsOut").innerHTML = `<div class="alert alert-danger">${err.message}</div>`;
      toast(err.message, true);
    } finally {
      setBusy(false);
    }
  }

  async function runResearch() {
    setBusy(true, "Running research experiments…");
    activateTab("research");
    $("research").innerHTML = `<div class="loading"><span class="spinner"></span> Running deployment + experiment suite…</div>`;
    try {
      const [deploy, experiments, metrics] = await Promise.all([
        api("/research/deploy"),
        api("/research/experiments"),
        api("/models/metrics"),
      ]);
      $("research").innerHTML = `
        <div class="grid-3">
          <div class="metric"><div class="k">Predictive AUC</div><div class="v">${deploy.predictive_auc.toFixed(3)}</div></div>
          <div class="metric"><div class="k">Uplift Qini</div><div class="v">${deploy.uplift_qini.toFixed(3)}</div></div>
          <div class="metric"><div class="k">Policy value</div><div class="v">${deploy.policy_value.toFixed(3)}</div></div>
        </div>
        <div class="grid-2">
          <div class="card">
            <h3>Deployment metrics</h3>
            <pre class="block">${JSON.stringify(deploy, null, 2)}</pre>
          </div>
          <div class="card">
            <h3>Experiment takeaways</h3>
            ${experiments
              .map(
                (e) => `<div style="margin-bottom:1rem">
                  <strong>${e.name}</strong>
                  <p>${e.summary}</p>
                  <ul>${(e.takeaways || []).map((t) => `<li>${t}</li>`).join("")}</ul>
                </div>`
              )
              .join("")}
          </div>
        </div>
        <details style="margin-top:1rem"><summary>Training metrics</summary>
          <pre class="block">${JSON.stringify(metrics, null, 2)}</pre>
        </details>`;
      toast("Research snapshot loaded");
    } catch (err) {
      $("research").innerHTML = `<div class="alert alert-danger">${err.message}</div>`;
      toast(err.message, true);
    } finally {
      setBusy(false);
    }
  }

  function renderArchitecture() {
    const a = state.architecture || { pipeline: [], models: [], agent_tools: [], responsible_ai: [] };
    $("architecture").innerHTML = `
      <div class="grid-2">
        <div class="card">
          <h3>Decision pipeline</h3>
          <ol>${a.pipeline.map((s) => `<li class="mono">${s}</li>`).join("")}</ol>
        </div>
        <div class="card">
          <h3>Models & tools</h3>
          <p><strong>Models</strong></p>
          <div>${a.models.map((m) => `<span class="badge badge-neutral" style="margin:.15rem">${m}</span>`).join("")}</div>
          <p style="margin-top:.8rem"><strong>Agent tools</strong></p>
          <div>${a.agent_tools.map((m) => `<span class="badge badge-ok" style="margin:.15rem">${m}</span>`).join("")}</div>
        </div>
      </div>
      <div class="card" style="margin-top:1rem">
        <h3>Responsible AI guards</h3>
        <ul>${a.responsible_ai.map((x) => `<li>${nice(x)}</li>`).join("")}</ul>
      </div>`;
  }

  async function init() {
    $("pipeline").innerHTML = PIPELINE.map(
      ([n, t], i) =>
        `<div class="pipe-step ${i === 0 ? "active" : ""}" data-step="${i}">
          <div class="n">STAGE ${n}</div><div class="t">${t}</div>
        </div>`
    ).join("");

    document.querySelectorAll(".side-nav button").forEach((btn) => {
      btn.addEventListener("click", () => activateTab(btn.dataset.tab));
    });

    setBusy(true, "Bootstrapping…");
    try {
      const [customers, interventions, architecture] = await Promise.all([
        api("/customers?limit=120"),
        api("/interventions"),
        api("/architecture"),
      ]);
      state.customers = customers;
      state.interventions = interventions;
      state.architecture = architecture;
      if (!customers.length) {
        $("overview").innerHTML = `<div class="empty">No customers available. Train/bootstrap the pipeline first.</div>`;
        setBusy(false);
        return;
      }
      $("customer").innerHTML = customers
        .map(
          (c) =>
            `<option value="${c.customer_id}">${c.customer_id} · ${c.line_of_business} · claims=${c.prior_claims_3y} · hazard=${c.hazard_exposure.toFixed(2)}</option>`
        )
        .join("");
      const opts = interventions
        .map((i) => `<option value="${i.intervention}">${i.label}</option>`)
        .join("");
      $("cfIntervention").innerHTML = opts;
      $("methodIntervention").innerHTML = opts;
      renderArchitecture();
      await loadRecommendation();
    } catch (err) {
      $("overview").innerHTML = `<div class="alert alert-danger">Startup failed: ${err.message}</div>`;
      toast(err.message, true);
      setBusy(false);
    }
  }

  $("loadBtn").addEventListener("click", loadRecommendation);
  $("cfBtn").addEventListener("click", runCounterfactual);
  $("methodBtn").addEventListener("click", loadMethodCompare);
  $("researchBtn").addEventListener("click", runResearch);

  // Auto-refresh recommendation when customer/method changes (debounced).
  let reloadTimer = null;
  function scheduleReload() {
    clearTimeout(reloadTimer);
    reloadTimer = setTimeout(() => {
      if (!state.busy) loadRecommendation();
    }, 280);
  }
  $("customer").addEventListener("change", scheduleReload);
  $("method").addEventListener("change", scheduleReload);

  // Keyboard: Alt+1..9 switches tabs; Enter on focused controls activates.
  const tabOrder = [
    "overview",
    "factors",
    "recs",
    "impact",
    "evidence",
    "cf",
    "methods",
    "research",
    "architecture",
  ];
  document.addEventListener("keydown", (ev) => {
    if (ev.altKey && ev.key >= "1" && ev.key <= "9") {
      const idx = Number(ev.key) - 1;
      if (tabOrder[idx]) {
        ev.preventDefault();
        activateTab(tabOrder[idx]);
      }
    }
  });

  window.LPApp = { state, loadRecommendation, runCounterfactual, activateTab, api };
  init();
})();
