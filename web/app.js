"use strict";

const DATA_DIR = "./data";
const app = document.getElementById("app");
const headerMeta = document.getElementById("header-meta");
const footerModel = document.getElementById("footer-model");

const CONDITION_TITLES = {
  routine: "Routine trade-off",
  taboo: "Taboo trade-off (sacred vs. secular)",
  tragic: "Tragic trade-off (sacred vs. sacred)",
};

const state = {
  index: null,
  data: null, // current scenario
  step: -1, // -1 = nothing revealed yet
  playing: false,
  timer: null,
  completed: false, // true once user has revealed the whole response
};

/* ----------------------------- helpers ----------------------------- */

function scenarioTitle(exp) {
  // e.g. "Improve flood protection  vs.  Renovate the village square"
  return `${exp.option_a} vs. ${exp.option_b}`;
}

function isSpecialToken(t) {
  return /^<\|.*\|>$/.test(t.trim());
}

function fmtPct(x) {
  return `${Math.round(x * 100)}%`;
}

async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
}

/* ----------------------------- routing ----------------------------- */

async function route() {
  stopPlay();
  const hash = location.hash || "#/";
  const m = hash.match(/^#\/exp\/(.+)$/);
  if (m) {
    await renderExperiment(decodeURIComponent(m[1]));
  } else {
    renderHome();
  }
}

/* ----------------------------- home ----------------------------- */

function renderHome() {
  headerMeta.innerHTML = state.index
    ? `${state.index.n_experiments} experiments · ${state.index.model}`
    : "";
  const tpl = document.getElementById("tpl-home").content.cloneNode(true);
  app.innerHTML = "";
  app.appendChild(tpl);

  const grid = document.getElementById("card-grid");
  for (const exp of state.index.experiments) {
    const card = el("button", "card");
    card.appendChild(
      (() => {
        const top = el("div", "card-top");
        top.appendChild(el("span", `badge ${exp.condition}`, exp.condition));
        top.appendChild(el("span", "card-sub", `Exp ${exp.experiment}`));
        return top;
      })()
    );
    card.appendChild(el("h3", null, CONDITION_TITLES[exp.condition] || exp.condition));

    const vs = el("div", "vs");
    vs.appendChild(el("div", "opt a", exp.option_a));
    vs.appendChild(el("div", "divider", "VS"));
    vs.appendChild(el("div", "opt b", exp.option_b));
    card.appendChild(vs);

    const foot = el("div", "card-foot");
    foot.appendChild(el("span", null, `${exp.domain.replace(/_/g, " ")}`));
    const coh =
      exp.mean_coherence != null ? fmtPct(exp.mean_coherence) : "—";
    foot.appendChild(el("span", null, `mean coherence ${coh} →`));
    card.appendChild(foot);

    card.addEventListener("click", () => {
      location.hash = `#/exp/${encodeURIComponent(exp.scenario_id)}`;
    });
    grid.appendChild(card);
  }
}

/* ----------------------------- experiment ----------------------------- */

async function renderExperiment(id) {
  app.innerHTML = `<div class="loading">Loading experiment…</div>`;
  let data;
  try {
    data = await loadJSON(`${DATA_DIR}/${id}.json`);
  } catch (e) {
    app.innerHTML = `<div class="loading">Could not load <code>${id}</code>.</div>`;
    return;
  }
  state.data = data;
  state.step = -1;
  state.playing = false;
  state.completed = false;

  headerMeta.innerHTML = `${CONDITION_TITLES[data.condition] || data.condition}`;

  app.innerHTML = "";
  const head = el("div", "exp-head");
  const back = el("button", "back", "&larr; All experiments");
  back.addEventListener("click", () => (location.hash = "#/"));
  head.appendChild(back);
  const title = el("div", "exp-title");
  title.appendChild(el("span", `badge ${data.condition}`, data.condition));
  title.appendChild(el("h2", null, scenarioTitle(data)));
  head.appendChild(title);
  head.appendChild(
    el(
      "div",
      "exp-meta",
      `Experiment ${data.experiment} · ${data.domain.replace(/_/g, " ")} · ${
        data.n_steps
      } generated tokens · model ${data.model}`
    )
  );
  app.appendChild(head);

  const grid = el("div", "exp-grid");
  grid.appendChild(buildScenarioPanel(data));
  grid.appendChild(buildChatPanel(data));
  grid.appendChild(buildReadoutPanel(data));
  app.appendChild(grid);

  renderStep();
}

function buildScenarioPanel(data) {
  const panel = el("div", "panel");
  panel.appendChild(el("div", "panel-head", "<span>The dilemma</span>"));
  const body = el("div", "panel-body");

  // scenario text = prompt minus the instruction wrapper, fall back to prompt
  const scenarioText = extractScenario(data.prompt);
  body.appendChild(el("p", "scenario-text", escapeHTML(scenarioText)));

  const opts = el("div", "options-block");
  const a = el("div", "option-card a");
  a.innerHTML = `<span class="tag">Option A · ${data.value_a.replace(
    /_/g,
    " "
  )}</span>${escapeHTML(data.option_a)}<div class="sacred">${
    data.sacredness_a || ""
  }</div>`;
  const b = el("div", "option-card b");
  b.innerHTML = `<span class="tag">Option B · ${data.value_b.replace(
    /_/g,
    " "
  )}</span>${escapeHTML(data.option_b)}<div class="sacred">${
    data.sacredness_b || ""
  }</div>`;
  opts.appendChild(a);
  opts.appendChild(b);
  body.appendChild(opts);
  panel.appendChild(body);
  return panel;
}

function extractScenario(prompt) {
  // The prompt wraps the raw scenario after the instruction block.
  const marker = "Final choice: Option 2";
  const idx = prompt.indexOf(marker);
  if (idx !== -1) {
    return prompt.slice(idx + marker.length).trim();
  }
  return prompt.trim();
}

function buildChatPanel(data) {
  const panel = el("div", "panel");
  panel.appendChild(
    el(
      "div",
      "panel-head",
      `<span>Model response</span><span class="hint">advance, or click a token</span>`
    )
  );
  const body = el("div", "panel-body");
  const chat = el("div", "chat");
  const user = el("div", "bubble user");
  user.innerHTML = `You must choose between <b>Option&nbsp;A</b> and <b>Option&nbsp;B</b>. Explain briefly, then commit.`;
  chat.appendChild(user);
  const assistant = el("div", "bubble assistant");
  assistant.id = "assistant-bubble";
  assistant.addEventListener("click", (e) => {
    if (!state.completed) return;
    const tok = e.target.closest(".tok");
    if (!tok || tok.dataset.step == null) return;
    stopPlay();
    setStep(parseInt(tok.dataset.step, 10));
  });
  chat.appendChild(assistant);
  body.appendChild(chat);
  panel.appendChild(body);

  // controls
  const controls = el("div", "controls");
  const btnRow = el("div", "btn-row");
  btnRow.innerHTML = `
    <button class="btn primary" id="btn-advance">Advance ▸</button>
    <button class="btn" id="btn-play">Play ⏵</button>
    <button class="btn" id="btn-back">◂ Back</button>
    <button class="btn" id="btn-decision">Jump to choice ⤓</button>
    <button class="btn" id="btn-reset">↺ Reset</button>
  `;
  controls.appendChild(btnRow);

  const prog = el("div", "progress-row");
  prog.innerHTML = `<span id="step-label">0 / ${data.n_steps}</span>`;
  const slider = el("input");
  slider.type = "range";
  slider.min = "-1";
  slider.max = String(data.n_steps - 1);
  slider.value = "-1";
  slider.id = "step-slider";
  prog.appendChild(slider);
  controls.appendChild(prog);
  panel.appendChild(controls);

  // wire
  btnRow.querySelector("#btn-advance").addEventListener("click", () => advance(1));
  btnRow.querySelector("#btn-back").addEventListener("click", () => advance(-1));
  btnRow.querySelector("#btn-reset").addEventListener("click", () => {
    stopPlay();
    setStep(-1);
  });
  btnRow.querySelector("#btn-decision").addEventListener("click", () => {
    stopPlay();
    const d = data.final.decision_step;
    setStep(d != null ? d : data.n_steps - 1);
  });
  btnRow.querySelector("#btn-play").addEventListener("click", togglePlay);
  slider.addEventListener("input", (e) => {
    stopPlay();
    setStep(parseInt(e.target.value, 10));
  });

  return panel;
}

function buildReadoutPanel(data) {
  const panel = el("div", "panel");
  panel.appendChild(
    el(
      "div",
      "panel-head",
      `<span>J-space readout</span><span class="hint">per layer · A ◀ ▶ B</span>`
    )
  );
  const body = el("div", "panel-body");
  body.id = "readout-body";
  panel.appendChild(body);
  return panel;
}

/* ----------------------------- stepping ----------------------------- */

function advance(delta) {
  setStep(clampStep(state.step + delta));
}

function clampStep(s) {
  return Math.max(-1, Math.min(state.data.n_steps - 1, s));
}

function setStep(s) {
  state.step = clampStep(s);
  if (state.step >= state.data.n_steps - 1) state.completed = true;
  renderStep();
}

function togglePlay() {
  if (state.playing) stopPlay();
  else startPlay();
}

function startPlay() {
  if (state.step >= state.data.n_steps - 1) setStep(-1);
  state.playing = true;
  updatePlayBtn();
  state.timer = setInterval(() => {
    if (state.step >= state.data.n_steps - 1) {
      stopPlay();
      return;
    }
    advance(1);
  }, 220);
}

function stopPlay() {
  state.playing = false;
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  updatePlayBtn();
}

function updatePlayBtn() {
  const b = document.getElementById("btn-play");
  if (b) b.textContent = state.playing ? "Pause ⏸" : "Play ⏵";
}

function renderStep() {
  if (!state.data) return;
  renderTranscript();
  renderReadout();
  const slider = document.getElementById("step-slider");
  if (slider) slider.value = String(state.step);
  const label = document.getElementById("step-label");
  if (label)
    label.textContent = `${state.step + 1} / ${state.data.n_steps}`;
  const advBtn = document.getElementById("btn-advance");
  if (advBtn) advBtn.disabled = state.step >= state.data.n_steps - 1;
  const backBtn = document.getElementById("btn-back");
  if (backBtn) backBtn.disabled = state.step < 0;
}

function renderTranscript() {
  const bubble = document.getElementById("assistant-bubble");
  if (!bubble) return;
  const n = state.data.n_steps;
  // Click-to-sync + full-response preview only unlock once the user has
  // revealed the whole response, so they engage with the dilemma first.
  const interactive = state.completed;
  bubble.classList.toggle("interactive", interactive);

  const frag = document.createDocumentFragment();
  if (state.step < 0) {
    frag.appendChild(
      el(
        "span",
        "placeholder",
        interactive
          ? "Click any token to sync the readout to that moment…"
          : "Press “Advance” or “Play” to reveal the model’s response token by token…"
      )
    );
  }
  // Before completion, only reveal up to the current step; after, show all
  // tokens (with the not-yet-passed ones dimmed) and make them clickable.
  const last = interactive ? n - 1 : state.step;
  for (let i = 0; i <= last; i++) {
    const t = state.data.steps[i].token;
    const special = isSpecialToken(t);
    const revealed = i <= state.step;
    const span = el(
      "span",
      `tok${i === state.step ? " current" : ""}${
        revealed ? "" : " future"
      }${special ? " special" : ""}`
    );
    span.textContent = t;
    if (interactive) {
      span.dataset.step = String(i);
      span.title = `token ${i + 1} / ${n} — click to sync readout`;
    }
    frag.appendChild(span);
    if (i === state.step && state.step < n - 1) {
      frag.appendChild(el("span", "caret", "&nbsp;"));
    }
  }
  bubble.innerHTML = "";
  bubble.appendChild(frag);
}

function renderReadout() {
  const body = document.getElementById("readout-body");
  if (!body) return;
  const data = state.data;

  if (state.step < 0) {
    body.innerHTML = `<p class="readout-status">Advance the response to see what each
      layer's internal state is leaning toward at that moment.</p>`;
    appendVerdict(body, false);
    return;
  }

  const step = data.steps[state.step];
  const jdir = step.j_direction;
  const bestLayer = data.final.best_layer;

  // headline: use best predictive layer (or mid) as the summary direction
  const summaryLayer =
    bestLayer != null && jdir[String(bestLayer)] != null
      ? bestLayer
      : data.layers[Math.floor(data.layers.length / 2)];
  const sVal = jdir[String(summaryLayer)] ?? 0;
  const lean = leanLabel(sVal, data);

  body.innerHTML = "";
  const status = el("p", "readout-status");
  status.innerHTML = `Token <b>${escapeHTML(JSON.stringify(step.token))}</b> ·
    layer ${summaryLayer} leans <b>${lean.text}</b>`;
  body.appendChild(status);

  const key = el("div", "axis-key");
  key.innerHTML = `<span class="ka">◀ ${escapeHTML(shorten(data.option_a))}</span>
    <span class="kb">${escapeHTML(shorten(data.option_b))} ▶</span>`;
  body.appendChild(key);

  const ladder = el("div", "ladder");
  for (const layer of data.layers) {
    const d = jdir[String(layer)];
    if (d == null) continue;
    const row = el("div", `lrow${layer === bestLayer ? " best" : ""}`);
    row.appendChild(el("div", "lname", String(layer)));
    const track = el("div", "track");
    const fill = el("div", `fill ${d < 0 ? "a" : "b"}`);
    const mag = Math.min(1, Math.abs(d)) * 50;
    if (d < 0) {
      fill.style.right = "50%";
      fill.style.width = `${mag}%`;
    } else {
      fill.style.left = "50%";
      fill.style.width = `${mag}%`;
    }
    track.appendChild(fill);
    row.appendChild(track);
    ladder.appendChild(row);
  }
  body.appendChild(ladder);

  // top-k lens tokens (if exported)
  if (step.topk && Object.keys(step.topk).length) {
    const topk = el("div", "topk");
    topk.appendChild(
      el(
        "h4",
        null,
        "Lens verbalization (top tokens the residual would emit)"
      )
    );
    for (const layer of data.topk_layers) {
      const toks = step.topk[String(layer)];
      if (!toks) continue;
      const tl = el("div", "topk-layer");
      tl.appendChild(el("div", "ll", `layer ${layer}`));
      const chips = el("div", "chips");
      toks.forEach((tk, i) => {
        const chip = el("span", `chip${i === 0 ? " rank0" : ""}`);
        chip.textContent = tk === "" ? "␠" : tk.replace(/\n/g, "⏎");
        chips.appendChild(chip);
      });
      tl.appendChild(chips);
      topk.appendChild(tl);
    }
    body.appendChild(topk);
  }

  appendVerdict(body, state.step >= (data.final.decision_step ?? 1e9));
}

function appendVerdict(body, reached) {
  const data = state.data;
  const f = data.final;
  const v = el("div", "verdict");
  const choiceTxt =
    f.choice === "A"
      ? `Option A — ${shorten(data.option_a)}`
      : f.choice === "B"
      ? `Option B — ${shorten(data.option_b)}`
      : "ambiguous";

  v.innerHTML = `<h4>${reached ? "Committed choice" : "Where this is heading"}</h4>`;
  const rows = el("div");
  rows.innerHTML = `
    <div class="row"><span>Model's text choice</span><b>${escapeHTML(
      choiceTxt
    )}</b></div>
    <div class="row"><span>Output direction</span><b>${fmtDir(
      f.output_direction_final
    )}</b></div>
    <div class="row"><span>Mean J↔text coherence</span><b>${
      f.mean_coherence != null ? fmtPct(f.mean_coherence) : "—"
    }</b></div>
    <div class="row"><span>Most predictive layer</span><b>${
      f.best_layer ?? "—"
    }</b></div>
  `;
  v.appendChild(rows);
  if (f.mean_coherence != null) {
    const gauge = el("div", "gauge");
    gauge.innerHTML = `<span style="width:${Math.round(
      f.mean_coherence * 100
    )}%"></span>`;
    v.appendChild(gauge);
  }
  body.appendChild(v);
}

/* ----------------------------- small utils ----------------------------- */

function leanLabel(d, data) {
  if (d < -0.15) return { text: `Option A` };
  if (d > 0.15) return { text: `Option B` };
  return { text: "neither strongly" };
}

function fmtDir(d) {
  if (d == null) return "—";
  if (d <= -0.15) return `A (${d.toFixed(2)})`;
  if (d >= 0.15) return `B (+${d.toFixed(2)})`;
  return `neutral (${d.toFixed(2)})`;
}

function shorten(s, n = 34) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function escapeHTML(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* ----------------------------- keyboard ----------------------------- */

document.addEventListener("keydown", (e) => {
  if (!state.data || location.hash.indexOf("#/exp/") !== 0) return;
  if (e.key === "ArrowRight") {
    e.preventDefault();
    stopPlay();
    advance(1);
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    stopPlay();
    advance(-1);
  } else if (e.key === " ") {
    e.preventDefault();
    togglePlay();
  }
});

/* ----------------------------- boot ----------------------------- */

async function boot() {
  try {
    state.index = await loadJSON(`${DATA_DIR}/index.json`);
    footerModel.textContent = state.index.model;
  } catch (e) {
    app.innerHTML = `<div class="loading">Could not load data index. Did you run
      <code>scripts/export_web_data.py</code> and serve this folder over HTTP?</div>`;
    return;
  }
  window.addEventListener("hashchange", route);
  route();
}

boot();
