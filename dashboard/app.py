"""
AEGIS Dashboard — FastAPI backend
Serves the threat intelligence report and proxies result/catalog data.

Run:  uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
"""
import json
import yaml
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="AEGIS")
BASE = Path(__file__).parent.parent


@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML


@app.get("/api/results")
def results():
    p = BASE / "blue/results/summary.json"
    if not p.exists():
        return JSONResponse({"error": "Blue stack results not found. Run: python -m blue.run_blue"}, status_code=503)
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/catalog")
def catalog():
    p = BASE / "attack_catalog.yaml"
    data = yaml.safe_load(p.read_text())
    return JSONResponse(data)


@app.post("/api/simulate/{vector_id}")
def simulate(vector_id: str):
    from blue import infer
    try:
        return JSONResponse(infer.simulate(vector_id.upper()))
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Simulation failed: {e}"}, status_code=500)


@app.get("/health")
def health():
    return {"status": "ok"}


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AEGIS · Adversarial Fraud Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Unbounded:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:     #09090B;
      --ink:    #E7E9EC;
      --mute:   #7D8590;
      --faint:  #454A52;
      --line:   #1E2126;
      --signal: #FF3B4E;
      --accent: #3DE8D0;
      --display: 'Unbounded', sans-serif;
      --sans:   'IBM Plex Sans', sans-serif;
      --mono:   'IBM Plex Mono', monospace;
    }

    html { background: var(--bg); }
    body {
      background:
        linear-gradient(var(--bg), var(--bg)),
        repeating-linear-gradient(0deg, rgba(255,255,255,0.025) 0 1px, transparent 1px 44px),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.025) 0 1px, transparent 1px 44px);
      color: var(--ink);
      font-family: var(--sans);
      font-size: 15px;
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
      opacity: 0;
      animation: pageIn 0.4s ease forwards;
    }
    @keyframes pageIn { to { opacity: 1; } }
    @media (prefers-reduced-motion: reduce) { body { animation: none; opacity: 1; } }

    a { color: inherit; }
    :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

    .page { max-width: 880px; margin: 0 auto; padding: 56px 24px 96px; }

    /* ── masthead ─────────────────────────────────────────────────── */
    .masthead { display: flex; align-items: baseline; justify-content: space-between;
                border-bottom: 1px solid var(--line); padding-bottom: 16px; margin-bottom: 4px; }
    .wordmark-wrap { position: relative; display: inline-block; }
    .wordmark { font-family: var(--display); font-weight: 600; font-size: 26px; letter-spacing: 0.02em; }
    .wordmark-mark { position: absolute; left: 1px; bottom: -8px; width: 34px; height: 2px; background: var(--accent); }
    .masthead-right { text-align: right; }
    .masthead-clock { font-family: var(--mono); font-size: 12px; color: var(--mute); display: flex; align-items: center; gap: 7px; justify-content: flex-end; }
    .live-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); animation: blink 2.4s infinite; }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
    @media (prefers-reduced-motion: reduce) { .live-dot { animation: none; } }
    .masthead-sub { border-bottom: 1px solid var(--line); padding-bottom: 22px; margin-bottom: 40px; }
    .masthead-sub p { font-size: 13px; color: var(--mute); margin-top: 8px; max-width: 560px; }

    /* ── quick nav ────────────────────────────────────────────────── */
    .quicknav { position: sticky; top: 0; z-index: 10; display: flex; align-items: center; gap: 22px;
                background: var(--bg); border-bottom: 1px solid var(--line);
                padding: 12px 0; margin-bottom: 40px; }
    .quicknav a { font-family: var(--mono); font-size: 11.5px; color: var(--mute); letter-spacing: 0.04em;
                  text-decoration: none; text-transform: uppercase; padding-bottom: 2px; border-bottom: 1px solid transparent; }
    .quicknav a:hover, .quicknav a:focus-visible { color: var(--accent); border-bottom-color: var(--accent); }
    .quicknav-hint { font-family: var(--mono); font-size: 11px; color: var(--faint); margin-left: auto; }
    .quicknav-hint kbd { font-family: var(--mono); border: 1px solid var(--line); border-radius: 3px; padding: 1px 5px; color: var(--mute); }

    /* ── ledger controls ─────────────────────────────────────────── */
    .controls { display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
                padding: 14px 0; margin-bottom: 8px; border-bottom: 1px solid var(--line); }
    .search-input { background: transparent; border: none; border-bottom: 1px solid var(--line);
                     color: var(--ink); font-family: var(--mono); font-size: 13px; padding: 5px 2px;
                     width: 200px; }
    .search-input::placeholder { color: var(--faint); }
    .search-input:focus { outline: none; border-bottom-color: var(--accent); }
    .chip-row { display: flex; gap: 6px; flex-wrap: wrap; }
    .chip { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.03em; color: var(--mute);
            background: transparent; border: 1px solid var(--line); border-radius: 3px;
            padding: 4px 9px; cursor: pointer; text-transform: lowercase; }
    .chip:hover { color: var(--ink); border-color: var(--mute); }
    .chip.active { color: var(--accent); border-color: var(--accent); }
    .control-group { display: flex; align-items: center; gap: 6px; }
    .control-label { font-family: var(--mono); font-size: 10.5px; color: var(--faint); text-transform: uppercase; letter-spacing: 0.05em; margin-right: 2px; }
    .toggle { display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 11.5px;
              color: var(--mute); cursor: pointer; user-select: none; }
    .toggle input { accent-color: var(--accent); width: 13px; height: 13px; cursor: pointer; }
    .toggle:hover { color: var(--ink); }
    .reset-link { background: none; border: none; font-family: var(--mono); font-size: 11px; color: var(--faint);
                   text-decoration: underline; cursor: pointer; padding: 0; }
    .reset-link:hover { color: var(--accent); }
    .result-count { font-family: var(--mono); font-size: 11px; color: var(--faint); margin-left: auto; }
    .channel-group.hidden-group { display: none; }

    /* ── live simulation ─────────────────────────────────────────── */
    .sim-controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    .sim-select { background: transparent; border: none; border-bottom: 1px solid var(--line);
                  color: var(--ink); font-family: var(--mono); font-size: 12.5px; padding: 7px 4px;
                  max-width: 320px; }
    .sim-select:focus { outline: none; border-bottom-color: var(--accent); }
    .btn { font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.04em; text-transform: lowercase;
           border-radius: 3px; padding: 8px 16px; cursor: pointer; background: transparent; }
    .btn-primary { border: 1px solid var(--accent); color: var(--accent); }
    .btn-primary:hover { background: var(--accent); color: var(--bg); }
    .btn-ghost { border: 1px solid var(--line); color: var(--mute); }
    .btn-ghost:hover { border-color: var(--mute); color: var(--ink); }
    .sim-tally { font-family: var(--mono); font-size: 11px; color: var(--faint); margin-left: auto; }
    .sim-tally strong { color: var(--ink); }

    .console { border: 1px solid var(--line); padding: 16px 18px; min-height: 140px; max-height: 380px;
               overflow-y: auto; font-family: var(--mono); font-size: 12px; line-height: 1.9; }
    .console-placeholder { color: var(--faint); }
    .console-line { color: var(--mute); opacity: 0; animation: lineIn 0.15s ease forwards; white-space: pre; }
    @keyframes lineIn { to { opacity: 1; } }
    @media (prefers-reduced-motion: reduce) { .console-line { animation: none; opacity: 1; } }
    .console-line.cl-info { color: var(--faint); }
    .console-line.cl-error { color: var(--signal); }
    .console-line.cl-legit { color: var(--faint); }
    .console-line.cl-fraud { color: var(--ink); font-weight: 500; }
    .console-line .amt { color: var(--ink); }
    .console-line.cl-fraud .amt { color: var(--accent); }

    .verdict { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--line); opacity: 0; animation: lineIn 0.25s ease forwards; }
    @media (prefers-reduced-motion: reduce) { .verdict { animation: none; opacity: 1; } }
    .verdict-label { font-family: var(--display); font-weight: 600; font-size: 22px; letter-spacing: 0.02em; }
    .verdict.verdict-caught .verdict-label { color: var(--accent); }
    .verdict.verdict-missed .verdict-label { color: var(--signal); }
    .verdict-detail { font-family: var(--sans); font-size: 12.5px; color: var(--mute); margin-top: 6px; }
    .verdict-detail strong { font-family: var(--mono); color: var(--ink); }

    .row-run-btn { font-family: var(--mono); font-size: 10px; color: var(--mute); border: 1px solid var(--line);
                   border-radius: 3px; padding: 5px 10px; cursor: pointer; background: transparent; }
    .row-run-btn:hover { color: var(--accent); border-color: var(--accent); }
    tr.hidden-row { display: none; }

    /* ── headline stats: the four judging axes ───────────────────── */
    .headline { display: grid; grid-template-columns: repeat(4, 1fr); margin-bottom: 56px; }
    .stat { padding: 0 20px; border-left: 1px solid var(--line); }
    .stat:first-child { border-left: none; padding-left: 0; }
    .stat-axis { font-family: var(--mono); font-size: 10px; letter-spacing: 0.14em; color: var(--faint);
                 text-transform: uppercase; display: flex; align-items: center; gap: 6px; }
    .stat-axis .icon { color: var(--faint); }
    .stat-num { font-family: var(--display); font-weight: 600; font-size: 30px; line-height: 1.15; margin-top: 8px; color: var(--ink); }
    .stat-num.accent { color: var(--accent); }
    .stat-label { font-size: 12px; color: var(--mute); margin-top: 5px; }

    /* ── section headers ─────────────────────────────────────────── */
    h2 { font-family: var(--display); font-weight: 600; font-size: 16px; letter-spacing: 0.01em; margin-bottom: 4px; }
    .section { margin-bottom: 52px; }
    .section-note { font-size: 12.5px; color: var(--mute); margin-bottom: 22px; max-width: 620px; }

    /* ── the ledger ───────────────────────────────────────────────── */
    .channel-group { margin-bottom: 30px; }
    .channel-head { display: flex; align-items: baseline; justify-content: space-between;
                    font-family: var(--mono); font-size: 12px; color: var(--mute);
                    border-bottom: 1px solid var(--line); padding-bottom: 6px; margin-bottom: 2px; }
    .channel-head strong { color: var(--ink); font-weight: 600; letter-spacing: 0.02em; }

    .icon { display: inline-block; vertical-align: middle; color: var(--mute); flex-shrink: 0; }

    table.ledger { width: 100%; border-collapse: collapse; }
    table.ledger tr.row { cursor: pointer; }
    table.ledger tr.row:hover td { color: var(--ink); }
    table.ledger tr.row:hover .icon { color: var(--accent); }
    table.ledger tr.row.expanded .col-name { color: var(--ink); }
    table.ledger tr.row.expanded .icon { color: var(--accent); }
    table.ledger td { padding: 9px 0; border-bottom: 1px solid var(--line); font-size: 13.5px; vertical-align: middle; }
    table.ledger tr.row:last-of-type td { border-bottom: none; }
    table.ledger tr.detail-row td { border-bottom: 1px solid var(--line); padding: 0; }
    table.ledger tr.detail-row:last-child td { border-bottom: none; }
    .col-chev  { width: 16px; color: var(--faint); }
    .col-chev .chevron { transition: transform 0.15s ease; }
    tr.expanded .col-chev .chevron { transform: rotate(90deg); }
    @media (prefers-reduced-motion: reduce) { .col-chev .chevron { transition: none; } }
    .col-id    { font-family: var(--mono); font-size: 12px; color: var(--mute); width: 52px; white-space: nowrap; }
    .col-name  { color: var(--ink); }
    .col-name .zd-flag { color: var(--accent); font-family: var(--mono); }
    .col-mod   { font-family: var(--mono); font-size: 10.5px; color: var(--mute); letter-spacing: 0.05em;
                 width: 100px; white-space: nowrap; }
    .col-mod-inner { display: flex; align-items: center; gap: 6px; }
    .col-bar   { width: 130px; }
    .col-val   { font-family: var(--mono); font-size: 12.5px; text-align: right; width: 48px; white-space: nowrap; }

    .bar-track { position: relative; height: 3px; background: var(--line); border-radius: 0; }
    .bar-fill  { position: absolute; left: 0; top: 0; height: 100%; background: var(--accent); }
    .bar-fill.miss { background: transparent; }
    .bar-miss-mark { position: absolute; left: 0; top: -3px; width: 2px; height: 9px; background: var(--signal); }
    .bar-none { font-family: var(--mono); font-size: 12px; color: var(--faint); }

    .ledger-footnote { font-size: 11.5px; color: var(--faint); margin-top: 10px; font-family: var(--sans); }
    .ledger-footnote em { font-style: normal; color: var(--accent); }

    /* ── detail panel ─────────────────────────────────────────────── */
    .detail { padding: 4px 0 22px 22px; animation: detailIn 0.15s ease; }
    @keyframes detailIn { from { opacity: 0.4; } to { opacity: 1; } }
    @media (prefers-reduced-motion: reduce) { .detail { animation: none; } }
    .detail-desc { font-size: 13.5px; color: var(--ink); max-width: 640px; line-height: 1.6; margin-bottom: 14px; }
    .detail-signal { font-family: var(--mono); font-size: 12px; color: var(--accent); background: var(--line);
                      border-left: 2px solid var(--accent); padding: 10px 12px; max-width: 640px;
                      line-height: 1.7; margin-bottom: 14px; white-space: pre-wrap; }
    .detail-grid { display: flex; gap: 40px; flex-wrap: wrap; margin-bottom: 14px; }
    .detail-block-label { font-family: var(--mono); font-size: 10px; letter-spacing: 0.1em; color: var(--faint);
                           text-transform: uppercase; margin-bottom: 7px; }
    .tag-row { display: flex; gap: 6px; flex-wrap: wrap; }
    .tag { font-family: var(--mono); font-size: 10.5px; color: var(--mute); border: 1px solid var(--line);
           border-radius: 3px; padding: 3px 7px; }
    .kv-list { font-family: var(--mono); font-size: 11.5px; color: var(--mute); display: flex; flex-direction: column; gap: 4px; }
    .kv-list .k { color: var(--faint); }
    .kv-list .v { color: var(--ink); }
    .detail-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px 24px; align-items: end; }
    .metric-mini { min-width: 0; }
    @media (max-width: 640px) { .detail-metrics { grid-template-columns: repeat(2, 1fr); } }
    .metric-mini-label { font-family: var(--mono); font-size: 10px; color: var(--faint); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-mini-row { display: flex; align-items: center; gap: 8px; }
    .metric-mini-val { font-family: var(--mono); font-size: 12px; color: var(--ink); width: 34px; text-align: right; }

    /* ── modality coverage ────────────────────────────────────────── */
    .modality-row { display: grid; grid-template-columns: 130px 1fr 44px; align-items: center;
                    gap: 12px; padding: 7px 0; border-bottom: 1px solid var(--line); }
    .modality-row:last-child { border-bottom: none; }
    .modality-name { font-family: var(--mono); font-size: 11.5px; color: var(--ink); letter-spacing: 0.03em; }
    .modality-track { height: 3px; background: var(--line); position: relative; }
    .modality-fill { position: absolute; left: 0; top: 0; height: 100%; background: var(--accent); }
    .modality-val { font-family: var(--mono); font-size: 12px; text-align: right; color: var(--mute); }

    /* ── zero-day callouts ───────────────────────────────────────── */
    .callout { padding: 18px 0 20px; border-bottom: 1px solid var(--line); }
    .callout:last-child { border-bottom: none; }
    .callout-id { font-family: var(--mono); font-size: 11px; color: var(--faint); letter-spacing: 0.08em; text-transform: uppercase; }
    .callout-text { font-family: var(--sans); font-size: 16px; line-height: 1.55; margin-top: 6px; color: var(--ink); }
    .callout-text .figure { font-family: var(--mono); font-weight: 600; }
    .callout-text .rise { color: var(--accent); font-family: var(--mono); font-style: normal; }

    /* ── policy distribution ─────────────────────────────────────── */
    .policy-track { display: flex; height: 22px; border: 1px solid var(--line); }
    .policy-seg { display: flex; align-items: center; justify-content: center;
                  font-family: var(--mono); font-size: 10px; letter-spacing: 0.03em; overflow: hidden; white-space: nowrap; }
    .policy-legend { display: flex; gap: 26px; margin-top: 12px; flex-wrap: wrap; }
    .policy-item { font-family: var(--mono); font-size: 12px; color: var(--mute); }
    .policy-item strong { color: var(--ink); font-weight: 600; }

    /* ── methodology footer ──────────────────────────────────────── */
    .methodology { border-top: 1px solid var(--line); padding-top: 22px; margin-top: 20px; }
    .methodology p { font-size: 12px; color: var(--mute); line-height: 1.7; }
    .methodology p + p { margin-top: 8px; }

    .loading { font-family: var(--mono); font-size: 12px; color: var(--mute); padding: 24px 0; }
    .err { font-family: var(--mono); font-size: 12px; color: var(--signal); border: 1px solid var(--signal); padding: 12px 16px; }

    @media (max-width: 640px) {
      .page { padding: 32px 16px 64px; }
      .wordmark { font-size: 20px; }
      .headline { grid-template-columns: repeat(2, 1fr); row-gap: 24px; }
      .stat:nth-child(3) { border-left: none; padding-left: 0; }
      .col-mod, .col-bar { display: none; }
    }
  </style>
</head>
<body>
<div class="page">

  <div class="masthead">
    <div class="wordmark-wrap">
      <div class="wordmark">AEGIS</div>
      <div class="wordmark-mark"></div>
    </div>
    <div class="masthead-right">
      <div class="masthead-clock"><div class="live-dot"></div><span id="clock"></span></div>
    </div>
  </div>
  <div class="masthead-sub">
    <p>Adversarial fraud intelligence — Mastercard Innovation Challenge 2026. A red-team simulator and three-layer defense stack, evaluated on held-out synthetic data.</p>
  </div>

  <nav class="quicknav">
    <a href="#simulate">Simulate</a>
    <a href="#ledger">Ledger</a>
    <a href="#modality">Modality</a>
    <a href="#zeroday">Zero-Day</a>
    <a href="#policy">Policy</a>
    <span class="quicknav-hint">press <kbd>/</kbd> to search</span>
  </nav>

  <div class="headline" id="headline">
    <div class="loading">Loading report…</div>
  </div>

  <div class="section" id="simulate">
    <h2>Live Simulation</h2>
    <p class="section-note">Generate a brand-new synthetic actor right now and watch the trained model score it, live. Every run is a fresh random seed — nothing here is pre-computed or scripted.</p>

    <div class="sim-controls">
      <select id="sim-select" class="sim-select"></select>
      <button id="sim-run" class="btn btn-primary">run attack</button>
      <button id="sim-random" class="btn btn-ghost" title="pick a random vector">random</button>
      <span class="sim-tally" id="sim-tally"></span>
    </div>

    <div class="console" id="console">
      <div class="console-placeholder">Pick a vector above and press run — or hit random.</div>
    </div>
  </div>

  <div class="section" id="ledger">
    <h2>The Ledger</h2>
    <p class="section-note">Every modeled attack vector, grouped by the channel its evidence arrives on. Recall is measured on events the model never trained on; vectors without a bar were fully absorbed into training and are not independently verifiable this run.</p>

    <div class="controls">
      <input type="text" id="search" class="search-input" placeholder="search by id, name, modality…" autocomplete="off" />
      <div class="chip-row" id="modality-chips"></div>
      <div class="control-group">
        <span class="control-label">sort</span>
        <button class="chip sort-chip active" data-sort="id">id</button>
        <button class="chip sort-chip" data-sort="recall">worst recall</button>
      </div>
      <label class="toggle">
        <input type="checkbox" id="measured-only" />
        <span>measured only</span>
      </label>
      <button class="reset-link" id="reset-filters">reset</button>
      <span class="result-count" id="result-count"></span>
    </div>

    <div id="ledger-groups"><div class="loading">Loading vectors…</div></div>
  </div>

  <div class="section" id="modality">
    <h2>Modality Coverage</h2>
    <p class="section-note">Average held-out recall by signal domain.</p>
    <div id="modality-bars"><div class="loading">Loading…</div></div>
  </div>

  <div class="section" id="zeroday">
    <h2>Zero-Day Story</h2>
    <p class="section-note">What the supervised classifier misses by design — and what the anomaly layer catches anyway.</p>
    <div id="zeroday-callouts"><div class="loading">Loading…</div></div>
  </div>

  <div class="section" id="policy">
    <h2>Policy Distribution</h2>
    <p class="section-note">Disposition of held-out events at the chosen operating point (≤1% false-positive budget).</p>
    <div class="policy-track" id="policy-track"></div>
    <div class="policy-legend" id="policy-legend"></div>
  </div>

  <div class="methodology">
    <p><strong>Split.</strong> 70/15/15 train/validation/test, ordered by timestamp — not randomised — so the model is always evaluated on events that happened after the ones it trained on.</p>
    <p><strong>Zero-day.</strong> A subset of vectors is withheld entirely from the supervised classifier's training labels. Detection on those depends only on the anomaly layer.</p>
    <p><strong>Combined score.</strong> 0.6 × classifier probability + 0.4 × normalised anomaly score, the latter calibrated against the training-legit distribution so it stays comparable across batches.</p>
  </div>

</div>

<script>
function tick() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }) + '  ' +
    now.toLocaleTimeString('en-IN', { hour12: false });
}
tick();
setInterval(tick, 1000);

const CHANNEL_LABELS = {
  'txn-sequence': 'txn-sequence', 'kyc-session': 'kyc-session',
  'agent-payment': 'agent-payment', 'chat-call': 'chat-call',
};
const CHANNEL_ORDER = ['txn-sequence', 'kyc-session', 'agent-payment', 'chat-call'];
const ZERO_DAY = new Set(['V011', 'V012', 'V013']);

// ── icon set: thin-stroke geometric glyphs, one mark per modality ─────────
function svg(inner, vb = 16) {
  return `<svg width="${vb===16?14:18}" height="${vb===16?14:18}" viewBox="0 0 ${vb} ${vb}" class="icon">${inner}</svg>`;
}
const MOD_ICON = {
  TXN: svg('<path d="M3 5.5h9M9.5 3l2.5 2.5-2.5 2.5" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/><path d="M13 10.5H4M6.5 13l-2.5-2.5L6.5 8" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>'),
  BENEFICIARY: svg('<circle cx="8" cy="5.5" r="2.3" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M3.3 13c0-2.6 2.1-4.2 4.7-4.2s4.7 1.6 4.7 4.2" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'),
  KYC: svg('<rect x="2.3" y="4.5" width="11.4" height="7.4" rx="1.1" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M2.3 7.2h11.4" stroke="currentColor" stroke-width="1.3"/><path d="M4.6 9.6h2.4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>'),
  AGENT: svg('<rect x="5" y="5" width="6" height="6" rx="1" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M8 2v2.4M8 11.6V14M2 8h2.4M11.6 8H14" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'),
  CONTEXT: svg('<rect x="2.5" y="3.8" width="11" height="6.6" rx="1.6" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M6 10.4l-1.7 2.6-.2-2.6" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>'),
  MODEL: svg('<circle cx="4" cy="4.6" r="1.3" fill="currentColor"/><circle cx="12" cy="4.6" r="1.3" fill="currentColor"/><circle cx="8" cy="11.4" r="1.3" fill="currentColor"/><path d="M4 4.6l4 6.8M12 4.6l-4 6.8M4 4.6h8" stroke="currentColor" stroke-width="1" fill="none"/>'),
  MEDIA: svg('<rect x="2.2" y="6.5" width="1.5" height="4" fill="currentColor"/><rect x="5.4" y="3" width="1.5" height="10.5" fill="currentColor"/><rect x="8.6" y="5" width="1.5" height="6.5" fill="currentColor"/><rect x="11.8" y="1.8" width="1.5" height="12.9" fill="currentColor"/>'),
  PROCEDURAL: svg('<rect x="2.6" y="3.2" width="2" height="2" fill="currentColor"/><path d="M7 4.2h6.4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/><rect x="2.6" y="7" width="2" height="2" fill="currentColor"/><path d="M7 8h6.4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/><rect x="2.6" y="10.8" width="2" height="2" fill="currentColor"/><path d="M7 11.8h6.4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>'),
};
const STAT_ICON = {
  diversity: svg('<circle cx="5.5" cy="5.5" r="1.5" fill="currentColor"/><circle cx="14.5" cy="5.5" r="1.5" fill="currentColor"/><circle cx="5.5" cy="14.5" r="1.5" fill="currentColor"/><circle cx="14.5" cy="14.5" r="1.5" fill="currentColor"/>', 20),
  fidelity: svg('<rect x="3" y="3.5" width="14" height="4.6" rx="1" fill="none" stroke="currentColor" stroke-width="1.4"/><rect x="3" y="10.9" width="14" height="4.6" rx="1" fill="none" stroke="currentColor" stroke-width="1.4"/>', 20),
  efficacy: svg('<circle cx="10" cy="10" r="6.8" fill="none" stroke="currentColor" stroke-width="1.3"/><circle cx="10" cy="10" r="3.4" fill="none" stroke="currentColor" stroke-width="1.3"/><circle cx="10" cy="10" r="1" fill="currentColor"/>', 20),
  novelty: svg('<path d="M2.5 10s3.2-5 7.5-5 7.5 5 7.5 5-3.2 5-7.5 5-7.5-5-7.5-5z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/><circle cx="10" cy="10" r="2.2" fill="none" stroke="currentColor" stroke-width="1.3"/>', 20),
};
const CHEVRON = '<svg width="10" height="10" viewBox="0 0 16 16" class="chevron"><path d="M5 3l4 5-4 5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';

let _catalog = null, _results = null, _controlsWired = false;

async function loadAll() {
  try {
    const [c, r] = await Promise.all([
      fetch('/api/catalog').then(x => x.json()),
      fetch('/api/results').then(x => x.json()),
    ]);
    if (r.error) { showError(r.error); return; }
    _catalog = c; _results = r;
    if (!_controlsWired) {
      buildModalityChips(_catalog);
      wireLedgerControls();
      populateSimSelect(_catalog);
      wireSimControls();
      _controlsWired = true;
    }
    render();
  } catch (e) {
    showError('Failed to reach API: ' + e.message);
  }
}
function showError(msg) {
  document.getElementById('ledger-groups').innerHTML = `<div class="err">${msg}</div>`;
}

function buildRecallMap(results) {
  const m = {};
  Object.values(results).forEach(ch => (ch.per_vector_recall || []).forEach(p => { m[p.vector_id] = p; }));
  return m;
}
function buildOodList(results) {
  const list = [];
  Object.values(results).forEach(ch => (ch.ood_story || []).forEach(o => list.push(o)));
  return list;
}
function fmtInt(n) { return n.toLocaleString('en-IN'); }

// ── headline ────────────────────────────────────────────────────────────
function renderHeadline(catalog, results) {
  const recallMap = buildRecallMap(results);
  const totalEvents = Object.values(results).reduce((s, c) => s + c.n_total, 0);

  const measured = Object.values(recallMap);
  const good = measured.filter(p => p.combined_recall >= 0.9).length;

  const oodList = buildOodList(results);
  let best = null;
  oodList.forEach(o => { if (o.category === 'zero_day' && (!best || o.ood_delta > best.ood_delta)) best = o; });

  const html = `
    <div class="stat">
      <div class="stat-axis">${STAT_ICON.diversity} Diversity</div>
      <div class="stat-num">${catalog.vectors.length}</div>
      <div class="stat-label">attack vectors modeled</div>
    </div>
    <div class="stat">
      <div class="stat-axis">${STAT_ICON.fidelity} Fidelity</div>
      <div class="stat-num">${fmtInt(totalEvents)}</div>
      <div class="stat-label">synthetic events simulated</div>
    </div>
    <div class="stat">
      <div class="stat-axis">${STAT_ICON.efficacy} Efficacy</div>
      <div class="stat-num">${good}/${measured.length}</div>
      <div class="stat-label">evaluated vectors ≥90% recall</div>
    </div>
    <div class="stat">
      <div class="stat-axis">${STAT_ICON.novelty} Novelty</div>
      <div class="stat-num accent">${best ? '+' + Math.round(best.ood_delta * 100) + 'pp' : '—'}</div>
      <div class="stat-label">${best ? best.vector_id + ' zero-day, caught only by anomaly layer' : 'no zero-day lift measured'}</div>
    </div>`;
  document.getElementById('headline').innerHTML = html;
}

// ── ledger ──────────────────────────────────────────────────────────────
function recallBarHtml(recall) {
  if (recall === undefined || recall === null) {
    return `<span class="bar-none">—</span>`;
  }
  const pct = Math.round(recall * 100);
  if (recall === 0) {
    return `<div class="bar-track"><div class="bar-miss-mark"></div></div>`;
  }
  return `<div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>`;
}

const filterState = { search: '', modality: null, sort: 'id', measuredOnly: false };

function buildModalityChips(catalog) {
  const modalities = [...new Set(catalog.vectors.map(v => v.modality))].sort();
  const html = modalities.map(m =>
    `<button class="chip mod-chip" data-mod="${m}">${MOD_ICON[m] || ''}${m.toLowerCase()}</button>`).join('');
  document.getElementById('modality-chips').innerHTML = html;
}

function formatParamValue(v) {
  if (Array.isArray(v)) return v.join(' – ');
  return String(v);
}

function buildDetailHtml(v, pvr) {
  const zd = ZERO_DAY.has(v.vector_id);
  const observedTags = (v.observed_by || []).map(o => `<span class="tag">${o}</span>`).join('');
  const params = Object.entries(v.generator_params || {})
    .filter(([k]) => k !== 'n_legit' && k !== 'n_fraud')
    .map(([k, val]) => `<div><span class="k">${k}</span> <span class="v">${formatParamValue(val)}</span></div>`)
    .join('');

  const mm = (label, val) => `
      <div class="metric-mini">
        <div class="metric-mini-label">${label}</div>
        <div class="metric-mini-row">
          <div class="bar-track" style="flex:1"><div class="bar-fill" style="width:${Math.round(val*100)}%"></div></div>
          <div class="metric-mini-val">${Math.round(val*100)}%</div>
        </div>
      </div>`;

  const metrics = pvr ? `
    <div class="detail-metrics">
      ${mm('supervised recall', pvr.clf_recall)}
      ${mm('combined recall (+ anomaly layer)', pvr.combined_recall)}
      ${mm('supervised precision', pvr.clf_precision)}
      ${mm('combined precision', pvr.combined_precision)}
      ${mm('supervised F1', pvr.clf_f1)}
      ${mm('combined F1', pvr.combined_f1)}
      <div class="metric-mini">
        <div class="metric-mini-label">fraud actors tested</div>
        <div class="metric-mini-row"><div class="metric-mini-val" style="width:auto">${pvr.n_fraud_actors}</div></div>
      </div>
    </div>
    <p class="section-note" style="margin-top:10px">Precision/F1 here are actor-level, measured against this channel's shared pool of wrongly-flagged legitimate customers — useful for comparing vectors, but not the operational false-positive rate (that's measured per-transaction, at the channel level, since that's the actual point of decision).</p>`
    : `<p class="section-note" style="margin:0">No fraud actors from this vector landed in the held-out test window this run — coverage relies on its dedicated specialist trained during this run.</p>`;

  return `<div class="detail">
    <div class="detail-desc">${v.description}${zd ? ' <span class="zd-flag">† withheld from supervised training entirely.</span>' : ''}</div>
    <div class="detail-signal">${v.expected_signal}</div>
    <div class="detail-grid">
      <div>
        <div class="detail-block-label">Observed by</div>
        <div class="tag-row">${observedTags}</div>
      </div>
      <div>
        <div class="detail-block-label">Simulation parameters</div>
        <div class="kv-list">${params}</div>
      </div>
    </div>
    ${metrics}
    <button class="row-run-btn run-live-btn" data-run-id="${v.vector_id}" style="margin-top:14px">run live simulation →</button>
  </div>`;
}

const expandedIds = new Set();

function renderLedger(catalog, results) {
  const recallMap = buildRecallMap(results);
  const q = filterState.search.trim().toLowerCase();

  let allVectors = catalog.vectors.filter(v => {
    if (filterState.modality && v.modality !== filterState.modality) return false;
    if (filterState.measuredOnly && !recallMap[v.vector_id]) return false;
    if (q) {
      const hay = `${v.vector_id} ${v.name} ${v.modality} ${v.channel}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  const byChannel = {};
  allVectors.forEach(v => (byChannel[v.channel] = byChannel[v.channel] || []).push(v));

  const groups = CHANNEL_ORDER.filter(ch => byChannel[ch] && byChannel[ch].length);

  const html = groups.map(ch => {
    let vectors = byChannel[ch];
    if (filterState.sort === 'recall') {
      vectors = [...vectors].sort((a, b) => {
        const ra = recallMap[a.vector_id] ? recallMap[a.vector_id].combined_recall : -1;
        const rb = recallMap[b.vector_id] ? recallMap[b.vector_id].combined_recall : -1;
        return ra - rb; // worst first
      });
    }
    const chResult = results[ch];
    const prauc = chResult ? chResult.prauc_combined.mean.toFixed(3) : '—';
    const rows = vectors.map(v => {
      const pvr = recallMap[v.vector_id];
      const zd = ZERO_DAY.has(v.vector_id);
      const recall = pvr ? pvr.combined_recall : null;
      const valText = recall === null ? '—' : Math.round(recall * 100) + '%';
      const isOpen = expandedIds.has(v.vector_id);
      return `<tr class="row${isOpen ? ' expanded' : ''}" data-id="${v.vector_id}" tabindex="0" role="button" aria-expanded="${isOpen}">
        <td class="col-chev">${CHEVRON}</td>
        <td class="col-id">${v.vector_id}</td>
        <td class="col-name">${v.name}${zd ? ' <span class="zd-flag">†</span>' : ''}</td>
        <td class="col-mod"><div class="col-mod-inner">${MOD_ICON[v.modality] || ''}${v.modality.toLowerCase()}</div></td>
        <td class="col-bar">${recallBarHtml(recall)}</td>
        <td class="col-val">${valText}</td>
      </tr>
      <tr class="detail-row" data-for="${v.vector_id}" ${isOpen ? '' : 'hidden'}>
        <td colspan="6">${buildDetailHtml(v, pvr)}</td>
      </tr>`;
    }).join('');

    return `<div class="channel-group">
      <div class="channel-head"><strong>${CHANNEL_LABELS[ch]}</strong><span>${vectors.length} vector${vectors.length===1?'':'s'} · PR-AUC ${prauc}</span></div>
      <table class="ledger"><tbody>${rows}</tbody></table>
    </div>`;
  }).join('');

  document.getElementById('ledger-groups').innerHTML =
    (html || '<div class="loading">No vectors match these filters.</div>') +
    `<div class="ledger-footnote"><em>†</em> held out entirely from supervised training — detection depends on the anomaly layer alone. Click any row for detail.</div>`;

  const total = catalog.vectors.length;
  const shown = allVectors.length;
  document.getElementById('result-count').textContent =
    shown === total ? `${total} vectors` : `${shown} of ${total} vectors`;

  document.querySelectorAll('.mod-chip').forEach(chip => {
    chip.classList.toggle('active', chip.dataset.mod === filterState.modality);
  });
}

// ── live simulation ─────────────────────────────────────────────────────
const simTally = { runs: 0, caught: 0, missed: 0 };

function populateSimSelect(catalog) {
  const byChannel = {};
  catalog.vectors.forEach(v => (byChannel[v.channel] = byChannel[v.channel] || []).push(v));
  const html = CHANNEL_ORDER.filter(ch => byChannel[ch]).map(ch => `
    <optgroup label="${ch}">
      ${byChannel[ch].map(v => `<option value="${v.vector_id}">${v.vector_id} — ${v.name}</option>`).join('')}
    </optgroup>`).join('');
  document.getElementById('sim-select').innerHTML = html;
}

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
function sleep(ms) {
  return new Promise(r => setTimeout(r, prefersReducedMotion() ? 0 : ms));
}
function appendConsoleLine(text, cls) {
  const el = document.getElementById('console');
  const line = document.createElement('div');
  line.className = `console-line ${cls || ''}`;
  line.innerHTML = text;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}
function updateTally() {
  document.getElementById('sim-tally').innerHTML =
    simTally.runs === 0 ? '' :
    `<strong>${simTally.runs}</strong> run${simTally.runs===1?'':'s'} · <strong>${simTally.caught}</strong> caught · <strong>${simTally.missed}</strong> missed`;
}

async function runSimulation(vectorId) {
  const consoleEl = document.getElementById('console');
  consoleEl.innerHTML = '';
  document.getElementById('sim-select').value = vectorId;

  appendConsoleLine(`&gt; generating a fresh actor for ${vectorId}…`, 'cl-info');

  let data;
  try {
    const res = await fetch(`/api/simulate/${vectorId}`, { method: 'POST' });
    data = await res.json();
    if (data.error) { appendConsoleLine(data.error, 'cl-error'); return; }
  } catch (e) {
    appendConsoleLine('request failed: ' + e.message, 'cl-error');
    return;
  }

  await sleep(150);
  appendConsoleLine(`${data.name} · actor ${data.actor_id} · seed ${data.seed}`, 'cl-info');
  await sleep(200);

  for (const ev of data.events) {
    const isFraud = ev.label === 1;
    const t = new Date(ev.timestamp);
    const time = isNaN(t) ? '' : t.toLocaleString('en-IN', { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' });
    const amt = (ev.amount !== null && ev.amount !== undefined) ? '₹' + Math.round(ev.amount).toLocaleString('en-IN') : '—';
    const line = `${(time || '—').padEnd(15)} <span class="amt">${amt.padEnd(12)}</span> clf ${ev.clf_score.toFixed(3)}  combined ${ev.combined_score.toFixed(3)}  ${ev.policy}`;
    appendConsoleLine(line, isFraud ? 'cl-fraud' : 'cl-legit');
    await sleep(isFraud ? 260 : 55);
  }

  await sleep(300);
  renderVerdict(data);

  simTally.runs++;
  if (data.caught) simTally.caught++; else simTally.missed++;
  updateTally();
}

function renderVerdict(data) {
  const consoleEl = document.getElementById('console');
  const div = document.createElement('div');
  div.className = `verdict ${data.caught ? 'verdict-caught' : 'verdict-missed'}`;
  const byText = !data.caught ? 'slipped past both layers this run'
    : data.caught_by === 'anomaly-layer' ? 'caught by the anomaly layer alone — the supervised model missed it'
    : 'caught by the supervised classifier';
  div.innerHTML = `
    <div class="verdict-label">${data.caught ? 'DETECTED' : 'MISSED'}</div>
    <div class="verdict-detail">score <strong>${data.max_score.toFixed(3)}</strong> vs operating threshold <strong>${data.operating_threshold.toFixed(3)}</strong> — ${byText}.</div>`;
  consoleEl.appendChild(div);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

function wireSimControls() {
  document.getElementById('sim-run').addEventListener('click', () => {
    const id = document.getElementById('sim-select').value;
    if (id) runSimulation(id);
  });
  document.getElementById('sim-random').addEventListener('click', () => {
    const ids = _catalog.vectors.map(v => v.vector_id);
    runSimulation(ids[Math.floor(Math.random() * ids.length)]);
  });
  document.getElementById('ledger-groups').addEventListener('click', e => {
    const btn = e.target.closest('.run-live-btn');
    if (!btn) return;
    e.stopPropagation();
    document.getElementById('simulate').scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
    runSimulation(btn.dataset.runId);
  });
}

function wireLedgerControls() {
  const search = document.getElementById('search');
  search.addEventListener('input', () => {
    filterState.search = search.value;
    renderLedger(_catalog, _results);
  });

  document.getElementById('modality-chips').addEventListener('click', e => {
    const chip = e.target.closest('.mod-chip');
    if (!chip) return;
    filterState.modality = filterState.modality === chip.dataset.mod ? null : chip.dataset.mod;
    renderLedger(_catalog, _results);
  });

  document.querySelectorAll('.sort-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      filterState.sort = btn.dataset.sort;
      document.querySelectorAll('.sort-chip').forEach(b => b.classList.toggle('active', b === btn));
      renderLedger(_catalog, _results);
    });
  });

  document.getElementById('measured-only').addEventListener('change', e => {
    filterState.measuredOnly = e.target.checked;
    renderLedger(_catalog, _results);
  });

  document.getElementById('reset-filters').addEventListener('click', () => {
    filterState.search = ''; filterState.modality = null; filterState.sort = 'id'; filterState.measuredOnly = false;
    search.value = '';
    document.getElementById('measured-only').checked = false;
    document.querySelectorAll('.sort-chip').forEach(b => b.classList.toggle('active', b.dataset.sort === 'id'));
    renderLedger(_catalog, _results);
  });

  function toggleRow(row) {
    const id = row.dataset.id;
    const detail = document.querySelector(`tr.detail-row[data-for="${id}"]`);
    if (!detail) return;
    const opening = detail.hasAttribute('hidden');
    if (opening) { detail.removeAttribute('hidden'); row.classList.add('expanded'); row.setAttribute('aria-expanded', 'true'); expandedIds.add(id); }
    else { detail.setAttribute('hidden', ''); row.classList.remove('expanded'); row.setAttribute('aria-expanded', 'false'); expandedIds.delete(id); }
  }
  document.getElementById('ledger-groups').addEventListener('click', e => {
    const row = e.target.closest('tr.row');
    if (row) toggleRow(row);
  });
  document.getElementById('ledger-groups').addEventListener('keydown', e => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const row = e.target.closest('tr.row');
    if (!row) return;
    e.preventDefault();
    toggleRow(row);
  });

  document.addEventListener('keydown', e => {
    const typing = document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);
    if (e.key === '/' && !typing) {
      e.preventDefault();
      search.focus();
    } else if (e.key === 'Escape' && document.activeElement === search) {
      search.blur();
    }
  });
}

// ── modality coverage ─────────────────────────────────────────────────
function renderModality(catalog, results) {
  const recallMap = buildRecallMap(results);
  const byModality = {};
  catalog.vectors.forEach(v => {
    (byModality[v.modality] = byModality[v.modality] || []).push(v.vector_id);
  });

  const rows = Object.entries(byModality).map(([mod, ids]) => {
    const recalls = ids.map(id => recallMap[id]).filter(Boolean).map(p => p.combined_recall);
    if (!recalls.length) return null;
    const avg = recalls.reduce((a, b) => a + b, 0) / recalls.length;
    return { mod, avg, n: recalls.length };
  }).filter(Boolean).sort((a, b) => b.avg - a.avg);

  const html = rows.map(r => `
    <div class="modality-row">
      <div class="modality-name">${r.mod.toLowerCase()}</div>
      <div class="modality-track"><div class="modality-fill" style="width:${Math.round(r.avg*100)}%"></div></div>
      <div class="modality-val">${Math.round(r.avg*100)}%</div>
    </div>`).join('');

  document.getElementById('modality-bars').innerHTML = html || '<div class="loading">No measured vectors yet.</div>';
}

// ── zero-day callouts ─────────────────────────────────────────────────
function renderZeroDay(catalog, results) {
  const catMap = {};
  catalog.vectors.forEach(v => { catMap[v.vector_id] = v; });
  const list = buildOodList(results);

  if (!list.length) {
    document.getElementById('zeroday-callouts').innerHTML = '<div class="loading">No zero-day vectors landed in the held-out window this run.</div>';
    return;
  }

  const html = list.map(o => {
    const v = catMap[o.vector_id] || {};
    const supPct = Math.round((o.supervised_recall || 0) * 100);
    const combPct = Math.round((o.combined_recall || 0) * 100);
    const isZd = o.category === 'zero_day';
    return `<div class="callout">
      <div class="callout-id">${o.vector_id} · ${isZd ? 'zero-day' : 'model evasion'}</div>
      <div class="callout-text">
        <em>${v.name || ''}.</em>
        The supervised model${isZd ? ', trained without ever seeing this attack,' : ''}
        catches <span class="figure">${supPct}%</span> on its own.
        With the anomaly layer added, recall reaches <span class="figure">${combPct}%</span>
        ${o.ood_delta > 0 ? `<span class="rise">(+${Math.round(o.ood_delta*100)}pp)</span>` : ''}.
      </div>
    </div>`;
  }).join('');

  document.getElementById('zeroday-callouts').innerHTML = html;
}

// ── policy distribution ───────────────────────────────────────────────
const POLICY_CFG = [
  { key: 'APPROVE', label: 'approve' },
  { key: 'STEP_UP', label: 'step-up' },
  { key: 'HOLD',    label: 'hold' },
  { key: 'DECLINE', label: 'decline' },
];
function renderPolicy(results) {
  const totals = {};
  POLICY_CFG.forEach(p => totals[p.key] = 0);
  Object.values(results).forEach(ch => {
    const pd = ch.policy_distribution || {};
    POLICY_CFG.forEach(p => totals[p.key] += (pd[p.key] || 0));
  });
  const sum = Object.values(totals).reduce((a, b) => a + b, 0);

  const track = POLICY_CFG.map((p, i) => {
    const pct = sum ? (totals[p.key] / sum * 100) : 0;
    const isDecline = p.key === 'DECLINE';
    const bg = isDecline ? 'var(--signal)' : (i % 2 === 0 ? 'var(--ink)' : 'var(--line)');
    const textColor = i % 2 === 0 && !isDecline ? 'var(--bg)' : 'var(--ink)';
    return `<div class="policy-seg" style="flex:${Math.max(pct,0.001)};background:${bg};color:${textColor}">${pct > 8 ? p.label : ''}</div>`;
  }).join('');

  const legend = POLICY_CFG.map(p => `<span class="policy-item">${p.label} <strong>${fmtInt(totals[p.key])}</strong></span>`).join('');

  document.getElementById('policy-track').innerHTML = track;
  document.getElementById('policy-legend').innerHTML = legend;
}

function render() {
  renderHeadline(_catalog, _results);
  renderLedger(_catalog, _results);
  renderModality(_catalog, _results);
  renderZeroDay(_catalog, _results);
  renderPolicy(_results);
}

loadAll();
setInterval(loadAll, 10000);
</script>
</body>
</html>"""
