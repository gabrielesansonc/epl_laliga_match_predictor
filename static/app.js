'use strict';

// ── Constants ──────────────────────────────────────────────────────────────────
const COLORS        = { H: '#2B6AF5', D: '#9ca3af', A: '#FFE600' };
const OUTCOME_LABELS = { H: 'Home Win', D: 'Draw', A: 'Away Win' };
const CHART_FONT    = "'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const CHART_COLOR   = '#b0c4de';
const CHART_GRID    = '#1A3A52';

// ── State ──────────────────────────────────────────────────────────────────────
const state = {
  page:      'predictions',
  charts:    {},
  dataCache: {},
};

// ── Utilities ──────────────────────────────────────────────────────────────────
function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function el(id) { return document.getElementById(id); }

function destroyCharts() {
  Object.values(state.charts).forEach(c => { try { c.destroy(); } catch (_) {} });
  state.charts = {};
}

// ── Chart.js defaults ─────────────────────────────────────────────────────────
function setupChartDefaults() {
  Chart.defaults.color                            = CHART_COLOR;
  Chart.defaults.borderColor                      = CHART_GRID;
  Chart.defaults.font.family                      = CHART_FONT;
  Chart.defaults.font.size                        = 11;
  Chart.defaults.plugins.tooltip.backgroundColor = '#0F2347';
  Chart.defaults.plugins.tooltip.borderColor     = '#1A3A52';
  Chart.defaults.plugins.tooltip.borderWidth     = 1;
  Chart.defaults.plugins.tooltip.titleColor      = '#ffffff';
  Chart.defaults.plugins.tooltip.bodyColor       = '#b0c4de';
  Chart.defaults.plugins.tooltip.padding         = 10;
  Chart.defaults.plugins.tooltip.cornerRadius    = 8;
  Chart.defaults.plugins.legend.labels.color     = CHART_COLOR;
  Chart.defaults.plugins.legend.labels.boxWidth  = 10;
  Chart.defaults.plugins.legend.labels.padding   = 14;
}

function baseScales(yFmt) {
  return {
    x: {
      grid:  { color: CHART_GRID, drawBorder: false },
      ticks: { color: CHART_COLOR, font: { size: 11, family: CHART_FONT } },
    },
    y: {
      grid:  { color: CHART_GRID, drawBorder: false },
      ticks: { color: CHART_COLOR, font: { size: 11, family: CHART_FONT }, callback: yFmt || (v => v) },
    },
  };
}

function makeChart(id, config) {
  const canvas = el(id);
  if (!canvas) return null;
  if (state.charts[id]) state.charts[id].destroy();
  const chart = new Chart(canvas, config);
  state.charts[id] = chart;
  return chart;
}

// ── Navigation ─────────────────────────────────────────────────────────────────
function navigate(page) {
  state.page = page;
  history.replaceState(null, '', '#' + page);

  // Sync sidebar + bottom nav
  document.querySelectorAll('.nav-item, .bnav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.page === page);
  });

  destroyCharts();

  const content = el('content');
  switch (page) {
    case 'predictions': renderPredictions(content); break;
    case 'ucl':         renderUCL(content);         break;
    case 'validation':  renderValidation(content);  break;
    default:            renderPredictions(content);
  }
}

// ── Expandable card logic (event delegation) ───────────────────────────────────
function attachCardToggle(containerId) {
  const container = el(containerId);
  if (!container) return;
  container.addEventListener('click', e => {
    const card = e.target.closest('.match-card');
    if (!card) return;
    const section = card.querySelector('.insights-section');
    const chevron = card.querySelector('.card-chevron');
    if (!section) return;
    const expanded = section.classList.toggle('expanded');
    if (chevron) chevron.classList.toggle('rotated', expanded);
  });
}

// ── Predictions page ──────────────────────────────────────────────────────────
function renderPredictions(content) {
  content.innerHTML = `
    <div class="page-header">
      <h1>Upcoming Fixtures</h1>
      <p>Predicted probabilities for this week's matches</p>
    </div>
    <div class="tabs" id="league-tabs">
      <button class="tab active" data-league="SP1">La Liga</button>
      <button class="tab" data-league="E0">Premier League</button>
    </div>
    <div id="fixtures-area"></div>
  `;

  el('league-tabs').addEventListener('click', e => {
    const btn = e.target.closest('.tab');
    if (!btn) return;
    el('league-tabs').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    fetchFixtures(btn.dataset.league);
  });

  attachCardToggle('fixtures-area');
  fetchFixtures('SP1');
}

async function fetchFixtures(league) {
  const area = el('fixtures-area');
  if (!area) return;

  const key = 'fixtures_' + league;
  if (state.dataCache[key]) { renderFixtureList(state.dataCache[key]); return; }

  area.innerHTML = loadingHTML('Loading fixtures…');

  try {
    const res  = await fetch(`/api/fixtures?league=${league}`);
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.statusText); }
    const data = await res.json();
    state.dataCache[key] = data;
    renderFixtureList(data);
  } catch (err) {
    const a = el('fixtures-area');
    if (a) a.innerHTML = errorHTML(err.message);
  }
}

function renderFixtureList(data) {
  const area = el('fixtures-area');
  if (!area) return;

  if (!data.fixtures.length) {
    area.innerHTML = '<div class="empty-state">No upcoming fixtures found.</div>';
    return;
  }

  area.innerHTML =
    `<p class="fixture-count">${data.count} fixture${data.count !== 1 ? 's' : ''}</p>` +
    `<div class="fixtures-list">` +
    data.fixtures.map(f => matchCardHTML(f)).join('') +
    `</div>`;
}

// ── Champions League page ──────────────────────────────────────────────────────
function renderUCL(content) {
  content.innerHTML = `
    <div class="page-header">
      <h1>Champions League</h1>
      <p>Predict any cross-league matchup — enter teams and press Predict</p>
    </div>
    <div class="ucl-form">
      <div class="form-row">
        <div class="form-group">
          <label for="home-input">Home team</label>
          <input id="home-input" type="text" placeholder="e.g. Real Madrid" autocomplete="off" spellcheck="false">
        </div>
        <div class="form-group">
          <label for="away-input">Away team</label>
          <input id="away-input" type="text" placeholder="e.g. Arsenal" autocomplete="off" spellcheck="false">
        </div>
      </div>
      <button class="btn-primary" id="predict-btn">Predict</button>
    </div>
    <div id="ucl-result"></div>
  `;

  el('predict-btn').addEventListener('click', runUCLPrediction);
  [el('home-input'), el('away-input')].forEach(inp => {
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') runUCLPrediction(); });
  });
  attachCardToggle('ucl-result');
}

async function runUCLPrediction() {
  const home = (el('home-input').value || '').trim();
  const away = (el('away-input').value || '').trim();
  const resultDiv = el('ucl-result');
  if (!resultDiv) return;

  if (!home || !away) {
    resultDiv.innerHTML = errorHTML('Please enter both a home and away team.');
    return;
  }

  resultDiv.innerHTML = loadingHTML('Running prediction…');

  try {
    const res = await fetch(
      `/api/predict?home_team=${encodeURIComponent(home)}&away_team=${encodeURIComponent(away)}`
    );
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.statusText); }
    const data = await res.json();

    const dateStr = new Date().toLocaleDateString('en-GB', {
      weekday: 'short', day: 'numeric', month: 'short',
    });

    resultDiv.innerHTML = matchCardHTML({
      home_team: data.home_team,
      away_team: data.away_team,
      date:      dateStr,
      league:    'Champions League',
      prob_h:    data.prob_h,
      prob_d:    data.prob_d,
      prob_a:    data.prob_a,
      predicted: data.predicted,
      insights:  [],
    });
  } catch (err) {
    const r = el('ucl-result');
    if (r) r.innerHTML = errorHTML(err.message);
  }
}

// ── Validation page ────────────────────────────────────────────────────────────
function renderValidation(content) {
  content.innerHTML = `
    <div class="page-header">
      <h1>Model Validation</h1>
      <p>Performance on the held-out 10% test set</p>
    </div>
    <div class="tabs" id="val-tabs">
      <button class="tab active" data-league="All">All Leagues</button>
      <button class="tab" data-league="SP1">La Liga</button>
      <button class="tab" data-league="E0">Premier League</button>
    </div>
    <div id="validation-area"></div>
  `;

  el('val-tabs').addEventListener('click', e => {
    const btn = e.target.closest('.tab');
    if (!btn) return;
    el('val-tabs').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    fetchValidation(btn.dataset.league);
  });

  fetchValidation('All');
}

async function fetchValidation(league) {
  const area = el('validation-area');
  if (!area) return;

  const key = 'val_' + league;
  if (state.dataCache[key]) { buildValidationUI(state.dataCache[key]); return; }

  area.innerHTML = loadingHTML('Scoring test set…');

  try {
    const res  = await fetch(`/api/validation?league=${league}`);
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.statusText); }
    const data = await res.json();
    state.dataCache[key] = data;
    buildValidationUI(data);
  } catch (err) {
    const a = el('validation-area');
    if (a) a.innerHTML = errorHTML(err.message);
  }
}

function buildValidationUI(d) {
  const area = el('validation-area');
  if (!area) return;
  const m   = d.metrics;
  const sim = d.betting_sim;

  area.innerHTML = `
    <div class="metrics-row">
      <div class="metric-card">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">${pct(m.accuracy)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Matches</div>
        <div class="metric-value">${m.matches.toLocaleString()}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Seasons</div>
        <div class="metric-value">${m.seasons}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Date Range</div>
        <div class="metric-value" style="font-size:1.25rem">${esc(m.date_range)}</div>
      </div>
    </div>

    <div class="section-header">
      <span class="section-title">Model Accuracy by Season</span>
      <span class="section-caption">test set</span>
    </div>
    <div class="chart-wrap" style="height:220px"><canvas id="ch-season-acc"></canvas></div>

    <div class="section-header">
      <span class="section-title">Betting Odds Baseline</span>
      <span class="section-caption">B365 implied probability · all matches</span>
    </div>
    <div class="chart-wrap" style="height:220px"><canvas id="ch-betting"></canvas></div>

    <div class="section-header">
      <span class="section-title">Model Drift</span>
      <span class="section-caption">rolling 100-match window</span>
    </div>
    <div class="chart-wrap" style="height:180px"><canvas id="ch-drift"></canvas></div>

    <div class="charts-grid">
      <div>
        <div class="section-header">
          <span class="section-title">ROC Curves</span>
          <span class="section-caption">one vs rest</span>
        </div>
        <div class="chart-wrap" style="height:280px"><canvas id="ch-roc"></canvas></div>
      </div>
      <div>
        <div class="section-header">
          <span class="section-title">By Outcome</span>
        </div>
        <div class="chart-wrap" style="height:280px"><canvas id="ch-by-outcome"></canvas></div>
      </div>
    </div>

    <div class="section-header">
      <span class="section-title">Calibration</span>
      <span class="section-caption">predicted probability vs actual frequency</span>
    </div>
    <div class="chart-wrap" style="height:240px"><canvas id="ch-calibration"></canvas></div>

    ${sim ? `
    <div class="section-header">
      <span class="section-title">Betting Simulation</span>
      <span class="section-caption">$1 per match · max model vs B365 edge · test set</span>
    </div>
    <div class="sim-metrics-row">
      <div class="metric-card">
        <div class="metric-label">Total P&amp;L</div>
        <div class="metric-value ${sim.total_pnl >= 0 ? 'positive' : 'negative'}">${sim.total_pnl >= 0 ? '+' : ''}$${sim.total_pnl.toFixed(2)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">ROI</div>
        <div class="metric-value ${sim.roi >= 0 ? 'positive' : 'negative'}">${sim.roi >= 0 ? '+' : ''}${(sim.roi * 100).toFixed(1)}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Win Rate</div>
        <div class="metric-value">${pct(sim.win_rate)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg Odds</div>
        <div class="metric-value">${sim.avg_odds.toFixed(2)}</div>
      </div>
    </div>
    <div class="chart-wrap" style="height:260px"><canvas id="ch-sim"></canvas></div>
    <p class="chart-footnote">Strategy: for each test match, the model probability is compared to the B365 implied probability (overround-adjusted). The outcome with the largest positive gap is backed at the B365 closing odds. $1 staked per match.</p>
    ` : '<div class="info-state" style="margin-top:24px">No betting simulation data available.</div>'}
  `;

  drawValidationCharts(d);
}

function drawValidationCharts(d) {
  const avgAcc = d.overall_accuracy * 100;
  const avgBet = d.betting_overall * 100;

  const avgLine = (labels, val) => ({
    type: 'line',
    data: labels.map(() => +val.toFixed(2)),
    borderColor: 'rgba(255,255,255,0.12)',
    borderWidth: 1,
    borderDash: [4, 4],
    pointRadius: 0,
    fill: false,
    label: `avg ${val.toFixed(1)}%`,
  });

  // Season accuracy
  makeChart('ch-season-acc', {
    type: 'bar',
    data: {
      labels: d.season_accuracy.map(r => r.Season),
      datasets: [
        { data: d.season_accuracy.map(r => +(r.accuracy * 100).toFixed(2)), backgroundColor: 'rgba(99,102,241,0.65)', borderWidth: 0, borderRadius: 4 },
        avgLine(d.season_accuracy, avgAcc),
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(1)}%` } } },
      scales: { ...baseScales(v => `${v.toFixed(0)}%`), x: { ...baseScales().x, grid: { display: false } }, y: { ...baseScales(v => `${v.toFixed(0)}%`).y, min: 0 } },
    },
  });

  // Betting baseline
  makeChart('ch-betting', {
    type: 'bar',
    data: {
      labels: d.betting_baseline.map(r => r.Season),
      datasets: [
        { data: d.betting_baseline.map(r => +(r.accuracy * 100).toFixed(2)), backgroundColor: 'rgba(99,102,241,0.65)', borderWidth: 0, borderRadius: 4 },
        avgLine(d.betting_baseline, avgBet),
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(1)}%` } } },
      scales: { ...baseScales(v => `${v.toFixed(0)}%`), x: { ...baseScales().x, grid: { display: false } }, y: { ...baseScales(v => `${v.toFixed(0)}%`).y, min: 0 } },
    },
  });

  // Drift
  makeChart('ch-drift', {
    type: 'line',
    data: {
      labels: d.drift.map(p => p.date),
      datasets: [
        { data: d.drift.map(p => +(p.acc * 100).toFixed(2)), borderColor: COLORS.H, backgroundColor: 'rgba(99,102,241,0.07)', fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2, label: 'Rolling accuracy' },
        { data: d.drift.map(() => +avgAcc.toFixed(2)), borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false, label: `avg ${avgAcc.toFixed(1)}%` },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(1)}%` } } },
      scales: { ...baseScales(v => `${v.toFixed(0)}%`), x: { ...baseScales().x, ticks: { ...baseScales().x.ticks, maxTicksLimit: 6, autoSkip: true } } },
    },
  });

  // ROC curves
  const rocDs = Object.entries(d.roc).map(([key, r]) => ({
    label: `${r.label}  ${r.auc.toFixed(3)}`,
    data: r.fpr.map((x, i) => ({ x, y: r.tpr[i] })),
    borderColor: COLORS[key], backgroundColor: 'transparent',
    borderWidth: 2, pointRadius: 0, tension: 0,
  }));
  rocDs.push({ label: 'Random', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: 'rgba(255,255,255,0.1)', backgroundColor: 'transparent', borderWidth: 1, borderDash: [4, 4], pointRadius: 0 });

  makeChart('ch-roc', {
    type: 'scatter',
    data: { datasets: rocDs },
    options: {
      responsive: true, maintainAspectRatio: false, showLine: true,
      plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: ctx => ` FPR ${ctx.parsed.x.toFixed(2)}  TPR ${ctx.parsed.y.toFixed(2)}` } } },
      scales: {
        x: { ...baseScales().x, title: { display: true, text: 'FPR', color: CHART_COLOR, font: { size: 10 } }, min: 0, max: 1 },
        y: { ...baseScales().y, title: { display: true, text: 'TPR', color: CHART_COLOR, font: { size: 10 } }, min: 0, max: 1 },
      },
    },
  });

  // By outcome
  const outKeys = Object.keys(d.by_outcome);
  makeChart('ch-by-outcome', {
    type: 'bar',
    data: {
      labels: outKeys.map(k => OUTCOME_LABELS[k] || k),
      datasets: [{ data: outKeys.map(k => +(d.by_outcome[k].accuracy * 100).toFixed(2)), backgroundColor: outKeys.map(k => hexAlpha(COLORS[k] || '#888', 0.65)), borderWidth: 0, borderRadius: 4 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(1)}%` } } },
      scales: { ...baseScales(v => `${v.toFixed(0)}%`), x: { ...baseScales().x, grid: { display: false } }, y: { ...baseScales(v => `${v.toFixed(0)}%`).y, min: 0 } },
    },
  });

  // Calibration
  const calDs = Object.entries(d.calibration).map(([key, r]) => ({
    label: r.label,
    data: r.mean_pred.map((x, i) => ({ x, y: r.fraction[i] })),
    borderColor: COLORS[key], backgroundColor: 'transparent',
    borderWidth: 2, pointRadius: 3.5, pointBackgroundColor: COLORS[key],
    pointBorderColor: '#0a0a0f', pointBorderWidth: 1.5, tension: 0.1, showLine: true,
  }));
  calDs.push({ label: 'Perfect', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: 'rgba(255,255,255,0.1)', backgroundColor: 'transparent', borderWidth: 1, borderDash: [4, 4], pointRadius: 0 });

  makeChart('ch-calibration', {
    type: 'scatter',
    data: { datasets: calDs },
    options: {
      responsive: true, maintainAspectRatio: false, showLine: true,
      plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: ctx => ` Predicted ${ctx.parsed.x.toFixed(2)}  Actual ${ctx.parsed.y.toFixed(2)}` } } },
      scales: {
        x: { ...baseScales().x, title: { display: true, text: 'Mean predicted probability', color: CHART_COLOR, font: { size: 10 } }, min: 0, max: 1 },
        y: { ...baseScales().y, title: { display: true, text: 'Actual frequency', color: CHART_COLOR, font: { size: 10 } }, min: 0, max: 1 },
      },
    },
  });

  // Betting simulation
  if (!d.betting_sim) return;
  const sim       = d.betting_sim;
  const lineColor = sim.total_pnl >= 0 ? '#4ade80' : '#f87171';
  const fillColor = sim.total_pnl >= 0 ? 'rgba(74,222,128,0.07)' : 'rgba(248,113,113,0.07)';

  makeChart('ch-sim', {
    type: 'line',
    data: {
      labels: sim.points.map(p => p.match),
      datasets: [
        { data: sim.points.map(p => p.cum_pnl), borderColor: lineColor, backgroundColor: fillColor, fill: true, tension: 0.2, pointRadius: 0, borderWidth: 2, label: 'Cumulative P&L' },
        { data: sim.points.map(() => 0), borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, pointRadius: 0, fill: false, label: 'Breakeven' },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` $${ctx.parsed.y.toFixed(2)}` } } },
      scales: {
        x: { ...baseScales().x, title: { display: true, text: 'Match (chronological)', color: CHART_COLOR, font: { size: 10 } } },
        y: { ...baseScales(v => `$${v.toFixed(0)}`).y, title: { display: true, text: 'Cumulative P&L', color: CHART_COLOR, font: { size: 10 } } },
      },
    },
  });
}

// ── Match card HTML ────────────────────────────────────────────────────────────
function matchCardHTML(match) {
  const ph = Math.round(match.prob_h * 100);
  const pd = Math.round(match.prob_d * 100);
  const pa = Math.round(match.prob_a * 100);

  const badgeClass = { H: 'outcome-h', D: 'outcome-d', A: 'outcome-a' }[match.predicted] || 'outcome-d';
  const badgeLabel = OUTCOME_LABELS[match.predicted] || match.predicted;

  const hasInsights = match.insights && match.insights.length > 0;
  const insightsHTML = hasInsights
    ? match.insights.map(i => {
        const teamClass = i.team === 'H' ? 'insight-home' : i.team === 'A' ? 'insight-away' : '';
        return `
        <div class="insight-line">
          <span class="insight-dot ${teamClass}">•</span>
          <span>${esc(i.text)}</span>
        </div>`;
      }).join('')
    : '';

  const chevron = `
    <div class="card-chevron">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>
    </div>`;

  return `
<div class="match-card">
  <div class="card-header">
    <span class="card-meta">${esc(match.date)}&ensp;&middot;&ensp;${esc(match.league)}</span>
    <span class="outcome-badge ${badgeClass}">${esc(badgeLabel)}</span>
  </div>
  <div class="match-layout">
    <div class="team-block team-home">
      <div class="team-name">${esc(match.home_team)}</div>
      <div class="team-prob home-prob">${ph}%</div>
      <div class="team-result-label">Home</div>
    </div>
    <div class="draw-block">
      <div class="draw-prob">${pd}%</div>
      <div class="team-result-label">Draw</div>
    </div>
    <div class="team-block team-away">
      <div class="team-name">${esc(match.away_team)}</div>
      <div class="team-prob away-prob">${pa}%</div>
      <div class="team-result-label">Away</div>
    </div>
  </div>
  <div class="prob-bar">
    <div class="prob-h" style="flex:${ph}"></div>
    <div class="prob-d" style="flex:${pd}"></div>
    <div class="prob-a" style="flex:${pa}"></div>
  </div>
  ${hasInsights ? `<div class="insights-section">${insightsHTML}</div>${chevron}` : ''}
</div>`;
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function loadingHTML(msg) {
  return `<div class="loading-inline"><div class="spinner"></div><span>${esc(msg)}</span></div>`;
}

function errorHTML(msg) {
  return `<div class="error-state">${esc(msg)}</div>`;
}

function pct(frac) { return (frac * 100).toFixed(1) + '%'; }

function hexAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Init ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupChartDefaults();

  // Sidebar nav
  document.getElementById('sidebar-nav').addEventListener('click', e => {
    const btn = e.target.closest('.nav-item');
    if (btn) navigate(btn.dataset.page);
  });

  // Bottom nav
  document.getElementById('bottom-nav').addEventListener('click', e => {
    const btn = e.target.closest('.bnav-item');
    if (btn) navigate(btn.dataset.page);
  });

  const hash = location.hash.replace('#', '') || 'predictions';
  navigate(hash);
});
