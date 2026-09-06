// ThermoWatch AI — map + status log + analysis wiring.

const logPanel = document.getElementById('logpanel');
const lastUpdatedEl = document.getElementById('lastupdated');
const hotspotCountEl = document.getElementById('hotspotcount');

function logLine(msg, isErr) {
  const div = document.createElement('div');
  if (isErr) div.className = 'err';
  div.textContent = msg;
  logPanel.appendChild(div);
  logPanel.scrollTop = logPanel.scrollHeight;
}

function logBatch(lines) {
  (lines || []).forEach(l => logLine(l, /error/i.test(l)));
}

// ---- map setup ----
// Default base layer is satellite imagery (Esri World Imagery), not an
// OpenStreetMap street map: OSM's standard raster tiles bake in disputed-
// boundary line styling for the Kashmir region that we have no ability to
// restyle (it's part of the raster image, not a separate layer we control).
// Satellite imagery has no political boundary lines drawn on it at all, so
// this sidesteps that entirely rather than us redrawing a disputed border
// ourselves. OpenStreetMap street view stays available from the layer
// control (top-right) for anyone who wants street/place-name context.
const map = L.map('map').setView([22.5, 82.0], 5);

const satelliteLayer = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  { attribution: 'Esri, Maxar, Earthstar Geographics', maxZoom: 18 }
).addTo(map);

const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors',
  maxZoom: 18,
});

L.control.layers({
  'Satellite (no borders drawn)': satelliteLayer,
  'Street map (OpenStreetMap)': osmLayer,
}, {}, { position: 'topright', collapsed: false }).addTo(map);

let hotspotLayer = L.layerGroup().addTo(map);
let markerById = {};
let selectedMarker = null;
let analyzedRiskByKey = {}; // "lat.toFixed(3),lon.toFixed(3)" -> risk_score
let currentHotspots = [];

function keyFor(lat, lon) {
  return `${(+lat).toFixed(3)},${(+lon).toFixed(3)}`;
}

async function loadPriorAnalyses() {
  try {
    const resp = await fetch('/api/analyses');
    const data = await resp.json();
    (data.analyses || []).forEach(a => {
      analyzedRiskByKey[keyFor(a.latitude, a.longitude)] = a.risk_score;
    });
  } catch (e) {
    logLine(`[WARN] Could not load prior analyses for risk filter: ${e}`, true);
  }
}

function frpColor(frp) {
  if (frp === null || frp === undefined) return '#ffcc00';
  if (frp >= 15) return '#cc0000';
  if (frp >= 5) return '#ff8800';
  return '#ffdd00';
}

function plotHotspots(hotspots) {
  currentHotspots = hotspots;
  hotspotLayer.clearLayers();
  markerById = {};
  hotspots.forEach(h => {
    const marker = L.circleMarker([h.latitude, h.longitude], {
      radius: 5,
      color: '#222',
      weight: 1,
      fillColor: frpColor(h.frp),
      fillOpacity: 0.85,
    });
    const facilityLine = h.nearest_facility_name
      ? `<br>Nearest facility: ${h.nearest_facility_name} (${h.nearest_facility_type}), ${h.nearest_facility_m}m`
      : '';
    marker.bindTooltip(
      `FRP: ${h.frp ?? 'n/a'} | Brightness: ${h.brightness ?? 'n/a'} | ${h.acq_date} ${h.acq_time || ''}${facilityLine}`
    );
    marker.on('click', () => onHotspotClick(h, marker));
    marker.addTo(hotspotLayer);
    markerById[h.id] = marker;
  });
  applyRiskFilter();
}

function applyRiskFilter() {
  const minRisk = parseInt(document.getElementById('riskmin').value || '0', 10);
  let shown = 0;
  currentHotspots.forEach(h => {
    const marker = markerById[h.id];
    if (!marker) return;
    const risk = analyzedRiskByKey[keyFor(h.latitude, h.longitude)];
    const visible = (risk === undefined) || (risk >= minRisk);
    if (visible) {
      if (!hotspotLayer.hasLayer(marker)) marker.addTo(hotspotLayer);
      shown++;
    } else {
      hotspotLayer.removeLayer(marker);
    }
  });
  hotspotCountEl.textContent = `${shown}/${currentHotspots.length} hotspots plotted` +
    (minRisk > 0 ? ` (risk >= ${minRisk}, unanalyzed always shown)` : '');
}

document.getElementById('riskmin').addEventListener('change', applyRiskFilter);

async function refreshHotspots() {
  const start = document.getElementById('startdate').value;
  const end = document.getElementById('enddate').value;
  const industrialOnly = document.getElementById('industrialonly').checked;
  const radiusKm = document.getElementById('radiuskm').value || '5';
  const params = new URLSearchParams({
    refresh: 'true',
    industrial_only: industrialOnly ? 'true' : 'false',
    radius_km: radiusKm,
  });
  if (start) params.set('start_date', start);
  if (end) params.set('end_date', end);

  if (industrialOnly) {
    logLine(`> Requesting hotspot refresh (industrial facilities only, within ${radiusKm}km)... this can take up to a minute the first time while the industrial facility index builds.`);
  } else {
    logLine('> Requesting hotspot refresh (all detections, no industrial filter)...');
  }
  try {
    const resp = await fetch(`/api/hotspots?${params.toString()}`);
    const data = await resp.json();
    logBatch(data.log);
    if (data.is_synthetic_demo) {
      logLine('[NOTICE] Displaying SYNTHETIC DEMO hotspots — set FIRMS_MAP_KEY in .env for live NASA FIRMS data.');
    }
    plotHotspots(data.hotspots);
    lastUpdatedEl.textContent = data.last_updated_utc;
  } catch (e) {
    logLine(`[ERROR] Failed to fetch hotspots: ${e}`, true);
  }
}

document.getElementById('btnrefresh').addEventListener('click', refreshHotspots);
document.getElementById('industrialonly').addEventListener('change', refreshHotspots);

// ---- selection / analysis ----
const selectedInfo = document.getElementById('selectedinfo');
const analysisPanel = document.getElementById('analysispanel');
const analysisContent = document.getElementById('analysiscontent');

function onHotspotClick(h, marker) {
  if (selectedMarker) {
    selectedMarker.setStyle({ weight: 1, color: '#222' });
  }
  marker.setStyle({ weight: 3, color: '#0033cc' });
  selectedMarker = marker;

  selectedInfo.innerHTML = `
    <table class="datatable">
      <tr><th>Latitude</th><td>${h.latitude}</td></tr>
      <tr><th>Longitude</th><td>${h.longitude}</td></tr>
      <tr><th>FRP</th><td>${h.frp ?? 'n/a'}</td></tr>
      <tr><th>Brightness</th><td>${h.brightness ?? 'n/a'}</td></tr>
      <tr><th>Acquired</th><td>${h.acq_date} ${h.acq_time || ''}</td></tr>
      <tr><th>Confidence</th><td>${h.confidence ?? 'n/a'}</td></tr>
      <tr><th>Satellite</th><td>${h.satellite ?? 'n/a'}</td></tr>
    </table>
    <button id="btnanalyze" style="margin-top:6px; width:100%;">RUN FULL ANALYSIS</button>
  `;
  document.getElementById('btnanalyze').addEventListener('click', () => runAnalysis(h));
}

async function runAnalysis(h) {
  analysisPanel.classList.add('open');
  analysisContent.innerHTML = '<p>Running pipeline... see System Log for progress.</p>';
  analysisPanel.scrollIntoView({ behavior: 'instant', block: 'start' });
  logLine(`> Starting full analysis pipeline for (${h.latitude}, ${h.longitude})...`);
  try {
    const resp = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ latitude: h.latitude, longitude: h.longitude, hotspot_id: h.id }),
    });
    const data = await resp.json();
    logBatch(data.log);
    renderAnalysis(data);
    analyzedRiskByKey[keyFor(h.latitude, h.longitude)] = data.result.fusion.risk_score;
    applyRiskFilter();
  } catch (e) {
    logLine(`[ERROR] Analysis pipeline failed: ${e}`, true);
    analysisContent.innerHTML = `<p style="color:#7a1f1f;">Analysis failed: ${e}</p>`;
  }
}

function tagHtml(text, kind) {
  return `<span class="tag ${kind}">${text}</span>`;
}

function renderAnalysis(data) {
  const r = data.result;
  const probs = r.fusion.class_probabilities;
  const probRows = Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<tr><td>${k}</td><td>${(v * 100).toFixed(1)}%</td></tr>`)
    .join('');

  const evidenceItems = r.fusion.evidence.map(e => `<li>${e}</li>`).join('');

  analysisContent.innerHTML = `
    <div class="verdict-box">
      <div class="cls">${r.fusion.classification.toUpperCase().replace(/_/g, ' ')}</div>
      Confidence: ${(r.fusion.confidence * 100).toFixed(1)}% &nbsp;|&nbsp; Risk score: ${r.fusion.risk_score}/100
      <div style="margin-top:4px;">
        ${tagHtml('Branch A: Gemini vision (placeholder for trained CNN)', 'placeholder')}
        ${tagHtml('Branch B: Random Forest (synthetic training data)', 'placeholder')}
        ${tagHtml(r.branch_c.simulated ? 'Branch C: simulated history' : 'Branch C: real archived history', r.branch_c.simulated ? 'simulated' : 'real')}
      </div>
      <div class="caption">Evidence:</div>
      <ul class="evidence-list">${evidenceItems}</ul>
    </div>

    <div class="grid3">
      <div class="panel">
        <div class="panel-title">Branch Votes</div>
        <img class="chartimg" src="data:image/png;base64,${r.charts.branch_votes}">
      </div>
      <div class="panel">
        <div class="panel-title">Class Probabilities</div>
        <table class="datatable">${probRows}</table>
      </div>
      <div class="panel">
        <div class="panel-title">Risk Gauge</div>
        <img class="chartimg" src="data:image/png;base64,${r.charts.risk_gauge}">
      </div>
    </div>

    <div class="grid3">
      <div class="panel"><div class="panel-title">Original Image</div><img class="chartimg" src="data:image/png;base64,${r.branch_a.images.original}"></div>
      <div class="panel"><div class="panel-title">Grayscale</div><img class="chartimg" src="data:image/png;base64,${r.branch_a.images.grayscale}"></div>
      <div class="panel"><div class="panel-title">Edge Detection</div><img class="chartimg" src="data:image/png;base64,${r.branch_a.images.edges}"></div>
      <div class="panel"><div class="panel-title">False Color (simulated heat)</div><img class="chartimg" src="data:image/png;base64,${r.branch_a.images.false_color}"></div>
      <div class="panel"><div class="panel-title">False Color Alt (simulated heat)</div><img class="chartimg" src="data:image/png;base64,${r.branch_a.images.false_color_alt}"></div>
      <div class="panel"><div class="panel-title">K-Means Segmentation</div><img class="chartimg" src="data:image/png;base64,${r.branch_a.images.kmeans}"></div>
    </div>

    <div class="grid2">
      <div class="panel"><div class="panel-title">Feature Importance (Random Forest)</div><img class="chartimg" src="data:image/png;base64,${r.charts.feature_importance}"></div>
      <div class="panel"><div class="panel-title">Feature Correlation Heatmap</div><img class="chartimg" src="data:image/png;base64,${r.charts.correlation_heatmap}"></div>
    </div>

    <div class="grid2">
      <div class="panel"><div class="panel-title">30-Day Thermal History ${r.branch_c.simulated ? '(simulated)' : '(real)'}</div><img class="chartimg" src="data:image/png;base64,${r.charts.history_line}"></div>
      <div class="panel"><div class="panel-title">Anomaly Z-Score by Day</div><img class="chartimg" src="data:image/png;base64,${r.charts.anomaly_bar}"></div>
    </div>

    <div class="grid3">
      <div class="panel"><div class="panel-title">Confusion Matrix (held-out)</div><img class="chartimg" src="data:image/png;base64,${r.charts.confusion_matrix}"></div>
      <div class="panel"><div class="panel-title">ROC Curve</div><img class="chartimg" src="data:image/png;base64,${r.charts.roc_curve}"></div>
      <div class="panel"><div class="panel-title">Precision-Recall Curve</div><img class="chartimg" src="data:image/png;base64,${r.charts.pr_curve}"></div>
    </div>

    <div class="panel">
      <div class="panel-title">Model Evaluation (Random Forest, held-out split, ${r.branch_b.model_metrics.n_test} test samples)</div>
      <table class="datatable">
        <tr><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr>
        <tr>
          <td>${r.branch_b.model_metrics.accuracy.toFixed(3)}</td>
          <td>${r.branch_b.model_metrics.precision.toFixed(3)}</td>
          <td>${r.branch_b.model_metrics.recall.toFixed(3)}</td>
          <td>${r.branch_b.model_metrics.f1.toFixed(3)}</td>
          <td>${r.branch_b.model_metrics.auc.toFixed(3)}</td>
        </tr>
      </table>
      <div class="caption">Trained on synthetic placeholder data (see /about) — evaluation methodology itself is real.</div>
    </div>

    <div class="panel">
      <div class="panel-title">Vision Ensemble / Provider Agreement</div>
      ${renderEnsembleTable(r.branch_a.ensemble)}
      <div class="caption">Independent-model agreement is itself evidence: providers without a configured API key show as UNCONFIGURED, not a crash.</div>
    </div>

    <div class="panel">
      <div class="panel-title">Context Branch — Raw Inputs</div>
      <table class="datatable">
        <tr><th>Nearest facility</th><td>${r.branch_b.facility.found ? `${r.branch_b.facility.facility_name} (${r.branch_b.facility.facility_type}), ${r.branch_b.facility.distance_m}m` : (r.branch_b.facility.status === 'unavailable' ? 'Overpass unavailable' : 'none within radius')}</td></tr>
        <tr><th>Temp / Humidity</th><td>${r.branch_b.weather.temp_c ?? 'n/a'}&deg;C / ${r.branch_b.weather.humidity_pct ?? 'n/a'}%</td></tr>
        <tr><th>Precip / Soil moisture</th><td>${r.branch_b.weather.precip_mm ?? 'n/a'}mm / ${r.branch_b.weather.soil_moisture ?? 'n/a'}</td></tr>
      </table>
    </div>
  `;
}

function renderEnsembleTable(ensemble) {
  const rows = ensemble.map(e => {
    if (e.status === 'unconfigured') {
      return `<tr><td>${e.provider}</td><td colspan="3" style="color:#7a1f1f;">NO API KEY CONFIGURED</td></tr>`;
    }
    if (e.status === 'error') {
      return `<tr><td>${e.provider}</td><td colspan="3" style="color:#7a1f1f;">ERROR: ${e.message}</td></tr>`;
    }
    return `<tr><td>${e.provider}</td><td>${e.classification}</td><td>${(e.confidence*100).toFixed(0)}%</td><td>${(e.evidence||[]).join('; ')}</td></tr>`;
  }).join('');
  return `<table class="datatable"><tr><th>Provider</th><th>Classification</th><th>Confidence</th><th>Evidence</th></tr>${rows}</table>`;
}

// ---- industrial facility index status polling ----
const indexStatusEl = document.getElementById('indexstatus');
let indexWasBuilding = false;

async function pollIndustrialIndexStatus() {
  try {
    const resp = await fetch('/api/industrial-index/status');
    const data = await resp.json();
    if (data.build_in_progress) {
      indexWasBuilding = true;
      indexStatusEl.textContent = `[building industrial facility index in background... ${data.cached_facility_count} cached so far]`;
      setTimeout(pollIndustrialIndexStatus, 8000);
    } else if (data.ready) {
      indexStatusEl.style.color = '#205723';
      indexStatusEl.textContent = `[industrial index ready: ${data.cached_facility_count} facilities cached]`;
      if (indexWasBuilding) {
        logLine(`[INDUSTRIAL-INDEX] Build finished (${data.cached_facility_count} facilities cached). Refreshing hotspots with the filter applied...`);
        indexWasBuilding = false;
        refreshHotspots();
      }
    } else {
      indexStatusEl.textContent = '[industrial index not built yet]';
      setTimeout(pollIndustrialIndexStatus, 8000);
    }
  } catch (e) {
    // non-fatal — just stop polling silently
  }
}

// initial load
loadPriorAnalyses().then(refreshHotspots);
pollIndustrialIndexStatus();
