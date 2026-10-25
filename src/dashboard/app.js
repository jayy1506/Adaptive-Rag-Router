// Adaptive RAG Router Dashboard Client

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(view => view.classList.remove('active'));

  if (tabId === 'live') {
    document.getElementById('tabLiveBtn').classList.add('active');
    document.getElementById('liveView').classList.add('active');
  } else if (tabId === 'bench') {
    document.getElementById('tabBenchBtn').classList.add('active');
    document.getElementById('benchView').classList.add('active');
    loadBenchmarkData();
  } else if (tabId === 'ablation') {
    document.getElementById('tabAblationBtn').classList.add('active');
    document.getElementById('ablationView').classList.add('active');
    loadAblationData();
  }
}

function setQuery(text) {
  document.getElementById('queryInput').value = text;
}

async function handleRouteOnly() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;

  const btn = document.getElementById('btnRouteOnly');
  btn.innerText = 'Routing...';
  btn.disabled = true;

  try {
    const res = await fetch('/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, use_cache: true }),
    });

    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    renderRoutingTelemetry(data);

    document.getElementById('answerStatusBadge').innerText = 'Route Decision Only';
    document.getElementById('answerText').innerText = 
      `Route selected: k = ${data.executed_k} (Predicted k=${data.predicted_k}, Fallback=${data.fallback_triggered}). Full execution skipped.`;
    document.getElementById('passagesList').innerHTML = '<p class="placeholder-text">Click "Execute Pipeline" to fetch passages and synthesize answer.</p>';
    document.getElementById('passagesCount').innerText = '0';
  } catch (err) {
    alert('Routing error: ' + err.message);
  } finally {
    btn.innerText = 'Route Only';
    btn.disabled = false;
  }
}

async function handleRunQuery() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;

  const btn = document.getElementById('btnExecuteQuery');
  btn.innerText = 'Executing...';
  btn.disabled = true;

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, use_cache: true }),
    });

    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    renderFullExecution(data);
  } catch (err) {
    alert('Execution error: ' + err.message);
  } finally {
    btn.innerText = 'Execute Pipeline';
    btn.disabled = false;
  }
}

function renderRoutingTelemetry(data) {
  document.getElementById('resultsGrid').style.display = 'grid';

  const kBadge = document.getElementById('executedKBadge');
  kBadge.innerText = `k = ${data.executed_k}`;
  kBadge.className = 'badge ' + (data.executed_k === 0 ? '' : 'badge-success');

  document.getElementById('metricPredK').innerText = data.predicted_k;
  document.getElementById('metricConfidence').innerText = (data.confidence * 100).toFixed(1) + '%';
  document.getElementById('metricFallback').innerText = data.fallback_triggered ? 'Triggered' : 'Normal';
  document.getElementById('metricFallback').style.color = data.fallback_triggered ? '#f59e0b' : '#94a3b8';
  document.getElementById('metricEscalation').innerText = data.escalation_triggered ? 'Triggered' : 'None';
  document.getElementById('metricEscalation').style.color = data.escalation_triggered ? '#f43f5e' : '#94a3b8';

  // Render probability bars
  const probContainer = document.getElementById('probBarsContainer');
  probContainer.innerHTML = '';
  const probs = data.probabilities || {};

  [0, 3, 8, 15].forEach(k => {
    const pVal = probs[k] || 0.0;
    const pct = Math.round(pVal * 100);

    const row = document.createElement('div');
    row.className = 'prob-item';
    row.innerHTML = `
      <span class="prob-k-label">k = ${k}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${pct}%;"></div>
      </div>
      <span class="prob-val">${pct}%</span>
    `;
    probContainer.appendChild(row);
  });
}

function renderFullExecution(data) {
  renderRoutingTelemetry(data);

  // Latencies
  const rLat = data.router_latency_ms || 0.1;
  const retLat = (data.retrieval_latency_ms || 0) + (data.escalation_retrieval_latency_ms || 0);
  const genLat = (data.generation_latency_ms || 0) + (data.escalation_generation_latency_ms || 0);
  const totLat = data.total_latency_ms || (rLat + retLat + genLat);

  document.getElementById('latRouterText').innerText = rLat.toFixed(1) + 'ms';
  document.getElementById('latRetrievalText').innerText = retLat.toFixed(1) + 'ms';
  document.getElementById('latGenText').innerText = genLat.toFixed(1) + 'ms';
  document.getElementById('latTotalText').innerText = totLat.toFixed(1) + 'ms';

  const sumLat = Math.max(rLat + retLat + genLat, 0.001);
  document.getElementById('barRouter').style.width = ((rLat / sumLat) * 100).toFixed(1) + '%';
  document.getElementById('barRetrieval').style.width = ((retLat / sumLat) * 100).toFixed(1) + '%';
  document.getElementById('barGeneration').style.width = ((genLat / sumLat) * 100).toFixed(1) + '%';

  // Tokens
  document.getElementById('promptTokensText').innerText = data.prompt_tokens || 0;
  document.getElementById('completionTokensText').innerText = data.completion_tokens || 0;
  document.getElementById('totalTokensText').innerText = data.total_tokens || 0;

  // Answer
  document.getElementById('answerStatusBadge').innerText = 'Synthesized';
  document.getElementById('answerText').innerText = data.answer || '(Empty or inconclusive response)';

  // Passages
  const pList = document.getElementById('passagesList');
  pList.innerHTML = '';
  const passages = data.passages || [];
  document.getElementById('passagesCount').innerText = passages.length;

  if (passages.length === 0) {
    pList.innerHTML = '<p class="placeholder-text">Parametric generation used (k=0). No passages retrieved.</p>';
  } else {
    passages.forEach((p, idx) => {
      const card = document.createElement('div');
      card.className = 'passage-card';
      const score = (p.score !== undefined) ? `Score: ${p.score.toFixed(3)}` : '';
      card.innerHTML = `
        <div class="passage-meta">
          <span class="passage-title">[Chunk ${idx + 1}] ${p.title || 'Untitled Passage'}</span>
          <span>${score}</span>
        </div>
        <p class="passage-text">${p.text}</p>
      `;
      pList.appendChild(card);
    });
  }
}

async function loadBenchmarkData() {
  const tbody = document.getElementById('benchmarkTableBody');
  try {
    const res = await fetch('/api/benchmark');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();

    tbody.innerHTML = '';
    data.forEach(row => {
      const tr = document.createElement('tr');
      const isRouter = (row.system === 'adaptive_router');
      if (isRouter) tr.className = 'highlight-row';

      const sysName = isRouter ? 'Adaptive Router (Ours)' : row.system.replace('_', ' ').toUpperCase();
      tr.innerHTML = `
        <td><strong>${sysName}</strong></td>
        <td>${row.avg_executed_k}</td>
        <td>${(row.exact_match * 100).toFixed(1)}%</td>
        <td><strong>${row.token_f1.toFixed(3)}</strong></td>
        <td>${row.avg_total_tokens.toFixed(0)}</td>
        <td>${row.p50_total_latency_ms.toFixed(1)}ms</td>
        <td>${row.p95_total_latency_ms.toFixed(1)}ms</td>
        <td>${(row.under_retrieval_rate * 100).toFixed(2)}%</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="color: #f43f5e;">Failed to load benchmark data: ${err.message}</td></tr>`;
  }
}

async function loadAblationData() {
  const tbody = document.getElementById('ablationTableBody');
  try {
    const res = await fetch('/api/ablation');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();

    tbody.innerHTML = '';
    data.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><code>${row.experiment_id}</code></td>
        <td><strong>${row.model_name}</strong></td>
        <td>${row.feature_type}</td>
        <td>${row.feature_dim}</td>
        <td>${(row.accuracy * 100).toFixed(1)}%</td>
        <td><strong>${row.weighted_f1.toFixed(3)}</strong></td>
        <td>${row.latency_ms_per_query.toFixed(3)}ms</td>
        <td>${row.train_time_ms.toFixed(0)}ms</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="color: #f43f5e;">Failed to load ablation data: ${err.message}</td></tr>`;
  }
}
