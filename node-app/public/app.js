const sensors = [
  { key: 'X', field: 'x', color: '#4f46e5' },
  { key: 'Y', field: 'y', color: '#0f766e' },
  { key: 'Z', field: 'z', color: '#ea580c' },
  { key: 'EDA', field: 'eda', color: '#9333ea' },
  { key: 'HR', field: 'hr', color: '#dc2626' },
  { key: 'TEMP', field: 'temp', color: '#0284c7' },
];

let source = 'simulated';
let timer = null;

const elements = {
  healthDot: document.querySelector('#healthDot'),
  healthLabel: document.querySelector('#healthLabel'),
  healthMeta: document.querySelector('#healthMeta'),
  stressLabel: document.querySelector('#stressLabel'),
  stressMeta: document.querySelector('#stressMeta'),
  predictionCard: document.querySelector('#predictionCard'),
  observedAt: document.querySelector('#observedAt'),
  sequenceIndex: document.querySelector('#sequenceIndex'),
  sensorGrid: document.querySelector('#sensorGrid'),
  valueGrid: document.querySelector('#valueGrid'),
  historyList: document.querySelector('#historyList'),
  refreshHistory: document.querySelector('#refreshHistory'),
};

renderPlaceholders();
bindEvents();
checkHealth();
loadSource(source);

function bindEvents() {
  document.querySelectorAll('.tab').forEach((button) => {
    button.addEventListener('click', () => {
      source = button.dataset.source;
      document.querySelectorAll('.tab').forEach((tab) => tab.classList.remove('is-active'));
      button.classList.add('is-active');
      loadSource(source);
    });
  });

  elements.refreshHistory.addEventListener('click', () => loadHistory());
}

function loadSource(nextSource) {
  if (timer) {
    clearInterval(timer);
  }
  source = nextSource;
  loadSnapshot();
  loadHistory();
  timer = setInterval(loadSnapshot, 5000);
}

async function checkHealth() {
  try {
    const health = await fetchJson('/api/health');
    elements.healthDot.className = 'status-dot online';
    elements.healthLabel.textContent = 'Node API online';
    elements.healthMeta.textContent = `Inference: ${health.inferenceBaseUrl}`;
  } catch (error) {
    elements.healthDot.className = 'status-dot offline';
    elements.healthLabel.textContent = 'API offline';
    elements.healthMeta.textContent = error.message;
  }
}

async function loadSnapshot() {
  try {
    const snapshot = await fetchJson(`/api/telemetry/${source}/latest`);
    renderSnapshot(snapshot);
  } catch (error) {
    elements.stressLabel.textContent = 'Feed unavailable';
    elements.stressMeta.textContent = error.message;
    elements.predictionCard.className = 'prediction-card high-stress';
  }
}

async function loadHistory() {
  try {
    const rows = await fetchJson(`/api/telemetry/${source}/history?limit=20`);
    renderHistory(rows);
  } catch (error) {
    elements.historyList.innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`;
  }
}

function renderSnapshot(snapshot) {
  const category = Number(snapshot.predictedStressCategory);
  elements.predictionCard.className = `prediction-card ${stressClass(category)}`;
  elements.stressLabel.textContent = snapshot.predictedStressLabel;
  elements.stressMeta.textContent = `${sourceLabel()} source classified as category ${category}`;
  elements.observedAt.textContent = formatTime(snapshot.observedAt);
  elements.sequenceIndex.textContent = `Sequence ${snapshot.sequenceIndex}`;

  elements.valueGrid.innerHTML = sensors
    .map((sensor) => {
      const value = snapshot.reading?.[sensor.field];
      return `
        <div class="value-card">
          <strong>${sensor.key}</strong>
          <span>${formatNumber(value)}</span>
        </div>
      `;
    })
    .join('');

  elements.sensorGrid.innerHTML = sensors
    .map((sensor) => {
      const latest = snapshot.reading?.[sensor.field];
      return `
        <article class="sensor-card">
          <header>
            <strong>${sensor.key}</strong>
            <span>${formatNumber(latest)}</span>
          </header>
          <canvas data-sensor="${sensor.key}" height="150"></canvas>
        </article>
      `;
    })
    .join('');

  for (const sensor of sensors) {
    const canvas = document.querySelector(`canvas[data-sensor="${sensor.key}"]`);
    drawChart(canvas, snapshot.series?.[sensor.key] || [], sensor.color);
  }
}

function renderHistory(rows) {
  if (!rows.length) {
    elements.historyList.innerHTML =
      '<p class="muted">No MongoDB history yet. Configure MONGODB_URI to persist mild and high stress events.</p>';
    return;
  }

  elements.historyList.innerHTML = rows
    .map((row) => {
      const reading = row.reading || {};
      return `
        <div class="history-item">
          <strong>${escapeHtml(row.stressLabel || 'Stress event')}</strong>
          <small>${formatTime(row.loggedAt)} | X ${formatNumber(reading.x)} | HR ${formatNumber(reading.hr)} | TEMP ${formatNumber(reading.temp)}</small>
        </div>
      `;
    })
    .join('');
}

function renderPlaceholders() {
  elements.sensorGrid.innerHTML = sensors
    .map(
      (sensor) => `
        <article class="sensor-card">
          <header>
            <strong>${sensor.key}</strong>
            <span>-</span>
          </header>
          <canvas data-sensor="${sensor.key}" height="150"></canvas>
        </article>
      `,
    )
    .join('');

  elements.valueGrid.innerHTML = sensors
    .map(
      (sensor) => `
        <div class="value-card">
          <strong>${sensor.key}</strong>
          <span>-</span>
        </div>
      `,
    )
    .join('');
}

function drawChart(canvas, points, color) {
  if (!canvas) {
    return;
  }

  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(150 * ratio));

  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);

  const width = rect.width;
  const height = 150;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#d9e2ec';
  ctx.lineWidth = 1;

  for (let i = 0; i < 4; i += 1) {
    const y = 16 + i * 36;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  if (points.length < 2) {
    return;
  }

  const values = points.map((point) => Number(point.v)).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const stepX = width / Math.max(points.length - 1, 1);

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();

  points.forEach((point, index) => {
    const x = index * stepX;
    const y = height - 18 - ((Number(point.v) - min) / range) * (height - 34);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.stroke();
}

async function fetchJson(url) {
  const response = await fetch(url);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `${response.status} ${response.statusText}`);
  }
  return body;
}

function stressClass(category) {
  if (category === 0) {
    return 'no-stress';
  }
  if (category === 1) {
    return 'mild-stress';
  }
  return 'high-stress';
}

function sourceLabel() {
  return source === 'real' ? 'Real device' : 'Simulated';
}

function formatTime(value) {
  if (!value) {
    return '-';
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(new Date(value));
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return '-';
  }
  return number.toFixed(Math.abs(number) >= 100 ? 1 : 2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
