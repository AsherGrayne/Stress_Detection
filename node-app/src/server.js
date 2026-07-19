import 'dotenv/config';

import express from 'express';
import { MongoClient } from 'mongodb';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..', '..');

const config = {
  port: Number(process.env.PORT || 8090),
  csvPath: path.resolve(process.env.STRESS_CSV_PATH || path.join(ROOT_DIR, 'balanced_data.csv')),
  inferenceBaseUrl: trimTrailingSlash(process.env.STRESS_INFERENCE_BASE_URL || 'http://127.0.0.1:8081'),
  intervalMs: Number(process.env.STRESS_INTERVAL_MS || 5000),
  historySize: Number(process.env.STRESS_CHART_HISTORY_SIZE || 60),
  firebaseUrl:
    process.env.FIREBASE_URL ||
    'https://stress-detection-c2bec-default-rtdb.asia-southeast1.firebasedatabase.app/data.json',
  mongodbUri: process.env.MONGODB_URI || process.env.STRESS_MONGODB_URI || '',
};

const labels = new Map([
  [0, 'No Stress'],
  [1, 'Mild Stress'],
  [2, 'High Stress'],
]);

const sourceNames = new Set(['simulated', 'real']);

class CsvFeed {
  constructor(csvPath) {
    const lines = readFileSync(csvPath, 'utf8')
      .trim()
      .split(/\r?\n/)
      .slice(1)
      .filter(Boolean);

    if (lines.length === 0) {
      throw new Error(`No data rows found in ${csvPath}`);
    }

    this.rows = lines.map((line, index) => {
      const [x, y, z, eda, hr, temp] = line.split(',').map(Number);
      return {
        sequenceIndex: index,
        reading: { x, y, z, eda, hr, temp },
      };
    });
    this.cursor = 0;
  }

  next() {
    const row = this.rows[this.cursor];
    this.cursor = (this.cursor + 1) % this.rows.length;
    return row;
  }
}

class TelemetryState {
  constructor(historySize) {
    this.historySize = historySize;
    this.snapshot = null;
    this.series = {
      X: [],
      Y: [],
      Z: [],
      EDA: [],
      HR: [],
      TEMP: [],
    };
  }

  append(observedAt, sequenceIndex, reading, category) {
    const t = observedAt.getTime();
    const values = {
      X: reading.x,
      Y: reading.y,
      Z: reading.z,
      EDA: reading.eda,
      HR: reading.hr,
      TEMP: reading.temp,
    };

    for (const [name, value] of Object.entries(values)) {
      this.series[name].push({ t, v: value });
      if (this.series[name].length > this.historySize) {
        this.series[name].shift();
      }
    }

    this.snapshot = {
      observedAt: observedAt.toISOString(),
      sequenceIndex,
      reading,
      predictedStressCategory: category,
      predictedStressLabel: labelFor(category),
      series: this.series,
    };

    return this.snapshot;
  }

  latest() {
    return this.snapshot;
  }
}

class StressLogStore {
  constructor(uri) {
    this.enabled = Boolean(uri);
    this.client = null;
    this.collections = {};
    this.ready = this.#connect(uri).catch((error) => {
      this.enabled = false;
      console.error('MongoDB connection failed:', error.message);
    });
  }

  async #connect(uri) {
    if (!uri) {
      console.warn('MongoDB URI not configured. Mild/high stress logs will remain disabled.');
      return;
    }

    this.client = new MongoClient(uri);
    await this.client.connect();
    const db = this.client.db('Stress_Detection');
    this.collections.simulated = db.collection('stress_log_simulated');
    this.collections.real = db.collection('stress_log_real');
    console.log('MongoDB stress logging enabled.');
  }

  async log(source, observedAt, category, reading) {
    if (!this.enabled || category < 1 || !sourceNames.has(source)) {
      return;
    }

    try {
      await this.ready;
      const collection = this.collections[source];
      if (!collection) {
        return;
      }
      await collection.insertOne({
        stressCategory: category,
        stressLabel: labelFor(category),
        loggedAt: observedAt,
        reading,
      });
    } catch (error) {
      console.error(`MongoDB insert failed for ${source}:`, error.message);
    }
  }

  async history(source, limit) {
    if (!this.enabled || !sourceNames.has(source)) {
      return [];
    }

    try {
      await this.ready;
      const collection = this.collections[source];
      if (!collection) {
        return [];
      }
      const safeLimit = Math.max(1, Math.min(Number(limit) || 50, 250));
      return collection
        .find({}, { projection: { _id: 0 } })
        .sort({ loggedAt: -1 })
        .limit(safeLimit)
        .toArray();
    } catch (error) {
      console.error(`MongoDB history lookup failed for ${source}:`, error.message);
      return [];
    }
  }

  async close() {
    if (this.client) {
      await this.client.close();
    }
  }
}

const csvFeed = new CsvFeed(config.csvPath);
const simulatedState = new TelemetryState(config.historySize);
const realState = new TelemetryState(config.historySize);
const stressLogs = new StressLogStore(config.mongodbUri);

const app = express();
app.use(express.json());
app.use((_, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  next();
});
app.use(express.static(path.join(__dirname, '..', 'public')));

app.get('/api/health', (_, res) => {
  res.json({
    status: 'ok',
    service: 'stress-detection-node',
    inferenceBaseUrl: config.inferenceBaseUrl,
  });
});

app.get('/api/telemetry/simulated/latest', (_, res) => {
  sendSnapshot(res, simulatedState.latest(), 'Simulated feed not ready yet');
});

app.get('/api/telemetry/real/latest', (_, res) => {
  sendSnapshot(res, realState.latest(), 'Real feed not ready yet');
});

app.get('/api/telemetry/next', async (_, res, next) => {
  try {
    const snapshot = await processSimulatedTick();
    res.json({
      sequenceIndex: snapshot.sequenceIndex,
      reading: snapshot.reading,
      predictedStressCategory: snapshot.predictedStressCategory,
      predictedStressLabel: snapshot.predictedStressLabel,
    });
  } catch (error) {
    next(error);
  }
});

app.get('/api/telemetry/:source/history', async (req, res) => {
  const { source } = req.params;
  if (!sourceNames.has(source)) {
    res.status(404).json({ error: 'Unknown telemetry source' });
    return;
  }
  res.json(await stressLogs.history(source, req.query.limit));
});

app.get(/.*/, (_, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'index.html'));
});

app.use((error, _, res, _next) => {
  console.error(error);
  res.status(error.status || 503).json({
    error: error.message || 'Service unavailable',
  });
});

app.listen(config.port, () => {
  console.log(`Stress Detection Node app running at http://localhost:${config.port}`);
  console.log(`Python inference expected at ${config.inferenceBaseUrl}`);
  startSchedulers();
});

function startSchedulers() {
  processSimulatedTick().catch((error) => console.error('Simulated tick failed:', error.message));
  processRealTick().catch((error) => console.error('Real tick failed:', error.message));

  setInterval(() => {
    processSimulatedTick().catch((error) => console.error('Simulated tick failed:', error.message));
  }, config.intervalMs);

  setInterval(() => {
    processRealTick().catch((error) => console.error('Real tick failed:', error.message));
  }, config.intervalMs);
}

async function processSimulatedTick() {
  const row = csvFeed.next();
  const category = await predict(row.reading);
  const observedAt = new Date();
  const snapshot = simulatedState.append(observedAt, row.sequenceIndex, row.reading, category);
  await stressLogs.log('simulated', observedAt, category, row.reading);
  return snapshot;
}

async function processRealTick() {
  const response = await fetch(config.firebaseUrl);
  if (!response.ok) {
    throw new Error(`Firebase returned ${response.status}`);
  }

  const data = await response.json();
  if (!data || data.accX == null) {
    return null;
  }

  const reading = {
    x: numberOrZero(data.accX),
    y: numberOrZero(data.accY),
    z: numberOrZero(data.accZ),
    eda: numberOrZero(data.gsr),
    hr: numberOrZero(data.pulse),
    temp: numberOrZero(data.temperature),
  };
  const category = await predict(reading);
  const observedAt = new Date();
  const sequenceIndex = (realState.latest()?.sequenceIndex || 0) + 1;
  const snapshot = realState.append(observedAt, sequenceIndex, reading, category);
  await stressLogs.log('real', observedAt, category, reading);
  return snapshot;
}

async function predict(reading) {
  const response = await fetch(`${config.inferenceBaseUrl}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reading),
  });

  if (!response.ok) {
    throw new Error(
      `Inference service unavailable (${response.status}). Start Python with: python -m uvicorn inference_service.main:app --host 127.0.0.1 --port 8081`,
    );
  }

  const body = await response.json();
  return Number(body.stressCategory);
}

function sendSnapshot(res, snapshot, message) {
  if (!snapshot) {
    res.status(503).json({ error: message });
    return;
  }
  res.json(snapshot);
}

function labelFor(category) {
  return labels.get(Number(category)) || 'Unknown';
}

function numberOrZero(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function trimTrailingSlash(value) {
  return value.replace(/\/+$/, '');
}

async function shutdown() {
  await stressLogs.close();
  process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
