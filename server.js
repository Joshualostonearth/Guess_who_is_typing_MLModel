const express = require('express');
const fs      = require('fs');
const path    = require('path');
const { spawn } = require('child_process');

const app  = express();
const PORT = 3000;

const CSV_FILE      = path.join(__dirname, 'user_typing_data.csv');
const MLSVM_PATH    = path.join(__dirname, 'MLSVM.py');
const MODEL_FILE    = path.join(__dirname, 'svm_model.pkl');

const CSV_HEADER = [
  'Timestamp','Name','WPM','Accuracy(%)','Characters','Errors','Regressions',
  'Time(s)','AvgDwellTime(ms)','AvgFlightTime(ms)','AvgDDTime(ms)','AvgUUTime(ms)',
  'LeftShiftCount','RightShiftCount','ShiftPreference','BackspaceCount','BackspaceStyle',
  'CtrlACount','CapsLockCount','RolloverCount','AvgRolloverDuration(ms)',
  'ErrorRate(%)','Burstiness','SpeedDecayRate','WPMOverTime','TopDigraphs','TopTrigraphs'
].join(',') + '\n';

if (!fs.existsSync(CSV_FILE)) {
  fs.writeFileSync(CSV_FILE, CSV_HEADER, 'utf-8');
  console.log(`Created ${CSV_FILE}`);
} else {
  const firstLine = fs.readFileSync(CSV_FILE, 'utf-8').split('\n')[0];
  if (!firstLine.includes('AvgDwellTime')) {
    const backupPath = CSV_FILE.replace('.csv', `_backup_${Date.now()}.csv`);
    fs.copyFileSync(CSV_FILE, backupPath);
    fs.writeFileSync(CSV_FILE, CSV_HEADER, 'utf-8');
    console.log(`Backed up old CSV; created new header.`);
  }
}



function getPythonPath() {
  const candidates = [
    path.join(__dirname, 'venv', 'Scripts', 'python.exe'),
    path.join(__dirname, 'venv', 'bin',     'python'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return 'python3';
}

function spawnPython(args, stdinData, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    const python = getPythonPath();
    const child  = spawn(python, args);
    let stdout = '', stderr = '';
    child.stdout.on('data', d => { stdout += d.toString(); });
    child.stderr.on('data', d => { stderr += d.toString(); });
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error('Python process timed out'));
    }, timeoutMs);
    child.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`Python exited ${code}: ${stderr.slice(0, 500)}`));
      } else {
        resolve({ stdout: stdout.trim(), stderr: stderr.trim() });
      }
    });
    if (stdinData) {
      child.stdin.write(stdinData);
      child.stdin.end();
    }
  });
}

function autoTrain() {
  const python = getPythonPath();
  console.log(`\n  Auto-training SVM model...`);
  const child = spawn(python, [MLSVM_PATH, '--train']);
  child.stdout.on('data', d => {
    try {
      const r = JSON.parse(d.toString().trim());
      console.log(`  Model trained — type: ${r.model_type}, samples: ${r.samples}, accuracy: ${r.accuracy ?? 'N/A'}`);
    } catch (_) {}
  });
  child.stderr.on('data', d => process.stderr.write(d));
}

if (!fs.existsSync(MODEL_FILE)) {
  console.log('  No model found — training on startup...');
  autoTrain();
}

app.use(express.json({ limit: '5mb' }));
app.use(express.static(__dirname));

app.post('/api/log-session', (req, res) => {
  const {
    name, wpm, accuracy, characters, errors, regressions, time,
    avgDwellMs, avgFlightMs, avgDDMs, avgUUMs,
    leftShiftCount, rightShiftCount, shiftPreference,
    backspaceCount, backspaceStyle, ctrlACount, capsLockCount,
    rolloverCount, avgRolloverDurationMs,
    errorRate, burstiness, speedDecayRate, wpmOverTime,
    digraphLatencies, trigraphLatencies
  } = req.body;

  const resolvedName = (typeof name === 'string' && name.trim()) ? name.trim() : 'Unknown';
  const timestamp    = new Date().toISOString();

  // ——— Skip CSV logging for Unknown sessions (keeps training data clean for OneClassSVM) ———
  if (resolvedName === 'Unknown') {
    console.log(`Skipped CSV for Unknown session — prediction-only, not training data.`);
    return res.json({ success: true, skippedCsv: true });
  }

  const safeName     = `"${resolvedName.replace(/"/g, '""')}"`;

  const wpmOverTimeStr = Array.isArray(wpmOverTime) ? `"${wpmOverTime.join(';')}"` : '""';
  const safeDigraphs   = `"${String(digraphLatencies  || '').replace(/"/g, '""')}"`;
  const safeTrigraphs  = `"${String(trigraphLatencies || '').replace(/"/g, '""')}"`;

  const csvRow = [
    timestamp, safeName,
    wpm ?? 0, accuracy ?? 0, characters ?? 0, errors ?? 0, regressions ?? 0, time ?? 0,
    avgDwellMs ?? 0, avgFlightMs ?? 0, avgDDMs ?? 0, avgUUMs ?? 0,
    leftShiftCount ?? 0, rightShiftCount ?? 0,
    `"${shiftPreference || 'none'}"`,
    backspaceCount ?? 0,
    `"${backspaceStyle || 'minimal'}"`,
    ctrlACount ?? 0, capsLockCount ?? 0, rolloverCount ?? 0, avgRolloverDurationMs ?? 0,
    errorRate ?? 0, burstiness ?? 0, speedDecayRate ?? 0,
    wpmOverTimeStr, safeDigraphs, safeTrigraphs
  ].join(',') + '\n';

  fs.appendFile(CSV_FILE, csvRow, 'utf-8', (csvErr) => {
    if (csvErr) {
      console.error('Failed to write CSV:', csvErr);
      return res.status(500).json({ error: 'Failed to save CSV row.' });
    }
    console.log(`CSV logged: ${safeName} — ${wpm} WPM`);

    res.json({ success: true });
  });
});

app.post('/api/predict', async (req, res) => {
  if (!fs.existsSync(MODEL_FILE)) {
    return res.json({ prediction: null, error: 'Model not trained yet.' });
  }
  try {
    const sessionJson = JSON.stringify(req.body);
    const { stdout } = await spawnPython([MLSVM_PATH, '--predict'], sessionJson);
    const result = JSON.parse(stdout);
    res.json(result);
  } catch (err) {
    console.error('Predict error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/train', async (req, res) => {
  try {
    const { stdout, stderr } = await spawnPython([MLSVM_PATH, '--train'], null, 60000);
    const result = JSON.parse(stdout);
    console.log(`Model retrained — type: ${result.model_type}, samples: ${result.samples}, accuracy: ${result.accuracy ?? 'N/A'}`);
    res.json({ ...result, log: stderr });
  } catch (err) {
    console.error('Train error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/log', (req, res) => {
  res.status(301).json({ error: 'Use /api/log-session instead.' });
});

app.listen(PORT, () => {
  console.log(`\n  ChronoType server running at http://localhost:${PORT}\n`);
});
