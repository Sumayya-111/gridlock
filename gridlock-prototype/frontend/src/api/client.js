
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const WS_BASE = API_BASE.replace(/^http/, 'ws');

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function detectImage(file, settings = {}) {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams({
    conf_threshold: settings.confThreshold ?? 0.35,
    overlap_threshold: settings.overlapThreshold ?? 0.45,
    triple_threshold: settings.tripleThreshold ?? 0.5,
    run_alpr: true,
  });

  const res = await fetch(`${API_BASE}/detect/image?${params.toString()}`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || 'Detection failed');
  }

  return res.json();
}

export async function readPlate(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/alpr/read`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || 'ALPR failed');
  }

  return res.json();
}

export async function fetchViolationLog(limit = 50) {
  const res = await fetch(`${API_BASE}/violations/log?limit=${limit}`);

  if (!res.ok) {
    throw new Error('Failed to fetch violation log');
  }

  return res.json();
}

export async function fetchViolationSummary() {
  const res = await fetch(`${API_BASE}/violations/summary`);

  if (!res.ok) {
    throw new Error('Failed to fetch violation summary');
  }

  return res.json();
}

export async function clearViolationLog() {
  const res = await fetch(`${API_BASE}/violations/log`, {
    method: 'DELETE',
  });

  if (!res.ok) {
    throw new Error('Failed to clear log');
  }

  return res.json();
}

export async function fetchRoboflowDataset(body) {
  const res = await fetch(`${API_BASE}/dataset/fetch/roboflow`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || 'Dataset fetch failed');
  }

  return res.json();
}

export async function fetchKaggleDataset(slug) {
  const res = await fetch(
    `${API_BASE}/dataset/fetch/kaggle?dataset_slug=${encodeURIComponent(slug)}`,
    { method: 'POST' }
  );

  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || 'Dataset fetch failed');
  }

  return res.json();
}

export function streamVideoDetection(
  file,
  settings,
  onFrame,
  onDone,
  onError,
  calibration = {}
) {
  const params = new URLSearchParams({
    conf_threshold: settings.confThreshold ?? 0.35,
    overlap_threshold: settings.overlapThreshold ?? 0.45,
    triple_threshold: settings.tripleThreshold ?? 3,
  });

  // Add traffic-light calibration coordinates when available.
  if (calibration.signalROI?.length === 4) {
    params.set('signal_roi', calibration.signalROI.join(','));
  }

  if (calibration.stopLine?.length === 4) {
    params.set('stop_line', calibration.stopLine.join(','));
  }

  if (calibration.stopDir) {
    params.set('stop_dir', calibration.stopDir);
  }

  // Add wrong-way road zone and allowed direction.
  if (calibration.wrongWay?.length === 6) {
    params.set('wrong_way', calibration.wrongWay.join(','));
}

  const ws = new WebSocket(
    `${WS_BASE}/ws/detect/video?${params.toString()}`
  );

  let finished = false;
  let failed = false;
  let intentionalClose = false;

  const finish = () => {
    if (!finished) {
      finished = true;
      onDone?.();
    }
  };

  ws.onopen = () => {
    if (!intentionalClose) {
      ws.send(file);
    }
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      if (data.error) {
        failed = true;
        onError?.(new Error(data.error));
        ws.close();
        return;
      }

      if (data.done || data.status === 'completed') {
        finish();
        intentionalClose = true;
        ws.close();
        return;
      }

      onFrame?.(data);
    } catch (err) {
      failed = true;
      onError?.(err);
      ws.close();
    }
  };

  ws.onerror = (err) => {
    failed = true;
    onError?.(err);
  };

  ws.onclose = () => {
    if (!failed && !intentionalClose) {
      finish();
    }
  };

  return {
    close: () => {
      intentionalClose = true;

      if (
        ws.readyState === WebSocket.CONNECTING ||
        ws.readyState === WebSocket.OPEN
      ) {
        ws.close();
      }
    },
  };
}

export async function fetchChallans(limit = 100) {
  const response = await fetch(`${API_BASE}/challans?limit=${limit}`);

  if (!response.ok) {
    throw new Error('Failed to fetch challans');
  }

  return response.json();
}

export async function downloadChallanPdf(challanId) {
  const response = await fetch(
    `${API_BASE}/challans/${encodeURIComponent(challanId)}/pdf`
  );

  if (!response.ok) {
    throw new Error('Failed to download challan PDF');
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');

  link.href = url;
  link.download = `${challanId}.pdf`;

  document.body.appendChild(link);
  link.click();
  link.remove();

  URL.revokeObjectURL(url);
}
