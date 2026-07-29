/**
 * RouteX API Client
 * Plain fetch wrapper with real-time history and stats.
 */

const API_BASE = '/api';

// ── Retry helper ──────────────────────────────────────────────────────────────
async function retryFetch(url, options = {}, retries = 3, delay = 1500) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, options);
      return res;
    } catch (err) {
      if (attempt === retries) throw err;
      await new Promise((r) => setTimeout(r, delay * (attempt + 1)));
    }
  }
}

// ── Health, Models & Telemetry ───────────────────────────────────────────────

export async function getHealth() {
  const res = await retryFetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function getModels() {
  const res = await retryFetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) return [];
  return res.json();
}

export async function getStats() {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) return null;
  return res.json();
}

// ── Process Text ──────────────────────────────────────────────────────────────

export async function processText(text, options = {}) {
  const res = await fetch(`${API_BASE}/process/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      model: options.model || 'llama3-8b',
      context_limit: options.contextLimit || 4000,
      enable_routing: options.enableRouting ?? true,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Text processing failed');
  }

  return res.json();
}

// ── Process PDF ───────────────────────────────────────────────────────────────

export async function processPdf(file, options = {}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', options.model || 'llama3-8b');
  formData.append('context_limit', String(options.contextLimit || 4000));
  formData.append('enable_routing', String(options.enableRouting ?? true));

  const res = await fetch(`${API_BASE}/process/pdf`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'PDF processing failed');
  }

  return res.json();
}

// ── SSE Progress Stream ───────────────────────────────────────────────────────

export function streamProgress(taskId, onProgress) {
  const eventSource = new EventSource(`${API_BASE}/process/stream/${taskId}`);

  eventSource.addEventListener('progress', (event) => {
    try {
      const data = JSON.parse(event.data);
      onProgress(data);
      if (data.done) eventSource.close();
    } catch (e) {
      console.error('SSE parse error:', e);
    }
  });

  eventSource.addEventListener('error', (event) => {
    try {
      const data = JSON.parse(event.data);
      onProgress({ error: data.error, done: true });
    } catch {
      onProgress({ error: 'Connection lost', done: true });
    }
    eventSource.close();
  });

  eventSource.onerror = () => eventSource.close();

  return eventSource;
}

// ── Email ─────────────────────────────────────────────────────────────────────

export async function sendEmail(summary, routing, pdfPath) {
  const res = await fetch(`${API_BASE}/email`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      summary,
      routing,
      pdf_path: pdfPath,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Email sending failed');
  }

  return res.json();
}
