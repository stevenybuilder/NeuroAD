// Thin wrappers over the FastAPI backend. All volume/overlay bytes are fetched
// by NiiVue directly from these URLs; JSON endpoints are fetched here.

export const volumeUrl = (scanId, modality) =>
  `/api/volume/${encodeURIComponent(scanId)}/${encodeURIComponent(modality)}`;

export const resourceUrl = (key) => `/api/resource/${encodeURIComponent(key)}`;

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

export const api = {
  scans: () => getJSON('/api/scans'),
  scan: (id) => getJSON(`/api/scans/${encodeURIComponent(id)}`),
  checks: () => getJSON('/api/checks'),
  run: (scanIds, checkIds) => postJSON('/api/run', { scan_ids: scanIds, check_ids: checkIds }),
  gallery: () => getJSON('/api/gallery'),
  adjudicate: (rec) => postJSON('/api/adjudications', rec),
};
