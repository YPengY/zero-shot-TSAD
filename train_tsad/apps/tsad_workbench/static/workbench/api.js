/**
 * Minimal HTTP helpers for the workbench frontend.
 *
 * These functions centralize JSON request/response conventions so controller
 * code can stay focused on UI orchestration.
 */

/** Send a JSON `POST` request and raise a user-facing error on failure. */
export async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
  }
  return payload;
}

/** Fetch JSON and raise a user-facing error on failure. */
export async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
  }
  return payload;
}

/** Build a query string while dropping empty filter values. */
export function buildQuery(path, params) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      query.set(key, String(value));
    }
  });
  return `${path}?${query.toString()}`;
}

/** Pretty-print JSON payloads for debug panes. */
export function safeJson(value) {
  return JSON.stringify(value, null, 2);
}
