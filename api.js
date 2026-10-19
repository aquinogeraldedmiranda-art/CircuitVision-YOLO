const config = () => window.CIRCUITVISION_CONFIG || {};

/**
 * Network-safe CircuitVision detection transport.
 * YOLO request fields and file bytes remain unchanged; retries are reserved
 * for temporary fetch/Render gateway errors in image-upload mode.
 */
function wait(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
function offline() { return typeof navigator !== 'undefined' && navigator.onLine === false; }

export async function detectImage(file, settings, options = {}) {
  if (config().mode !== 'api') throw new Error('API mode is disabled. The Python backend must be enabled.');
  if (!file) throw new Error('Select an image before running detection.');
  const isLive = options.live === true;
  const retries = isLive ? 0 : 1; // Avoid amplifying traffic from live camera.
  const api = String(config().baseUrl || '/api').replace(/\/$/, '');
  const form = new FormData();
  form.append('image', file, file.name || 'capture.png');
  form.append('confidence', String(settings.confidence / 100));
  form.append('model', settings.model || 'auto');
  form.append('live', String(isLive));
  form.append('show_boxes', String(settings.showBoxes));
  form.append('show_labels', String(settings.showLabels));
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (offline()) throw new Error('Your phone is offline. Reconnect to Wi-Fi or mobile data, then retry.');
    const controller = new AbortController();
    const timeoutMs = isLive ? 30000 : (config().requestTimeoutMs || 120000);
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(api + '/detect', {
        method: 'POST', body: form, signal: controller.signal, cache: 'no-store'
      });
      if (!response.ok) {
        let message = 'Detection server returned HTTP ' + response.status;
        try {
          const data = await response.json();
          if (typeof data.detail === 'string') message = data.detail;
          else if (typeof data.error === 'string') message = data.error;
        } catch (_) { /* Gateway may return HTML, not JSON. */ }
        if (attempt < retries && [502, 504, 429].includes(response.status)) {
          await wait(900 * (attempt + 1));
          continue;
        }
        if ([502, 503, 504].includes(response.status) && message.startsWith('Detection server returned')) {
          message = 'CircuitVision server is temporarily unavailable (HTTP ' + response.status + '). Please retry shortly.';
        }
        throw new Error(message);
      }
      let result;
      try { result = await response.json(); }
      catch (_) { throw new Error('The detection server returned an incomplete response. Please retry.'); }
      if (!Array.isArray(result?.detections)) {
        throw new Error('Detection server returned an invalid result. Please retry.');
      }
      return {
        detections: result.detections.map((d, index) => ({
          id: index + 1, label: String(d.label || d.class_name || 'Object'),
          confidence: Math.max(0, Math.min(1, Number(d.confidence) || 0)),
          bbox: Array.isArray(d.bbox) ? d.bbox.map(Number) : null
        })),
        annotatedImageUrl: result.annotated_image_url || null,
        processingMs: result.processing_ms ?? null,
        model: result.model || settings.model,
        isDemo: false
      };
    } catch (err) {
      if (err?.name === 'AbortError') {
        throw new Error('Detection took too long. The server may be busy; wait a moment and try again.');
      }
      // A browser fetch TypeError means the request never produced a readable
      // response (offline, DNS, Render connection reset, or client network).
      const networkFailure = err instanceof TypeError || /failed to fetch|networkerror|network request failed/i.test(String(err?.message || ''));
      if (networkFailure) {
        if (offline()) throw new Error('Your phone is offline. Reconnect and retry detection.');
        if (attempt < retries) {
          await wait(900 * (attempt + 1));
          continue;
        }
        throw new Error('Connection to CircuitVision was interrupted. The image was not analyzed. Check your connection and retry; the YOLO11 detector remains available.');
      }
      throw err;
    } finally {
      clearTimeout(timeout);
    }
  }
  throw new Error('Detection could not be completed. Please try again.');
}

export function backendConfigured() { return config().mode === 'api'; }
