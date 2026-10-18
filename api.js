const config = () => window.CIRCUITVISION_CONFIG || {};

/**
 * Backend contract:
 * POST {baseUrl}/detect, multipart/form-data:
 *   image: File, confidence: number 0..1, model: string,
 *   show_boxes: 'true'|'false', show_labels: 'true'|'false'
 * Returns JSON:
 * { detections: [{label: string, confidence: 0..1, bbox:[x1,y1,x2,y2]}],
 *   annotated_image_url?: string, processing_ms?: number }
 * bbox is normalized (0..1) relative to original image dimensions.
 * No backend calls happen in 'demo' mode.
 */
export async function detectImage(file, settings) {
  if (config().mode !== 'api') throw new Error('API mode is disabled. Change config.js when your backend is ready.');
  if (!file) throw new Error('Select an image first.');
  const form = new FormData();
  form.append('image', file, file.name || 'capture.png');
  form.append('confidence', String(settings.confidence / 100));
  form.append('model', settings.model);
  form.append('show_boxes', String(settings.showBoxes));
  form.append('show_labels', String(settings.showLabels));
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config().requestTimeoutMs || 90000);
  try {
    const response = await fetch(`${String(config().baseUrl || '/api').replace(/\/$/, '')}/detect`, {
      method: 'POST', body: form, signal: controller.signal,
    });
    if (!response.ok) {
      let reason = `Backend returned ${response.status}`;
      try { const json = await response.json(); reason = json.detail || json.error || reason; } catch (_) { /* ignore */ }
      throw new Error(reason);
    }
    const result = await response.json();
    if (!Array.isArray(result?.detections)) throw new Error('Invalid backend response: missing detections array.');
    return {
      detections: result.detections.map((d, index) => ({
        id: index + 1, label: String(d.label || d.class_name || 'Object'),
        confidence: Math.max(0, Math.min(1, Number(d.confidence) || 0)),
        bbox: Array.isArray(d.bbox) ? d.bbox.map(Number) : null
      })),
      annotatedImageUrl: result.annotated_image_url || null,
      processingMs: result.processing_ms ?? null,
      isDemo: false
    };
  } catch (err) {
    if (err?.name === 'AbortError') throw new Error('Request timed out. Check the YOLO backend.');
    throw err;
  } finally { clearTimeout(timeout); }
}
export function backendConfigured() { return config().mode === 'api'; }
