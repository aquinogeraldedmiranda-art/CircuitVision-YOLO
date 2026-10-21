const config = () => window.CIRCUITVISION_CONFIG || {};

/**
 * Network-safe CircuitVision detection transport.
 * YOLO request fields and file bytes remain unchanged; retries are reserved
 * for temporary fetch/Render gateway errors in image-upload mode.
 */
function wait(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
function offline() { return typeof navigator !== 'undefined' && navigator.onLine === false; }

// Phone galleries sometimes yield PNG/WEBP/JPEG files that Chrome displays
// but Pillow rejects because of partial or nonstandard encoding. Decode via
// the browser first, then send a standard RGB JPEG to the same inference API.
// The original File remains unchanged so output boxes align with its preview.
const normalizedUploads = new WeakMap();

// The API accepts JPEG/PNG/WEBP up to 16 million decoded pixels.
// An original photo that is inside those limits is a safe fallback when
// Android Chrome can display it but cannot allocate/encode a large canvas.
// Let Pillow decode that original rather than wrongly telling the user that
// the selected photo is unreadable.
function originalIsWithinServerLimits(file, width, height) {
  const maxBytes = (config().maxFileMB || 10) * 1024 * 1024;
  return Number(file?.size) > 0 && file.size <= maxBytes &&
    (!(width > 0 && height > 0) || width * height <= 16_000_000);
}

async function normalizeUploadForDetection(file) {
  if (typeof document === 'undefined' || typeof Blob === 'undefined' ||
      (typeof createImageBitmap !== 'function' &&
       (typeof Image === 'undefined' || typeof URL === 'undefined'))) {
    // Node tests and nonbrowser environments preserve direct-file requests.
    return {blob:file,filename:file.name || 'image.jpg'};
  }
  if (normalizedUploads.has(file)) return normalizedUploads.get(file);
  const work = (async () => {
    let bitmap=null, url=null, visual=null, width=0, height=0;
    try {
      if (typeof createImageBitmap === 'function') {
        try { bitmap=await createImageBitmap(file,{imageOrientation:'from-image'}); }
        catch (_) { bitmap=null; }
      }
      if (bitmap) visual=bitmap;
      else if (typeof Image !== 'undefined' && typeof URL !== 'undefined') {
        try {
          url=URL.createObjectURL(file);
          visual=await new Promise((resolve,reject)=>{
            const el=new Image();
            el.onload=()=>resolve(el);
            el.onerror=()=>reject(new Error('Photo decoder rejected image'));
            el.src=url;
          });
        } catch (_) { visual=null; }
      }
      width=Number(visual?.width||visual?.naturalWidth||0);
      height=Number(visual?.height||visual?.naturalHeight||0);
      if (!(width>0 && height>0)) {
        if (originalIsWithinServerLimits(file,0,0))
          return {blob:file,filename:file.name || 'photo.jpg'};
        throw new Error('Photo dimensions unavailable');
      }

      // Phones sometimes fail to create or export a 3000px canvas under
      // memory pressure. Try progressively smaller, still useful images.
      // This also avoids the browser-toBlob(null) failure on some Androids.
      const sizes=[3000,2048,1280,768];
      for (const maxEdge of sizes) {
        let canvas=null;
        try {
          const scale=Math.min(1,maxEdge/Math.max(width,height));
          const w=Math.max(1,Math.round(width*scale));
          const h=Math.max(1,Math.round(height*scale));
          canvas=document.createElement('canvas');
          canvas.width=w;canvas.height=h;
          const ctx=canvas.getContext('2d',{alpha:false});
          if (!ctx) continue;
          ctx.fillStyle='#ffffff';ctx.fillRect(0,0,w,h);
          ctx.drawImage(visual,0,0,w,h);
          const output=await new Promise((resolve,reject)=>{
            try { canvas.toBlob(resolve,'image/jpeg',0.92); }
            catch (err) { reject(err); }
          });
          if (output?.size)
            return {blob:output,filename:(file.name||'component-photo').replace(/\.[^./\\]+$/,'')+'-circuitvision.jpg'};
        } catch (_) {
          // A smaller canvas often succeeds without reselecting the photo.
        } finally {
          // Explicitly release GPU-backed canvas memory before retrying.
          if(canvas){canvas.width=0;canvas.height=0;}
        }
      }
      if (originalIsWithinServerLimits(file,width,height))
        return {blob:file,filename:file.name || 'photo.jpg'};
      throw new Error('The selected photo is too large for direct upload.');
    } catch (_) {
      // Avoid hiding a recoverable local canvas/decoder failure behind a
      // generic error: the backend can safely decode supported source files.
      if (originalIsWithinServerLimits(file,width,height))
        return {blob:file,filename:file.name || 'photo.jpg'};
      throw new Error('This photo is too large for your phone to resize. Try a smaller photo, or resave it as a JPG.');
    } finally {
      bitmap?.close?.();
      if(url)URL.revokeObjectURL(url);
    }
  })();
  normalizedUploads.set(file,work);
  try {return await work;}
  catch(err){normalizedUploads.delete(file);throw err;}
}

export async function detectImage(file, settings, options = {}) {
  if (config().mode !== 'api') throw new Error('API mode is disabled. The Python backend must be enabled.');
  if (!file) throw new Error('Select an image before running detection.');
  const isLive = options.live === true;
  const retries = isLive ? 0 : 1; // Avoid amplifying traffic from live camera.
  const api = String(config().baseUrl || '/api').replace(/\/$/, '');
  // Live frames are already browser-generated JPEGs, so skip extra encoding.
  const prepared = isLive ? {blob:file, filename:file.name || 'live-frame.jpg'}
                          : await normalizeUploadForDetection(file);
  const form = new FormData();
  form.append('image', prepared.blob, prepared.filename);
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
          bbox: Array.isArray(d.bbox) ? d.bbox.map(Number) : null,
          sourceModel: d.source_model || null,
          reviewStatus: d.review_status || null,
          componentInfo: d.component_info?.reference_source === 'sqlite' ? d.component_info : null
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
