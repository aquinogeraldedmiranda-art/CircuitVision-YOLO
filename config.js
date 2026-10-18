// Production and local frontend share the same-origin YOLO API.
window.CIRCUITVISION_CONFIG = {
  mode: 'api',
  baseUrl: '/api',
  requestTimeoutMs: 120000,
  liveFrameWidth: 416,
  liveFrameIntervalMs: 650,
  maxFileMB: 10
};
