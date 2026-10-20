import assert from 'node:assert/strict';
import fs from 'node:fs';
const app=fs.readFileSync('app.js','utf8');
const api=fs.readFileSync('api.js','utf8');
const backend=fs.readFileSync('backend/main.py','utf8');
assert(app.includes('YOLOv11 (1)') && app.includes('YOLOv11 (2)'));
assert(app.includes("function detectorChoices(id='model')"));
assert(app.includes("detectorChoices('camera-model')"));
assert(app.includes("detectorChoices('model')"));
assert(app.includes("state.settings.model=next"));
assert(app.includes("option.disabled=!modelStatus.unifiedReady"));
assert(app.includes("if(state.settings.model==='unified'&&!modelStatus.unifiedReady)"));
assert(app.includes("model:'auto'"),'Model (1) must remain default');
assert(api.includes("form.append('model', settings.model || 'auto')"),
  'Selected model must be included in requests');
assert(backend.includes("if kind == 'unified':"));
assert(backend.includes("'yolov11_2_available': unified_candidate_installed"));
assert(backend.includes("'auto', 'custom', 'unified'"));
console.log('YOLOv11 (1)/(2) UI and server selection contract: PASS');
