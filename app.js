import {detectImage,backendConfigured} from './api.js';
import {diagramSvg,demoResults} from './demo.js';

const $ = (sel,ctx=document)=>ctx.querySelector(sel);
const escapeHTML = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cfg = window.CIRCUITVISION_CONFIG || {};
const pages = new Set(['home','upload','camera','results','about']);
const modelStatus = {online:false,customReady:false,message:'Checking YOLO backend…'};
let cameraSession = 0;
const formatBytes = n => n < 1024*1024 ? `${Math.max(.01,n/1024).toFixed(1)} KB` : `${(n/(1024*1024)).toFixed(2)} MB`;
const iconPaths = {
  check:'<path d="m5 12 4 4L19 6"/>',upload:'<path d="M12 16V3m0 0L7 8m5-5 5 5M5 16v4h14v-4"/>',
  camera:'<path d="M14 4h-4l-2 3H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-4z"/><circle cx="12" cy="13" r="4"/>',
  layers:'<path d="m12 2 9 5-9 5-9-5 9-5zm-9 10 9 5 9-5M3 17l9 5 9-5"/>',
  home:'<path d="m3 10 9-7 9 7v11h-6v-7H9v7H3z"/>',info:'<circle cx="12" cy="12" r="10"/><path d="M12 11v6m0-10h.01"/>',
  grid:'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  arrow:'<path d="M5 19 19 5M9 5h10v10"/>',arrowright:'<path d="M4 12h16m-6-6 6 6-6 6"/>',
  menu:'<path d="M4 7h16M4 12h16M4 17h16"/>',close:'<path d="M6 6l12 12M18 6 6 18"/>',
  sparkle:'<path d="m12 2 2.3 7.7L22 12l-7.7 2.3L12 22l-2.3-7.7L2 12l7.7-2.3z"/>',
  chart:'<path d="M4 20h16M7 16V8m5 8V4m5 12v-5"/>',file:'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8m-8 4h8"/>',
  image:'<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 16-5-5-12 9"/>',
  download:'<path d="M12 3v12m-5-5 5 5 5-5M4 18v3h16v-3"/>',
  play:'<path d="m7 4 14 8-14 8z"/>', stop:'<rect x="5" y="5" width="14" height="14" rx="2"/>',
  refresh:'<path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/>',
  alert:'<path d="m12 3 10 18H2L12 3zM12 9v4m0 4h.01"/>',
  folder:'<path d="M3 6h7l2 3h9v11H3z"/>',
  shield:'<path d="m12 2 9 4v7c0 5-4 8-9 9-5-1-9-4-9-9V6z"/>',
  bolt:'<path d="m13 2-9 12h8l-1 8 9-12h-8z"/>',
  chevron:'<path d="m6 9 6 6 6-6"/>'
};
const I=(name,size)=>`<svg class="icon" ${size?`style="width:${size}px;height:${size}px"`:''} viewBox="0 0 24 24" aria-hidden="true">${iconPaths[name]||iconPaths.info}</svg>`;
const brand=()=>`<span class="brand"><span class="logo">${I('check')}</span><span>CIRCUIT <span class="word-accent">VISION</span></span></span>`;
const link=(href,label,cls='')=>`<a href="#/${href}" class="${cls}" data-link>${label}</a>`;
const button=(label,action,cls='button',attr='')=>`<button type="button" class="${cls}" data-action="${action}" ${attr}>${label}</button>`;
const app=document.getElementById('app');
const state={page:readHash(),menuOpen:false,file:null,fileURL:null,fileSource:null,settings:{confidence:35,model:'custom',showBoxes:true,showLabels:true,summary:true,zoom:1},
  result:{detections:[],isDemo:false},resultSource:'not-run',resultFile:null,processing:false,stream:null,cameras:[],cameraId:'',cameraError:'',capturedCount:0,liveEnabled:false,liveBusy:false,liveTimer:null,liveDetections:[],liveFrameUrl:null,liveStatus:'',liveFrames:0,liveProcessingMs:0};
function readHash(){const slug=location.hash.replace(/^#\/?/,'').split('?')[0].toLowerCase();return pages.has(slug)?slug:'home';}
function getMode(){return backendConfigured()?'api':'demo';}
function detectionModeSwitch(active){return `<nav class="detection-mode-switch" aria-label="Detection mode">${link('camera',I('camera')+' <span>Live camera detector</span>',active==='camera'?'active':'')}${link('upload',I('upload')+' <span>Upload image</span>',active==='upload'?'active':'')}</nav>`;}
function modelStatusBadge(){return `<div class="model-status ${modelStatus.online&&modelStatus.customReady?'ready':modelStatus.online?'missing':'offline'}"><span class="status-dot"></span><span id="backend-health">${escapeHTML(modelStatus.message)}</span></div>`;}
async function refreshBackendStatus(){
  if(!backendConfigured()){modelStatus.online=false;modelStatus.customReady=false;modelStatus.message='Preview mode • no YOLO backend';updateModelStatus();return modelStatus;}
  try{const response=await fetch(`${String(cfg.baseUrl||'/api').replace(/\/$/,'')}/health`,{cache:'no-store',signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw Error(`HTTP ${response.status}`);const data=await response.json();
    modelStatus.online=!!data.ok;modelStatus.customReady=!!data.custom_model_available;
    modelStatus.message=modelStatus.customReady?(data.model_file==='best.onnx'?'Student-trained YOLO11 ready • 57 lab classes':data.model_file==='eesob.onnx'?'YOLO detector ready • EESOB pretrained model':'YOLO detector ready • project model'):'Backend online • trained model not installed yet';
  }catch(_){modelStatus.online=false;modelStatus.customReady=false;modelStatus.message='Website service unavailable • refresh or try again later';}
  updateModelStatus();return modelStatus;
}
function updateModelStatus(){document.querySelectorAll('#backend-health').forEach(n=>n.textContent=modelStatus.message);document.querySelectorAll('.model-status').forEach(n=>n.className='model-status '+(modelStatus.online&&modelStatus.customReady?'ready':modelStatus.online?'missing':'offline'));}

function route(page){page=pages.has(page)?page:'home';location.hash=`/${page}`;}
function toast(message,error=false){const area=document.getElementById('toast-root');const t=document.createElement('div');t.className='toast'+(error?' error':'');t.setAttribute('role','status');t.textContent=message;area.appendChild(t);setTimeout(()=>t.remove(),4700);}
function navLinks(){return [['home','Overview','grid'],['upload','Detect image','upload'],['camera','Live camera','camera'],['results','Results','layers'],['about','About project','info']];}
function sidebar(){return `<aside class="sidebar ${state.menuOpen?'open':''}" id="sidebar">${link('home',brand())}<div><div class="nav-header">WORKSPACE</div><nav aria-label="Workspace" class="app-nav">${navLinks().map(([p,l,ic])=>link(p,I(ic)+`<span>${l}</span>`,state.page===p?'active':'')).join('')}</nav></div><div class="sidebar-bottom"><div class="sidebar-promo"><div class="actions-inline accent">${I('sparkle')}<strong>YOLO-powered</strong></div><p>Upload, camera and detections are ready to connect to your custom trained model.</p></div><div class="app-version">CIRCUITVISION / FRONTEND V1.0</div></div></aside>`;}
function mobileHeader(){return `<div class="mobile-header">${link('home',brand())}${button(I('menu'),'toggle-menu','hamburger','aria-label="Open navigation menu"')}</div>`;}
function bottomNav(){const nav=[['home','Home','home'],['upload','Upload','upload'],['camera','Camera','camera'],['results','Results','layers']];return `<nav class="mobile-bottom" aria-label="Mobile navigation">${nav.map(([p,s,ic])=>link(p,I(ic)+`<span>${s}</span>`,state.page===p?'active':'')).join('')}</nav>`;}
function heading(eyebrow,title,sub){return `<div class="workspace-head"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${sub}</p></div><div class="account"><span class="avatar">CV</span> Student</div></div>`;}
function appShell(eyebrow,title,sub,content){return `<div class="app-shell"><div class="scrim ${state.menuOpen?'show':''}" data-action="close-menu"></div>${sidebar()}<main class="main-app">${mobileHeader()}${heading(eyebrow,title,sub)}<div class="workspace-body">${content}</div></main>${bottomNav()}</div>`;}
function landing(){return `<div class="landing"><header class="site-header">${link('home',brand())}<nav aria-label="Main navigation">${link('home','Home','active')}${link('camera','Live camera')}${link('upload','Upload image')}${link('about','About')}</nav>${link('camera','Open live detector '+I('arrow'),'button primary')}${button(I('menu'),'mobile-links','hamburger landing-mobile-nav','aria-label="Open website navigation"')}</header>${state.menuOpen?`<nav class="site-mobile-drawer" aria-label="Website mobile menu">${link("home","Home")}${link("camera","Live camera")}${link("upload","Upload image")}${link("about","About")}</nav>`:""}<section class="hero"><div><span class="badge">✦ YOLO: CircuitVision · AI-ASSISTED ELECTRONICS LAB</span><h1>Understand circuits.<br>One detection<br><span class="accent">at a time.</span></h1><p>Explore object detection for electronics components through an intuitive, YOLO-powered workspace.</p><div class="hero-actions">${link('camera',I('camera')+' Start live camera detector','button primary')}${link('upload',I('upload')+' Upload an image','button ghost')}</div><div class="hero-points"><span>${I('camera')} Live YOLO bounding boxes</span><span>${I('upload')} Upload photos too</span><span>${I('layers')} Confidence scores</span></div></div><div class="preview"><div class="preview-head"><i class="dot"></i><i class="dot yellow"></i><i class="dot green"></i><span style="margin-left:14px">LIVE DETECTION PREVIEW</span><span class="badge green">DESIGN PREVIEW</span></div><div class="preview-graphic">${diagramSvg()}</div><div class="preview-foot"><span>Illustrative detection overlay</span><b>03</b></div></div></section><section class="feature-grid"><div class="feature"><div class="tile">${I('sparkle',24)}</div><div><strong>Live camera detection</strong><p>Watch model bounding boxes appear as you show components to your webcam.</p></div></div><div class="feature"><div class="tile">${I('upload',24)}</div><div><strong>Optional image uploads</strong><p>Analyze still photos using the same YOLO model as the live camera.</p></div></div><div class="feature"><div class="tile">${I('layers',24)}</div><div><strong>Organized insights</strong><p>Review detections and export reports.</p></div></div></section></div>`;}
function uploadBox(){if(state.file){return `<div class="selected-image"><div class="thumbnail">${state.fileURL?`<img src="${escapeHTML(state.fileURL)}" alt="Selected electronics image"/>`:diagramSvg({labels:false,boxes:false,compact:true})}</div><div><span class="badge green">${I('check')} IMAGE READY</span><p class="file-name" style="margin-top:17px">${escapeHTML(state.file.name)}</p><div class="file-meta">${formatBytes(state.file.size)} · ${escapeHTML(state.file.type||'image')}</div><div class="actions-inline">${button('Replace image','browse-image','button smallbtn ghost')}${button(I('close')+' Remove','remove-image','button smallbtn ghost')}</div></div></div>`;}
return `<div id="dropzone" class="dropzone" tabindex="0" role="button" aria-label="Upload image, or drop a JPG PNG or WebP file"><div class="upload-icon">${I('upload')}</div><strong>Drag and drop your image here</strong><p>or select a file from your computer</p>${button('Browse image','browse-image','button primary')}<span class="tag">JPG · PNG · WEBP &nbsp; UP TO ${cfg.maxFileMB||10} MB</span></div>`;}
function settingPanel(){return `<section class="surface"><h3>Detection settings</h3><div class="settings"><div><label for="model" class="field-label">Model</label><select id="model" class="select" name="model"><option value="custom" ${state.settings.model==='custom'?'selected':''}>YOLO custom model</option><option value="yolo11n" ${state.settings.model==='yolo11n'?'selected':''}>YOLO11n — generic objects (not lab components)</option></select></div><div><label for="confidence" class="field-label"><span>Confidence threshold</span><b id="confidence-value">${state.settings.confidence}%</b></label><input id="confidence" class="range" type="range" min="1" max="99" step="1" value="${state.settings.confidence}" aria-valuetext="${state.settings.confidence} percent"></div><hr class="field-separator"><div><div class="field-label">Output options</div><div class="checks"><label class="check"><input type="checkbox" id="showBoxes" ${state.settings.showBoxes?'checked':''}> Show bounding boxes</label><label class="check"><input type="checkbox" id="showLabels" ${state.settings.showLabels?'checked':''}> Display confidence labels</label><label class="check"><input type="checkbox" id="summary" ${state.settings.summary?'checked':''}> Include detection summary</label></div></div><div class="info-line">${I('info')} ${getMode()==='demo'?'Demo mode: results are illustrative.': 'Upload and webcam detection use the trained CircuitVision YOLO model. Lower confidence increases sensitivity but can produce more false detections.'}</div>${button(I('play')+' Run detection','run-detection','button wide '+(state.file?'primary':'ghost'),state.processing?'disabled':'')}</div></section>`;}
function steps(){return `<div class="steps-title eyebrow">HOW IT WORKS</div><div class="steps"><div class="step"><span class="num">01</span><div><b>Select image</b><p>Choose an electronics photo</p></div></div><div class="step"><span class="num">02</span><div><b>Analyze</b><p>Run the connected YOLO model</p></div></div><div class="step"><span class="num">03</span><div><b>Review</b><p>Inspect detections and export</p></div></div></div>`;}
function upload(){return appShell('ELECTRONICS LAB','Image detection','Upload a still photo and let the same YOLO model identify the electronics.',`${detectionModeSwitch('upload')}${modelStatusBadge()}<div class="two-col"><section class="surface"><div class="surface-head"><h3>Upload image</h3><p>Supported formats: JPG, PNG, WEBP</p></div>${uploadBox()}${state.file?`<div class="file-note">Your selected image stays in the browser until you run an API request. ${getMode()==='demo'?'Demo analysis uses a separate sample PCB diagram.':''}</div>`:''}</section>${settingPanel()}</div>${steps()}<input class="sr-only" id="file-input" type="file" accept="image/jpeg,image/png,image/webp" aria-label="Choose electronics image"/>`);}
function imageResult(){const demo=state.result.isDemo;const source=state.resultSource;return demo?`<div class="result-stage">${diagramSvg({boxes:state.settings.showBoxes,labels:state.settings.showLabels})}</div>`:
`<div class="result-stage"><div class="annotated"><img src="${escapeHTML(state.result.annotatedImageUrl||state.resultFileURL||state.fileURL||'')}" alt="Analyzed electronics image">${!state.result.annotatedImageUrl&&state.settings.showBoxes?state.result.detections.filter(d=>d.bbox&&d.bbox.length===4).map(d=>{const [x1,y1,x2,y2]=d.bbox;return `<div class="overlay" style="left:${x1*100}%;top:${y1*100}%;width:${(x2-x1)*100}%;height:${(y2-y1)*100}%">${state.settings.showLabels?`<span>${escapeHTML(d.label)} ${d.confidence.toFixed(2)}</span>`:''}</div>`;}).join(''):''}</div></div>`;}
function results(){
  if(state.resultSource==='not-run'){
    return appShell('ELECTRONICS LAB','Detection results','Your real detections will appear here after you analyze an image.',
      `<div class="surface" style="min-height:390px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;gap:14px">${I('image',44)}<h2 style="margin:0">No analysis yet</h2><p class="small" style="max-width:480px;margin:0">Upload a photo or capture an image from your camera to see model predictions. CircuitVision will not invent results before your detector has run.</p><div class="actions-inline" style="justify-content:center;margin-top:14px">${link('camera',I('camera')+' Live camera','button primary')}${link('upload',I('upload')+' Upload image','button ghost')}</div></div>`);
  }
  const r=state.result;const sample=r.isDemo;const avg=r.detections.length?r.detections.reduce((a,d)=>a+d.confidence,0)/r.detections.length:0;const rows=r.detections.map(d=>`<div class="result-row"><div style="flex:1"><div class="labelgroup"><i class="mark" style="background:${escapeHTML(d.color||'#3dd8d0')}"></i>${escapeHTML(d.label)}</div><div class="confidence-bar"><i style="width:${Math.round(d.confidence*100)}%;background:${escapeHTML(d.color||'#3dd8d0')}"></i></div></div><b>${Math.round(d.confidence*100)}%</b></div>`).join('');return appShell('ELECTRONICS LAB','Detection results',sample?'Illustrative sample results — not predictions from your uploaded image.':'Review your model predictions and prepare a report.',`<div class="result-grid"><section class="surface"><div class="result-subheading"><h3 style="margin:0">Analyzed image</h3><span class="badge ${sample?'green':''}">${sample?'SAMPLE DATA':'MODEL OUTPUT'}</span></div>${r.detections.length?imageResult():`<div class="result-stage"><div class="empty">${I('image')}<h3>No components detected</h3><p>Try a clearer image or adjust the confidence threshold.</p>${link('upload','Try another image','button primary')}</div></div>`}<div class="file-note">${sample?'electronics_sample.svg · Illustrative detection preview':escapeHTML(state.resultFile?.name||'Analyzed image')}</div></section><section class="surface"><h3>Detected objects</h3><p class="small">${sample?'Example only · not model output':'Returned from your YOLO API'}</p>${r.detections.length?rows:`<div class="empty"><p>No objects above the chosen confidence threshold.</p></div>`}<div class="actions-inline" style="margin-top:23px">${button(I('download')+' Export CSV','export-csv','button ghost wide')}${button(I('download')+' Export JSON','export-json','button ghost wide')}${button(I('image')+' Download image','export-image','button primary wide')}</div></section></div>${state.settings.summary?`<div class="stat-row"><div class="stat"><p>Objects identified</p><b>${String(r.detections.length).padStart(2,'0')}</b><br><span>${sample?'sample detections':'model predictions'}</span></div><div class="stat"><p>Avg. confidence</p><b style="color:var(--purple)">${(avg*100).toFixed(1)}%</b><br><span>${sample?'illustrative values':'across detected objects'}</span></div><div class="stat"><p>Processing mode</p><b style="font-size:1.5rem">${sample?'DEMO':'IMAGE'}</b><br><span>${sample?'static example':'backend inference'}</span></div></div>`:''}<div class="actions-inline" style="margin-top:22px">${link('upload',I('refresh')+' Analyze another image','button ghost')}${link('camera',I('camera')+' Use camera','button ghost')}</div>`);}
function liveObjectList(){if(!state.liveEnabled)return '<div class="live-list-empty">Detection is off. Start YOLO to see identified components.</div>';if(!state.liveDetections.length)return '<div class="live-list-empty">No components above confidence threshold yet.</div>';return state.liveDetections.slice(0,6).map(d=>`<div class="live-object-row"><b>${escapeHTML(d.label)}</b><span>${Math.round(d.confidence*100)}%</span></div>`).join('');}
function camera(){const on=!!state.stream;return appShell('ELECTRONICS LAB','Live YOLO detection','Hold an electronics module in front of the webcam to see its predicted class, box, and confidence.',`${detectionModeSwitch('camera')}${modelStatusBadge()}<div class="cam-grid"><section class="surface"><div class="surface-head"><div class="actions-inline" style="justify-content:space-between"><h3 style="margin:0">Real-time detection view</h3><span class="badge ${on?'green':''}">${on?'● CAMERA ON':'● CAMERA OFF'}</span></div></div><div class="video-container">${on?'<video id="camera-video" autoplay playsinline muted aria-label="Live camera preview"></video><div id="live-overlay" class="live-overlay" aria-hidden="true"></div><div id="live-status" class="live-status">'+escapeHTML(state.liveStatus||'Camera on · press Start YOLO detection')+'</div>':`<div class="camera-idle">${I('camera')}<h3>Start your live electronics detector</h3><p>Show a Power Module, resistor, capacitor or other trained component to your webcam. YOLO will label it automatically when your custom model is loaded.</p></div>`}</div><div class="camera-controls">${button(I('camera')+' Enable webcam','start-camera','button '+(!on?'primary':'ghost'),on?'disabled':'')}${button(I('image')+' Capture frame','capture-camera','button ghost',!on?'disabled':'')}${button(I('stop')+' Stop camera','stop-camera','button ghost',!on?'disabled':'')}${button(I('bolt')+(state.liveEnabled?' Stop YOLO':' Start YOLO detection'),'toggle-live','button '+(state.liveEnabled?'danger':'ghost'),!on?'disabled':'')}</div>${state.cameraError?`<div class="info-line warning" style="margin-top:17px">${I('alert')}${escapeHTML(state.cameraError)}</div>`:''}<p class="file-note">Browser webcam → YOLO API → colored boxes over your live video. Frames are processed in memory and are not stored by the app. HTTPS is required on mobile.</p></section><div style="display:grid;gap:20px"><section class="surface"><h3>YOLO settings</h3><label class="field-label" for="camera-source">Video source</label><select id="camera-source" class="select"><option value="">Default camera</option>${state.cameras.map(x=>`<option value="${escapeHTML(x.deviceId)}" ${x.deviceId===state.cameraId?'selected':''}>${escapeHTML(x.label||'Camera')}</option>`).join('')}</select><div style="margin-top:15px"><label for="camera-model" class="field-label">Detection model</label><select id="camera-model" class="select"><option value="custom" ${state.settings.model==='custom'?'selected':''}>Laboratory YOLO detector</option></select></div><hr class="field-separator"><label for="live-zoom" class="field-label">Camera focus / zoom</label><select id="live-zoom" class="select"><option value="1" ${state.settings.zoom===1?'selected':''}>1× · Full view (modules, breadboards)</option><option value="2" ${state.settings.zoom===2?'selected':''}>2× · Small components (LED, resistor, LDR)</option></select><p class="file-note">For small parts, keep the component near the center of the camera. The 2× mode crops the center of each frame to make small objects larger for YOLO.</p><hr class="field-separator"><label for="confidence" class="field-label">Confidence threshold <b id="confidence-value">${state.settings.confidence}%</b></label><input id="confidence" class="range" type="range" min="1" max="99" value="${state.settings.confidence}"><div class="checks" style="margin-top:15px"><label class="check"><input type="checkbox" id="showBoxes" ${state.settings.showBoxes?'checked':''}> Show bounding boxes</label><label class="check"><input type="checkbox" id="showLabels" ${state.settings.showLabels?'checked':''}> Show confidence labels</label></div><hr class="field-separator"><div class="field-label">Captured frames</div><strong style="font-size:1.7rem">${state.capturedCount}</strong></section><section class="surface"><h3>Session summary</h3><div class="side-stat">Status <strong>${on?'Camera active':'Not started'}</strong></div><div class="side-stat">YOLO inference <strong>${backendConfigured()?'API configured':'Backend pending'}</strong></div><div class="side-stat">Objects detected <strong id="live-count">${state.liveDetections.length||'0'}</strong></div><div class="side-stat">Frames analyzed <strong id="live-frames">${state.liveFrames}</strong></div><div class="side-stat">Last inference <strong id="live-speed">${state.liveProcessingMs?state.liveProcessingMs+' ms':'—'}</strong></div><div id="live-detection-list" class="live-object-list">${liveObjectList()}</div><div class="info-line" style="margin-top:20px">${I('info')} Frames and bounding boxes are synchronized to the same captured image. Free CPU hosting cannot process at full video frame rate. Try 2× focus for tiny parts, strong light, and 25–35% confidence to find missed objects; lower thresholds may also create false detections.</div></section></div></div>`);}
function about(){return appShell('PROJECT OVERVIEW','About the project','The idea behind an AI-assisted electronics learning experience.',`<section class="about-intro"><span class="badge">✦ AI-ASSISTED ELECTRONICS LAB</span><h2>YOLO: You Only Look Once</h2><h3 style="color:var(--purple)">An AI-Assisted Electronics Laboratory</h3><p>A student project exploring electronics component recognition with computer vision. CircuitVision supports live webcam detection and optional photo upload. Both modes use the same FastAPI/Ultralytics YOLO inference endpoint. A pretrained EESOB model, credited to Yacin Hamdi, provides initial electronics recognition. Our team-trained model can replace it later.</p></section><div class="steps-title eyebrow">PROJECT WORKFLOW</div><div class="steps about-steps"><div class="step"><span class="num">01</span><b>Prepare dataset</b><p>Collect and label images of resistors, capacitors, ICs and other electronics components.</p></div><div class="step"><span class="num">02</span><b>Train YOLO</b><p>Use annotated data and validate your trained detection model.</p></div><div class="step"><span class="num">03</span><b>Connect backend</b><p>Deploy the model and connect its detection API to the finished frontend.</p></div></div><div class="info-line" style="margin-top:25px">${I('shield')} Full-stack YOLO integration is included. The initial live detector uses a public EESOB electronics model. We will replace it with our team's own trained YOLO11s model after validation. See MODEL_CREDITS.md for attribution.</div>`);}
function processing(){return `<div class="processing-wrap" role="status" aria-live="polite"><div class="processing-box"><span class="badge">${I('bolt')} ${backendConfigured()?'ANALYZING IMAGE':'RUNNING SAMPLE DEMO'}</span><h2 style="margin-top:20px">${backendConfigured()?'Detecting components...':'Preparing demo results...'}</h2><p>${backendConfigured()?'Sending the image to your YOLO detection API.':'Loading the sample electronics detection preview.'}</p><div class="progress"><span></span></div><div class="small">${backendConfigured()?'Waiting for a model response':'Illustrative demonstration • no model inference'}</div></div></div>`;}
function render(){document.title=`${state.page==='home'?'YOLO: CircuitVision | Electronics Laboratory Object Detection':state.page[0].toUpperCase()+state.page.slice(1)+' | YOLO: CircuitVision'}`;app.innerHTML=state.page==='home'?landing():state.page==='upload'?upload():state.page==='camera'?camera():state.page==='results'?results():about();if(state.processing)app.insertAdjacentHTML('beforeend',processing());if(state.stream){const video=$('#camera-video');if(video){video.srcObject=state.stream;video.addEventListener('loadedmetadata',updateLiveOverlay,{once:true});updateLiveOverlay();}}const drag=$('#dropzone');if(drag){drag.addEventListener('dragover',e=>{e.preventDefault();drag.classList.add('drag-over')});drag.addEventListener('dragleave',()=>drag.classList.remove('drag-over'));drag.addEventListener('drop',e=>{e.preventDefault();drag.classList.remove('drag-over');if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0]);});drag.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#file-input')?.click();}})}}
async function handleFile(file){const accepted=['image/jpeg','image/png','image/webp'];if(!file||!accepted.includes(file.type)){toast('Choose a JPG, PNG or WEBP image.',true);return}if(file.size>(cfg.maxFileMB||10)*1024*1024){toast(`Image must be under ${cfg.maxFileMB||10} MB.`,true);return}if(state.fileURL)URL.revokeObjectURL(state.fileURL);state.file=file;state.fileURL=URL.createObjectURL(file);state.fileSource='upload';if(state.page!=='upload')route('upload');else render();toast('Image selected successfully.');}
function clearImage(){if(state.fileURL)URL.revokeObjectURL(state.fileURL);state.file=null;state.fileURL=null;state.fileSource=null;render();}
async function runDetection(){if(!state.file){toast('Select an image before running detection.',true);return}if(state.processing)return;state.processing=true;render();try{if(backendConfigured()){state.result=await detectImage(state.file,state.settings);state.resultFile=state.file;if(state.resultFileURL)URL.revokeObjectURL(state.resultFileURL);state.resultFileURL=URL.createObjectURL(state.file);state.resultSource='api';}else{await new Promise(resolve=>setTimeout(resolve,700));state.result=demoResults();state.resultSource='sample';state.resultFile=null;state.resultFileURL=null;}state.processing=false;route('results');render();toast(backendConfigured()?'YOLO analysis completed.':'Sample results loaded (demo only).');}catch(e){state.processing=false;render();toast(e.message||'Detection failed.',true);}}
async function startCamera(){if(!navigator.mediaDevices?.getUserMedia){state.cameraError='Camera access is unavailable. Serve the website via localhost or HTTPS.';render();return}if(state.stream)stopCamera(false);state.cameraError='';try{const stream=await navigator.mediaDevices.getUserMedia({video:state.cameraId?{deviceId:{ideal:state.cameraId}}:{facingMode:'environment'},audio:false});state.stream=stream;const devices=await navigator.mediaDevices.enumerateDevices();state.cameras=devices.filter(d=>d.kind==='videoinput').map((d,i)=>({deviceId:d.deviceId,label:d.label||`Camera ${i+1}`}));render();toast('Camera enabled.'); await refreshBackendStatus(); if(backendConfigured()&&(state.settings.model!=='custom'||modelStatus.customReady))await toggleLiveDetection();}catch(e){state.cameraError=e.name==='NotAllowedError'?'Camera permission denied. Allow permission in your browser and try again.':`Camera unavailable: ${e.message}`;render();}}
function stopCamera(reRender=true){stopLiveDetection();if(state.stream){state.stream.getTracks().forEach(t=>t.stop());state.stream=null}if(reRender){render();toast('Camera stopped.')}}
function stopLiveDetection(){
  state.liveEnabled=false;cameraSession+=1;clearTimeout(state.liveTimer);state.liveTimer=null;
  state.liveDetections=[];state.liveStatus='';
  if(state.liveFrameUrl){URL.revokeObjectURL(state.liveFrameUrl);state.liveFrameUrl=null;}
}

async function toggleLiveDetection(){
  if(state.liveEnabled){stopLiveDetection();render();toast('Live object detection stopped.');return;}
  if(!state.stream){toast('Enable the camera first.',true);return;}
  if(!backendConfigured()){toast('Start the included Python server for live YOLO detection.',true);return;}
  await refreshBackendStatus();
  if(!modelStatus.online){toast('YOLO service is unavailable. Refresh and try again.',true);return;}
  if(state.settings.model==='custom'&&!modelStatus.customReady){toast('The trained custom model has not been installed on this website yet.',true);state.liveStatus='Camera ready • trained model missing';updateLiveOverlay();return;}
  state.liveEnabled=true;cameraSession+=1;state.liveFrames=0;state.liveProcessingMs=0;state.liveStatus='Waiting for first YOLO result…';render();
  runLiveLoop(cameraSession);toast('Live YOLO started. Point your webcam at a trained electronics component.');
}
function updateLiveOverlay(){
  const overlay=$('#live-overlay');const video=$('#camera-video');
  if(!overlay||!video)return;
  const container=video.parentElement;
  if(!container)return;
  const W=container.clientWidth,H=container.clientHeight;
  const vw=video.videoWidth||W,vh=video.videoHeight||H;
  const scale=Math.min(W/vw,H/vh),dx=(W-vw*scale)/2,dy=(H-vh*scale)/2;
  // Display the *same captured frame* sent for inference. Drawing older boxes on
  // a continuously moving live <video> caused boxes to appear above objects.
  // The <video> keeps recording behind the matched inference frame.
  const frozenFrame=state.liveEnabled&&state.liveFrameUrl
    ? `<img class="live-inference-frame" src="${escapeHTML(state.liveFrameUrl)}" alt="" draggable="false">`
    : '';
  const boxes=state.settings.showBoxes?state.liveDetections.filter(d=>d.bbox?.length===4).map(d=>{
    const [x1,y1,x2,y2]=d.bbox;
    return `<div class="live-box" style="left:${(dx+x1*vw*scale)/W*100}%;top:${(dy+y1*vh*scale)/H*100}%;width:${(x2-x1)*vw*scale/W*100}%;height:${(y2-y1)*vh*scale/H*100}%">${state.settings.showLabels?`<span>${escapeHTML(d.label)} ${d.confidence.toFixed(2)}</span>`:''}</div>`;
  }).join(''):'';
  overlay.innerHTML=frozenFrame+boxes;
  const count=$('#live-count');if(count)count.textContent=String(state.liveDetections.length);
  const frames=$('#live-frames');if(frames)frames.textContent=String(state.liveFrames);
  const speed=$('#live-speed');if(speed)speed.textContent=state.liveProcessingMs?`${state.liveProcessingMs} ms`:'—';
  const list=$('#live-detection-list');if(list)list.innerHTML=liveObjectList();
  const badge=$('#live-status');if(badge)badge.textContent=state.liveStatus||'Camera on · press Start YOLO detection';
}
async function runLiveLoop(session){
  if(session!==cameraSession||!state.liveEnabled||!state.stream)return;
  if(state.liveBusy){state.liveTimer=setTimeout(()=>runLiveLoop(session),400);return;}
  const video=$('#camera-video');
  if(!video?.videoWidth){state.liveTimer=setTimeout(()=>runLiveLoop(session),250);return;}
  state.liveBusy=true;
  try{
    const canvas=document.createElement('canvas');const maxW=cfg.liveFrameWidth||640;
    canvas.width=Math.min(video.videoWidth,maxW);
    canvas.height=Math.max(1,Math.round(video.videoHeight*canvas.width/video.videoWidth));
    const ctx=canvas.getContext('2d',{alpha:false});
    const zoom=Number(state.settings.zoom)===2?2:1;
    const sourceW=video.videoWidth/zoom,sourceH=video.videoHeight/zoom;
    ctx.drawImage(video,(video.videoWidth-sourceW)/2,(video.videoHeight-sourceH)/2,sourceW,sourceH,0,0,canvas.width,canvas.height);
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.86));
    if(!blob)throw new Error('Unable to capture frame');
    const file=new File([blob],'live-frame.jpg',{type:'image/jpeg'});
    const result=await detectImage(file,state.settings);
    if(state.liveEnabled&&session===cameraSession){
      // Keep this exact captured JPEG displayed with the boxes inferred from it.
      const nextFrameURL=URL.createObjectURL(blob);
      const previousFrameURL=state.liveFrameUrl;
      state.liveFrameUrl=nextFrameURL;
      state.liveDetections=result.detections;
      state.liveFrames++;
      state.liveProcessingMs=Math.round(result.processingMs||0);
      state.liveStatus=`● LIVE YOLO • ${result.detections.length} detected • frame synced`;
      updateLiveOverlay();
      if(previousFrameURL)URL.revokeObjectURL(previousFrameURL);
    }
  }catch(e){if(state.liveEnabled&&session===cameraSession){state.liveStatus=`YOLO API: ${e.message}`;updateLiveOverlay();}}
  finally{state.liveBusy=false;if(state.liveEnabled&&session===cameraSession)state.liveTimer=setTimeout(()=>runLiveLoop(session),Math.max(100,cfg.liveFrameIntervalMs||300))}
}
async function captureCamera(){const video=$('#camera-video');if(!video||!state.stream){toast('Enable the camera before capturing.',true);return}if(!video.videoWidth){toast('Waiting for your camera to be ready. Try again.',true);return}const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.getContext('2d').drawImage(video,0,0);canvas.toBlob(blob=>{if(!blob){toast('Could not capture the frame.',true);return}const file=new File([blob],`circuitvision-capture-${Date.now()}.png`,{type:'image/png'});state.capturedCount++;stopCamera(false);handleFile(file);toast('Camera frame captured. Review it in Image Upload.');},'image/png');}
function downloadBlob(blob,name){const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),2500);}
function exportResults(format){const r=state.result;const filename=`circuitvision-${r.isDemo?'sample':'detection'}`;if(format==='json'){downloadBlob(new Blob([JSON.stringify({is_demo:r.isDemo,model:state.settings.model,confidence_threshold:state.settings.confidence/100,detections:r.detections},null,2)],{type:'application/json'}),filename+'.json');return}const lines=[['label','confidence','x1','y1','x2','y2'],...r.detections.map(d=>[d.label,d.confidence,...(d.bbox||['','','',''])])];const csv=lines.map(row=>row.map(cell=>'"'+String(cell??'').replace(/"/g,'""')+'"').join(',')).join('\r\n');downloadBlob(new Blob([csv],{type:'text/csv'}),filename+'.csv')}
async function downloadImage(){
  if(state.result.isDemo){
    const svg=diagramSvg({boxes:state.settings.showBoxes,labels:state.settings.showLabels});
    downloadBlob(new Blob([svg],{type:'image/svg+xml'}),'circuitvision-sample-annotated.svg');
    toast('Sample diagram exported as SVG.');return;
  }
  try{
    if(state.result.annotatedImageUrl){
      const response=await fetch(state.result.annotatedImageUrl);
      if(!response.ok)throw Error('Annotated image not available');
      downloadBlob(await response.blob(),'circuitvision-result.png');return;
    }
    if(!state.resultFile)throw Error('No analyzed image available');
    const bitmap=await createImageBitmap(state.resultFile);
    const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;
    const ctx=canvas.getContext('2d');ctx.drawImage(bitmap,0,0);bitmap.close();
    if(state.settings.showBoxes){
      const scale=Math.max(2,Math.round(canvas.width/450));
      ctx.lineWidth=scale*2;ctx.font=`bold ${13*scale}px sans-serif`;
      for(const d of state.result.detections){
        if(!d.bbox||d.bbox.length!==4)continue;
        const [x1,y1,x2,y2]=d.bbox;
        const x=x1*canvas.width,y=y1*canvas.height,w=(x2-x1)*canvas.width,h=(y2-y1)*canvas.height;
        ctx.strokeStyle='#37DBD5';ctx.strokeRect(x,y,w,h);
        if(state.settings.showLabels){
          const label=`${d.label} ${Math.round(d.confidence*100)}%`;
          const labelW=ctx.measureText(label).width+15*scale;
          const labelH=20*scale;
          ctx.fillStyle='#37DBD5';ctx.fillRect(x,Math.max(0,y-labelH),labelW,labelH);
          ctx.fillStyle='#081827';ctx.fillText(label,x+7*scale,Math.max(15*scale,y-5*scale));
        }
      }
    }
    canvas.toBlob(blob=>{if(blob)downloadBlob(blob,'circuitvision-result.png');else toast('Image export failed.',true)},'image/png');
  }catch(e){toast(e.message||'Could not export the image.',true)}
}
const actions={
  'toggle-menu':()=>{state.menuOpen=true;render()},'mobile-links':()=>{state.menuOpen=!state.menuOpen;render()}, 'close-menu':()=>{state.menuOpen=false;render()},
  'browse-image':()=>$('#file-input')?.click(),'remove-image':clearImage,'run-detection':runDetection,
  'start-camera':startCamera,'stop-camera':()=>stopCamera(),'capture-camera':captureCamera,'toggle-live':toggleLiveDetection,
  'export-csv':()=>exportResults('csv'),'export-json':()=>exportResults('json'),'export-image':downloadImage,
};
document.addEventListener('click',e=>{const a=e.target.closest('[data-action]');if(a){const action=a.dataset.action;if(actions[action]){if(a.tagName==='A')e.preventDefault();actions[action]();}return;}if(e.target.closest('[data-link]'))state.menuOpen=false;});
document.addEventListener('change',e=>{const t=e.target;if(t.id==='live-zoom'){
    state.settings.zoom=Number(t.value)===2?2:1;
    state.liveDetections=[];
    if(state.liveFrameUrl){URL.revokeObjectURL(state.liveFrameUrl);state.liveFrameUrl=null;}
    state.liveStatus='Zoom updated • analyzing next frame…';
    updateLiveOverlay();
  }else if(t.id==='file-input'){if(t.files?.[0])handleFile(t.files[0]);t.value='';}else if(t.id==='model'||t.id==='camera-model'){state.settings.model=t.value;if(state.liveEnabled){stopLiveDetection();render();toast('Model changed. Restart YOLO detection to use it.');}}else if(t.id==='confidence')state.settings.confidence=+t.value;else if(['showBoxes','showLabels','summary'].includes(t.id)){state.settings[t.id]=t.checked;if(state.page==='results')render();if(state.page==='camera')updateLiveOverlay();}else if(t.id==='camera-source'){state.cameraId=t.value;if(state.stream)startCamera();}});
document.addEventListener('input',e=>{if(e.target.id==='confidence'){state.settings.confidence=Number(e.target.value);$('#confidence-value').textContent=`${e.target.value}%`;}});
window.addEventListener('hashchange',()=>{const next=readHash();if(state.page==='camera'&&next!=='camera'&&state.stream)stopCamera(false);state.page=next;state.menuOpen=false;render();window.scrollTo({top:0,behavior:'instant'});});
window.addEventListener('beforeunload',()=>{if(state.liveFrameUrl)URL.revokeObjectURL(state.liveFrameUrl);if(state.stream)state.stream.getTracks().forEach(t=>t.stop());if(state.fileURL)URL.revokeObjectURL(state.fileURL);if(state.resultFileURL)URL.revokeObjectURL(state.resultFileURL);});
render();
refreshBackendStatus();
