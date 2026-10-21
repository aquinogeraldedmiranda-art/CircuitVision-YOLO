import {detectImage,backendConfigured} from './api.js';
import {diagramSvg,demoResults} from './demo.js';
import {MotionTracker} from './motion_tracker.mjs';
import {decodeResistorBands,DIGITS,MULTIPLIERS,TOLERANCES,COLOR_HEX,GUIDE_CHOICES,getComponentDetails,formatOhms} from './component_details.mjs';

const $ = (sel,ctx=document)=>ctx.querySelector(sel);
const escapeHTML = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cfg = window.CIRCUITVISION_CONFIG || {};
const pages = new Set(['home','upload','camera','results','guide','about']);
const modelStatus = {online:false,customReady:false,round3Ready:false,samReady:false,unifiedReady:false,message:'Connecting to YOLOv11…'};
let cameraSession = 0;
let trackerAnimationFrame=0;
let trackerLastSample=0;
const liveTracker=new MotionTracker({maxHistory:90,maxTracks:18,maxDetectionAgeMs:3000});
const trackingCanvas=document.createElement('canvas');
const trackingContext=trackingCanvas.getContext('2d',{willReadFrequently:true});
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
const state={page:readHash(),menuOpen:false,file:null,fileURL:null,fileSource:null,settings:{confidence:35,model:'auto',showBoxes:true,showLabels:true,summary:true,zoom:1},
  result:{detections:[],isDemo:false},resultSource:'not-run',resultFile:null,processing:false,stream:null,cameras:[],cameraId:'',cameraError:'',capturedCount:0,requestError:'',compareResults:null,selectedComponent:'Resistor',referenceByLabel:{},referenceLoadingFor:'',referenceAttempted:{},referenceError:'',resistorBandCount:4,resistorBands4:['brown','black','red','gold'],resistorBands5:['brown','black','black','red','brown'],liveEnabled:false,liveBusy:false,liveTimer:null,liveDetections:[],liveFrameUrl:null,liveFrameSize:null,liveStatus:'',liveFrames:0,liveProcessingMs:0};
function readHash(){const slug=location.hash.replace(/^#\/?/,'').split('?')[0].toLowerCase();return pages.has(slug)?slug:'home';}
function getMode(){return backendConfigured()?'api':'demo';}
function detectionModeSwitch(active){return `<nav class="detection-mode-switch" aria-label="Detection mode">${link('camera',I('camera')+' <span>Live camera detector</span>',active==='camera'?'active':'')}${link('upload',I('upload')+' <span>Upload image</span>',active==='upload'?'active':'')}</nav>`;}
function modelStatusBadge(){return `<div class="model-status ${modelStatus.online&&modelStatus.customReady?'ready':modelStatus.online?'missing':'offline'}"><span class="status-dot"></span><span id="backend-health">${escapeHTML(modelStatus.message)}</span></div>`;}
async function refreshBackendStatus(){
  if(!backendConfigured()){modelStatus.online=false;modelStatus.customReady=false;modelStatus.round3Ready=false;modelStatus.unifiedReady=false;modelStatus.message='Demo mode · No detector connected';updateModelStatus();return modelStatus;}
  try{const response=await fetch(`${String(cfg.baseUrl||'/api').replace(/\/$/,'')}/health`,{cache:'no-store',signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw Error(`HTTP ${response.status}`);const data=await response.json();
    modelStatus.online=!!data.ok;modelStatus.customReady=!!data.custom_model_available;modelStatus.round3Ready=!!data.experimental_round3_available;modelStatus.samReady=!!data.sam_specialist_available;modelStatus.unifiedReady=!!data.yolov11_2_available;
    modelStatus.message=modelStatus.customReady?'YOLOv11 ready · Camera and image detection':'YOLOv11 detector unavailable';
  }catch(_){modelStatus.online=false;modelStatus.customReady=false;modelStatus.round3Ready=false;modelStatus.unifiedReady=false;modelStatus.message='Unable to connect · Please retry';}
  updateModelStatus();return modelStatus;
}
function updateModelStatus(){
  document.querySelectorAll('#backend-health').forEach(n=>n.textContent=modelStatus.message);
  document.querySelectorAll('.model-status').forEach(n=>n.className='model-status '+(modelStatus.online&&modelStatus.customReady?'ready':modelStatus.online?'missing':'offline'));
  // Health checks finish asynchronously. Enable the already-rendered selector
  // without resetting the camera preview or clearing tracking state.
  document.querySelectorAll('option[value="round3"]').forEach(option=>{
    option.disabled=!modelStatus.round3Ready;
    option.textContent='Additional model (developer only)'+(modelStatus.round3Ready?'':' — unavailable');
  });
  document.querySelectorAll('option[value="unified"]').forEach(option=>{
    option.disabled=!modelStatus.unifiedReady;
    option.textContent=modelStatus.unifiedReady?'YOLOv11 (2)':'YOLOv11 (2) — Not installed';
  });
    const compare=document.getElementById('compare-models-btn');
  if(compare)compare.disabled=!modelStatus.round3Ready||!state.file||state.processing;
}

function route(page){page=pages.has(page)?page:'home';location.hash=`/${page}`;}
let lastToastText='',lastToastAt=0;
function toast(message,error=false){
  const area=document.getElementById('toast-root');if(!area)return;
  const text=String(message||'Unknown error'),now=Date.now();
  if(text===lastToastText&&now-lastToastAt<4500)return;
  lastToastText=text;lastToastAt=now;
  if(error)area.querySelectorAll('.toast.error').forEach(t=>t.remove());
  const t=document.createElement('div');t.className='toast'+(error?' error':'');
  t.setAttribute('role','status');t.textContent=text;area.appendChild(t);
  setTimeout(()=>t.remove(),5200);
}
function navLinks(){return [['home','Overview','grid'],['upload','Detect image','upload'],['camera','Live camera','camera'],['results','Results','layers'],['guide','Component guide','info'],['about','About','info']];}
function sidebar(){return `<aside class="sidebar ${state.menuOpen?'open':''}" id="sidebar">${link('home',brand())}<div><div class="nav-header">WORKSPACE</div><nav aria-label="Workspace" class="app-nav">${navLinks().map(([p,l,ic])=>link(p,I(ic)+`<span>${l}</span>`,state.page===p?'active':'')).join('')}</nav></div><div class="sidebar-bottom"><div class="sidebar-promo"><div class="actions-inline accent">${I('sparkle')}<strong>YOLO-powered</strong></div><p>Electronic component detection with YOLOv11.</p></div><div class="app-version">CIRCUITVISION · YOLOv11</div></div></aside>`;}
function mobileHeader(){return `<div class="mobile-header">${link('home',brand())}${button(I('menu'),'toggle-menu','hamburger','aria-label="Open navigation menu"')}</div>`;}
function bottomNav(){const nav=[['home','Home','home'],['upload','Upload','upload'],['camera','Camera','camera'],['results','Results','layers']];return `<nav class="mobile-bottom" aria-label="Mobile navigation">${nav.map(([p,s,ic])=>link(p,I(ic)+`<span>${s}</span>`,state.page===p?'active':'')).join('')}</nav>`;}
function heading(eyebrow,title,sub){return `<div class="workspace-head"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${sub}</p></div><div class="account"><span class="workspace-version">YOLOv11</span></div></div>`;}
function appShell(eyebrow,title,sub,content){return `<div class="app-shell"><div class="scrim ${state.menuOpen?'show':''}" data-action="close-menu"></div>${sidebar()}<main class="main-app">${mobileHeader()}${heading(eyebrow,title,sub)}<div class="workspace-body">${content}</div></main>${bottomNav()}</div>`;}
function landing(){return `<div class="landing"><header class="site-header">${link('home',brand())}<nav aria-label="Main navigation">${link('home','Home','active')}${link('camera','Live camera')}${link('upload','Upload image')}${link('guide','Component guide')}${link('about','About')}</nav>${link('camera','Open live detector '+I('arrow'),'button primary')}${button(I('menu'),'mobile-links','hamburger landing-mobile-nav','aria-label="Open website navigation"')}</header>${state.menuOpen?`<nav class="site-mobile-drawer" aria-label="Website mobile menu">${link("home","Home")}${link("camera","Live camera")}${link("upload","Upload image")}${link("guide","Component guide")}${link("about","About")}</nav>`:""}<section class="hero"><div><span class="badge">CIRCUITVISION · YOLOv11 OBJECT DETECTION</span><h1>Identify components.<br><span class="accent">See the details.</span></h1><p>Explore electronic components through live camera detection or image analysis.</p><div class="hero-actions">${link('camera',I('camera')+' Start live camera detector','button primary')}${link('upload',I('upload')+' Upload an image','button ghost')}${link('guide',I('info')+' Component guide','button ghost')}</div><div class="hero-points"><span>${I('camera')} Live YOLO bounding boxes</span><span>${I('upload')} Upload photos too</span><span>${I('layers')} Confidence scores</span></div></div><div class="preview"><div class="preview-head"><i class="dot"></i><i class="dot yellow"></i><i class="dot green"></i><span style="margin-left:14px">LIVE DETECTION PREVIEW</span><span class="badge green">DESIGN PREVIEW</span></div><div class="preview-graphic">${diagramSvg()}</div><div class="preview-foot"><span>Illustrative detection overlay</span><b>03</b></div></div></section><section class="feature-grid"><div class="feature"><div class="tile">${I('sparkle',24)}</div><div><strong>Live camera detection</strong><p>Watch model bounding boxes appear as you show components to your webcam.</p></div></div><div class="feature"><div class="tile">${I('upload',24)}</div><div><strong>Image analysis</strong><p>Upload a photo to inspect predicted component labels and locations.</p></div></div><div class="feature"><div class="tile">${I('layers',24)}</div><div><strong>Clear results</strong><p>Review component predictions and export your results.</p></div></div></section></div>`;}
function uploadBox(){if(state.file){return `<div class="selected-image"><div class="thumbnail">${state.fileURL?`<img src="${escapeHTML(state.fileURL)}" alt="Selected electronics image"/>`:diagramSvg({labels:false,boxes:false,compact:true})}</div><div><span class="badge green">${I('check')} IMAGE READY</span><p class="file-name" style="margin-top:17px">${escapeHTML(state.file.name)}</p><div class="file-meta">${formatBytes(state.file.size)} · ${escapeHTML(state.file.type||'image')}</div><div class="actions-inline">${button('Replace image','browse-image','button smallbtn ghost')}${button(I('close')+' Remove','remove-image','button smallbtn ghost')}</div></div></div>`;}
return `<div id="dropzone" class="dropzone" tabindex="0" role="button" aria-label="Upload image, or drop a JPG PNG or WebP file"><div class="upload-icon">${I('upload')}</div><strong>Drag and drop your image here</strong><p>or select a file from your computer</p>${button('Browse image','browse-image','button primary')}<span class="tag">JPG · PNG · WEBP &nbsp; UP TO ${cfg.maxFileMB||10} MB</span></div>`;}
function detectorChoices(id='model'){
  return '<select id="'+id+'" class="select" aria-label="Choose YOLOv11 detector">'+
    '<option value="auto" '+(state.settings.model==='auto'?'selected':'')+'>YOLOv11 (1)</option>'+
    '<option value="unified" '+(state.settings.model==='unified'?'selected':'')+
      (modelStatus.unifiedReady?'':' disabled')+'>'+
      (modelStatus.unifiedReady?'YOLOv11 (2)':'YOLOv11 (2) — Not installed')+
    '</option></select>';
}
function settingPanel(){return `<section class="surface"><h3>Detection settings</h3><div class="settings"><div><div class="field-label">Detection model</div>${detectorChoices('model')}</div><div><label for="confidence" class="field-label"><span>Confidence threshold</span><b id="confidence-value">${state.settings.confidence}%</b></label><input id="confidence" class="range" type="range" min="1" max="99" step="1" value="${state.settings.confidence}" aria-valuetext="${state.settings.confidence} percent"></div><hr class="field-separator"><div><div class="field-label">Output options</div><div class="checks"><label class="check"><input type="checkbox" id="showBoxes" ${state.settings.showBoxes?'checked':''}> Show bounding boxes</label><label class="check"><input type="checkbox" id="showLabels" ${state.settings.showLabels?'checked':''}> Display confidence labels</label><label class="check"><input type="checkbox" id="summary" ${state.settings.summary?'checked':''}> Include detection summary</label></div></div><div class="info-line">${I('info')} ${getMode()==='demo'?'Demo mode: results are illustrative.': 'Lower confidence may detect more objects, but can also increase false positives.'}</div>${button(I('play')+' Run detection','run-detection','button wide '+(state.file?'primary':'ghost'),state.processing?'disabled':'')}${state.requestError?'<div class="info-line warning" role="alert">'+escapeHTML(state.requestError)+'</div>':''}</div></section>`;} 
function steps(){return `<div class="steps-title eyebrow">HOW IT WORKS</div><div class="steps"><div class="step"><span class="num">01</span><div><b>Select image</b><p>Choose an electronics photo</p></div></div><div class="step"><span class="num">02</span><div><b>Analyze</b><p>Run the connected YOLO model</p></div></div><div class="step"><span class="num">03</span><div><b>Review</b><p>Inspect detections and export</p></div></div></div>`;}
function upload(){return appShell('ELECTRONICS LAB','Image detection','Upload a photo to identify electronic components using YOLOv11.',`${detectionModeSwitch('upload')}${modelStatusBadge()}<div class="two-col"><section class="surface"><div class="surface-head"><h3>Upload image</h3><p>Supported formats: JPG, PNG, WEBP</p></div>${uploadBox()}${state.file?`<div class="file-note">Your selected image stays in the browser until you run an API request. ${getMode()==='demo'?'Demo analysis uses a separate sample PCB diagram.':''}</div>`:''}</section>${settingPanel()}</div>${steps()}<input class="sr-only" id="file-input" type="file" accept="image/jpeg,image/png,image/webp" aria-label="Choose electronics image"/>`);}
function imageResult(){const demo=state.result.isDemo;const source=state.resultSource;return demo?`<div class="result-stage">${diagramSvg({boxes:state.settings.showBoxes,labels:state.settings.showLabels})}</div>`:
`<div class="result-stage"><div class="annotated"><img src="${escapeHTML(state.result.annotatedImageUrl||state.resultFileURL||state.fileURL||'')}" alt="Analyzed electronics image">${!state.result.annotatedImageUrl&&state.settings.showBoxes?state.result.detections.filter(d=>d.bbox&&d.bbox.length===4).map(d=>{const [x1,y1,x2,y2]=d.bbox;return `<div class="overlay" style="left:${x1*100}%;top:${y1*100}%;width:${(x2-x1)*100}%;height:${(y2-y1)*100}%">${state.settings.showLabels?`<span>${escapeHTML(d.label)} ${d.confidence.toFixed(2)}</span>`:''}</div>`;}).join(''):''}</div></div>`;}
function comparisonPreview(model,description){
  const src=state.resultFileURL||state.fileURL||'';
  const boxes=model.detections.filter(d=>Array.isArray(d.bbox)&&d.bbox.length===4)
    .map(d=>{const [x1,y1,x2,y2]=d.bbox;
      return '<div class="overlay" style="left:'+(x1*100)+'%;top:'+(y1*100)+'%;width:'+((x2-x1)*100)+'%;height:'+((y2-y1)*100)+'%"><span>'+escapeHTML(d.label)+'</span></div>';}).join('');
  const labels=model.detections.slice(0,10).map(d=>'<div class="comparison-label"><span>'+escapeHTML(d.label)+'</span><b>'+Math.round(d.confidence*100)+'%</b></div>').join('');
  return '<section class="surface"><h3>'+description+'</h3><p class="small">'+model.detections.length+' detected above '+state.settings.confidence+'% confidence</p><div class="result-stage"><div class="annotated"><img src="'+escapeHTML(src)+'" alt="'+description+' detection results">'+boxes+'</div></div><div class="comparison-label-list">'+(labels||'<p class="small">No matching components detected</p>')+'</div></section>';
}
function comparisonPanel(){
  if(!state.compareResults)return '';
  return '<section style="margin-top:22px"><h2>Developer model comparison</h2><p class="small">Both models analyzed the SAME uploaded photo and threshold, sequentially. Inspect box positions, missing components and incorrect labels. Confidence scores and detection counts do not measure ground-truth accuracy.</p><div class="comparison-grid">'+comparisonPreview(state.compareResults.round2,'Reference model')+comparisonPreview(state.compareResults.round3,'Comparison model')+'</div></section>';
}
function results(){
  if(state.resultSource==='not-run'){
    return appShell('ELECTRONICS LAB','Detection results','Your real detections will appear here after you analyze an image.',
      `<div class="surface" style="min-height:390px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;gap:14px">${I('image',44)}<h2 style="margin:0">No analysis yet</h2><p class="small" style="max-width:480px;margin:0">Upload a photo or capture an image from your camera to see model predictions. CircuitVision will not invent results before your detector has run.</p><div class="actions-inline" style="justify-content:center;margin-top:14px">${link('camera',I('camera')+' Live camera','button primary')}${link('upload',I('upload')+' Upload image','button ghost')}</div></div>`);
  }
  const r=state.result;const sample=r.isDemo;const avg=r.detections.length?r.detections.reduce((a,d)=>a+d.confidence,0)/r.detections.length:0;const rows=r.detections.map(d=>`<div class="result-row"><div style="flex:1"><div class="labelgroup"><i class="mark" style="background:${escapeHTML(d.color||'#3dd8d0')}"></i>${escapeHTML(d.label)}</div><div class="confidence-bar"><i style="width:${Math.round(d.confidence*100)}%;background:${escapeHTML(d.color||'#3dd8d0')}"></i></div></div><b>${Math.round(d.confidence*100)}%</b>${d.reviewStatus==='experimental_extra_class'?'<span class="badge">Review prediction</span>':''}${d.reviewStatus==='specialist_visual_estimate'?'<span class="badge">Visual estimate</span>':''}${d.reviewStatus==='specialist_conflict'?'<span class="badge">Value unclear</span>':''}${d.reviewStatus==='specialist_candidate'?'<span class="badge">Review prediction</span>':''}<button type="button" class="detail-mini" data-action="component-detail" data-component="${escapeHTML(d.label)}">Details</button></div>`).join('');return appShell('ELECTRONICS LAB','Detection results',sample?'Illustrative sample results — not predictions from your uploaded image.':'Review your model predictions and prepare a report.',`<div class="result-grid"><section class="surface"><div class="result-subheading"><h3 style="margin:0">Analyzed image</h3><span class="badge ${sample?'green':''}">${sample?'SAMPLE DATA':'MODEL OUTPUT'}</span></div>${r.detections.length?imageResult():`<div class="result-stage"><div class="empty">${I('image')}<h3>No components detected</h3><p>Try a clearer image or adjust the confidence threshold.</p>${link('upload','Try another image','button primary')}</div></div>`}<div class="file-note">${sample?'electronics_sample.svg · Illustrative detection preview':escapeHTML(state.resultFile?.name||'Analyzed image')}</div></section><section class="surface"><h3>Detected objects</h3><p class="small">${sample?'Example only · not model output':'Returned from your YOLO API'}</p>${r.detections.length?rows:`<div class="empty"><p>No objects above the chosen confidence threshold.</p></div>`}<div class="actions-inline" style="margin-top:23px">${button(I('download')+' Export CSV','export-csv','button ghost wide')}${button(I('download')+' Export JSON','export-json','button ghost wide')}${button(I('image')+' Download image','export-image','button primary wide')}</div></section></div>${state.settings.summary?`<div class="stat-row"><div class="stat"><p>Objects identified</p><b>${String(r.detections.length).padStart(2,'0')}</b><br><span>${sample?'sample detections':'model predictions'}</span></div><div class="stat"><p>Avg. confidence</p><b style="color:var(--purple)">${(avg*100).toFixed(1)}%</b><br><span>${sample?'illustrative values':'across detected objects'}</span></div><div class="stat"><p>Processing mode</p><b style="font-size:1.5rem">${sample?'DEMO':'IMAGE'}</b><br><span>${sample?'static example':'backend inference'}</span></div></div>`:''}<div class="actions-inline" style="margin-top:22px">${link('upload',I('refresh')+' Analyze another image','button ghost')}${link('camera',I('camera')+' Use camera','button ghost')}</div>`);}
function liveObjectList(){if(!state.liveEnabled)return '<div class="live-list-empty">Detection is off. Start YOLO to see identified components.</div>';if(!state.liveDetections.length)return '<div class="live-list-empty">No components above confidence threshold yet.</div>';return state.liveDetections.slice(0,6).map(d=>`<div class="live-object-row"><b>${escapeHTML(d.label)}</b><span>${Math.round(d.confidence*100)}%</span><button type="button" class="detail-mini" data-action="component-detail" data-component="${escapeHTML(d.label)}">Details</button></div>`).join('');}
function camera(){const on=!!state.stream;return appShell('ELECTRONICS LAB','Live detection','Use your camera to view electronic component predictions with YOLOv11.',`${detectionModeSwitch('camera')}${modelStatusBadge()}<div class="cam-grid"><section class="surface"><div class="surface-head"><div class="actions-inline" style="justify-content:space-between"><h3 style="margin:0">Real-time detection view</h3><span class="badge ${on?'green':''}">${on?'● CAMERA ON':'● CAMERA OFF'}</span></div></div><div class="video-container">${on?'<video id="camera-video" autoplay playsinline muted aria-label="Live camera preview"></video><div id="live-overlay" class="live-overlay" aria-hidden="true"></div><div id="live-status" class="live-status">'+escapeHTML(state.liveStatus||'Camera on · press Start YOLO detection')+'</div>':`<div class="camera-idle">${I('camera')}<h3>Start your live electronics detector</h3><p>Point your camera at an electronic component, then start detection to see its predicted label.</p></div>`}</div><div class="camera-controls">${button(I('camera')+' Enable webcam','start-camera','button '+(!on?'primary':'ghost'),on?'disabled':'')}${button(I('image')+' Capture frame','capture-camera','button ghost',!on?'disabled':'')}${button(I('stop')+' Stop camera','stop-camera','button ghost',!on?'disabled':'')}${button(I('bolt')+(state.liveEnabled?' Stop YOLO':' Start YOLO detection'),'toggle-live','button '+(state.liveEnabled?'danger':'ghost'),!on?'disabled':'')}</div>${state.cameraError?`<div class="info-line warning" style="margin-top:17px">${I('alert')}${escapeHTML(state.cameraError)}</div>`:''}<p class="file-note">Live camera uses the faster core YOLO detector with browser motion tracking. For closer component details, use Capture frame and run image detection. Verify electrical values independently. Frames are not stored.</p></section><div style="display:grid;gap:20px"><section class="surface"><h3>YOLO settings</h3><label class="field-label" for="camera-source">Video source</label><select id="camera-source" class="select"><option value="">Default camera</option>${state.cameras.map(x=>`<option value="${escapeHTML(x.deviceId)}" ${x.deviceId===state.cameraId?'selected':''}>${escapeHTML(x.label||'Camera')}</option>`).join('')}</select><div style="margin-top:15px"><div class="field-label">Detection model</div>${detectorChoices('camera-model')}</div><hr class="field-separator"><label for="live-zoom" class="field-label">Camera focus / zoom</label><select id="live-zoom" class="select"><option value="1" ${state.settings.zoom===1?'selected':''}>1× · Full view (modules, breadboards)</option><option value="2" ${state.settings.zoom===2?'selected':''}>2× · Small components (LED, resistor, LDR)</option></select><p class="file-note">For small parts, keep the component near the center of the camera. The 2× mode crops the center of each frame to make small objects larger for YOLO.</p><hr class="field-separator"><label for="confidence" class="field-label">Confidence threshold <b id="confidence-value">${state.settings.confidence}%</b></label><input id="confidence" class="range" type="range" min="1" max="99" value="${state.settings.confidence}"><div class="checks" style="margin-top:15px"><label class="check"><input type="checkbox" id="showBoxes" ${state.settings.showBoxes?'checked':''}> Show bounding boxes</label><label class="check"><input type="checkbox" id="showLabels" ${state.settings.showLabels?'checked':''}> Show confidence labels</label></div><hr class="field-separator"><div class="field-label">Captured frames</div><strong style="font-size:1.7rem">${state.capturedCount}</strong></section><section class="surface"><h3>Session summary</h3><div class="side-stat">Status <strong>${on?'Camera active':'Not started'}</strong></div><div class="side-stat">Detector <strong>${modelStatus.customReady?'Model ready':modelStatus.online?'Model unavailable':'Checking status'}</strong></div><div class="side-stat">Objects detected <strong id="live-count">${state.liveDetections.length||'0'}</strong></div><div class="side-stat">Frames analyzed <strong id="live-frames">${state.liveFrames}</strong></div><div class="side-stat">Last response <strong id="live-speed">${state.liveProcessingMs?state.liveProcessingMs+' ms':'—'}</strong></div><div id="live-detection-list" class="live-object-list">${liveObjectList()}</div><div class="info-line" style="margin-top:20px">${I('info')} Browser motion tracking estimates box movement between YOLO results, then corrects using the next detection. Fast movements, blur and occlusion can still cause mistakes. Try 2× focus for tiny parts, strong light, and 25–35% confidence to find missed objects; lower thresholds may also create false detections.</div></section></div></div>`);}

async function loadComponentReference(label){
  // Component details are fetched from the same relational database that
  // enriches real detection results. The built-in guide stays as a fallback.
  const key=String(label||'').trim().toLowerCase();
  if(!backendConfigured()||!key||state.referenceAttempted[key])return;
  state.referenceAttempted[key]=true;
  state.referenceLoadingFor=key;
  state.referenceError='';
  try{
    const base=String(cfg.baseUrl||'/api').replace(/\/$/,'');
    const response=await fetch(base+'/components/'+encodeURIComponent(label),{cache:'no-store'});
    if(!response.ok)throw Error('Component database HTTP '+response.status);
    const info=await response.json();
    if(info?.reference_source!=='sqlite')throw Error('Unexpected component reference format');
    state.referenceByLabel[key]=info;
  }catch(err){
    state.referenceError='Database reference unavailable; showing the built-in guide.';
  }finally{
    if(state.referenceLoadingFor===key)state.referenceLoadingFor='';
    if(state.page==='guide'&&state.selectedComponent.toLowerCase()===key)render();
  }
}
function componentGuide(){
  const fallback=getComponentDetails(state.selectedComponent);
  const dbEntry=state.referenceByLabel[state.selectedComponent.toLowerCase()];
  const info=dbEntry?{...fallback,name:dbEntry.name,role:dbEntry.role,learn:dbEntry.learn,value:dbEntry.value}:fallback;
  const sourceLine=dbEntry
    ? '<div class="info-line">Component reference retrieved from the SQLite catalog. Confirm exact specifications with a datasheet.</div>'
    : state.referenceLoadingFor===state.selectedComponent.toLowerCase()
      ? '<div class="info-line">Loading component details from the database…</div>'
      : state.referenceError
        ? '<div class="info-line warning">'+escapeHTML(state.referenceError)+'</div>'
        : '<div class="info-line">Educational reference; exact part specifications require verification.</div>';
  const bands=state.resistorBandCount===4?state.resistorBands4:state.resistorBands5;
  let decoded=null,error='';
  try{decoded=decodeResistorBands(bands)}catch(ex){error=ex.message;}
  const choices=GUIDE_CHOICES.slice();
  if(state.selectedComponent&&!choices.includes(state.selectedComponent))choices.unshift(state.selectedComponent);
  const opt=choices.map(s=>'<option value="'+escapeHTML(s)+'" '+(s===state.selectedComponent?'selected':'')+'>'+escapeHTML(s)+'</option>').join('');
  const digitsCount=bands.length===4?2:3;
  const bandControls=bands.map((color,i)=>{
    const kind=i<digitsCount?'Digit '+(i+1):i===digitsCount?'Multiplier':'Tolerance';
    const pool=i<digitsCount?Object.keys(DIGITS).filter(c=>i!==0||c!=='black'):i===digitsCount?Object.keys(MULTIPLIERS):Object.keys(TOLERANCES);
    const options=pool.map(c=>'<option value="'+c+'" '+(c===color?'selected':'')+'>'+c[0].toUpperCase()+c.slice(1)+'</option>').join('');
    return '<label class="value-band-select"><span>'+kind+'</span><select class="select" id="resistor-band-'+i+'" aria-label="'+kind+'">'+options+'</select><i class="band-dot" style="background:'+COLOR_HEX[color]+'"></i></label>';
  }).join('');
  const colorBars=bands.map(c=>'<i style="background:'+COLOR_HEX[c]+'"></i>').join('');
  const showValue=decoded?'<div class="value-result" role="status"><span>Calculated resistance from selected colors</span><strong>'+decoded.formatted+'</strong><span>±'+decoded.tolerancePercent+'% tolerance · '+formatOhms(decoded.range[0])+' to '+formatOhms(decoded.range[1])+'</span></div>':
    '<div class="info-line warning">'+escapeHTML(error)+'</div>';
  const about='<section class="surface"><h3>Component reference</h3><label for="guide-component" class="field-label">Choose a component</label><select class="select" id="guide-component">'+opt+'</select>'+
    '<div class="component-summary"><span class="badge green">REFERENCE · NOT A MEASUREMENT</span><h2>'+escapeHTML(info.name)+'</h2>'+
    '<h3>Purpose</h3><p>'+escapeHTML(info.role)+'</p><h3>Important details</h3><p>'+escapeHTML(info.learn)+'</p><h3>How to determine the value</h3><p>'+escapeHTML(info.value)+'</p></div>'+
    sourceLine+'<div class="info-line">A YOLO bounding box identifies a component type. It does not measure resistance, read printed values, or guarantee a part number.</div></section>';
  const calculator='<section class="surface component-calculator"><h3>Resistor color-band calculator</h3>'+
    '<p class="small">Enter the band colors you see on a real resistor. This calculation is from <strong>your inputs</strong>, not an automatic AI measurement.</p>'+
    '<label class="field-label" for="resistor-band-count">Number of bands</label><select class="select" id="resistor-band-count">'+
    '<option value="4" '+(bands.length===4?'selected':'')+'>4-band resistor</option><option value="5" '+(bands.length===5?'selected':'')+'>5-band resistor</option></select>'+
    '<div class="value-bands">'+bandControls+'</div>'+
    '<div class="resistor-graphic" role="img" aria-label="Resistor colors: '+bands.join(', ')+'"><span class="resistor-lead"></span><span class="resistor-body">'+colorBars+'</span><span class="resistor-lead"></span></div>'+
    showValue+'<p class="file-note">Read from the end opposite the separated tolerance band, often gold or silver. Verify with a multimeter. 6-band and SMD codes are not supported here yet.</p></section>';
  return appShell('COMPONENT GUIDE','Electronics component details','Explore what components do and determine resistor values from their color bands.',
    '<div class="component-guide-grid">'+about+calculator+'</div>'+
    '<section class="surface" style="margin-top:18px"><p class="small">The resistor calculator uses colors you enter manually; it does not measure resistance.</p></section>');
}

function about(){return appShell('ABOUT','About CircuitVision','A clear view of electronic components.',
  '<section class="about-intro about-clean"><span class="badge">CIRCUITVISION · YOLOv11</span><h2>See components more clearly.</h2>'+
  '<p>CircuitVision uses YOLOv11 to identify electronic components in uploaded images and live camera views. Explore detected objects through bounding boxes, class labels and confidence scores.</p>'+
  '<div class="about-quicklinks"><a href="#/camera" class="button primary">Open live camera</a><a href="#/upload" class="button ghost">Analyze an image</a></div></section>'+
  '<div class="feature-grid about-feature-grid"><div class="feature"><div class="tile">'+I('camera',24)+'</div><div><strong>Live camera</strong><p>View component predictions as you move your camera.</p></div></div>'+
  '<div class="feature"><div class="tile">'+I('upload',24)+'</div><div><strong>Image detection</strong><p>Upload a photo to inspect individual detections.</p></div></div>'+
  '<div class="feature"><div class="tile">'+I('layers',24)+'</div><div><strong>Results</strong><p>Review labels, bounding boxes and confidence scores.</p></div></div></div>'+
  '<p class="about-disclaimer">Results are visual predictions. Verify component identity and electrical values before practical use.</p>');
}
function processing(){return `<div class="processing-wrap" role="status" aria-live="polite"><div class="processing-box"><span class="badge">${I('bolt')} ${backendConfigured()?'ANALYZING IMAGE':'RUNNING SAMPLE DEMO'}</span><h2 style="margin-top:20px">${backendConfigured()?'Detecting components...':'Preparing demo results...'}</h2><p>${backendConfigured()?'Sending the image to your YOLO detection API.':'Loading the sample electronics detection preview.'}</p><div class="progress"><span></span></div><div class="small">${backendConfigured()?'Waiting for a model response':'Illustrative demonstration • no model inference'}</div></div></div>`;}
function render(){document.title=`${state.page==='home'?'YOLO: CircuitVision | Electronics Laboratory Object Detection':state.page[0].toUpperCase()+state.page.slice(1)+' | YOLO: CircuitVision'}`;app.innerHTML=state.page==='home'?landing():state.page==='upload'?upload():state.page==='camera'?camera():state.page==='results'?results():state.page==='guide'?componentGuide():about();if(state.processing)app.insertAdjacentHTML('beforeend',processing());if(state.page==='guide')void loadComponentReference(state.selectedComponent);if(state.stream){const video=$('#camera-video');if(video){video.srcObject=state.stream;video.style.transform=state.settings.zoom===2?'scale(2)':'scale(1)';video.addEventListener('loadedmetadata',updateLiveOverlay,{once:true});updateLiveOverlay();}}const drag=$('#dropzone');if(drag){drag.addEventListener('dragover',e=>{e.preventDefault();drag.classList.add('drag-over')});drag.addEventListener('dragleave',()=>drag.classList.remove('drag-over'));drag.addEventListener('drop',e=>{e.preventDefault();drag.classList.remove('drag-over');if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0]);});drag.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#file-input')?.click();}})}}
async function handleFile(file){const accepted=['image/jpeg','image/png','image/webp'];if(!file||!accepted.includes(file.type)){toast('Choose a JPG, PNG or WEBP image.',true);return}if(file.size>(cfg.maxFileMB||10)*1024*1024){toast(`Image must be under ${cfg.maxFileMB||10} MB.`,true);return}if(state.fileURL)URL.revokeObjectURL(state.fileURL);state.file=file;state.fileURL=URL.createObjectURL(file);state.fileSource='upload';state.requestError='';state.compareResults=null;if(state.page!=='upload')route('upload');else render();toast('Image selected successfully.');}
function clearImage(){if(state.fileURL)URL.revokeObjectURL(state.fileURL);state.file=null;state.fileURL=null;state.fileSource=null;render();}
async function runDetection(){
  if(!state.file){toast('Select an image before running detection.',true);return;}
  if(state.processing)return;
  state.processing=true;state.requestError='';state.compareResults=null;render();
  try{
    if(backendConfigured()){
      const result=await detectImage(state.file,{...state.settings});
      state.result=result;
      for(const detection of result.detections||[]){
        if(detection.componentInfo?.reference_source==='sqlite')
          state.referenceByLabel[detection.label.toLowerCase()]=detection.componentInfo;
      }
      state.resultFile=state.file;
      if(state.resultFileURL)URL.revokeObjectURL(state.resultFileURL);
      state.resultFileURL=URL.createObjectURL(state.file);state.resultSource='api';
    }else{
      await new Promise(resolve=>setTimeout(resolve,700));
      state.result=demoResults();state.resultSource='sample';
      state.resultFile=null;state.resultFileURL=null;
    }
    state.processing=false;route('results');render();
    toast(backendConfigured()?'YOLO analysis completed.':'Sample results loaded (demo only).');
  }catch(e){
    state.processing=false;state.requestError=e.message||'Detection failed.';
    render();toast(state.requestError,true);
  }
}
async function compareModels(){
  if(!state.file){toast('Choose an image to compare.',true);return;}
  if(state.processing)return;
  if(!backendConfigured()||!modelStatus.round3Ready){
    toast('Comparison mode is unavailable.',true);return;
  }
  state.processing=true;state.requestError='';state.compareResults=null;render();
  try{
    // Sequential inferences avoid simultaneous ONNX session work on 512 MB Render.
    const common={...state.settings};
    const round2=await detectImage(state.file,{...common,model:'custom'});
    const round3=await detectImage(state.file,{...common,model:'round3'});
    state.compareResults={round2,round3};
    state.result=round2;state.resultSource='api';state.resultFile=state.file;
    if(state.resultFileURL)URL.revokeObjectURL(state.resultFileURL);
    state.resultFileURL=URL.createObjectURL(state.file);
    state.processing=false;route('results');render();
    toast('Comparison complete.');
  }catch(e){
    state.processing=false;state.requestError=e.message||'Comparison failed.';
    render();toast(state.requestError,true);
  }
}
async function startCamera(){if(!navigator.mediaDevices?.getUserMedia){state.cameraError='Camera access is unavailable. Serve the website via localhost or HTTPS.';render();return}if(state.stream)stopCamera(false);state.cameraError='';try{const stream=await navigator.mediaDevices.getUserMedia({video:state.cameraId?{deviceId:{ideal:state.cameraId}}:{facingMode:'environment'},audio:false});state.stream=stream;const devices=await navigator.mediaDevices.enumerateDevices();state.cameras=devices.filter(d=>d.kind==='videoinput').map((d,i)=>({deviceId:d.deviceId,label:d.label||`Camera ${i+1}`}));render();toast('Camera enabled.'); await refreshBackendStatus(); if(backendConfigured()&&modelStatus.customReady&&(state.settings.model!=='unified'||modelStatus.unifiedReady))await toggleLiveDetection();}catch(e){state.cameraError=e.name==='NotAllowedError'?'Camera permission denied. Allow permission in your browser and try again.':`Camera unavailable: ${e.message}`;render();}}
function stopCamera(reRender=true){stopLiveDetection();if(state.stream){state.stream.getTracks().forEach(t=>t.stop());state.stream=null}if(reRender){render();toast('Camera stopped.')}}
function stopLiveDetection(){
  state.liveEnabled=false;cameraSession+=1;clearTimeout(state.liveTimer);state.liveTimer=null;
  cancelAnimationFrame(trackerAnimationFrame);trackerAnimationFrame=0;trackerLastSample=0;liveTracker.reset();
  state.liveBusy=false;state.liveDetections=[];state.liveStatus='';
  if(state.liveFrameUrl){URL.revokeObjectURL(state.liveFrameUrl);state.liveFrameUrl=null;}
  state.liveFrameSize=null;
}

async function toggleLiveDetection(){
  if(state.liveEnabled){stopLiveDetection();render();toast('Live object detection stopped.');return;}
  if(!state.stream){toast('Enable the camera first.',true);return;}
  if(!backendConfigured()){toast('Start the included Python server for live YOLO detection.',true);return;}
  await refreshBackendStatus();
  if(!modelStatus.online){toast('YOLO service is unavailable. Refresh and try again.',true);return;}
  if(!modelStatus.customReady){toast('The YOLO11 detector is unavailable. Retry shortly.',true);return;}
  if(state.settings.model==='unified'&&!modelStatus.unifiedReady){toast('YOLOv11 (2) is not installed yet.',true);return;}
   
  state.liveEnabled=true;cameraSession+=1;state.liveFrames=0;state.liveProcessingMs=0;state.liveStatus='Waiting for first YOLO result…';render();
  liveTracker.reset();trackerLastSample=0;
  trackerAnimationFrame=requestAnimationFrame(t=>runMotionTracking(cameraSession,t));
  runLiveLoop(cameraSession);toast('Live YOLO started. Point your webcam at a trained electronics component.');
}
// Tracks grayscale patches locally; YOLO remains the only source of labels.
function captureTrackingFrame(source,at,zoom=1){
  if(!trackingContext)return null;
  const sw=source.videoWidth||source.width,sh=source.videoHeight||source.height;
  if(!sw||!sh)return null;
  const width=160,height=Math.max(1,Math.round(sh*width/sw));
  if(trackingCanvas.width!==width)trackingCanvas.width=width;
  if(trackingCanvas.height!==height)trackingCanvas.height=height;
  const srcW=sw/zoom,srcH=sh/zoom;
  trackingContext.drawImage(source,(sw-srcW)/2,(sh-srcH)/2,srcW,srcH,0,0,width,height);
  const pixels=trackingContext.getImageData(0,0,width,height).data;
  const gray=new Uint8Array(width*height);
  for(let i=0,j=0;i<pixels.length;i+=4,j++)
    gray[j]=(pixels[i]*77+pixels[i+1]*150+pixels[i+2]*29)>>8;
  return liveTracker.addFrame(gray,width,height,at);
}
function runMotionTracking(session,at){
  if(session!==cameraSession||!state.liveEnabled||!state.stream)return;
  if(at-trackerLastSample>=105){
    const video=$('#camera-video');
    if(video?.readyState>=2&&video.videoWidth){
      try{
        captureTrackingFrame(video,at,Number(state.settings.zoom)===2?2:1);
        state.liveDetections=liveTracker.getDetections();
        if(!state.liveDetections.length&&liveTracker.lastDetectionAt&&at-liveTracker.lastDetectionAt>liveTracker.maxDetectionAgeMs)
          state.liveStatus='Waiting for a fresh detection…';
        updateLiveOverlay();
      }catch(_){/* Camera stays live if tracking temporarily fails. */}
    }
    trackerLastSample=at;
  }
  trackerAnimationFrame=requestAnimationFrame(t=>runMotionTracking(session,t));
}
function updateLiveOverlay(){
  const overlay=$('#live-overlay');const video=$('#camera-video');
  if(!overlay||!video)return;
  const container=video.parentElement;
  if(!container)return;
  const W=container.clientWidth,H=container.clientHeight;
  // ONNX boxes are relative to the JPEG actually sent, NOT to a newer frame.
  // The phone's video stream can change dimensions on rotation or camera reset.
  const hasSyncedFrame=!!(state.liveEnabled&&state.liveFrameUrl&&state.liveFrameSize);
  overlay.classList.toggle('has-frame',false);
  const vw=hasSyncedFrame?state.liveFrameSize.width:(video.videoWidth||W);
  const vh=hasSyncedFrame?state.liveFrameSize.height:(video.videoHeight||H);
  const scale=Math.min(W/vw,H/vh),dx=(W-vw*scale)/2,dy=(H-vh*scale)/2;
  // Smooth mode: keep the browser's <video> visible at native camera FPS.
  // Predictions are the latest completed server result and may visibly trail
  // objects in motion, because inference happens on a different (older) frame.
  const frozenFrame='';
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
    // Anchor motion to the exact JPEG source that the server will analyze.
    const trackingAnchor=captureTrackingFrame(canvas,performance.now());
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.86));
    if(!blob)throw new Error('Unable to capture frame');
    const file=new File([blob],'live-frame.jpg',{type:'image/jpeg'});
    const requestStart=performance.now();
    const result=await detectImage(file,{...state.settings},{live:true});
    const responseTime=Math.round(performance.now()-requestStart);
    if(state.liveEnabled&&session===cameraSession&&zoom===(Number(state.settings.zoom)===2?2:1)){
      // Re-anchor to this YOLO prediction, replaying all camera motion since
      // inference capture. Ignore results invalidated by camera/zoom changes.
      const accepted=trackingAnchor!==null&&liveTracker.ingestDetections(result.detections,trackingAnchor,performance.now());
      state.liveFrames++;
      // Actual delay to the phone includes upload, hosting and model inference.
      // This is more meaningful for tracking lag than server-only processing_ms.
      state.liveProcessingMs=responseTime;
      if(accepted){
        state.liveFrameSize={width:canvas.width,height:canvas.height};
        state.liveDetections=liveTracker.getDetections();
        state.liveStatus=`● LIVE YOLO • ${state.liveDetections.length} detected • motion tracking`;
        updateLiveOverlay();
      }
    }
  }catch(e){if(state.liveEnabled&&session===cameraSession){state.liveStatus=`YOLO API: ${e.message}`;updateLiveOverlay();}}
  finally{if(session===cameraSession){state.liveBusy=false;if(state.liveEnabled)state.liveTimer=setTimeout(()=>runLiveLoop(session),Math.max(100,cfg.liveFrameIntervalMs||300));}}
}
async function captureCamera(){const video=$('#camera-video');if(!video||!state.stream){toast('Enable the camera before capturing.',true);return}if(!video.videoWidth){toast('Waiting for your camera to be ready. Try again.',true);return}const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.getContext('2d').drawImage(video,0,0);canvas.toBlob(blob=>{if(!blob){toast('Could not capture the frame.',true);return}const file=new File([blob],`circuitvision-capture-${Date.now()}.png`,{type:'image/png'});state.capturedCount++;stopCamera(false);handleFile(file);toast('Camera frame captured. Review it in Image Upload.');},'image/png');}
function downloadBlob(blob,name){const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),2500);}
function exportResults(format){const r=state.result;const filename=`circuitvision-${r.isDemo?'sample':'detection'}`;if(format==='json'){downloadBlob(new Blob([JSON.stringify({is_demo:r.isDemo,model:r.model||state.settings.model,confidence_threshold:state.settings.confidence/100,detections:r.detections},null,2)],{type:'application/json'}),filename+'.json');return}const lines=[['label','confidence','x1','y1','x2','y2'],...r.detections.map(d=>[d.label,d.confidence,...(d.bbox||['','','',''])])];const csv=lines.map(row=>row.map(cell=>'"'+String(cell??'').replace(/"/g,'""')+'"').join(',')).join('\r\n');downloadBlob(new Blob([csv],{type:'text/csv'}),filename+'.csv')}
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
  'browse-image':()=>$('#file-input')?.click(),'remove-image':clearImage,'run-detection':runDetection,'compare-models':compareModels,
  'start-camera':startCamera,'stop-camera':()=>stopCamera(),'capture-camera':captureCamera,'toggle-live':toggleLiveDetection,
  'component-detail':()=>{},'export-csv':()=>exportResults('csv'),'export-json':()=>exportResults('json'),'export-image':downloadImage,
};
document.addEventListener('click',e=>{const a=e.target.closest('[data-action]');if(a){const action=a.dataset.action;if(actions[action]){if(a.tagName==='A')e.preventDefault();if(action==='component-detail'){state.selectedComponent=a.dataset.component||'Resistor';route('guide');}else actions[action]();}return;}if(e.target.closest('[data-link]'))state.menuOpen=false;});
document.addEventListener('change',e=>{const t=e.target;if(t.id==='guide-component'){state.selectedComponent=t.value;render();return;}if(t.id==='resistor-band-count'){state.resistorBandCount=Number(t.value)===5?5:4;render();return;}if(/^resistor-band-[0-4]$/.test(t.id)){const bands=state.resistorBandCount===4?state.resistorBands4:state.resistorBands5;bands[Number(t.id.slice(-1))]=t.value;render();return;}if(t.id==='live-zoom'){
    state.settings.zoom=Number(t.value)===2?2:1;
    liveTracker.reset();
    const video=$('#camera-video');if(video)video.style.transform=state.settings.zoom===2?'scale(2)':'scale(1)';
    state.liveDetections=[];
    if(state.liveFrameUrl){URL.revokeObjectURL(state.liveFrameUrl);state.liveFrameUrl=null;}
    state.liveFrameSize=null;
    state.liveStatus='Zoom updated • analyzing next frame…';
    updateLiveOverlay();
  }else if(t.id==='file-input'){if(t.files?.[0])handleFile(t.files[0]);t.value='';}else if(t.id==='model'||t.id==='camera-model'){
    const next=t.value==='unified'?'unified':'auto';
    if(next==='unified'&&!modelStatus.unifiedReady){
      toast('YOLOv11 (2) is not installed yet. Please use YOLOv11 (1).',true);
      t.value=state.settings.model;return;
    }
    if(state.settings.model!==next){
      state.settings.model=next;
      if(state.liveEnabled)stopLiveDetection();
      render();
      toast('Selected '+(next==='unified'?'YOLOv11 (2)':'YOLOv11 (1)')+'.'+(state.stream?' Restart detection to apply.':''));
    }
  }else if(t.id==='confidence')state.settings.confidence=+t.value;else if(['showBoxes','showLabels','summary'].includes(t.id)){state.settings[t.id]=t.checked;if(state.page==='results')render();if(state.page==='camera')updateLiveOverlay();}else if(t.id==='camera-source'){state.cameraId=t.value;if(state.stream)startCamera();}});
document.addEventListener('input',e=>{if(e.target.id==='confidence'){state.settings.confidence=Number(e.target.value);$('#confidence-value').textContent=`${e.target.value}%`;}});
window.addEventListener('hashchange',()=>{const next=readHash();if(state.page==='camera'&&next!=='camera'&&state.stream)stopCamera(false);state.page=next;state.menuOpen=false;render();window.scrollTo({top:0,left:0,behavior:'instant'});});
window.addEventListener('beforeunload',()=>{if(state.liveFrameUrl)URL.revokeObjectURL(state.liveFrameUrl);if(state.stream)state.stream.getTracks().forEach(t=>t.stop());if(state.fileURL)URL.revokeObjectURL(state.fileURL);if(state.resultFileURL)URL.revokeObjectURL(state.resultFileURL);});
render();
refreshBackendStatus();
