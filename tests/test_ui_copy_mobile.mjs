import assert from 'node:assert/strict';
import fs from 'node:fs';

const app = fs.readFileSync('app.js','utf8');
const styles = fs.readFileSync('styles.css','utf8');
const readme = fs.readFileSync('README.md','utf8');

assert(app.includes('Round 2 YOLO11n runs live-camera detection'),
  'About page must identify current trained student checkpoint');
assert(app.includes('Sam\'s specialist model is not yet active in production'),
  'Do not claim unpublished Sam model is active');
assert(app.includes('not an electrical measurement'),
  'Do not present a resistor-label estimate as a measurement');
assert(!app.includes('initial live detector uses a public EESOB electronics model'),
  'Stale EESOB production claim');
assert(!app.includes('trained YOLO11s model after validation'),
  'Wrong production architecture');
assert(!app.includes('We are gathering licensed close-up images'),
  'Do not state unverified data collection');
assert(app.includes('modelStatus.customReady'),
  'Camera model status should reflect actual model readiness');
assert(app.includes("window.scrollTo({top:0,left:0"),
  'Mobile route must reset horizontal scroll');
assert(styles.includes('.about-steps .step .num{flex:0 0 42px'),
  'About step numbers should not stretch into full-width bars');
assert(styles.includes('.cam-grid>div{min-width:0;grid-template-columns:minmax(0,1fr)}'),
  'Mobile camera summary grid must be shrinkable');
assert(styles.includes('html,body,#app{max-width:100%;overflow-x:clip}'),
  'Mobile page should not scroll horizontally');
assert(readme.includes('Production:') || readme.includes('**Production:**'),
  'Project README should link to live site');
console.log('Verified factual About copy, model readiness and mobile overflow CSS safeguards');
