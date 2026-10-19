import assert from 'node:assert/strict';
import {MotionTracker} from '../motion_tracker.mjs';
function image(w,h,x=18,y=18){
  const data=new Uint8Array(w*h).fill(24);
  for(let py=0;py<14;py++)for(let px=0;px<14;px++)
    data[(y+py)*w+x+px]=60+((px*73+py*37+px*py*19)%190);
  return data;
}
const w=96,h=72;
const tracker=new MotionTracker({maxHistory:30,maxDetectionAgeMs:5000});
const anchor=tracker.addFrame(image(w,h,18,18),w,h,0);
const bbox=[18/w,18/h,32/w,32/h];
const detections=[{label:'Resistor',confidence:0.86,bbox}];
for(let f=1;f<=6;f++)tracker.addFrame(image(w,h,18+2*f,18+f),w,h,f*90);
assert.equal(tracker.ingestDetections(detections,anchor,700),true);
let current=tracker.getDetections()[0];
assert.equal(current.label,'Resistor');
assert.equal(current.confidence,0.86);
assert(Math.abs(current.bbox[0]-30/w)<.025,'replay x mismatch: '+current.bbox[0]);
assert(Math.abs(current.bbox[1]-24/h)<.025,'replay y mismatch: '+current.bbox[1]);
tracker.addFrame(image(w,h,32,25),w,h,720);
current=tracker.getDetections()[0];
assert(Math.abs(current.bbox[0]-32/w)<.025,'tracks newly arrived frame');
assert.equal(tracker.ingestDetections([],tracker.frames.at(-1).id,721),true);
assert.equal(tracker.getDetections().length,0,'zero YOLO detections clear tracks');
assert.equal(tracker.ingestDetections(detections,anchor,800),true);
tracker.addFrame(image(w,h,32,25),w,h,6000);
assert.equal(tracker.getDetections().length,0,'stale detections expire');
tracker.addFrame(image(100,72,20,20),100,72,6100);
assert.equal(tracker.ingestDetections(detections,anchor,6200),false,'changed geometry rejects stale');
const history=new MotionTracker({maxHistory:2});
const older=history.addFrame(image(w,h),w,h,0);
history.addFrame(image(w,h),w,h,1);
history.addFrame(image(w,h),w,h,2);
assert.equal(history.ingestDetections(detections,older,3),false,'expired history rejected');
console.log('Motion tracking: replay, movement, no invented classes, staleness, geometry passed');
