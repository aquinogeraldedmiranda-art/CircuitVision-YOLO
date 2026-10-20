import assert from 'node:assert/strict';
import {MotionTracker} from '../motion_tracker.mjs';

function image(w,h,x=15,y=18,visible=true){
  const pixels=new Uint8Array(w*h).fill(18);
  if(visible)for(let j=0;j<14;j++)for(let i=0;i<14;i++)
    pixels[(y+j)*w+x+i]=50+((i*73+j*39+i*j*23)%196);
  return pixels;
}
const w=96,h=72,tracker=new MotionTracker({maxDetectionAgeMs:1600,maxMissingFrames:4});
const first=tracker.addFrame(image(w,h),w,h,1);
const det=[{label:'Resistor',confidence:.82,bbox:[15/w,18/h,29/w,32/h]}];
assert(tracker.ingestDetections(det,first,2));
for(let i=1;i<=4;i++)tracker.addFrame(image(w,h,15+i,18+i),w,h,i*95);
const moved=tracker.getDetections();
assert.equal(moved.length,1);
assert(Math.abs(moved[0].bbox[0]-19/w)<.03);
assert.equal(moved[0].confidence,.82,'tracker must not invent YOLO confidence');
assert(Number.isInteger(moved[0].trackId));

for(let i=5;i<=10;i++)tracker.addFrame(image(w,h,30,30,false),w,h,i*95);
assert.equal(tracker.getDetections().length,0,'missing component boxes must expire');
assert(tracker.ingestDetections([{label:'LED',confidence:.9,bbox:[-.1,0,.2,.3]}],tracker.frames.at(-1).id,960));
assert.equal(tracker.getDetections().length,0,'invalid model box must not be tracked');
tracker.addFrame(image(108,h),108,h,1000);
assert.equal(tracker.getDetections().length,0,'camera rotation resets stale boxes');
assert.equal(tracker.ingestDetections(det,first,1050),false,'old geometry must be rejected');
console.log('Occlusion, real confidence, movement, and changed camera dimensions passed');
