/**
 * CircuitVision camera tracker. Only real YOLO predictions provide labels and
 * confidence. This optical-flow estimate adjusts boxes between detections.
 * No prediction is synthesized for an object not seen by YOLO.
 */
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));

function patchShift(prev,next,bbox,radius=7){
  const {width:w,height:h,gray:a}=prev,b=next.gray;
  if(w!==next.width||h!==next.height)return null;
  const cx=Math.round((bbox[0]+bbox[2])*w/2),cy=Math.round((bbox[1]+bbox[3])*h/2);
  const half=Math.max(3,Math.min(5,Math.floor(Math.min((bbox[2]-bbox[0])*w,(bbox[3]-bbox[1])*h)*0.40)));
  if(cx-half-radius<0||cy-half-radius<0||cx+half+radius>=w||cy+half+radius>=h)return null;
  let sum=0,sum2=0,n=0;
  for(let py=-half;py<=half;py++)for(let px=-half;px<=half;px++){
    const val=a[(cy+py)*w+cx+px];sum+=val;sum2+=val*val;n++;
  }
  if(sum2/n-(sum/n)**2<85)return null; // A uniform patch has no reliable motion signal.
  let best=Infinity,bestDx=0,bestDy=0,bestError=Infinity;
  for(let dy=-radius;dy<=radius;dy++)for(let dx=-radius;dx<=radius;dx++){
    let difference=0;
    for(let py=-half;py<=half;py++){
      const row=(cy+py)*w,shifted=(cy+py+dy)*w;
      for(let px=-half;px<=half;px++)difference+=Math.abs(a[row+cx+px]-b[shifted+cx+px+dx]);
    }
    const avg=difference/n,score=avg+0.11*(Math.abs(dx)+Math.abs(dy));
    if(score<best){best=score;bestError=avg;bestDx=dx;bestDy=dy;}
  }
  if(bestError>38)return null;
  return {dx:bestDx,dy:bestDy,error:bestError};
}
function translate(box,dx,dy,w,h){
  const width=box[2]-box[0],height=box[3]-box[1];
  const x=clamp(box[0]+dx/w,0,Math.max(0,1-width));
  const y=clamp(box[1]+dy/h,0,Math.max(0,1-height));
  return [x,y,x+width,y+height];
}
// Temporal stabilization is only applied when two genuine YOLO results
// strongly agree. Fast movement and changing labels must use the newest result.
function iou(a,b){
  const ix=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]));
  const iy=Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
  const areaA=(a[2]-a[0])*(a[3]-a[1]);
  const areaB=(b[2]-b[0])*(b[3]-b[1]);
  return ix*iy/(areaA+areaB-ix*iy||1);
}
function keepStableIds(current,previous){
  const used=new Set();
  for(const next of current){
    let match=null,best=0;
    for(const old of previous){
      if(used.has(old.trackId)||old.label!==next.label)continue;
      const score=iou(next.bbox,old.bbox);
      if(score>best){best=score;match=old;}
    }
    if(match&&best>=0.72){
      used.add(match.trackId);
      next.trackId=match.trackId;
      // A small 15% blend dampens 1–2px localization jitter.
      // Restrict to high IoU so rapidly moving objects are never held back.
      if(best>=0.84){
        next.bbox=next.bbox.map((v,i)=>v*0.85+match.bbox[i]*0.15);
      }
    }
  }
  return current;
}
function validBox(d){
  const b=d?.bbox;
  return Array.isArray(b)&&b.length===4&&b.every(Number.isFinite)&&
    b[0]>=0&&b[1]>=0&&b[2]<=1&&b[3]<=1&&b[0]<b[2]&&b[1]<b[3];
}
export class MotionTracker {
  constructor({maxHistory=90,maxTracks=18,maxDetectionAgeMs=2400,
               maxMissingFrames=5,searchRadius=7}={}){
    this.maxHistory=maxHistory;this.maxTracks=maxTracks;
    this.maxDetectionAgeMs=maxDetectionAgeMs;
    this.maxMissingFrames=maxMissingFrames;this.searchRadius=searchRadius;
    this.reset();
  }
  reset(){
    this.frames=[];this.tracks=[];this.nextId=(this.nextId||0)+1;
    this.nextTrackId=(this.nextTrackId||0)+1;this.lastDetectionAt=0;
  }
  step(prev,next){
    for(const item of this.tracks){
      const motion=patchShift(prev,next,item.bbox,this.searchRadius);
      if(motion){
        item.bbox=translate(item.bbox,motion.dx,motion.dy,prev.width,prev.height);
        item._missing=0;
      }else item._missing=(item._missing||0)+1;
    }
    this.tracks=this.tracks.filter(x=>x._missing<=this.maxMissingFrames);
  }
  addFrame(gray,width,height,at=Date.now()){
    if(!(gray instanceof Uint8Array)||gray.length!==width*height)
      throw new Error('Tracking frame must be 8-bit grayscale with matching dimensions');
    let old=this.frames[this.frames.length-1];
    if(old&&(old.width!==width||old.height!==height)){
      this.reset();old=null;
    }
    const frame={id:this.nextId++,gray,width,height,at};
    if(old&&this.tracks.length)this.step(old,frame);
    this.frames.push(frame);
    if(this.frames.length>this.maxHistory)this.frames.shift();
    if(this.lastDetectionAt&&at-this.lastDetectionAt>this.maxDetectionAgeMs)
      this.tracks=[]; // do not leave ghost boxes from a lagging server
    return frame.id;
  }
  ingestDetections(detections,captureId,at=Date.now()){
    const index=this.frames.findIndex(x=>x.id===captureId);
    if(index<0||at-this.frames[index].at>this.maxDetectionAgeMs)return false;
    const previous=this.tracks;
    this.tracks=(Array.isArray(detections)?detections:[])
      .filter(validBox)
      .sort((a,b)=>b.confidence-a.confidence).slice(0,this.maxTracks)
      .map(d=>({...d,bbox:d.bbox.slice(),trackId:this.nextTrackId++,_missing:0}));
    // First move each new YOLO box from its captured frame to the latest
    // video frame. Comparing boxes before replay mixes two different times.
    for(let i=index+1;i<this.frames.length;i++)this.step(this.frames[i-1],this.frames[i]);
    this.tracks=keepStableIds(this.tracks,previous);
    this.lastDetectionAt=at;
    return true;
  }
  getDetections(){
    return this.tracks.map(({_missing,...d})=>({...d,bbox:d.bbox.slice()}));
  }
}
