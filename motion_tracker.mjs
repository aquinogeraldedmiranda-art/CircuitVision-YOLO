/**
 * Dependency-free browser motion tracker for CircuitVision.
 * YOLO supplies object identity and confidence; motion tracking only moves
 * those already-detected boxes between server responses.
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
export class MotionTracker{
  constructor({maxHistory=90,maxTracks=18,maxDetectionAgeMs=4500,searchRadius=7}={}){
    this.maxHistory=maxHistory;this.maxTracks=maxTracks;
    this.maxDetectionAgeMs=maxDetectionAgeMs;this.searchRadius=searchRadius;this.reset();
  }
  reset(){this.frames=[];this.tracks=[];this.nextId=(this.nextId||0)+1;this.lastDetectionAt=0;}
  step(prev,next){
    for(const item of this.tracks){
      const motion=patchShift(prev,next,item.bbox,this.searchRadius);
      if(motion)item.bbox=translate(item.bbox,motion.dx,motion.dy,prev.width,prev.height);
    }
  }
  addFrame(gray,width,height,at=Date.now()){
    if(!(gray instanceof Uint8Array)||gray.length!==width*height)
      throw new Error('Tracking frame must be 8-bit grayscale with matching dimensions');
    const old=this.frames[this.frames.length-1];
    if(old&&(old.width!==width||old.height!==height))this.reset();
    const frame={id:this.nextId++,gray,width,height,at};
    if(old&&this.tracks.length)this.step(old,frame);
    this.frames.push(frame);
    if(this.frames.length>this.maxHistory)this.frames.shift();
    if(this.lastDetectionAt&&at-this.lastDetectionAt>this.maxDetectionAgeMs)this.tracks=[];
    return frame.id;
  }
  ingestDetections(detections,captureId,at=Date.now()){
    const atIndex=this.frames.findIndex(x=>x.id===captureId);
    if(atIndex<0)return false;
    if(at-this.frames[atIndex].at>this.maxDetectionAgeMs)return false;
    this.lastDetectionAt=at;
    this.tracks=(Array.isArray(detections)?detections:[])
      .filter(d=>d.bbox?.length===4&&d.bbox.every(Number.isFinite))
      .sort((a,b)=>b.confidence-a.confidence).slice(0,this.maxTracks)
      .map(d=>({...d,bbox:d.bbox.slice()}));
    // Replay movement over actual captured camera frames while YOLO was busy.
    for(let i=atIndex+1;i<this.frames.length;i++)this.step(this.frames[i-1],this.frames[i]);
    return true;
  }
  getDetections(){return this.tracks.map(d=>({...d,bbox:d.bbox.slice()}));}
}
