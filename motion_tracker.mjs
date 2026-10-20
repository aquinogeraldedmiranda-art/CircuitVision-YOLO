/**
 * CircuitVision camera tracker. Only real YOLO predictions provide labels and
 * confidence. This optical-flow estimate adjusts boxes between detections.
 * No prediction is synthesized for an object not seen by YOLO.
 */
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));

function patchShift(prev,next,bbox,radius=7) {
  const {width:w,height:h,gray:a}=prev, b=next.gray;
  if(w!==next.width||h!==next.height) return null;
  const cx=(bbox[0]+bbox[2])*w/2,cy=(bbox[1]+bbox[3])*h/2;
  const bw=(bbox[2]-bbox[0])*w,bh=(bbox[3]-bbox[1])*h;
  // A center point alone often lies on a uniform resistor body or background.
  // Sample spatially separate points and choose a consensus movement.
  const offsets = bw<15||bh<15 ? [[0,0]] :
    [[0,0],[-.22,-.19],[.22,.19]];
  const half=Math.max(3,Math.min(5,Math.floor(Math.min(bw,bh)*.33)));
  const estimates=[];
  for(const [ox,oy] of offsets) {
    const px=Math.round(cx+ox*bw),py=Math.round(cy+oy*bh);
    if(px-half-radius<0||py-half-radius<0||
       px+half+radius>=w||py+half+radius>=h) continue;
    const n=(half*2+1)**2;
    let sum=0,sum2=0;
    for(let yy=-half;yy<=half;yy++) for(let xx=-half;xx<=half;xx++){
      const q=a[(py+yy)*w+px+xx];sum+=q;sum2+=q*q;
    }
    if(sum2/n-(sum/n)**2<65) continue; // no useful image texture
    let best=Infinity,bestError=Infinity,dxBest=0,dyBest=0;
    for(let dy=-radius;dy<=radius;dy++)for(let dx=-radius;dx<=radius;dx++){
      let difference=0;
      for(let yy=-half;yy<=half;yy++){
        const row=(py+yy)*w+px, shifted=(py+yy+dy)*w;
        for(let xx=-half;xx<=half;xx++)
          difference+=Math.abs(a[row+xx]-b[shifted+xx+dx]);
      }
      const error=difference/n;
      const score=error+0.13*(Math.abs(dx)+Math.abs(dy));
      if(score<best){best=score;bestError=error;dxBest=dx;dyBest=dy;}
    }
    if(bestError<=34) estimates.push({dx:dxBest,dy:dyBest,error:bestError});
  }
  if(!estimates.length) return null;
  // Require nearby displacement agreement if multiple image patches succeed.
  // Reject conflicting background/object shifts rather than moving boxes wildly.
  estimates.sort((a,b)=>a.error-b.error);
  const strongest=estimates[0];
  const consistent=estimates.filter(x=>
    Math.abs(x.dx-strongest.dx)<=2 && Math.abs(x.dy-strongest.dy)<=2);
  if(estimates.length>=3 && consistent.length<2) return null;
  const dx=consistent.reduce((s,x)=>s+x.dx,0)/consistent.length;
  const dy=consistent.reduce((s,x)=>s+x.dy,0)/consistent.length;
  return {dx,dy,error:strongest.error};
}

function translate(box,dx,dy,w,h){
  const width=box[2]-box[0],height=box[3]-box[1];
  const x=clamp(box[0]+dx/w,0,Math.max(0,1-width));
  const y=clamp(box[1]+dy/h,0,Math.max(0,1-height));
  return [x,y,x+width,y+height];
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
    this.lastDetectionAt=at;
    this.tracks=(Array.isArray(detections)?detections:[])
      .filter(validBox)
      .sort((a,b)=>b.confidence-a.confidence).slice(0,this.maxTracks)
      .map(d=>({...d,bbox:d.bbox.slice(),trackId:this.nextTrackId++,_missing:0}));
    // Reproject predictions from their original capture time to the current
    // video frame. Late, wrong-sized and expired requests are never rendered.
    for(let i=index+1;i<this.frames.length;i++)this.step(this.frames[i-1],this.frames[i]);
    return true;
  }
  getDetections(){
    return this.tracks.map(({_missing,...d})=>({...d,bbox:d.bbox.slice()}));
  }
}
