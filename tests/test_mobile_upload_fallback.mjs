// Regression: real mobile browsers sometimes return null for toBlob or
// throw while drawing a large image even though the source photo is valid.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../api.js',import.meta.url),'utf8').replace(/\bexport /g,'');
const settings={model:'auto',confidence:35,showBoxes:true,showLabels:true};

function fixture({width,height,convert,decodeFails=false}){
  const attempts=[],requests=[];
  class FD{constructor(){this.values=[];}append(...v){this.values.push(v);}}
  const context={
    window:{CIRCUITVISION_CONFIG:{mode:'api',baseUrl:'/api',maxFileMB:10}},
    navigator:{onLine:true},FormData:FD,AbortController,Blob:class Blob{},
    TypeError,Error,setTimeout,clearTimeout,
    createImageBitmap:async()=>{
      if(decodeFails)throw new Error('ImageBitmap failed');
      return {width,height,close(){}};
    },
    document:{createElement(){
      const canvas={
        width:0,height:0,
        getContext(){return {fillRect(){},drawImage(){},set fillStyle(_){}};},
        toBlob(resolve){
          attempts.push([canvas.width,canvas.height]);
          if(convert==='throw') throw new Error('Canvas quota exceeded');
          if(convert==='null')resolve(null);
          else if(convert==='second'&&attempts.length===1)resolve(null);
          else resolve({size:100000,type:'image/jpeg'});
        }
      };
      return canvas;
    }},
    fetch:async(_url,{body})=>{
      requests.push(body.values.find(x=>x[0]==='image'));
      return {ok:true,json:async()=>({model:'YOLOv11 (1)',detections:[]})};
    }
  };
  vm.createContext(context);
  vm.runInContext(source.replace(/\\bexport /g,'')+'\nglobalThis.go=detectImage;',context);
  return {go:context.go,attempts,requests};
}

{
  const f=fixture({width:4000,height:3000,convert:'null'});
  const raw={name:'android-shot.png',size:750000,type:'image/png'};
  await f.go(raw,settings);
  assert.equal(f.requests.length,1,'The model server should still receive the valid photo');
  assert.equal(f.requests[0][1],raw,'Use original bytes when JPEG encoding failed');
  assert.equal(f.requests[0][2],'android-shot.png','Keep correct original file name');
  assert.equal(f.attempts.length,4,'Multiple canvas sizes were attempted');
}
{
  const f=fixture({width:6250,height:6250,convert:'second'});
  await f.go({name:'oversized-gallery.png',size:600000,type:'image/png'},settings);
  assert.equal(f.requests.length,1);
  assert.equal(f.requests[0][2],'oversized-gallery-circuitvision.jpg');
  assert.deepEqual(f.attempts,[[3000,3000],[2048,2048]]);
}
{
  const f=fixture({width:6250,height:6250,convert:'null'});
  await assert.rejects(()=>f.go({name:'huge.png',size:200000,type:'image/png'},settings),
    /too large for your phone to resize/);
  assert.equal(f.requests.length,0,'A 39MP original must not be uploaded to 16MP server');
}
{
  const f=fixture({width:0,height:0,convert:'throw',decodeFails:true});
  const raw={name:'already-small.jpg',size:300000,type:'image/jpeg'};
  await f.go(raw,settings);
  assert.equal(f.requests[0][1],raw,'Fallback to server decoding when ImageBitmap fails');
}
console.log('Android photo upload: null canvas, adaptive resize, oversized guard and raw fallback PASS');
