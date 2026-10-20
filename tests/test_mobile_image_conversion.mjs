import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const script=readFileSync(new URL('../api.js',import.meta.url),'utf8').replace(/\bexport /g,'');
const settings={model:'auto',confidence:35,showBoxes:true,showLabels:true};
let count=0,requests=[],surfaces=[],lastQuality=0,bitmapsClosed=0;
const blob={size:930_000,type:'image/jpeg'};
class FakeFormData {
  constructor(){this.entries=[];}
  append(...args){this.entries.push(args);}
}
const context={
  window:{CIRCUITVISION_CONFIG:{mode:'api',baseUrl:'/api',requestTimeoutMs:5000}},
  navigator:{onLine:true},FormData:FakeFormData,AbortController,
  Blob:class FakeBlob{},setTimeout,clearTimeout,TypeError,Error,
  createImageBitmap:async(file,options)=>{
    count++;assert.equal(options.imageOrientation,'from-image');
    return {width:6250,height:6250,close(){bitmapsClosed++;}};
  },
  document:{createElement(name){
    assert.equal(name,'canvas');
    const out={width:0,height:0, getContext(){
      return {fillRect(){},drawImage(){},set fillStyle(x){assert.equal(x,'#ffffff');}};
    },toBlob(resolve,format,quality){lastQuality=quality;assert.equal(format,'image/jpeg');surfaces.push([out.width,out.height]);resolve(blob);}};
    return out;
  }},
  fetch:async(url,{body})=>{
    requests.push({url,body});
    return {ok:true,json:async()=>({model:'CircuitVision YOLO11',detections:[]})};
  }
};
vm.createContext(context);
vm.runInContext(script+'\nglobalThis.run = detectImage;',context);
const source={name:'1000098022.png',size:532400,type:'image/png'};
await context.run(source,settings);
await context.run(source,settings);
assert.equal(count,1,'Same file normalized only once when pressing retry');
assert.equal(bitmapsClosed,1,'Browser bitmaps must be released');
assert.deepEqual(surfaces,[[3000,3000]],'Oversized PNG shrinks below 16MP limit');
assert(lastQuality>=.90);
assert.equal(requests.length,2);
const files=requests[0].body.entries.find(e=>e[0]==='image');
assert.equal(files[1],blob);
assert.equal(files[2],'1000098022-circuitvision.jpg');
await context.run({name:'live-frame.jpg',type:'image/jpeg'},settings,{live:true});
assert.equal(count,1,'Live frames must skip conversion for camera performance');
assert.equal(requests.at(-1).body.entries.find(e=>e[0]==='image')[2],'live-frame.jpg');
console.log('Android 6250px PNG conversion, filename, caching and live bypass: PASS');
