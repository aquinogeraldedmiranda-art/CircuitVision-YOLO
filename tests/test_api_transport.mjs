import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('../api.js',import.meta.url),'utf8').replace(/\bexport /g,'');
function fixture(fetchFn,online=true){
  class TestFormData { constructor(){this.entries=[];} append(...v){this.entries.push(v);} }
  const context={
    window:{CIRCUITVISION_CONFIG:{mode:'api',baseUrl:'/api',requestTimeoutMs:1000}},
    fetch:fetchFn,FormData:TestFormData,AbortController,navigator:{onLine:online},
    setTimeout,clearTimeout,TypeError,Error
  };
  vm.createContext(context);vm.runInContext(source+'\nglobalThis.testDetect = detectImage;',context);
  return context.testDetect;
}
const settings={model:'custom',confidence:35,showBoxes:true,showLabels:true};
const image={name:'real-lab.jpg'};
const good={
  ok:true,json:async()=>({detections:[{label:'Resistor',confidence:.77,bbox:[.1,.2,.3,.4]}],
    model:'custom',processing_ms:950})
};
let attempts=0;
let detect=fixture(async()=>{attempts++;if(attempts===1)throw new TypeError('Failed to fetch');return good;});
let result=await detect(image,settings);
assert.equal(attempts,2,'an interrupted upload retries exactly once');
assert.equal(result.detections[0].label,'Resistor');
assert.equal(result.model,'custom');
attempts=0;
detect=fixture(async()=>{attempts++;throw new TypeError('Failed to fetch');});
await assert.rejects(()=>detect(image,settings),/Connection to CircuitVision was interrupted/);
assert.equal(attempts,2,'upload does not retry indefinitely');
attempts=0;
await assert.rejects(()=>detect(image,settings,{live:true}),/Connection to CircuitVision was interrupted/);
assert.equal(attempts,1,'live camera does not resend failed frames');
attempts=0;
detect=fixture(async()=>{attempts++;return {ok:false,status:422,json:async()=>({detail:'Invalid model selection'})};});
await assert.rejects(()=>detect(image,settings),/Invalid model selection/);
assert.equal(attempts,1,'nontransient 422 preserves actual server response');
detect=fixture(async()=>{throw new TypeError('Failed to fetch');},false);
await assert.rejects(()=>detect(image,settings),/phone is offline/);
console.log('Detection transport: upload retry, live no-retry, HTTP error, offline diagnostic PASS');
