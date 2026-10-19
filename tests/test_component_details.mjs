import assert from 'node:assert/strict';
import {decodeResistorBands,formatOhms,getComponentDetails} from '../component_details.mjs';
const cases=[
  [['red','red','brown','gold'],220,5,'220 Ω'],
  [['brown','black','red','gold'],1000,5,'1 kΩ'],
  [['brown','black','orange','gold'],10000,5,'10 kΩ'],
  [['brown','black','black','red','brown'],10000,1,'10 kΩ'],
  [['yellow','violet','gold','silver'],4.7,10,'4.7 Ω'],
  [['brown','green','black','brown','brown'],1500,1,'1.5 kΩ']
];
for(const [bands,ohms,tol,formatted] of cases){
  const r=decodeResistorBands(bands);
  assert(Math.abs(r.ohms-ohms)<1e-8);
  assert.equal(r.tolerancePercent,tol);
  assert.equal(r.formatted,formatted);
}
assert.throws(()=>decodeResistorBands(['black','brown','red','gold']),/first significant/);
assert.throws(()=>decodeResistorBands(['brown','gold','red','gold']),/first significant/);
assert.throws(()=>decodeResistorBands(['red','red','black','yellow']),/tolerance/);
assert.throws(()=>decodeResistorBands(['brown','black','red']),/4 or 5/);
assert.equal(formatOhms(1000000),'1 MΩ');
assert.equal(getComponentDetails('Resistor').resistor,true);
assert.equal(getComponentDetails('Power Module').resistor,false);
assert.equal(getComponentDetails('LED').name,'LED');
console.log('CircuitVision component guide: all resistor band/tolerance tests passed');
