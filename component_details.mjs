/**
 * CircuitVision classroom electronics component guide.
 * The existing YOLO model returns class labels, not a measured component value.
 * Band-code values are derived ONLY from colors explicitly entered by the user.
 */
export const DIGITS={black:0,brown:1,red:2,orange:3,yellow:4,green:5,blue:6,violet:7,gray:8,white:9};
export const MULTIPLIERS={...Object.fromEntries(Object.entries(DIGITS).map(([k,v])=>[k,10**v])),gold:0.1,silver:0.01};
export const TOLERANCES={brown:1,red:2,orange:3,green:0.5,blue:0.25,violet:0.1,gray:0.05,gold:5,silver:10};
export const COLOR_HEX={black:'#1a2430',brown:'#8b4b2d',red:'#e94245',orange:'#ee9233',yellow:'#ffd34b',green:'#27a566',blue:'#3988eb',violet:'#914dc4',gray:'#9099a5',white:'#fbfbfd',gold:'#c8a044',silver:'#c0c6d0'};
export const COLOR_OPTIONS=Object.keys(DIGITS);
export function formatOhms(ohms){
  if(!Number.isFinite(ohms)||ohms<0)throw new RangeError('Invalid resistance');
  const round=(n)=>Number(n.toPrecision(5)).toLocaleString('en-US',{maximumFractionDigits:5});
  if(ohms>=1e9)return round(ohms/1e9)+' GΩ';
  if(ohms>=1e6)return round(ohms/1e6)+' MΩ';
  if(ohms>=1e3)return round(ohms/1e3)+' kΩ';
  return round(ohms)+' Ω';
}
export function decodeResistorBands(colors){
  if(!Array.isArray(colors)||![4,5].includes(colors.length))throw new Error('Choose 4 or 5 resistor bands');
  const clean=colors.map(x=>String(x).toLowerCase().trim());
  const count=clean.length===4?2:3;
  const digits=clean.slice(0,count);
  if(digits[0]==='black'||digits.some(x=>!(x in DIGITS)))throw new Error('Choose a valid first significant band (not black)');
  const multiplier=MULTIPLIERS[clean[count]],tolerance=TOLERANCES[clean[count+1]];
  if(!Number.isFinite(multiplier)||!Number.isFinite(tolerance))throw new Error('Invalid multiplier or tolerance band');
  const ohms=digits.reduce((n,c)=>n*10+DIGITS[c],0)*multiplier;
  return {ohms,tolerancePercent:tolerance,formatted:formatOhms(ohms),bands:clean.slice(),
    range:[ohms*(1-tolerance/100),ohms*(1+tolerance/100)]};
}
const GUIDE={
 resistor:{name:'Resistor',role:'Limits current and sets voltages in electronic circuits.',learn:'Resistance is measured in ohms (Ω). Axial resistors use color bands; SMD resistors may have printed numbers.',value:'Enter 4 or 5 bands below, or confirm with a multimeter. YOLO cannot read the value yet.'},
 led:{name:'LED',role:'Emits light when forward biased with the correct current.',learn:'Polarity matters. Use a current-limiting resistor; typical forward voltage depends on color and part.',value:'LED voltage and current rating cannot be identified reliably from appearance alone.'},
 diode:{name:'Diode',role:'Allows current mainly in one direction; used for rectification and protection.',learn:'Check the cathode stripe and datasheet. Use diode-test mode to measure forward drop.',value:'Exact type, voltage and current ratings require markings or a part number.'},
 capacitor:{name:'Capacitor',role:'Stores electric charge; used for filtering, timing and decoupling.',learn:'Capacitance is measured in farads (often µF, nF or pF); electrolytic types can be polarized.',value:'Read printed capacitance and voltage ratings or measure with an LCR meter.'},
 ldr:{name:'LDR / photoresistor',role:'Changes resistance as surrounding light changes.',learn:'Typical resistance is lower in bright light and higher in darkness.',value:'No fixed resistance value; measure in the current lighting conditions.'},
 breadboard:{name:'Breadboard',role:'Allows temporary solderless prototyping of circuits.',learn:'Power rails may be split. Check internal connection groups and voltage limits.',value:'There is no electrical component value to decode.'},
 potentiometer:{name:'Potentiometer',role:'Variable resistor used for adjustment or voltage division.',learn:'The wiper moves along the resistive track between end terminals.',value:'Measure outer-pin resistance or read the printed resistance marking.'},
 transistor:{name:'Transistor',role:'Amplifies or switches electronic signals.',learn:'Pinout and switching limits differ by part and package.',value:'Exact part number and ratings must come from a marking or datasheet.'},
 ic:{name:'Integrated circuit / logic IC',role:'Combines many electronic functions in one package.',learn:'74HC-series logic chips can look alike; the printed part number determines the logic function.',value:'OCR of the chip marking or manual part entry is needed to identify the precise IC.'},
 arduino:{name:'Arduino board',role:'Programmable microcontroller development board.',learn:'Useful for sensor reading, embedded logic, digital and analog I/O.',value:'Voltage/current limits depend on the exact board variant and pins; check its datasheet.'},
 esp32:{name:'ESP32 board',role:'Microcontroller board commonly used with Wi-Fi and Bluetooth.',learn:'Used for IoT and embedded prototypes.',value:'Supply and GPIO limits depend on the specific board; verify the datasheet.'},
 battery:{name:'Battery',role:'Portable DC power source.',learn:'Voltage depends on chemistry, cell count and state of charge.',value:'Read the label and verify with a multimeter; appearance does not measure actual voltage.'},
 power:{name:'Power module',role:'Regulates or distributes DC electrical power.',learn:'A breadboard power module can output multiple voltages and must be connected correctly.',value:'Read the regulator label and selected jumpers; confirm the actual output with a multimeter.'},
 sensor:{name:'Sensor module',role:'Converts a physical measurement into an electrical signal.',learn:'Common examples include ultrasonic, IR, temperature and motion sensors.',value:'Range, supply voltage and calibration depend on the exact part number.'},
 display:{name:'Display / indicator',role:'Presents readings, digits or status to a user.',learn:'Can use LED segments, character LCDs, LED matrices or OLED panels.',value:'Resolution, pinout and required voltage vary by module.'},
 instrument:{name:'Lab instrument',role:'Measures or generates electronic signals in a workbench lab.',learn:'Examples include multimeters, oscilloscopes and function generators.',value:'Read the instrument display and use the correct probes and measurement range.'}
};
export const GUIDE_CHOICES=['Resistor','LED','Diode','Capacitor','LDR','Breadboard','Potentiometer','Transistor','Integrated Circuit','Arduino Uno','ESP32','Battery 9V','Power Module','Ultrasonic Sensor','7-Segment Display','Multimeter'];
export function getComponentDetails(label){
  const s=String(label??'').trim().toLowerCase();
  const key = s.includes('resistor')&&!s.includes('photo')?'resistor':
    s==='ldr'||s.includes('photoresistor')?'ldr':
    s.includes('capacitor')?'capacitor':
    s.includes('led')?'led':
    s.includes('diode')?'diode':
    s.includes('breadboard')?'breadboard':
    s.includes('potentiometer')?'potentiometer':
    s.includes('transistor')||s.includes('mosfet')?'transistor':
    s.includes('arduino')?'arduino':
    s.includes('esp32')||s.includes('esp8266')?'esp32':
    s.includes('battery')?'battery':
    s.includes('power module')||s.includes('power supply')?'power':
    s.includes('ic')||s.includes('chip')||s.includes('logic')?'ic':
    s.includes('sensor')?'sensor':
    s.includes('display')||s.includes('segment')||s.includes('lcd')||s.includes('oled')?'display':
    s.includes('multimeter')||s.includes('oscilloscope')||s.includes('generator')?'instrument':null;
  return {label:label||'Unspecified component', ...(GUIDE[key]||{name:'Electronic laboratory component',role:'Component identified or selected for study.',learn:'Check the printed part number, pinout and manufacturer documentation.',value:'The exact electrical value is not provided by the current object-detection model.'}),resistor:key==='resistor'};
}
