# CircuitVision: Computer Engineering Lab Dataset Research

**Research date:** 2026-10-19. This file is a research catalog, not a downloaded or retrained model. Website still uses the student's actual YOLO11 Round 2, **57 classes**.

## Verified candidate public datasets

The publishers' public descriptions list these as CC BY 4.0; confirm the export license per version before reuse and preserve credits. Sizes are those reported on public project pages, not audited training sample counts.

| Dataset / author | Coverage | Public URL |
|---|---|---|
| Electronic Component Detection / Personal | ~2,121 images; 61 classes: resistor, diode, inductor, 9V battery, Arduino, ESP32, LED, LDR, buck converter, servo, motor driver, sensors, IC, MOSFET, etc. | https://universe.roboflow.com/personal-8hq68/electronic-component-detection-m6sop |
| Electronic components recognition / CPEN291G8 | ~611 images; 16 classes including resistor values, colored LEDs, capacitors, inductors, transistors, IC chips | https://universe.roboflow.com/cpen291g8/electronic-components-recognition |
| Electronic components detection / CV2024 | ~278 images; 18 classes including capacitor types, diode, crystal, IC, power resistor, potentiometer, TO-92/TO-220 | https://universe.roboflow.com/cv2024-4skvf/electronic-components-detection-jzf42 |
| Deteksi alat elektronik / myspace | ~546 images; 8 classes including oscilloscope, function generator, multimeter, trainer kit, breadboard | https://universe.roboflow.com/myspace-natcf/deteksi-alat-elektronik |
| ProjectAI / supanat-sreewichai | ~3,453 images; multimeter, oscilloscope, soldering iron and voltage source | https://universe.roboflow.com/supanat-sreewichai/projectai-mmrgp |
| PCB Component Detection / wangshnegqi | Public repository with YOLO object labels for 10 PCB classes: resistor, capacitor, diode, LED, IC, connector, switch/button, header, crystal, test point. Images+labels CC BY 4.0; code/weights Apache 2.0. | https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo |
| Digital_Circuit_Component_Identifier / Majd Osseiran | ~211 labeled **drawn circuit gate symbols**, not photographs of physical DIP chips. AND, NAND, NOR, NOT, OR, XNOR, XOR | https://universe.roboflow.com/search?q=Digital_Circuit_Component_Identifier |

## Curriculum-based proposed physical class taxonomy

**Discrete + analog:** fixed resistor, power resistor, variable resistor, trimmer, LDR, thermistor, ceramic capacitor, electrolytic capacitor, film capacitor, diode, Zener diode, Schottky diode, LED, RGB LED, transistor (BJT), MOSFET, IGBT, op amp, voltage regulator, rectifier, inductor, transformer, crystal oscillator, relay, optocoupler, fuse, heat sink.

**Logic + digital electronics:** general DIP/SMD IC, 74HC00/7400 NAND, 74HC02 NOR, 74HC04 NOT, 74HC08 AND, 74HC32 OR, 74HC86 XOR, 74HC74 flip-flop, 74HC138 decoder, 74HC151 multiplexer, 74HC157 multiplexer, 74HC595 shift register, 74HC165, 74HC161 counter, 74HC373 latch, NE555 timer, seven-segment module, binary trainer board, DIP switch, logic probe. **These named chips usually look physically identical**; chip-level identities should use readable markings (OCR) or device metadata. Start with a broad Logic IC detection class, and keep circuit diagram symbol recognition separate.

**Prototyping:** full/mini breadboard, jumper wire, DuPont cable, jumper leads, pin header, connector, terminal block, USB connector, perfboard, protoboard, printed circuit board, pushbutton, slide switch, toggle switch, rotary encoder, power-module breadboard adapter, battery holder, 9V battery, AA/AAA cell, buck/boost module.

**Embedded systems:** Arduino Uno, Nano, Mega, ESP8266, ESP32, Raspberry Pi, Pico, PIC/AVR board, programmer, USB-UART adapter, relay module, Bluetooth module, RFID reader, SD-card module, EEPROM.

**Sensors, actuators, indicators:** IR sensor, ultrasonic sensor, PIR motion detector, LDR module, temperature/humidity sensor, barometer, gas sensor, accelerometer, gyro, Hall sensor, microphone, joystick, LCD, OLED, LED matrix, keypad, servo motor, DC motor, stepper motor, motor driver, speaker, buzzer.

**Lab instruments:** digital multimeter, analog meter, bench power supply, function generator, oscilloscope, logic analyzer, soldering iron/station, solder sucker, hot-air rework gun, wire stripper, side cutters, pliers, breadboard trainer, oscilloscope probe, variable power module.

This is a **proposed scope**, not a claim the existing model can detect them all. Adding classes means a new mapping and a new GPU training/export cycle.

## Before Round 3 training

1. Obtain permitted YOLO object-detection exports with complete provenance; do NOT scrape product photos and assume use is permitted.
2. Merge only compatible classes; reconcile e.g. battery_9v, 9-Volt-Battery, Battery 9V. Keep *power module* distinct from battery.
3. Do not mix circuit diagram symbols with real photographs of physical gates.
4. Audit labels and bounding boxes, deduplicate across sources and splits, freeze a held-out test set and report per-class AP.
5. Collect or curate more challenging lab-bench photos and videos whenever available to close domain gaps. Public images alone may not represent this phone, its lighting and movement.
6. Re-train on a Colab GPU only after taxonomy/mapping/data validation; compare with the existing 49.87% test mAP@50 using a consistent holdout and only then deploy.

## Motion-tracking deployment notes

The existing **Round 2 trained ONNX** continues to provide class labels, scores and exact detection anchors. Browser-side grayscale patch tracking estimates in-between position updates, replays frames buffered during each server inference, and resets on zoom/camera session changes. It does not add classes, raise detection mAP or guarantee tracking during occlusion, blur or fast movement.


## Resistor value-reading research datasets (candidate for a separate value model)

This addresses a task distinct from YOLO object detection: accurately reading each colored band, ordering 4/5-band stripes, calculating resistance and optionally checking the value against a multimeter. Individual band masks alone do not supply the color and digit of every band.

| Candidate source | Public description | License as published | Link |
|---|---|---|---|
| Resistor Color Band Detection / image | ~134 images, 11 color classes, YOLO11 object detection | Public Domain | https://universe.roboflow.com/image-gpmgb/resistor-color-band-detection-fnbak |
| Band Detection / ResistorAI | ~1,603 images, one segmentation class resistor-color-bands (locates areas, does not identify colors) | CC BY 4.0 | https://universe.roboflow.com/resistorai/band-detection |
| Resistor Color Band Detection5 / Rochester Institute Of Technology Dubai | ~356 project images; 12 color classes including gold and silver | Public Domain | https://universe.roboflow.com/rochester-institute-of-technology-dubai-6oda9/resistor-color-band-detection5 |
| Resistor Value Determination / Resistor Color Code | ~296 images; semantic segmentation | CC BY 4.0 | https://universe.roboflow.com/resistor-color-code/resistor-value-determination |
| Resistor Detection / Circuits Project | ~756 images: resistors, wires, breadboards | CC BY 4.0 | https://universe.roboflow.com/circuits-project/resistor-detection-5azes |

**Status**: verified links and published licenses only; the images have NOT yet been downloaded, audited, merged or trained here. Some online entries report inconsistent counts for overview vs model versions, so verify the chosen version after export. Recheck rights, annotation types and author credits.

**Required training-label design**: resistor bbox, per-band bbox/mask and color label, left-to-right or tolerance-side orientation, band_count, independently checked resistance_ohms, tolerance_percent, specimen/scene grouping and photo conditions. Leave hard-to-read bands as unknown; never silently guess a measured value. Compare automatic decoding to a fixed photo-level test set and report exact-value accuracy, value-within-tolerance accuracy, false confident readings, and inference time. A close-up capture workflow is likely necessary on a phone.

**Implemented now**: a 4-/5-band manual color decoder and component reference page, powered by deterministic color-code tables. The camera still identifies only the original **57** trained categories; manual values are explicitly NOT attributed to YOLO predictions. Color standard cross-checked with DigiKey's published 4-/5-band calculator: https://www.digikey.com/en/resources/conversion-calculators/conversion-calculator-resistor-color-code .
