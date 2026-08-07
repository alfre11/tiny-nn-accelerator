# Tiny Neural Network Accelerator

A fixed-point neural network inference accelerator written in SystemVerilog for 8×8 handwritten digit classification. This project trains a small neural network in Python, quantizes the model to integer fixed-point values, exports memory files, and verifies the RTL accelerator against a Python fixed-point reference model.

## Overview

This project implements a simple hardware accelerator for neural network inference. The accelerator runs a fully connected neural network with the architecture:

```text
64 → 16 → 10
```

The input is an 8×8 handwritten digit image from the `sklearn` digits dataset. The model classifies the image as one of ten digit classes, `0` through `9`.

The project demonstrates an end-to-end hardware/software flow:

```text
Train floating-point model in Python
        ↓
Quantize weights, biases, and inputs
        ↓
Run fixed-point Python reference inference
        ↓
Export signed values to .mem files
        ↓
Load .mem files into SystemVerilog RTL
        ↓
Simulate accelerator with testbench
        ↓
Compare RTL outputs against Python expected outputs
```

## Motivation

The goal of this project is to understand how neural network inference maps onto hardware. Instead of only training a model in Python, this project implements the inference computation at the RTL level using:

* Fixed-point arithmetic
* Signed multiply-accumulate operations
* FSM-controlled layer execution
* ReLU activation
* Argmax classification
* Cycle-level RTL verification

This project is intended to connect AI, computer architecture, digital design, and hardware/software co-design.

## Model Architecture

The neural network is a small fully connected classifier:

```text
Input layer:   64 values
Hidden layer:  16 neurons
Output layer:  10 classes
```

The model structure is:

```text
Linear(64, 16)
ReLU
Linear(16, 10)
Argmax
```

The input is an 8×8 image flattened into a 64-element vector.

## Dataset

The model uses the `sklearn.datasets.load_digits` dataset.

Each image is:

```text
8 × 8 pixels = 64 input features
```

Pixel values are normalized in Python and then quantized to signed 8-bit fixed-point values before being used by the accelerator.

## Fixed-Point Quantization

The accelerator uses fixed-point integer arithmetic instead of floating point. This makes the design simpler and more hardware-friendly.

The fixed-point scale is:

```text
SCALE_BITS = 7
SCALE = 2^7 = 128
```

A floating-point value is quantized as:

```text
quantized_value = round(float_value × 128)
```

### Numeric Formats

| Value Type         |       Format |
| ------------------ | -----------: |
| Inputs             |  signed int8 |
| Weights            |  signed int8 |
| Biases             | signed int32 |
| Accumulator        | signed int32 |
| Hidden activations | signed int32 |
| Output logits      | signed int32 |

Biases are quantized using `SCALE × SCALE` because they are added before the accumulator is shifted back down.

## Hardware Architecture

The baseline accelerator uses a sequential single-MAC datapath.

### Main Hardware Blocks

```text
Input Memory
     |
     v
Weight Memory ---> MAC Unit ---> Accumulator ---> ReLU ---> Hidden Memory
                                      |
                                      v
                              Output Layer MAC
                                      |
                                      v
                                  Output Logits
                                      |
                                      v
                                    Argmax
                                      |
                                      v
                                  Prediction
```

### Layer 1

Layer 1 computes the hidden activations:

```text
hidden[j] = ReLU(bias1[j] + sum(input[i] × weight1[j][i]))
```

For this model:

```text
16 hidden neurons × 64 inputs = 1024 MAC operations
```

### Layer 2

Layer 2 computes the output logits:

```text
output[k] = bias2[k] + sum(hidden[j] × weight2[k][j])
```

For this model:

```text
10 output neurons × 16 hidden values = 160 MAC operations
```

### Total MAC Operations

```text
1024 + 160 = 1184 MAC operations
```

The measured RTL cycle count is slightly higher because of FSM overhead for initialization, storing results, shifting, ReLU, argmax, and done signaling.

## RTL Modules

| File                      | Description                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| `rtl/mac_unit.sv`         | Signed multiply-accumulate datapath                                                      |
| `rtl/nn_accelerator.sv`   | Top-level FSM-controlled neural network accelerator                                      |
| `tb/nn_accelerator_tb.sv` | RTL testbench that compares accelerator outputs against Python-generated expected values |

## Python Files

| File                              | Description                                                          |
| --------------------------------- | -------------------------------------------------------------------- |
| `python/train_model.py`           | Trains the floating-point 64→16→10 neural network                    |
| `python/quantize_model.py`        | Quantizes weights, biases, and test inputs                           |
| `python/fixed_point_reference.py` | Runs integer-only fixed-point inference in Python                    |
| `python/export_mem.py`            | Exports signed fixed-point values to `.mem` files for RTL simulation |

## Memory Files

The RTL testbench loads generated `.mem` files from:

```text
data/mem/
```

Expected files include:

| File                      | Description                           |
| ------------------------- | ------------------------------------- |
| `input.mem`               | Quantized input sample                |
| `weights_l1.mem`          | Quantized layer 1 weights             |
| `biases_l1.mem`           | Quantized layer 1 biases              |
| `weights_l2.mem`          | Quantized layer 2 weights             |
| `biases_l2.mem`           | Quantized layer 2 biases              |
| `expected_hidden.mem`     | Python fixed-point hidden activations |
| `expected_output.mem`     | Python fixed-point output logits      |
| `expected_prediction.mem` | Python fixed-point argmax prediction  |

## Verification Results

The SystemVerilog accelerator was verified against the Python fixed-point reference model.

Current verified sample:

```text
Sample index: 0
Expected prediction: 7
RTL prediction:      7
```

## Multi-Sample Verification Results

The RTL accelerator was tested across 100 input samples exported from the Python fixed-point reference flow.

| Metric | Result |
|---|---:|
| Samples tested | 100 |
| Prediction errors | 0 |
| Hidden activation mismatches | 0 |
| Output logit mismatches | 0 |
| Total errors | 0 |
| Average cycles per inference | 1249 |
| Overall result | PASS |

### RTL Output Verification

The RTL testbench verifies:

* Final prediction
* All 16 hidden activations
* All 10 output logits

Current result:

```text
PASS: prediction matches
PASS: all hidden values match
PASS: all output logits match
OVERALL RESULT: PASS
```

## Performance Results

The sequential accelerator completes one inference in:

```text
1249 cycles
```

Assuming a 100 MHz clock:

```text
Clock period: 10 ns
Inference latency: 12.49 µs
Estimated throughput: 80,064 inferences/sec
```

At a simulated 100 MHz clock, the accelerator completes each inference in 1249 cycles:

```text
Inference latency = 1249 cycles × 10 ns = 12.49 µs
Throughput ≈ 80,064 inferences/sec

### Performance Table

| Metric                            |                Result |
| --------------------------------- | --------------------: |
| Model architecture                |          64 → 16 → 10 |
| Layer 1 MACs                      |                  1024 |
| Layer 2 MACs                      |                   160 |
| Total MAC operations              |                  1184 |
| Measured cycles per inference     |                  1249 |
| Estimated clock frequency         |               100 MHz |
| Estimated inference latency       |              12.49 µs |
| Estimated throughput              | 80,064 inferences/sec |
| Hidden activations matched Python |                   Yes |
| Output logits matched Python      |                   Yes |
| Prediction matched Python         |                   Yes |
```

## Example Testbench Output

```text
Cycle count:
Cycles from start wait to done: 1249
Inference time at 100 MHz: 12490 ns
Approx throughput at 100 MHz: 80064 inferences/sec

Prediction check:
Prediction:          7
Expected prediction: 7
PASS: prediction matches

Hidden values:
hidden_mem[0] = 0, expected = 0
hidden_mem[1] = 0, expected = 0
hidden_mem[2] = 11, expected = 11
hidden_mem[3] = 0, expected = 0
hidden_mem[4] = 0, expected = 0
hidden_mem[5] = 507, expected = 507
hidden_mem[6] = 355, expected = 355
hidden_mem[7] = 0, expected = 0
hidden_mem[8] = 17, expected = 17
hidden_mem[9] = 787, expected = 787
hidden_mem[10] = 232, expected = 232
hidden_mem[11] = 534, expected = 534
hidden_mem[12] = 464, expected = 464
hidden_mem[13] = 95, expected = 95
hidden_mem[14] = 0, expected = 0
hidden_mem[15] = 506, expected = 506
PASS: all hidden values match

Output logits:
output_mem[0] = -244, expected = -244
output_mem[1] = -266, expected = -266
output_mem[2] = -579, expected = -579
output_mem[3] = -217, expected = -217
output_mem[4] = -184, expected = -184
output_mem[5] = -467, expected = -467
output_mem[6] = -865, expected = -865
output_mem[7] = 706, expected = 706
output_mem[8] = 24, expected = 24
output_mem[9] = 304, expected = 304
PASS: all output logits match

OVERALL RESULT: PASS
```

## Sequential vs 4-MAC Accelerator

| Design | MAC Units | Samples Tested | Prediction Errors | Hidden Mismatches | Output Mismatches | Avg Cycles | Latency @ 100 MHz | Throughput @ 100 MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequential | 1 | 100 | 0 | 0 | 0 | 1249 | 12.49 µs | 80,064 inf/s |
| 4-MAC Parallel | 4 | 100 | 0 | 0 | 0 | 361 | 3.61 µs | 277,008 inf/s |

The 4-MAC parallel datapath reduced inference latency from 1249 cycles to 361 cycles, achieving a 3.46× speedup while preserving exact agreement with the Python fixed-point reference across predictions, hidden activations, and output logits.

## Latency / Resource Tradeoff

Both accelerator designs were synthesized with Yosys to compare the latency improvement from parallel MAC execution against the additional hardware cost.

| Design | MAC Units | Avg Cycles | Speedup | Total Cells | Multipliers | Adders | Muxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential | 1 | 1249 | 1.00× | 1544 | 1 | 8 | 1360 |
| 4-MAC Parallel | 4 | 361 | 3.46× | 5393 | 4 | 20 | 5188 |

The 4-MAC accelerator reduced inference latency from 1249 cycles to 361 cycles, achieving a 3.46× speedup. This performance improvement came at the cost of increasing synthesized cell count from 1544 to 5393 cells, or about 3.49× more cells. The multiplier count increased from 1 to 4, matching the intended parallel MAC datapath.

Calculations:  
Speedup = 1249 / 361 = 3.46×  
Cell increase = 5393 / 1544 = 3.49×

## How to Run

### 1. Train the model

```bash
python python/train_model.py
```

This creates the trained floating-point model and test data.

### 2. Quantize the model

```bash
python python/quantize_model.py
```

This creates quantized weights, biases, and test inputs.

### 3. Run the Python fixed-point reference

```bash
python python/fixed_point_reference.py
```

This verifies the integer-only inference behavior in Python.

### 4. Export `.mem` files

```bash
python python/export_mem.py
```

This creates the `.mem` files used by the SystemVerilog simulation.

### 5. Compile the RTL simulation

```bash
iverilog -g2012 -o sim.out rtl/mac_unit.sv rtl/nn_accelerator.sv tb/nn_accelerator_tb.sv
```

### 6. Run the simulation

```bash
vvp sim.out
```

Expected final result:

```text
OVERALL RESULT: PASS
```

## Current Project Status

Completed:

* Trained a 64→16→10 neural network on the sklearn digits dataset
* Implemented fixed-point quantized inference in Python
* Exported signed fixed-point values to `.mem` files for RTL simulation
* Implemented a signed single-MAC unit in SystemVerilog
* Implemented a sequential FSM-controlled neural network accelerator
* Implemented a 4-MAC parallel accelerator datapath
* Built SystemVerilog testbenches for both accelerator versions
* Verified RTL outputs against Python fixed-point reference values
* Tested 100 samples automatically with 0 prediction errors
* Matched RTL hidden activations and output logits against Python expected values
* Added cycle counting to measure inference latency
* Compared sequential and 4-MAC accelerator performance
* Ran Yosys synthesis for both designs
* Collected synthesized resource usage for latency/resource tradeoff analysis

## Future Improvements

Planned or possible extensions:

* Add an 8-MAC parallel datapath
* Compare 1-MAC, 4-MAC, and 8-MAC latency/resource tradeoffs
* Run FPGA-specific synthesis to report LUTs, flip-flops, DSPs, and BRAM usage
* Add timing estimation or maximum clock frequency results
* Add waveform screenshots from GTKWave
* Add a memory-mapped input/output interface
* Add valid/ready handshake signals
* Add support for larger fully connected models
* Add support for convolutional layers
* Explore OpenLane ASIC synthesis/place-and-route
* Create a block diagram image for the README
* Add automated run scripts for training, export, simulation, and synthesis


## Skills Demonstrated

* SystemVerilog RTL design
* Fixed-point arithmetic
* Neural network inference
* Quantization
* Multiply-accumulate datapath design
* FSM control logic
* RTL simulation
* Hardware/software co-design
* Python model training and verification
* Testbench development
* Cycle-level performance measurement
