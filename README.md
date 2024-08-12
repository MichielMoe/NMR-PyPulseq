# MR Sequence Simulation Template

## Introduction

This repository provides a comprehensive Jupyter notebook template for setting up and simulating magnetic resonance (MR) sequences using the `pypulseq` and `MRzeroCore` libraries. The notebook guides users through the process of configuring system parameters, defining various MR events, constructing sequences, and simulating MR data using custom or pre-defined phantoms.

### Key Sections:
1. **System Setup**: Configure essential system parameters like gradient strength, slew rate, and RF pulse properties.
2. **Event Definitions**: Define MR events such as RF pulses, gradient events, and ADC events.
3. **Sequence Construction**: Combine the defined events into a coherent MR sequence and write it to external files.
4. **Sequence Application**: Apply the constructed sequences to MR phantoms, simulate MR data, and visualize results.
5. **Simulating the Sequence**: Execute the sequence on a phantom model and analyze the outcomes.

## Table of Contents
- [1. System Setup](#1-system-setup)
- [2. Event Definitions](#2-event-definitions)
  - [2.1 RF Events](#21-rf-events)
  - [2.2 Gradient Events](#22-gradient-events)
  - [2.3 ADC and Delay Events](#23-adc-and-delay-events)
- [3. Sequence Construction](#3-sequence-construction)
  - [3.1 Sequence Building](#31-sequence-building)
  - [3.2 Writing Sequence to an External File](#32-writing-sequence-to-an-external-file)
  - [3.3 Loading a Sequence from an External File](#33-loading-a-sequence-from-an-external-file)
- [4. Sequence Application](#4-sequence-application)
  - [4.1 Loading a Custom Phantom](#41-loading-a-custom-phantom)
  - [4.2 Making a Custom Phantom](#42-making-a-custom-phantom)
  - [4.3 Converting the Phantom into Simulation Data](#43-converting-the-phantom-into-simulation-data)
- [5. Simulating the Sequence](#5-simulating-the-sequence)
- [6. Installation and Requirements](#6-installation-and-requirements)

## 0. Installation and Requirements

Before running any of the notebooks, ensure that you have installed the necessary dependencies. You can install them directly by running into your terminal window:

```bash
pip install -r requirements.txt
```

## 1. System Setup

To set up the MR system parameters:

```python
import pypulseq as pp
import numpy as np
import MRzeroCore as mr0
import torch
import matplotlib.pyplot as plt

system = pp.Opts(max_grad=28,             
                 grad_unit='mT/m',                   
                 max_slew=150,                        
                 slew_unit='T/m/s',     
                 rf_ringdown_time=20e-6,                                     
                 rf_dead_time=100e-6,             
                 adc_dead_time=20e-6,   
                 grad_raster_time=50 * 10e-6 
                )

seq = pp.Sequence(system)
```

## 2. Event Definitions

### 2.1 RF Events

RF events define the characteristics of radio-frequency pulses:

```python
rf, gz, gzr = pp.make_sinc_pulse(flip_angle= 90.0 * np.pi / 180,     
                                 duration=2e-3,                      
                                 slice_thickness=8e-3,              
                                 apodization=0.5,                   
                                 time_bw_product=4,                  
                                 system=system,                      
                                 return_gz=True)
```

### 2.2 Gradient Events

Gradient events define the characteristics of gradient pulses:

```python
gx = pp.make_trapezoid(channel='x', 
                       area=80, 
                       duration=2e-3, 
                       system=system)
```

### 2.3 ADC Event and Delay

ADC events capture the MR signal during a readout window:

```python
adc = pp.make_adc(num_samples=128, duration=200e-3, system=system)
delay = pp.make_delay(0.1)
```

## 3. Sequence Construction

### 3.1 Sequence Building

Construct an MR sequence by combining events:

```python
seq.add_block(rf)
seq.add_block(delay, adc)
seq.add_block(pp.make_trapezoid('x', duration=1e-3, flat_time=-1, area=1))
```

### 3.2 Writing Sequence to an External File

Save the constructed sequence for further analysis:

```python
seq.set_definition('FOV', [1000e-3, 1000e-3, 8e-3])
seq.set_definition('Name', 'Spin Echo')
seq.write('./out/PP_SE_1.seq')
```

### 3.3 Loading a Sequence from an External File

Load a previously saved sequence:

```python
seq0 = mr0.Sequence.import_file('./out/PP_SE_1.seq')
```

## 4. Sequence Application

### 4.1 Loading a Custom Phantom

Load and manipulate a custom phantom:

```python
obj_p = mr0.VoxelGridPhantom.load_mat('./data/numerical_brain_cropped.mat')
obj_p = obj_p.interpolate(64, 64, 1)
obj_p.B0 *= 1
obj_p.D *= 0
obj_p.plot()
```

### 4.2 Making a Custom Phantom

Create a custom phantom:

```python
obj_p = mr0.CustomVoxelPhantom(
    pos=[[-0.25, -0.25, 0]],
    PD=[1.0],
    T1=[3.0],
    T2=[0.5],
    T2dash=[30e-3],
    D=[0.0],
    B0=0,
    voxel_size=0.25,
    voxel_shape="box"
)
obj_p.plot()
```

### 4.3 Converting the Phantom into Simulation Data

Prepare the phantom for simulation:

```python
obj_p.size = torch.tensor([1000e-3, 1000e-3, 8e-3])
obj_p = obj_p.build()
```

## 5. Simulating the Sequence

Run the simulation and visualize the results:

```python
graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0, obj_p)
sp_adc, t_adc = mr0.util.pulseq_plot(seq, clear=False, signal=signal.numpy())
seq0.plot_kspace_trajectory()
```
