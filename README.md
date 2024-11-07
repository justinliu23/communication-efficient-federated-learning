# communication-efficient-federated-learning

This repository contains a Python-based quantization module designed to optimize federated learning by reducing the communication cost associated with model updates. By converting client models from full precision (32-bit) to low-bit (e.g., 8-bit) representations, this project aims to improve efficiency in federated learning environments where communication bandwidth is a constraint.

Click [here](https://github.com/justinliu23/federated-learning-simulation) for my related repo on creating a federated learning simulation with multiple devices that each train independently on local data.

## Project Overview

In federated learning, multiple clients collaboratively train a shared model without sharing their local data. However, each communication round between clients and the central server can be bandwidth-intensive due to the large size of model parameters. This project addresses this challenge by using **model quantization**, a technique that reduces the bit-width of model parameters before transmission, thereby decreasing the model size and communication overhead.

The repository implements quantization strategies specifically for federated learning, focusing on minimizing communication while maintaining model performance.

## Key Features

- **Quantization of Model Weights**:
  - Converts 32-bit floating-point representations of model weights to lower-bit formats (e.g., 8-bit), resulting in approximately a 4x reduction in model size.
  - Preserves essential model information while reducing the number of bits, allowing for efficient transmission without extensive accuracy loss.
  
- **Customizable Quantization Levels**:
  - Supports variable bit-width quantization, allowing users to choose the appropriate trade-off between compression and model fidelity.
  - Implements both full-precision and low-precision representations, demonstrating the impact of different bit-width choices on model performance and communication efficiency.

- **Federated Learning Integration**:
  - Designed to integrate seamlessly with federated learning workflows, where clients independently quantize their models before sending updates to the central server.
  - The quantization method is particularly useful for edge devices with limited bandwidth, such as mobile phones or IoT devices.

## Core Functions

- **Quantize Weights**:
  - Applies quantization to model weights by projecting full-precision weights to a discrete set of values determined by the chosen bit-width.
  - Ensures efficient compression while preserving essential characteristics of the original model, minimizing the trade-off between model size and accuracy.

- **Bit-Width Conversion**:
  - The code includes utilities for converting weights between different bit-width representations (e.g., 32-bit to 8-bit).
  - Enables flexible quantization settings, allowing for experimentation with different bit-widths based on network constraints and model requirements.

## Technical Details

- **Data Representation**:
  - 32-bit floating-point (IEEE 754 Format) is reduced to lower-bit fixed-point representations, constraining weights to a limited range of discrete values.
  - Quantization reduces each weight’s range and precision by mapping it to the nearest neighbor within the discrete set of values.

- **Compression Ratios**:
  - For example, 8-bit quantization results in a model size reduction of roughly 4x compared to 32-bit full precision, significantly reducing the communication bandwidth needed per round.

- **Trade-offs**:
  - Quantization achieves compression at the cost of precision and range, affecting model convergence and accuracy. The project includes mechanisms to evaluate these trade-offs, allowing users to assess the impact of various quantization schemes.

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Torchvision
- Matplotlib
