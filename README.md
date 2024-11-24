# keras_sig: Very Fast Path Signature Computation for Keras

This package started as backend-agnostic Keras implementation of path signature computations, focusing on simplicity and ease of integration.
Since we proposed a GPU-optimized computation methods that leverages fully parallel operations, it has become the fastest and most efficient path signature computation package available at date. This method is available either in full Keras for model training, but also as a standalone JAX function for direct computation.

## Overview

`keras_sig` provides path signature computations as a Keras layer. It aims to offer:

- Native Keras implementation supporting all backends (JAX, PyTorch, TensorFlow)
- Simple integration within Keras models
- Pure Python implementation avoiding C++ dependencies
- Consistent API across different backends
- GPU-optimized computation for faster training

The package builds upon several key projects in the signature computation ecosystem:

### Historical Context

1. **iisignature** ([repo](https://github.com/bottler/iisignature/)): The foundational C++ implementation providing highly optimized signature computations with a python wrapper
2. **signatory** ([repo](https://github.com/patrick-kidger/signatory)): A PyTorch-specific implementation using C++ level optimizations for GPU acceleration
3. **iisignature-tensorflow-2** ([repo](https://github.com/remigenet/iisignature-tensorflow-2/)): An attempt at wrapping iisignature for TensorFlow 2, which faced limitations with model compilation
4. **signax** ([repo](https://github.com/anh-tong/signax)): A breakthrough pure JAX implementation showing that C++ optimization could be avoided
5. **keras_sig** (this package): Bringing the pure Python approach to all Keras backends and optimizing further the computation for the GPU.

## Installation

```bash
pip install keras_sig
```

Or install from source:
```bash
git clone https://github.com/yourusername/keras_sig
cd keras_sig
pip install -e .
```

## Quick Start

Basic usage with Keras:
```python
import keras
from keras_sig import SigLayer

model = keras.Sequential([
    keras.layers.Input(shape=(timesteps, features)),
    SigLayer(depth=3, stream=False, gpu_optimized=True),  # Enable GPU optimization
    keras.layers.Dense(output_dim)
])
```

Direct JAX computation (fastest option):
```python
from keras_sig import jax_gpu_signature

# Pre-compiled GPU-optimized computation
signatures = jax_gpu_signature(paths, depth=3, stream=False)
```

## Performance & Implementation Options

### Computation Methods

1. **GPU-Optimized** (Recommended when GPU available)
   - Uses parallel operations instead of loops
   - 5x faster than standard implementation
   - Higher memory usage
   - Enable with `gpu_optimized=True` (automaticaly selected if GPU detected) or use `jax_gpu_signature`
   
2. **Standard Implementation**
   - Loop-based computation with scan operations
   - Lower memory footprint
   - Better for CPU-only systems
   - Default when GPU unavailable

### Performance Benchmarks

All benchmarks run on AMD EPYC-7302P 16-cores with RTX-3090.

#### Forward Pass (128 batch, 100 sequence, 5 features, depth 4)

| Backend | Version | GPU Time | CPU Time |
|---------|---------|----------|-----------|
| JAX | Pure Jax-GPU function | 163µs | 46.5ms |
| JAX | keras Standard | 713ms | 378ms |
| JAX | keras GPU-optimized | - | 80.5ms |
| JAX | signax | 668µs | 11.7ms |
| TensorFlow | keras GPU-optimized | 55.2ms | 180ms |
| TensorFlow | keras Standard | 375ms | 317ms |
| Torch | keras GPU-optimized | 2.84ms | 50.6ms |
| Torch | keras Standard | 92.4ms | 91.4ms |
| None | iisignature | 36.4ms | 36.4ms |

Here the Keras version are not performing optimally as direct Jax function because the keras operation are not runned on GPU nor compiled with jit. This phase is only happening at training time.
However we can easily compare the performance of the Pure Jax function with signax and iisignature and see that our proposed approach is the fastest when a GPU is available.
When no GPU is available, the standard version is very similar to the signax implementation.

#### Training Performance

Test conditions: We created a model following the [SigKAN paper](https://arxiv.org/abs/2406.17890) the following way
```python
model = keras.Sequential([
    Input(shape=X.shape[1:]),
    Dense(7),
    SigDense(10, depth, SigLayer),
    Flatten(),
    Dense(10, 'relu'),
    Dense(n_ahead),
])
```
and trained it with jit_compilation enable when possible for 10 epochs with Adam optimizer on randomly generated datas.

##### Long Sequences (length=500)
| Backend | Version | Compile Time (GPU) | Compile Time (CPU) | Step Time (GPU) | Step Time (CPU) |
|---------|---------|-------------------|-------------------|----------------|----------------|
| JAX | GPU-opt | 5s | 25s | 2ms | 213ms |
| JAX | Standard | 7s | 14s | 14ms | 108ms |
| JAX | Signax | 6s | 12s | 14ms | 83ms |
| TensorFlow | GPU-opt | 9s | 26s | 2ms | 214ms |
| TensorFlow | Standard | Compile fail | Compile fail | - | - |
| TensorFlow | iisignature | No compile | No compile | 340-345ms | 340-345ms |
| Torch | GPU-opt | 53s | 26s | 21ms | 218ms |
| Torch | Standard | No compile | No compile | 590ms | 643ms |

##### Short Sequences (length=20)
| Backend | Version | Compile Time (GPU) | Compile Time (CPU) | Step Time (GPU) | Step Time (CPU) |
|---------|---------|-------------------|-------------------|----------------|----------------|
| JAX | GPU-opt | 4s | 6s | 1ms | 19ms |
| JAX | Standard | 8s | 8s | 1ms | 9ms |
| JAX | signax | 5s | 4s | 2ms | 6ms |
| TensorFlow | GPU-opt | 4s | 13s | 1ms | 102ms |
| TensorFlow | Standard | 19s | 14s | 2ms | 28ms |
| TensorFlow | iisignature | No compile | No compile | 27ms | 27ms |
| Torch | GPU-opt | 9s | 8s | 21ms | 17ms |
| Torch | Standard | No compile | No compile | 38ms | 31ms |



Key Findings:
1. Pure JAX GPU-optimized version is fastest for forward pass (4x faster than signax)
2. GPU-optimized variants excel with GPU availability across all backends
3. For training:
   - JAX: Best balance of compilation/execution
   - TensorFlow: GPU-optimized version required for long sequences
   - PyTorch: Longer compilation but good runtime with GPU-optimization
4. Standard implementations struggle with:
   - PyTorch: Compilation issues
   - TensorFlow: Long sequence compilation
   - All backends: Slower execution without GPU optimization

### Implementation Recommendations

1. **JAX + GPU (Best Overall)**
   - Use pure JAX implementation for forward pass
   - Use GPU-optimized SigLayer for training

2. **PyTorch + GPU**
   - Use GPU-optimized version only
   - Expect longer compilation times

3. **TensorFlow + GPU**
   - Use GPU-optimized version
   - Avoid standard version for long sequences

4. **CPU-Only Systems**
   - JAX standard implementation offers best balance
   - GPU-optimized versions still usable but with performance penalty

## Features

Currently implements:
- Standard signature computations
- Support for both streaming and non-streaming modes
- Configurable signature depth
- Backend-agnostic implementation

Not yet implemented (available in other packages):
- Log signatures
- Lyndon words
- Other advanced signature computations

## Citations

If using this package, please cite both this work and the foundational packages that inspired it:

```bibtex
@article{reizenstein2017iisignature,
  title={iisignature: A python package for computing iterated-integral signatures},
  author={Reizenstein, Jeremy},
  journal={Journal of Open Source Software},
  volume={2},
  number={10},
  pages={189},
  year={2017}
}

@article{kidger2021signatory,
  title={Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU},
  author={Kidger, Patrick and Lyons, Terry},
  journal={International Conference on Learning Representations},
  year={2021}
}

@software{signax2024github,
  author = {Anh Tong},
  title = {signax: Path Signatures in JAX},
  url = {https://github.com/anh-tong/signax},
  year = {2024},
}

@software{genet2024iisignaturetf2,
  author = {Remi Genet, Hugo Inzirillo},
  title = {iisignature-tensorflow-2: TensorFlow 2 Wrapper for iisignature},
  url = {https://github.com/remigenet/iisignature-tensorflow-2},
  year = {2024},
}
```

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

Would you like me to adjust any section or add more details?
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg