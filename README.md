# keras_sig: Easy Path Signature in Keras 

A backend-agnostic Keras implementation of path signature computations, focusing on simplicity and ease of integration.

## Overview

`keras_sig` provides path signature computations as a Keras layer. It aims to offer:

- Native Keras implementation supporting all backends (JAX, PyTorch, TensorFlow)
- Simple integration within Keras models
- Pure Python implementation avoiding C++ dependencies
- Consistent API across different backends

The package builds upon several key projects in the signature computation ecosystem:

### Historical Context

1. **iisignature** ([repo](https://github.com/bottler/iisignature/)): The foundational C++ implementation providing highly optimized signature computations
2. **signatory** ([repo](https://github.com/patrick-kidger/signatory)): A PyTorch-specific implementation using C++ level optimizations
3. **iisignature-tensorflow-2** ([repo](https://github.com/remigenet/iisignature-tensorflow-2/)): An attempt at wrapping iisignature for TensorFlow 2, which faced limitations with model compilation
4. **signax** ([repo](https://github.com/anh-tong/signax)): A breakthrough pure JAX implementation showing that C++ optimization could be avoided
5. **keras_sig** (this package): Bringing the pure Python approach to all Keras backends

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

```python
import keras
from keras_sig import SigLayer

# Create a model with signature computation
model = keras.Sequential([
    keras.layers.Input(shape=(timesteps, features)),
    SigLayer(depth=3, stream=False),
    keras.layers.Dense(output_dim)
])

# Use it like any other Keras layer
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=10)
```

## Performance Considerations

### Backend-Specific Performance

- **JAX**: Best performance and most stable for compilation
- **PyTorch**: Good performance at simple forward pass, compilation often fails but works fine without (but is much slower than JAX)
- **TensorFlow**: Bad performance at simple forward pass, but is able to compile and run with JIT (so will ran quicker than torch, but slower than JAX) - main issue is that due to the non-ability to use keras.ops.scan in tensorflow the loop unrolling is not handled directly and for long sequence the compilation can be very problematic. For long sequence go to jax..

For raw forward pass computation:
- Faster than iisignature for big batch/sequence sizes, a bit slower for small sizes
- Slower than signax (due to signax's pure JAX JIT compilation and possible use of GPU, while keras only compile at train time), but similar to it when it is at train time

### Training Performance

When integrated in training loops:
- Comparable efficiency to signax in JAX with compatibility with all backend
- JIT compilation available during training
- Seamless integration with Keras training features

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

@software{signax2023github,
  author = {Anh Tong},
  title = {signax: Path Signatures in JAX},
  url = {https://github.com/anh-tong/signax},
  year = {2023},
}

@misc{genet2023iisignaturetf2,
  author = {Remi Genet, Hugo Inzirillo},
  title = {iisignature-tensorflow-2: TensorFlow 2 Wrapper for iisignature},
  url = {https://github.com/remigenet/iisignature-tensorflow-2},
  year = {2023},
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