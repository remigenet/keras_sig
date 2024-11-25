"""Pure JAX implementation of memory-intensive but GPU-optimized signature computation.

This standalone module provides a highly efficient JAX implementation optimized for
scenarios where:
- GPU memory is sufficient for large intermediate tensors
- Maximum computational speed is required
- Pure forward pass computation is needed (vs. keras-sig's model integration)

Can serve as a faster alternative to signax when GPU memory constraints allow.
Trades increased VRAM usage for superior computational efficiency through 
parallel operations.
"""

import jax
import jax.numpy as jnp
from typing import List, Union, Optional
from functools import partial
from jaxtyping import Array

@jax.jit
def batch_otimes_pure_jax(x: Array, y: Array) -> Array:
    """GPU-optimized batched tensor product preserving batch dimension."""
    xdim = x.ndim
    ydim = y.ndim
    for i in range(ydim - 1):
        x = jnp.expand_dims(x, axis=-1)
    for i in range(xdim - 1):
        y = jnp.expand_dims(y, axis=1)
    return x * y

@jax.jit 
def batch_seq_otimes_pure_jax(x: Array, y: Array) -> Array:
    """GPU-optimized tensor product preserving both batch and sequence dimensions."""
    xdim = x.ndim
    ydim = y.ndim
    for i in range(ydim - 2):
        x = jnp.expand_dims(x, axis=-1)
    for i in range(xdim - 2):
        y = jnp.expand_dims(y, axis=2)
    return x * y

@partial(jax.jit, static_argnames="depth")
def batch_restricted_exp_pure_jax(input: Array, depth: int) -> list[Array]:
    """Computes restricted exponential with full GPU parallelization."""
    ret = [input]
    for i in range(2, depth + 1):
        ret.append(batch_otimes_pure_jax(ret[-1], input / i))
    return ret

@partial(jax.jit, static_argnames=["depth", "stream"])
def batch_signature_pure_jax(path: Array, depth: int, stream: bool = False) -> Array:
    """Highly optimized signature computation maximizing GPU parallelism.

    Key optimizations:
    - Pre-computes scaled increments for all depths
    - Uses cumsum for parallel sequence processing
    - Replaces sequential operations with parallel matrix ops
    - Fully JIT-compiled for maximum GPU utilization
    """
    batch_size, seq_len, n_features = path.shape
    path_increments = path[:, 1:] - path[:, :-1]

    stacked = [jnp.cumsum(path_increments, axis=1)]

    exp_term = batch_restricted_exp_pure_jax(path_increments[:, 0], depth=depth)

    path_increment_divided = jnp.stack([path_increments / i for i in range(2, depth+1)], axis=0)

    for depth_index in range(1, depth):
        current = stacked[0][:,:-1] + path_increment_divided[depth_index-1,:,1:]
        for j in range(depth_index-1):
            current = stacked[j+1][:,:-1] + batch_seq_otimes_pure_jax(current, path_increment_divided[depth_index-j-2,:,1:])
        current = batch_seq_otimes_pure_jax(current, path_increments[:,1:])
        current = jnp.concatenate([jnp.expand_dims(exp_term[depth_index], axis=1), current], axis=1)
        stacked.append(jnp.cumsum(current, axis=1))

    if not stream:
        return jnp.concatenate([jnp.reshape(c[:,-1], (batch_size,n_features**(1+idx))) for idx, c in enumerate(stacked)], axis=1)
    else:
        return jnp.concatenate([jnp.reshape(r, (batch_size, seq_len-1, n_features**(1+idx))) for idx, r in enumerate(stacked)], axis=2)
