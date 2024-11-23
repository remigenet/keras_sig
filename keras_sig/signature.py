import logging

logger = logging.getLogger(__name__)

import keras
from keras import ops
from typing import Optional, Union, List

backend = keras.backend.backend()
if backend == 'tensorflow':
    import tensorflow as tf
    Array = tf.Tensor
    has_gpu = len(tf.config.get_visible_devices('GPU'))>0
elif backend == 'jax':
    from jaxtyping import Array, Float
    import jax
    try:
        has_gpu = len(jax.devices('gpu'))
    except:
        has_gpu = False
elif backend == 'torch':
    import torch
    Array = torch.Tensor
    has_gpu = True #As GPU implementation is working anyway better with torch compiler

from keras_sig.tensor_ops import restricted_exp, mult_fused_restricted_exp, batch_restricted_exp, batch_mult_fused_restricted_exp, batch_seq_otimes

# has_alert_useless_unroll = False

def get_largest_divisor_under_20(n: int) -> int:
    """Gets largest divisor under or equal to 20 for scan unrolling optimization.

    Args:
        n: Integer to find divisor for
        
    Returns:
        Largest integer i ≤ 20 that divides n. Returns 10 if no such divisor exists.
    """
    for i in range(20, 0, -1):
        if i <= n and n % i == 0:
            return i
    return 10

def signature(
    path: Array,
    depth: int,
    stream: bool = False,
    unroll:Optional[Union[bool, int]] = None,
    gpu_optimized: Optional[bool] = None,
) -> Array:
    """Computes the signature of a path with automatic dispatch to batched/non-batched variants.
    
    Args:
        path: Array of shape (length, dim) or (batch, length, dim)
        depth: Maximum depth to truncate signature computation
        stream: If True, computed signatures is returned for each steps. Default False
        unroll: Level of unrolling for scan operations. If None, automatically determined. 
               If False, no unrolling.

    Returns:
        If path shape is (length, dim):
            If stream=False: Array of shape (dim + dim² + ... + dim^depth,)
            If stream=True: Array of shape (length-1, dim + dim² + ... + dim^depth)
            
        If path shape is (batch, length, dim):
            If stream=False: Array of shape (batch, dim + dim² + ... + dim^depth)
            If stream=True: Array of shape (batch, length-1, dim + dim² + ... + dim^depth)

    Raises:
        ValueError: If path does not have shape (length, dim) or (batch, length, dim)
    """
    if gpu_optimized is None:
        gpu_optimized = has_gpu
    
    # this is just to handle shape errors
    if path.ndim == 2:
        if backend == 'tensorflow':
            return _tf_single_signature(path, depth=depth, stream=stream)
        else:
            return _single_signature(path, depth=depth, stream=stream, unroll=unroll)
    if path.ndim == 3:  # batch case (mimics signatory)
        if gpu_optimized:
            # if not has_alert_useless_unroll and unroll is not None:
            #     logger.warning("Unroll parameter is ignored when using GPU-optimized implementation as it uses parallel operations instead of scan")
            #     has_alert_useless_unroll = True
            return _batch_signature_gpu(path, depth=depth, stream=stream)
        else:
            if backend == 'tensorflow':
                # if not has_alert_useless_unroll and unroll is not None:
                #     logger.warning("Unroll parameter is ignored when using Tensorflow Backend as its compiler do not handle well the scan")
                return _tf_batch_signature(path, depth=depth, stream=stream)
            else:
                return _batch_signature(path, depth=depth, stream=stream, unroll=unroll)
    msg = f"Path must be of shape (path_length, path_dim) or (batch, path_length, path_dim), got {path.shape}"
    raise ValueError(msg)

def _single_signature(
    path: Array,
    depth: int,
    stream: bool = False,
    unroll: Optional[Union[bool, int]] = False,
) -> Array:
    """Computes signature for single (non-batched) path.
    
    Args:
        path: Array of shape (length, dim)
        depth: Maximum depth to truncate signature computation
        stream: If True, computed signatures is returned for each steps. Default False ## NOT IMPLEMENTED
        unroll: Level of unrolling for scan operations
        
    Returns:
        If stream=False: Array of shape (dim + dim² + ... + dim^depth,)
        If stream=True: Array of shape (length-1, dim + dim² + ... + dim^depth)
    """
    path_increments = path[1:,:] - path[:-1,:]
    exp_term = restricted_exp(path_increments[0], depth=depth)

    carry = exp_term
    if stream:
        stacked =[carry]
    for inc_idx in range(1, path_increments.shape[0]):
        inc = path_increments[inc_idx]
        carry = mult_fused_restricted_exp(inc, carry)
        if stream:
            stacked.append(carry)
        
    if stream:
        res = ops.stack([
            ops.concatenate([ops.reshape(c, (-1,)) for c in res])
            for res in stacked
        ], axis=0)

    else:
        res = carry
        # `res` has shape [(dim,), (dim, dim), ...]
        res = ops.concatenate([ops.reshape(c, (-1,)) for c in res])
           
    return res


def _batch_signature(
    path: Array,
    depth: int,
    stream: bool = False,
    unroll: Optional[Union[bool, int]] = None,
) -> Array:
    """Computes signatures for batched paths using scan operations.
    
    Args:
        path: Array of shape (batch, length, dim)
        depth: Maximum depth to truncate signature computation  
        stream: If True, computed signatures is returned for each steps. Default False
        unroll: Level of unrolling for scan operations. If None, uses largest divisor under 20
        
    Returns:
        If stream=False: Array of shape (batch, dim + dim² + ... + dim^depth)
        If stream=True: Array of shape (batch, length-1, dim + dim² + ... + dim^depth)
    """
    batch_size, seq_len, n_features = path.shape
    path_increments = path[:, 1:] - path[:, :-1]
    exp_term = batch_restricted_exp(path_increments[:, 0], depth=depth)
    unroll_level = unroll if unroll is not None else get_largest_divisor_under_20(seq_len-1)
    
    if not stream:
        # Non-streaming case
        def scan_fn(carry, increment):
            new_carry = batch_mult_fused_restricted_exp(increment, carry)
            return new_carry, None
            
        final_carry, _ = ops.scan(
            scan_fn,
            init=exp_term,
            xs=ops.moveaxis(path_increments[:, 1:], 1, 0),  # Reshape to (steps, batch, dim)
            unroll=unroll_level,
            length=path_increments.shape[1] - 1
        )
        # Match original format: concatenate at the very end
        return ops.concatenate([ops.reshape(c, (batch_size,n_features**(1+idx))) for idx, c in enumerate(final_carry)], axis=1)
    else:
        # Streaming case
        def scan_fn(carry, increment):
            new_carry = batch_mult_fused_restricted_exp(increment, carry)
            return new_carry, new_carry
        
        _, stacked = ops.scan(
            scan_fn,
            init=exp_term,
            xs=ops.moveaxis(path_increments[:, 1:], 1, 0),  # Shape: (batch, steps, dim)
            unroll=unroll_level,
            length=path_increments.shape[1] - 1
        )
        res = [
            ops.concatenate([first[None, ...], rest], axis=0)
            for first, rest in zip(exp_term, stacked)
        ]
        # Match original stacking and concatenation
        return ops.concatenate([ops.reshape(ops.moveaxis(r, 1, 0), (batch_size, seq_len-1, n_features**(1+idx))) for idx, r in enumerate(res)], axis=2)

#Needed a tensorflow variant without scan 

def _tf_single_signature(
    path,
    depth: int,
    stream: bool = False,
) -> Array:
    """TensorFlow-specific signature computation for single (non-batched) path.
    
    Implementation without using scan operations.
    
    Args:
        path: Array of shape (length, dim)
        depth: Maximum depth to truncate signature computation
        stream: If True, computed signatures is returned for each steps.
        
    Returns:
        If stream=False: Array of shape (dim + dim² + ... + dim^depth,)
        If stream=True: Array of shape (length-1, dim + dim² + ... + dim^depth)
    """
    path_increments = path[1:,:] - path[:-1,:]
    exp_term = restricted_exp(path_increments[0], depth=depth)

    carry = exp_term
    if stream:
        stacked =[carry]
    for inc_idx in range(1, path_increments.shape[0]):
        inc = path_increments[inc_idx]
        carry = mult_fused_restricted_exp(inc, carry)
        if stream:
            stacked.append(carry)
        
    if stream:
        res = ops.stack([
            ops.concatenate([ops.reshape(c, (-1,)) for c in res])
            for res in stacked
        ], axis=0)

    else:
        res = carry
        # `res` has shape [(dim,), (dim, dim), ...]
        res = ops.concatenate([ops.reshape(c, (-1,)) for c in res])
           
    return res

def _tf_batch_signature(
    path,
    depth: int,
    stream: bool = False,
) -> Array:
    """TensorFlow-specific signature computation for batched paths.
    
    Implementation without using scan operations.
    
    Args:
        path: Array of shape (batch, length, dim) 
        depth: Maximum depth to truncate signature computation
        stream: If True, computed signatures is returned for each steps.
        
    Returns:
        If stream=False: Array of shape (batch, dim + dim² + ... + dim^depth)
        If stream=True: Array of shape (batch, length-1, dim + dim² + ... + dim^depth)
    """
    batch_size = tf.shape(path)[0]
    n_features = tf.shape(path)[2]
    path_increments = path[:,1:] - path[:,:-1]
    exp_term = batch_restricted_exp(path_increments[:, 0], depth=depth)
    
    carry = exp_term
    if stream:
        stacked =[carry]
    for inc_idx in range(1, path_increments.shape[1]):
        inc = path_increments[:, inc_idx]
        carry = batch_mult_fused_restricted_exp(inc, carry)   
        if stream:
            stacked.append(carry)
    if stream:
        res = ops.stack([
            ops.concatenate([ops.reshape(c, (-1,n_features**(idx+1))) for idx, c in enumerate(res)], axis=1)
            for res in stacked
        ], axis=1)
    else:
        res = carry
        res = ops.concatenate([ops.reshape(c, (-1,n_features**(idx+1))) for idx, c in enumerate(res)], axis=1)
    return res

def _batch_signature_gpu(
    path: Array,
    depth: int,
    stream: bool = False,
) -> Array:
    """Computes batched signature optimized for GPU execution.

    A memory-intensive but computationally efficient implementation that:
    - Replaces sequential scan operations with parallel matrix operations
    - Pre-computes path increment divisions 
    - Uses cumulative sums for parallel sequence processing
    - Trades increased memory usage for reduced sequential operations

    Best suited when:
    - GPU VRAM can accommodate larger intermediate tensors
    - Batch/sequence sizes benefit from parallel processing
    - Computation speed is prioritized over memory efficiency

    Args:
        path: Array of shape (batch, seq_len, features)
        depth: Maximum signature truncation depth
        stream: If True, returns signatures at each timestep

    Returns:
        If stream=False: Array of shape (batch, features + features^2 + ... + features^depth)
        If stream=True: Array of shape (batch, seq_len-1, features + features^2 + ... + features^depth)
    """
    batch_size, seq_len, n_features = path.shape
    if batch_size is None: #For tensorflow
        batch_size=-1
    path_increments = path[:, 1:] - path[:, :-1]

    stacked = [ops.cumsum(path_increments, axis=1)]

    exp_term = batch_restricted_exp(path_increments[:, 0], depth=depth)

    path_increment_divided = ops.stack([path_increments / i for i in range(2, depth+1)], axis=0)

    for depth_index in range(1, depth):
        current = stacked[0][:,:-1] + path_increment_divided[depth_index-1,:,1:]
        for j in range(depth_index-1):
            current = stacked[j+1][:,:-1] + batch_seq_otimes(current, path_increment_divided[depth_index-j-2,:,1:])
        current = batch_seq_otimes(current, path_increments[:,1:])
        current = ops.concatenate([ops.expand_dims(exp_term[depth_index], axis=1), current], axis=1)
        stacked.append(ops.cumsum(current, axis=1))

    if not stream:
        return ops.concatenate([ops.reshape(c[:,-1], (batch_size,n_features**(1+idx))) for idx, c in enumerate(stacked)], axis=1)
    else:
        return ops.concatenate([ops.reshape(r, (batch_size, seq_len-1, n_features**(1+idx))) for idx, r in enumerate(stacked)], axis=2)