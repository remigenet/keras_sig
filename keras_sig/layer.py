from functools import partial
from typing import Optional, Union
import keras
from keras_sig import signature

backend = keras.backend.backend()
if backend == 'tensorflow':
    import tensorflow as tf
    Array = tf.Tensor
elif backend == 'jax':
    from jaxtyping import Array, Float
elif backend == 'torch':
    import torch
    Array = torch.Tensor

class SigLayer(keras.layers.Layer):
    """A Keras layer that computes path signatures with optional GPU optimization.

    This layer transforms input path data into its signature representation up to a specified depth.
    It can handle both single paths and batches of paths, with optional streaming computation.

    Two computation methods are available:
    - Standard: Uses scan operations with configurable unrolling
    - GPU-optimized: Uses parallel operations trading memory for speed (recommended when GPU available)

    Attributes:
        depth (int): Maximum depth for truncating signature computation
        stream (bool): If True, computes signatures for each prefix of the path
        unroll (Optional[Union[bool, int]]): Level of unrolling for scan operations (ignored if gpu_optimized=True)
            - If int: Uses specified unroll level
            - If True: Uses default unroll level
            - If False: No unrolling
            - If None: Automatically determines optimal unroll level
        gpu_optimized (Optional[bool]): Whether to use GPU-optimized computation
            - If True: Uses parallel operations (higher memory, faster computation)
            - If False: Uses sequential scan operations
            - If None: Auto-detects based on backend and GPU availability
    """
   
    def __init__(self, 
                depth: int, 
                stream: bool = False,
                unroll: Optional[Union[bool, int]] = None,
                gpu_optimized: Optional[bool] = None,
                *args, **kwargs):
        """Initializes the signature layer.

        Args:
            depth: Maximum depth for truncating signature computation
            stream: If True, computes signatures for each prefix. Default False
            unroll: Controls unrolling of scan operations (ignored if gpu_optimized=True). Default None
            gpu_optimized: If True, uses memory-intensive but faster parallel operations.
                        If None, auto-detects based on backend and GPU availability.
            *args: Additional positional arguments passed to keras.layers.Layer
            **kwargs: Additional keyword arguments passed to keras.layers.Layer
        """
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.stream = stream
        self.unroll = unroll
        self.gpu_optimized = gpu_optimized
        self.signature_func = partial(signature, 
                                    depth=depth, 
                                    stream=stream, 
                                    unroll=unroll,
                                    gpu_optimized=gpu_optimized)
    
    def call(self, inputs: Array) -> Array:
        """Computes the signature transform of the input paths.
        
        Args:
            inputs: Path data with shape:
                - (length, dim) for single path
                - (batch, length, dim) for batched paths
                where:
                    - length is the number of points in each path
                    - dim is the dimension of the space
                    - batch is the batch size
        
        Returns:
            Signature transforms with shape:
            For single path (length, dim):
                If stream=False: (dim + dim² + ... + dim^depth,)
                If stream=True: (length-1, dim + dim² + ... + dim^depth)
                
            For batched paths (batch, length, dim):
                If stream=False: (batch, dim + dim² + ... + dim^depth)
                If stream=True: (batch, length-1, dim + dim² + ... + dim^depth)
                
        Note:
            The output size grows exponentially with both the input dimension 
            and the signature depth.
        """
        return self.signature_func(inputs)