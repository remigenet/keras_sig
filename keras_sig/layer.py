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
    """A Keras layer that computes path signatures.
    
    This layer transforms input path data into its signature representation up to a specified depth.
    It can handle both single paths and batches of paths, with optional streaming computation.
    
    Attributes:
        depth (int): Maximum depth for truncating signature computation
        stream (bool): If True, computes signatures for each prefix of the path
        unroll (Optional[Union[bool, int]]): Level of unrolling for scan operations
            - If int: Uses specified unroll level
            - If True: Uses default unroll level
            - If False: No unrolling
            - If None: Automatically determines optimal unroll level
            
    Example:
        ```python
        # Create a signature layer with depth 3
        sig_layer = SigLayer(depth=3)
        
        # For a single path
        path = keras.random.normal(shape=(100, 2))  # path of length 100 in 2D
        sig = sig_layer(path)  # Computes signature up to depth 3
        
        # For batched paths
        paths = keras.random.normal(shape=(32, 100, 2))  # 32 paths
        sigs = sig_layer(paths)  # Computes signatures for all paths
        ```
    """
    
    def __init__(self, 
                 depth: int, 
                 stream: bool = False,
                 unroll: Optional[Union[bool, int]] = None,
                 *args, **kwargs):
        """Initializes the signature layer.
        
        Args:
            depth: Maximum depth for truncating signature computation
            stream: If True, computes signatures for each prefix. Default False
            unroll: Controls unrolling of scan operations. Default None
            *args: Additional positional arguments passed to keras.layers.Layer
            **kwargs: Additional keyword arguments passed to keras.layers.Layer
        """
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.stream = stream
        self.unroll = unroll
        self.signature_func = partial(signature, 
                                    depth=depth, 
                                    stream=stream, 
                                    unroll=unroll)
    
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