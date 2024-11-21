import keras
from keras import ops

backend = keras.backend.backend()
if backend == 'tensorflow':
    import tensorflow as tf
    Array = tf.Tensor
elif backend == 'jax':
    from jaxtyping import Array, Float
elif backend == 'torch':
    import torch
    Array = torch.Tensor


def otimes(x: Array, y: Array) -> Array:
    """Tensor product

    Args:
        x: size=(n,n,...,n), ndim=ndim_x
        y: size=(n,n,...,n), ndim=ndim_y
    Return:
        Tensor size (n,n,...,n) with ndim=ndim_x + ndim_y
    """
    xdim = x.ndim
    ydim = y.ndim
    for i in range(ydim):
        x = ops.expand_dims(x, axis=-1)
    for i in range(xdim):
        y = ops.expand_dims(y, axis=0)
    return x * y

def addcmul(x: Array, y: Array, z: Array) -> Array:
    """Similar to `torch.addcmul` returning
        x + y * z
    Here `*` is the tensor product
    """
    return x + otimes(y, z)


def restricted_exp(input: Array, depth: int) -> list[Array]:
    """Restricted exponentiate

    As `depth` is fixed so we can make it as a static argument.
    This allows us to `jit` this function
    Args:
        input: shape (n, )
        depth: the depth of signature
    Return:
        A list of `jnp.ndarray` contains tensors
    """
    ret = [input]
    for i in range(2, depth + 1):
        ret.append(otimes(ret[-1], input / i))
    return ret


def mult_fused_restricted_exp(z: Array, A: list[Array]) -> list[Array]:
    """
    Multiply-fused-exponentiate

    Args:
        z: shape (n,)
        A: a list of `jnp.array` [(n, ), (n x n), (n x n x n), ...]
    Return:
        A list of which elements have the same shape with `A`
    """
    depth = len(A)
    ret = []

    for depth_index in range(depth):
        current = ops.array(1.0, dtype=z.dtype)
        for i in range(depth_index + 1):
            current = addcmul(x=A[i], y=current, z=z / (depth_index + 1 - i))
        ret.append(current)

    return ret

def batch_otimes(x: Array, y: Array) -> Array:
    """Tensor product

    Args:
        x: size=(n,n,...,n), ndim=ndim_x
        y: size=(n,n,...,n), ndim=ndim_y
    Return:
        Tensor size (n,n,...,n) with ndim=ndim_x + ndim_y
    """
    xdim = x.ndim
    ydim = y.ndim
    for i in range(ydim - 1):
        x = ops.expand_dims(x, axis=-1)
    for i in range(xdim - 1):
        y = ops.expand_dims(y, axis=1)
    return x * y
    
def batch_restricted_exp(input: Array, depth: int) -> list[Array]:
    """Restricted exponentiate

    As `depth` is fixed so we can make it as a static argument.
    This allows us to `jit` this function
    Args:
        input: shape (n, )
        depth: the depth of signature
    Return:
        A list of `jnp.ndarray` contains tensors
    """
    ret = [input]
    for i in range(2, depth + 1):
        ret.append(batch_otimes(ret[-1], input / i))
    return ret

def batch_mult_fused_restricted_exp(z: Array, A: list[Array]) -> list[Array]:
    """
    Multiply-fused-exponentiate

    Args:
        z: shape (n,)
        A: a list of `jnp.array` [(n, ), (n x n), (n x n x n), ...]
    Return:
        A list of which elements have the same shape with `A`
    """
    depth = len(A)
    ret = []

    for depth_index in range(depth):
        current = ops.array(1.0, dtype=z.dtype)
        for i in range(depth_index + 1):
            current = batch_addcmul(x=A[i], y=current, z=z / (depth_index + 1 - i))
        ret.append(current)

    return ret

def batch_addcmul(x: Array, y: Array, z: Array) -> Array:
    """Similar to `torch.addcmul` returning
        x + y * z
    Here `*` is the tensor product
    """
    return x + batch_otimes(y, z)