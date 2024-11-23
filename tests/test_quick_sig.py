
import pytest

from keras_sig import jax_gpu_signature
import signax
import numpy as np


def generate_random_tensor(shape):
    """Generates random tensor for testing.
    
    Creates normally distributed random array using numpy's random generator
    
    Args:
        shape: Tuple defining the shape of the tensor to generate
        
    Returns:
        Array: Random tensor of specified shape
    """
    return np.random.randn(*shape)

@pytest.fixture(params=[2, 3, 4])  # Test different signature depths
def depth(request):
    """Fixture providing different signature depths for testing.
    
    Parametrizes tests with signature depths of 2, 3, and 4 to ensure 
    the signature computation works correctly at different truncation levels.
    
    Args:
        request: pytest request object containing the parameter value
    
    Returns:
        int: Signature computation depth (2, 3, or 4)
    """
    return request.param

@pytest.fixture(params=[False, True])  # Test different signature depths
def stream(request):
    """Fixture for testing streaming vs non-streaming signature computation.
    
    Parametrizes tests to run with both streaming and non-streaming modes 
    to verify both computational approaches work correctly.
    
    Args:
        request: pytest request object containing the parameter value
    
    Returns:
        bool: If True, compute streaming signatures; if False, compute terminal signatures
    """
    return request.param

def test_batch_signature_pure_jax(depth, stream):
    batch_size, time_steps, features = 32, 10, 5

    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output_signature = jax_gpu_signature(input_sequence, depth=depth, stream=stream)

    if stream:
        target_output_shape = (batch_size, time_steps-1, sum([features ** i for i in range(1, depth + 1)]))
    else:
        target_output_shape = (batch_size, sum([features ** i for i in range(1, depth + 1)]))
    
    assert output_signature.shape == target_output_shape, f"Expected shape {target_output_shape}, but got {output_signature.shape}"

    signax_sig = signax.signature(input_sequence, depth=depth, stream=stream)

    assert np.max(np.abs(output_signature - signax_sig)) < 1e-3, "Output should match signax signature"