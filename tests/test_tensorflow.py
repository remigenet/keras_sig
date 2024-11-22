import os
import tempfile
BACKEND = 'tensorflow' 
os.environ['KERAS_BACKEND'] = BACKEND

import pytest
import keras
from keras import ops
from keras import backend
from keras import random
from keras.models import Model, load_model
from keras_sig import SigLayer
from keras.layers import Dense, Input

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

@pytest.fixture(params=[True, False])  # Test different signature depths
def jit_compile(request):
    """Fixture for testing JIT compilation settings.
    
    Parametrizes tests to run with and without JIT compilation to ensure 
    the layer works correctly under both execution modes.
    
    Args:
        request: pytest request object containing the parameter value
    
    Returns:
        bool: If True, enable JIT compilation; if False, use standard execution
    """
    return request.param

def test_siglayer_forward(depth, stream):
    """Tests forward pass of SigLayer against signax reference implementation.
    
    Verifies that:
    1. SigLayer produces output tensors of correct shape
    2. Shape calculations properly account for streaming vs non-streaming modes
    3. Backend is correctly set to JAX
    4. Output values match signax's signature computation within tolerance
    
    Args:
        depth: Signature computation depth from fixture
        stream: Streaming mode flag from fixture
    
    The test:
    - Creates random input sequences
    - Computes signatures using both SigLayer and signax
    - Verifies shape correctness
    - Compares outputs element-wise with 1e-3 tolerance
    
    Raises:
        AssertionError: If output shape doesn't match expected shape,
                       backend is incorrect, or if outputs differ 
                       from signax by more than tolerance
    """
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 5
    
    sig_layer = SigLayer(depth=depth, stream=stream)

    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output_signature = sig_layer(input_sequence)

    if stream:
        target_output_shape = (batch_size, time_steps-1, sum([features ** i for i in range(1, depth + 1)]))
    else:
        target_output_shape = (batch_size, sum([features ** i for i in range(1, depth + 1)]))
    assert output_signature.shape == target_output_shape, f"Expected shape {target_output_shape}, but got {output_signature.shape}"

    signax_sig = signax.signature(input_sequence, depth=depth, stream=stream)
    if keras.backend.backend() == 'torch':
        output_signature = output_signature.detach().numpy()
    assert ops.max(ops.abs(output_signature - signax_sig)) < 1e-3, "Output should match signax signature"


def test_siglayer_training(jit_compile):
    """Tests SigLayer in training configuration with different compilation modes.
    
    Verifies that:
    1. SigLayer can be integrated into a Keras model
    2. Model successfully trains with SigLayer
    3. Loss is properly tracked during training
    4. Works with both JIT and non-JIT compilation
    
    Args:
        jit_compile: JIT compilation flag from fixture
    
    Raises:
        AssertionError: If model fails to train or loss history is missing
    """
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 325, 10, 8
    output_len = 5

    model = keras.Sequential([
        keras.layers.Input(shape=(time_steps, features)),
        SigLayer(depth=3),
        keras.layers.Dense(output_len)
    ])

    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output_target = generate_random_tensor((batch_size, output_len))
    
    model.compile(optimizer='adam', loss='mse', jit_compile=jit_compile)
    history = model.fit(input_sequence, output_target, epochs=2, batch_size=16, verbose=0)

    assert 'loss' in history.history, "Model should train successfully"

def test_siglayer_serialization_alone():
    """Tests serialization and deserialization of standalone SigLayer.
    
    Verifies that:
    1. SigLayer can be converted to config
    2. SigLayer can be reconstructed from config
    3. Reconstructed layer maintains original parameters
    4. Backend is correctly set
    
    Raises:
        AssertionError: If serialization/deserialization fails or parameters don't match
    """
    assert keras.backend.backend() == BACKEND
    depth = 3
    sig_layer = SigLayer(depth=depth)
    config = sig_layer.get_config()
    sig_layer_reconstructed = SigLayer.from_config(config)

    assert sig_layer.depth == sig_layer_reconstructed.depth, "Depth should be the same after serialization"

def test_siglayer_serialization_in_model():
    """Tests end-to-end serialization of model containing SigLayer.
    
    Verifies that:
    1. Model with SigLayer can be saved and loaded
    2. Predictions remain consistent after loading
    3. Loaded model can continue training
    4. Backend is correctly set
    
    The test:
    - Creates a model with SigLayer
    - Trains it briefly
    - Saves and loads the model
    - Verifies predictions match
    - Continues training loaded model
    
    Raises:
        AssertionError: If model serialization fails or predictions don't match
    """
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    units = 16

    # Create and compile the model
    inputs = Input(shape=(time_steps, features))
    sig_layer = SigLayer(depth=3)
    outputs = keras.layers.Dense(units)(sig_layer(inputs))
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    # Generate some random data
    x_train = generate_random_tensor((batch_size, time_steps, features))
    y_train = generate_random_tensor((batch_size, units))

    # Train the model
    model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=False)

    # Get predictions before saving
    predictions_before = model.predict(x_train, verbose=False)

    # Save the model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'sig_model.keras')
        model.save(model_path)

        # Load the model
        loaded_model = load_model(model_path)

    # Get predictions after loading
    predictions_after = loaded_model.predict(x_train, verbose=False)

    # Compare predictions
    assert ops.all(ops.equal(predictions_before, predictions_after)), "Predictions should be the same after loading"

    # Test that the loaded model can be used for further training
    loaded_model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=False)

    print("Sig model successfully saved, loaded, and reused.")
