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


def generate_random_tensor(shape):
    return random.normal(shape=shape, dtype=backend.floatx())

@pytest.fixture(params=[2, 3, 4])  # Test different signature depths
def depth(request):
    """Fixture providing different signature computation depths.
    
    Args:
        request: pytest request object containing the parameter value
    
    Returns:
        int: The signature computation depth to test with
    """
    return request.param

@pytest.fixture(params=[False, True])  # Test different signature depths
def stream(request):
    """Fixture providing different signature computation depths.
    
    Args:
        request: pytest request object containing the parameter value
    
    Returns:
        int: The signature computation depth to test with
    """
    return request.param

@pytest.fixture(params=[True, False])  # Test different signature depths
def jit_compile(request):
    """Fixture providing different signature computation depths.
    
    Args:
        request: pytest request object containing the parameter value
    
    Returns:
        int: The signature computation depth to test with
    """
    return request.param

def test_siglayer_forward(depth, stream):
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


def test_siglayer_training(jit_compile):
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 320, 10, 8
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
    assert keras.backend.backend() == BACKEND
    depth = 3
    sig_layer = SigLayer(depth=depth)
    config = sig_layer.get_config()
    sig_layer_reconstructed = SigLayer.from_config(config)

    assert sig_layer.depth == sig_layer_reconstructed.depth, "Depth should be the same after serialization"

def test_siglayer_serialization_in_model():
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
