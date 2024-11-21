from functools import partial

import keras
from keras_sig import signature


class SigLayer(keras.layers.Layer):
    def __init__(self, depth, stream = False, unroll = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.stream = stream
        self.unroll = unroll
        self.signature_func = partial(signature, depth=depth, stream=stream, unroll=unroll)
        
    def call(self, inputs):
        return self.signature_func(inputs)

