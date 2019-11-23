from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers

class MaskOfMaxPooling(layers.Layer):
    def __init__(self):
        super(MaskOfMaxPooling, self).__init__()

    def call(self,inputs):
        out = inputs[0] >= inputs[1]
        return tf.cast(out,tf.float32)