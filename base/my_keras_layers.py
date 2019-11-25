from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import numpy as np

class MaskOfMaxPooling(layers.Layer):
    def __init__(self):
        super(MaskOfMaxPooling, self).__init__()

    def call(self,inputs):
        out = inputs[0] >= inputs[1]
        return tf.cast(out,tf.float32)

class UnPooling(layers.Layer):
    def __init__(self):
        super(UnPooling, self).__init__()

    def call(self, inputs, pool_size=2, strides=2):

        before_pooling = inputs[0]
        after_pooling = inputs[1]
        rindex = tf.range(0, before_pooling.shape[1], 1)
        cindex = tf.range(0, before_pooling.shape[2], 1)

        sub_squares = before_pooling[:,rindex:rindex+pool_size, cindex:cindex+pool_size, :]
        return sub_squares

if __name__ == '__main__':
    input1 = layers.Input(shape=(64,64,4))
    input2 = layers.Input(shape=(32,32,2))

    x = UnPooling()(input1, input2)
    m = Model(inputs=[input1,input2],outputs=x)

    input1 = np.random.uniform(size=(1,64,64,4))
    input2 = np.random.uniform(size=(1,32,32,4))

    out = m.predict([input1,input2])
    print('')