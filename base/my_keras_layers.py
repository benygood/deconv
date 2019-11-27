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
    def __init__(self, pool_size=2, strides=2):
        super(UnPooling, self).__init__()

    def call(self, inputs, pool_size=(1,2,2,1), strides=(1,2,2,1), padding='VALID' ):
        out, argmax = tf.nn.max_pool_with_argmax(inputs[0], pool_size, strides, padding)

        a = tf.unravel_index(tf.reshape(argmax,[-1]), tf.shape(inputs[0],out_type=tf.int64))
        # print(a.shape)
        return tf.expand_dims(a,0)
if __name__ == '__main__':

    input1 = layers.Input(shape=(64,64,4))
    input2 = layers.Input(shape=(32,32,4))
    outs = UnPooling()([input1,input2])
    in1 = np.random.uniform(size=(1, 64, 64, 4))
    in2 = np.random.uniform(size=(1, 32, 32, 4))
    m = Model(inputs=[input1,input2], outputs=outs)
    res = m.predict([in1,in2])
    print("")