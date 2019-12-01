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

    def build(self, input_shape):
        self.shape=input_shape

    def call(self, inputs, pool_size=(1,2,2,1), strides=(1,2,2,1), padding='VALID' ):
        out, argmax = tf.nn.max_pool_with_argmax(inputs[0], pool_size, strides, padding)
        a = tf.unravel_index(tf.reshape(argmax,[-1]), tf.shape(inputs[0],out_type=tf.int64))
        a = tf.transpose(a)
        tl = tf.unravel_index(tf.range(0,tf.size(inputs[1],out_type=tf.int64),dtype=tf.int64), tf.shape(inputs[1],out_type=tf.int64))
        tl = tf.transpose(tl)*strides
        e = (a-tl)[:,1:3]
        a = tf.concat([a,e],axis=1)
        out_shape = tf.concat([ tf.shape(inputs[0], out_type=tf.int64), pool_size[1:3] ], axis=0)
        b = tf.SparseTensor(indices=a,
                            values=tf.reshape(inputs[1], [-1]),
                            dense_shape= out_shape)
        c = tf.sparse_tensor_to_dense(b, validate_indices=False)
        c = tf.reduce_max(c,[-2,-1])
        # b = tf.SparseTensor(indices=a,
        #                      values=tf.reshape(inputs[1], [-1]),
        #                      dense_shape= tf.shape(inputs[0], out_type=tf.int64))
        # c = tf.sparse_tensor_to_dense(b,validate_indices=False)
        #return c, out, tf.expand_dims(a,0), tf.expand_dims(e,0), tf.expand_dims(out_shape,0)
        return c

    def call_(self, inputs, pool_size=(1,2,2,1), strides=(1,2,2,1), padding='VALID' ):
        out, argmax = tf.nn.max_pool_with_argmax(inputs[0], pool_size, strides, padding)
        a = tf.unravel_index(tf.reshape(argmax,[-1]), tf.shape(inputs[0],out_type=tf.int64))
        a = tf.transpose(a)
        tl = tf.unravel_index(tf.range(0,tf.size(inputs[1],out_type=tf.int64),dtype=tf.int64), tf.shape(inputs[1],out_type=tf.int64))
        tl = tf.transpose(tl)*strides
        # e = (a-tl)[:,1:3]
        # a = tf.concat([a,e],axis=1)
        # out_shape = tf.concat([ tf.shape(inputs[0], out_type=tf.int64), pool_size[1:3] ], axis=0)
        #out_shape = [1, 64,64, 64,3,3]
        out_shape = tf.shape(inputs[0], out_type=tf.int64)
        b = tf.SparseTensor(indices=tl,
                            values=tf.reshape(inputs[1], [-1]),
                            dense_shape= out_shape)
        c = tf.sparse_tensor_to_dense(b,validate_indices=False)
        c = tf.reduce_max(c,[-1])
        # b = tf.SparseTensor(indices=a,
        #                      values=tf.reshape(inputs[1], [-1]),
        #                      dense_shape= tf.shape(inputs[0], out_type=tf.int64))
        # c = tf.sparse_tensor_to_dense(b,validate_indices=False)
        #return c, out, tf.expand_dims(a,0), tf.expand_dims(e,0), tf.expand_dims(out_shape,0)
        return c

if __name__ == '__main__':
    input1 = layers.Input(shape=(64,64,4))
    input2 = layers.Input(shape=(31,31,4))
    outs = UnPooling()([input1,input2], pool_size=(1,3,3,1), strides=(1,2,2,1))
    in1 = np.random.uniform(size=(1, 64, 64, 4))
    in2 = np.random.uniform(size=(1, 31, 31, 4))
    m = Model(inputs=[input1,input2], outputs=outs)
    res = m.predict([in1,in2])
    print("")