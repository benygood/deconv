from __future__ import absolute_import, division, print_function, unicode_literals


from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras.layers as layers
from base.my_keras_layers import UnPooling
from tensorflow.keras import Model



class DeconvResnet50:

    def __init__(self):
        m = ResNet50()
        self.conv_model = m
        self.conv_layers = m.layers
        self.conv_layer_names = [ l.name for l in m.layers ]
        self.inputPoints = {}
        self.deconv_model = {}
        input_shape = self.conv_model.get_layer(self.get_layer_name(5,'c')).output_shape[1:]
        self.inputPoints[(5, 'c')] = layers.Input(shape = input_shape)
        self.inputPoints[(5, 'b')] = self.deconv_identity_block(self.inputPoints[(5, 'c')], 3, [2048, 512, 512], stage=5, block='c')
        self.inputPoints[(5, 'a')] = self.deconv_identity_block(self.inputPoints[(5, 'b')], 3, [2048, 512, 512], stage=5, block='b')
        self.inputPoints[(4, 'f')] = self.deconv_block(self.inputPoints[(5, 'a')], 3, [1024, 512, 512], stage=5, block='a')
        self.inputPoints[(4, 'e')] = self.deconv_identity_block(self.inputPoints[(4, 'f')], 3, [1024, 256, 256], stage=4, block='f')
        self.inputPoints[(4, 'd')] = self.deconv_identity_block(self.inputPoints[(4, 'e')], 3, [1024, 256, 256], stage=4, block='e')
        self.inputPoints[(4, 'c')] = self.deconv_identity_block(self.inputPoints[(4, 'd')], 3, [1024, 256, 256], stage=4, block='d')
        self.inputPoints[(4, 'b')] = self.deconv_identity_block(self.inputPoints[(4, 'c')], 3, [1024, 256, 256], stage=4, block='c')
        self.inputPoints[(4, 'a')] = self.deconv_identity_block(self.inputPoints[(4, 'b')], 3, [1024, 256, 256], stage=4, block='b')
        self.inputPoints[(3, 'd')] = self.deconv_block(self.inputPoints[(4, 'a')], 3, [512, 256, 256], stage=4, block='a')
        self.inputPoints[(3, 'c')] = self.deconv_identity_block(self.inputPoints[(3, 'd')], 3, [512, 128, 128], stage=3,block='d')
        self.inputPoints[(3, 'b')] = self.deconv_identity_block(self.inputPoints[(3, 'c')], 3, [512, 128, 128], stage=3, block='c')
        self.inputPoints[(3, 'a')] = self.deconv_identity_block(self.inputPoints[(3, 'b')], 3, [512, 128, 128], stage=3, block='b')
        self.inputPoints[(2, 'c')] = self.deconv_block(self.inputPoints[(3, 'a')], 3, [256, 128, 128], stage=3, block='a')
        self.inputPoints[(2, 'b')] = self.deconv_identity_block(self.inputPoints[(2, 'c')], 3, [256, 64, 64], stage=2, block='c')
        self.inputPoints[(2, 'a')] = self.deconv_identity_block(self.inputPoints[(2, 'b')], 3, [256, 64, 64], stage=2, block='b')
        self.inputPoints[(1, 'b')] = self.deconv_block(self.inputPoints[(2, 'a')], 3, [64, 64, 64], stage=2, block='a')
        out = self.deconv_stage1(self.inputPoints[(1, 'b')])
        self.deconv_model[(5, 'c')] = Model(self.inputPoints[(5, 'c')], out); self.set_weights(self.deconv_model[(5, 'c')])
        self.deconv_model[(4, 'f')] = Model(self.inputPoints[(4, 'f')], out); self.set_weights(self.deconv_model[(4, 'f')])
        self.deconv_model[(3, 'd')] = Model(self.inputPoints[(3, 'd')], out); self.set_weights(self.deconv_model[(3, 'd')])
        self.deconv_model[(2, 'c')] = Model(self.inputPoints[(2, 'c')], out); self.set_weights(self.deconv_model[(2, 'c')])


    def deconv_identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        #deconv like a mirror of conv process, ignore BN temporally

        x = layers.Activation('relu')(input_tensor)
        x = layers.Conv2DTranspose( filters3, (1, 1),
                                    kernel_initializer='he_normal',
                                    name=conv_name_base + '2c')(x)
        #BN can be added here, also after ConvT?
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters2, kernel_size,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   name=conv_name_base + '2b')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters1, (1, 1),
                                   kernel_initializer='he_normal',
                                   name=conv_name_base + '2a')(x)
        x = layers.Activation('relu')(x)
        x = layers.add([x, input_tensor])
        return x

    def deconv_block(self, input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2)):
        """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        #deconv like a mirror of conv process, ignore BN temporally
        x = layers.Activation('relu')(input_tensor)
        shortcut = layers.Conv2DTranspose(filters1, (1, 1), strides=strides,
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '1')(input_tensor)
        x = layers.Conv2DTranspose(filters3, (1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters2, kernel_size, padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters1, (1, 1), strides=strides,
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a')(x)
        x = layers.add([x, shortcut])
        return x

    def set_weights(self, model):
        layer_names = [l.name for l in model.layers]
        for ln in layer_names:
            if ln.startswith('res'):
                w = self.conv_model.get_layer(ln).get_weights()[0]
                model.get_layer(ln).set_weights([w])


    def deconv_stage1(self, x):
        x = UnPooling()(x, pool_size=(1,3,3,1), strides=(1,2,2,1))
        x = layers.Cropping2D(1)(x)
        x = layers.Conv2DTranspose(3, (7,7), strides=(2,2), padding='valid', name='conv1')(x)
        x = layers.Cropping2D(3)(x)
        return x

    def get_layer_name(self, stage, block ):
        conv_name_base = 'res' + str(stage) + block + '_branch2c'
        return conv_name_base

    def gen_model(self, inputs, outputs, conv_model):
        pass
        return

if __name__ == '__main__':
    d = DeconvResnet50()
    print(d.conv_layer_names)