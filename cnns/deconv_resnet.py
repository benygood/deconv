from __future__ import absolute_import, division, print_function, unicode_literals


from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras.layers as layers

def deconv_identity_block(input_tensor, kernel_size, filters, stage, block):
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
    # bn_axis = 3
    # bn_name_base = 'bn' + str(stage) + block + '_branch'

    # x = layers.Conv2D(filters1, (1, 1),
    #                   kernel_initializer='he_normal',
    #                   name=conv_name_base + '2a')(input_tensor)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = layers.Activation('relu')(x)
    #
    # x = layers.Conv2D(filters2, kernel_size,
    #                   padding='same',
    #                   kernel_initializer='he_normal',
    #                   name=conv_name_base + '2b')(x)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = layers.Activation('relu')(x)
    #
    # x = layers.Conv2D(filters3, (1, 1),
    #                   kernel_initializer='he_normal',
    #                   name=conv_name_base + '2c')(x)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    #
    # x = layers.add([x, input_tensor])
    # x = layers.Activation('relu')(x)

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

def conv_block(input_tensor,
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
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # bn_axis = 3

    # conv_name_base = 'res' + str(stage) + block + '_branch'
    # bn_name_base = 'bn' + str(stage) + block + '_branch'
    #
    # x = layers.Conv2D(filters1, (1, 1), strides=strides,
    #                   kernel_initializer='he_normal',
    #                   name=conv_name_base + '2a')(input_tensor)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = layers.Activation('relu')(x)
    #
    # x = layers.Conv2D(filters2, kernel_size, padding='same',
    #                   kernel_initializer='he_normal',
    #                   name=conv_name_base + '2b')(x)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = layers.Activation('relu')(x)
    #
    # x = layers.Conv2D(filters3, (1, 1),
    #                   kernel_initializer='he_normal',
    #                   name=conv_name_base + '2c')(x)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    #
    # shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
    #                          kernel_initializer='he_normal',
    #                          name=conv_name_base + '1')(input_tensor)
    # shortcut = layers.BatchNormalization(
    #     axis=bn_axis, name=bn_name_base + '1')(shortcut)
    #
    # x = layers.add([x, shortcut])
    # x = layers.Activation('relu')(x)

    #deconv like a mirror of conv process, ignore BN temporally
    x = layers.Activation('relu')(input_tensor)
    shortcut = layers.Conv2DTranspose(filters3, (1, 1), strides=strides,
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

def deconv_stage1():

    # x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    # x = layers.Conv2D(64, (7, 7),
    #                   strides=(2, 2),
    #                   padding='valid',
    #                   kernel_initializer='he_normal',
    #                   name='conv1')(x)
    # x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # #


class DeconvResnet50:
    layers = []

    def __init__(self):
        m = ResNet50()
        self.conv_model = m
        self.conv_layers = m.layers
        self.conv_layer_names = [ l.name for l in m.layers ]

    def create_deconv_model(self, top_layer_name=None):
        assert top_layer_name != None
        deconv_input_shape = self.conv_model.get_layer(top_layer_name).output_shape[1:]
        deconv_origin_input = layers.Input(shape = deconv_input_shape)


if __name__ == '__main__':
    d = DeconvResnet50()
    print(d.conv_layer_names)