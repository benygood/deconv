from __future__ import absolute_import, division, print_function, unicode_literals


from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Input
from base.my_keras_layers import UnPooling
from tensorflow.keras import Model
import tensorflow.keras.layers as layers
from cnns.deconv_base import DeConv
import numpy as np
from tensorflow.keras.preprocessing import image
from base import image_ops, imagenet_id_word


class DeconvResnet50(DeConv):

    def __init__(self, top_names=['5c', '4f', '3d', '2c', '1b']):
        super(DeconvResnet50, self).__init__()
        m = ResNet50()
        self.conv_model = m
        self.conv_sub_model = {}
        self.conv_layers = m.layers
        self.conv_layer_names = [ l.name for l in m.layers ]
        self.tops = {}
        self.deconv_model = {}
        self.before_maxpooling_model = Model(inputs=m.input, outputs=self.get_conv_layer('pool1_pad').output)
        self.top_names = top_names
        for top_ln in self.top_names:
            self.conv_sub_model[top_ln] = Model(inputs=m.input, outputs=self.get_conv_layer(top_ln).output)
            self.__create_model(top_ln)
            print("init finished for {}".format(top_ln))

    def conv_predict(self, input, layer_label = ''):
        if layer_label == '':
            return self.conv_model.predict(input)
        else:
            return self.conv_sub_model[layer_label].predict(input)

    def before_max_pooling_predict(self, input):
        self.before_maxpooling_acts = [self.before_maxpooling_model.predict(input)]
        return self.before_maxpooling_acts

    def deconv_predict(self, input, layer_label = ''):
        assert len(layer_label) > 0
        return self.deconv_model[layer_label].predict([input] + self.before_maxpooling_acts)

    def preprocess(self, input = ''):
        assert len(input) > 0
        path = image_ops.get_path_from_id(input)
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def __create_model(self, top = None):
        assert type(top) == str
        beforeMaxPooling = Input(shape = [114,114,64])
        x = Input(shape=(7,7,2048))
        if top=='5c': self.tops['5c'] = x
        x = self.deconv_identity_block(x, 3, [2048, 512, 512], stage=5, block='c')
        if top=='5b': x=Input(shape=[7,7,2048]); self.tops['5b'] = x
        x = self.deconv_identity_block(x, 3, [2048, 512, 512], stage=5, block='b')
        if top=='5a': x=Input(shape=[7,7,2048]); self.tops['5a']=x
        x = self.deconv_block(x, 3, [1024, 512, 512], stage=5, block='a')
        if top=='4f': x=Input(shape=[14,14,1024]); self.tops['4f']=x
        x = self.deconv_identity_block(x, 3, [1024, 256, 256], stage=4, block='f')
        if top=='4e': x=Input(shape=[14,14,1024]); self.tops['4e']=x
        x = self.deconv_identity_block(x, 3, [1024, 256, 256], stage=4, block='e')
        if top=='4d': x=Input(shape=[14,14,1024]); self.tops['4d']=x
        x = self.deconv_identity_block(x, 3, [1024, 256, 256], stage=4, block='d')
        if top=='4c': x=Input(shape=[14,14,1024]); self.tops['4c']=x
        x = self.deconv_identity_block(x, 3, [1024, 256, 256], stage=4, block='c')
        if top=='4b': x=Input(shape=[14,14,1024]); self.tops['4b']=x
        x = self.deconv_identity_block(x, 3, [1024, 256, 256], stage=4, block='b')
        if top=='4a': x=Input(shape=[14,14,1024]); self.tops['4a']=x
        x = self.deconv_block(x, 3, [512, 256, 256], stage=4, block='a')
        if top=='3d': x=Input(shape=[28,28,512]); self.tops['3d']=x

        x = self.deconv_identity_block(x, 3, [512, 128, 128], stage=3, block='d')
        if top=='3c': x=Input(shape=[28,28,512]); self.tops['3c']=x
        x = self.deconv_identity_block(x, 3, [512, 128, 128], stage=3, block='c')
        if top=='3b': x=Input(shape=[28,28,512]); self.tops['3b']=x
        x = self.deconv_identity_block(x, 3, [512, 128, 128], stage=3, block='b')
        if top=='3a': x=Input(shape=[28,28,512]); self.tops['3a']=x
        x = self.deconv_block(x, 3, [256, 128, 128], stage=3, block='a')
        if top=='2c': x=Input(shape=[56,56,256]); self.tops['2c']=x

        x = self.deconv_identity_block(x, 3, [256, 64, 64], stage=2, block='c')
        if top=='2b': x=Input(shape=[56,56,256]); self.tops['2b']=x
        x = self.deconv_identity_block(x, 3, [256, 64, 64], stage=2, block='b')
        if top=='2a': x=Input(shape=[56,56,256]); self.tops['2a']=x
        x = self.deconv_block(x, 3, [64, 64, 64], stage=2, block='a', strides=(1, 1))
        if top=='1b': x=Input(shape=[56,56,64]); self.tops['1b']=x
        out, x1, x2 = self.deconv_stage1(x, beforeMaxPooling)

        self.deconv_model[top] = Model([self.tops[top]] + [beforeMaxPooling], [out, x1, x2])
        self.set_weights(self.deconv_model[top])

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
                                    use_bias=False,
                                    name=conv_name_base + '2c')(x)
        #BN can be added here, also after ConvT?
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters2, kernel_size,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   use_bias=False,
                                   name=conv_name_base + '2b')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters1, (1, 1),
                                   kernel_initializer='he_normal',
                                   use_bias=False,
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
                                 use_bias=False,
                                 name=conv_name_base + '1')(input_tensor)
        x = layers.Conv2DTranspose(filters3, (1, 1),
                                    kernel_initializer='he_normal',
                                    use_bias=False,
                                    name=conv_name_base + '2c')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters2, kernel_size, padding='same',
                                    kernel_initializer='he_normal',
                                    use_bias=False,
                                    name=conv_name_base + '2b')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters1, (1, 1), strides=strides,
                                    kernel_initializer='he_normal',
                                    use_bias=False,
                                    name=conv_name_base + '2a')(x)
        x = layers.add([x, shortcut])
        return x

    def set_weights(self, model):
        layer_names = [l.name for l in model.layers]
        for ln in layer_names:
            if ln.startswith('res') or ln == 'conv1':
                w = self.conv_model.get_layer(ln).get_weights()[0]
                model.get_layer(ln).set_weights([w])


    def deconv_stage1(self, x, before_pooling):
        x = UnPooling()([before_pooling, x], pool_size=(1,3,3,1), strides=(1,2,2,1)); x1=x
        x = layers.Cropping2D(1)(x); x2 = x
        x = layers.Conv2DTranspose(3, (7,7), strides=(2,2), padding='valid',  name='conv1', use_bias=False )(x)
        x = layers.Cropping2D(3)(x)
        return x, x1, x2

    def get_conv_layer(self, ln=None ):
        assert  type(ln) == str
        if ln == '1b':
            conv_layer_name = 'max_pooling2d'
        elif ln == 'pool1_pad':
            conv_layer_name = 'pool1_pad'
        else:
            conv_layer_name = 'res' + ln + '_branch2c'
        return self.conv_model.get_layer(conv_layer_name)


if __name__ == '__main__':
    d = DeconvResnet50(['3d', '1b'])
    print(d.conv_layer_names)
    for id in ['11337']:
        img = d.preprocess(id)
        pred = d.conv_predict(img)
        #print top3 preict class
        indexes = pred[0].argsort()[-3:][::-1]
        pred_word = []
        for ind in indexes:
            pred_word.append(imagenet_id_word.get_word(ind))
        print ("imgae: {} => pred class: {}.{}".format(id, indexes, pred_word))
        d.before_max_pooling_predict(img)
        d.project_multiple_layer_filters(img, id)


