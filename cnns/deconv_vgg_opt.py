from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow.keras.layers as Layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Conv2DTranspose,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from cnns import vgg
import numpy as np
from base import image_ops,imagenet_id_word,my_keras_layers
from PIL import Image
import matplotlib.pyplot as plt
import copy

class DeconvVgg16:
    vgg_layer_names = ['', 'block1_conv1', 'block1_conv2', 'block1_pool',
                          'block2_conv1', 'block2_conv2', 'block2_pool',
                          'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool',
                          'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
                          'block5_conv1', 'block5_conv2', 'block5_conv3']

    def __init__(self, conv_base_model=None):
        self.conv_base_model = conv_base_model if conv_base_model else VGG16()
        self.conv_sub_models = [None] + [vgg.VGG16NET(i, self.conv_base_model) for i in range(1,14)]
        # This attributes will be filled by 'project_down' method
        self.array = None  # Tensor being projected down from feature space to image space
        self.activation_maxpools = None  # Activation for max_pool layer 1 and 2, needed for switches
        self.current_layer = None  # Changes as array is passed on
        self.f = None  # Filter whose activation is projected down
        self.deconv_sub_models = {'block5_conv3': self.create_devgg_model('block5_conv3'),
                                  'block4_conv3': self.create_devgg_model('block4_conv3'),
                                  'block3_conv3': self.create_devgg_model('block3_conv3')}

    def create_devgg_model(self,  top_layer_name = None):
        assert top_layer_name != None
        conv_before_pool = {}
        conv_before_pool['block1_pool'] = Input(shape = self.conv_base_model.get_layer('block1_conv2').output_shape[1:])
        conv_before_pool['block2_pool'] = Input(shape = self.conv_base_model.get_layer('block2_conv2').output_shape[1:])
        conv_before_pool['block3_pool'] = Input(shape=self.conv_base_model.get_layer('block3_conv3').output_shape[1:])
        conv_before_pool['block4_pool'] = Input(shape=self.conv_base_model.get_layer('block4_conv3').output_shape[1:])
        deconv_input_shape = self.conv_base_model.get_layer(top_layer_name).output_shape[1:]
        deconv_origin_input = Input(shape = deconv_input_shape)
        deconv_curr_input = deconv_origin_input
        ind = DeconvVgg16.vgg_layer_names.index(top_layer_name)
        for i in range(ind, 0, -1):
            conv_layer = self.conv_base_model.get_layer(DeconvVgg16.vgg_layer_names[i])
            deconv_out_shape = conv_layer.input_shape[1:]

            if conv_layer.name.endswith('_pool'):
                #use unsample instead for simply
                deconv_out = UpSampling2D()(deconv_curr_input)
                mask = my_keras_layers.MaskOfMaxPooling()([conv_before_pool[conv_layer.name], deconv_out])
                deconv_out = Layers.multiply([deconv_out, mask])

            else:
                #deconv
                deconv_out = Conv2DTranspose(filters = deconv_out_shape[2],
                                      kernel_size=conv_layer.kernel_size,
                                      padding=conv_layer.padding,
                                      strides=conv_layer.strides,
                                      activation='relu',
                                      use_bias=False,
                                      name = 'de_{}'.format(conv_layer.name))(deconv_curr_input)
                assert deconv_out.shape[1:] == deconv_out_shape
            deconv_curr_input = deconv_out

        m = Model(inputs=[deconv_origin_input] + list(conv_before_pool.values()), outputs=deconv_out)
        for i in range(ind, 0, -1):
            conv_layer = self.conv_base_model.get_layer(DeconvVgg16.vgg_layer_names[i])
            ln = conv_layer.name
            if not ln.endswith('_pool'):
                w = self.conv_base_model.get_layer(ln).get_weights()[0]
                m.get_layer('de_{}'.format(ln)).set_weights([w])
        return m

    def project_down(self, x, layer, max_filter=True, use_bias=False):
        assert type(layer) == int
        self.current_layer = layer
        self.f = None
        self.use_bias = use_bias
        self.array = self.conv_sub_models[self.current_layer].predict(x)
        if max_filter:
            self.f = image_ops.get_strongest_filter(self.array)
            self._set_zero_except_maximum()
        print("layer {}'s strongest filter No. is: {}".format(self.current_layer, self.f))
        deconv_layer_name = vgg.VGG16NET.conv_layer_names[layer]
        deconv_sub_model = self.deconv_sub_models[deconv_layer_name]
        self.array = deconv_sub_model.predict([self.array]+self.activation_maxpools)
        return self.array

    def _set_zero_except_maximum(self):
        assert self.f >= 1
        # Set other layers to zero
        new_array = np.zeros_like(self.array)
        new_array[0, :, :, self.f - 1] = self.array[0, :, :, self.f - 1]

        # Set other activations in same layer to zero
        max_index_flat = np.nanargmax(new_array)
        max_index = np.unravel_index(max_index_flat, new_array.shape)
        self.array = np.zeros_like(new_array)
        self.array[max_index] = new_array[max_index]

def get_path_from_id(img_id):
    img_id_str = str(img_id)
    while len(img_id_str) < 5:
        img_id_str = '0' + img_id_str

    folder = 'D:\data\imagenet-data\ILSVRC2012_img_val\\'
    file = 'ILSVRC2012_val_000' + img_id_str + '.JPEG'

    path = folder + file

    return path

def project_multiple_layer_filters(img_id=None, deconv_base_model=None):
    assert img_id != None and deconv_base_model != None

    path = get_path_from_id(img_id)
    save_to_folder = '../output'

    projections = []
    box_borders = []
    contrast = [i for i in range(1,40,2)]

    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = deconv_base_model.conv_base_model.predict(x)
    #print top3 preict class
    indexes = pred[0].argsort()[-3:][::-1]
    pred_word = []
    for ind in indexes:
        pred_word.append(imagenet_id_word.get_word(ind))
    print ("imgae: {} => pred class: {}.{}".format(img_id, indexes, pred_word))

    deconv_base_model.activation_maxpools = []
    for i in (2, 4, 7, 10):
        deconv_base_model.activation_maxpools.append(deconv_base_model.conv_sub_models[i].predict(x))

    for layer in [13, 10]:
        print('image:{}=> layer: {}'.format(img_id, layer))
        projection = deconv_base_model.project_down(x, layer)

        if layer != 1:
            # Increase Contrast
            percentile = 99
            # max_val = np.nanargmax(unarranged_array)
            max_val = np.percentile(projection, percentile)
            if max_val == 0.0: max_val= 100.0
            projection *= (contrast[layer] / max_val)
        else:
            projection *= 0.3
        box_borders.append(image_ops.get_bounding_box_coordinates(projection))
        projections.append(projection)

    superposed_projections = np.maximum.reduce(projections)
    assert superposed_projections.shape == projections[0].shape
    DeconvOutput(superposed_projections,mode="BGR").save_as(save_to_folder, '{}_activations.JPEG'.format(img_id))
    out_img = DeconvOutput(x,mode="BGR")
    out_img.array = image_ops.draw_bounding_box_cv2(out_img.array, box_borders)
    out_img.save_as(save_to_folder, '{}.JPEG'.format(img_id))

class DeconvOutput:
    def __init__(self, unarranged_array, format="channel_last", mode="RGB", contrast=None):  # Takes output of DeconvNet
        self.contrast = contrast
        self.array = self._rearrange_array(unarranged_array,format,mode)
        self.image = None
        self.format = format

    def _rearrange_array(self, unarranged_array, format,mode):
        assert len(unarranged_array.shape) in (3, 4)
        if len(unarranged_array.shape) == 4:
            assert unarranged_array.shape[0] == 1
            unarranged_array = unarranged_array[0, :, :, :]  # Eliminate batch size dimension
        # If Array is not yet rearranged
        if  format == 'channel_first':
            unarranged_array = np.moveaxis(unarranged_array, 0, -1)  # Put channels last

        # Contrast
        if self.contrast is not None:
            percentile = 99
            # max_val = np.nanargmax(unarranged_array)
            max_val = np.percentile(unarranged_array, percentile)
            unarranged_array *= (self.contrast / max_val)

        if mode == "RGB":
            # Undo sample mean subtraction
            unarranged_array[:, :, 0] += 123.68
            unarranged_array[:, :, 1] += 116.779
            unarranged_array[:, :, 2] += 103.939
        elif mode == "BGR":
            unarranged_array = unarranged_array[..., ::-1]
            unarranged_array[:, :, 0] += 123.68
            unarranged_array[:, :, 1] += 116.779
            unarranged_array[:, :, 2] += 103.939
        else:
            raise Exception("illegal mode:{}".format(mode))

        return unarranged_array

    def save_as(self, folder=None, filename='test.JPEG'):
        self.image = Image.fromarray(self.array.astype(np.uint8), "RGB")
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')

        if folder is not None:
            assert type(folder) is str
            filename = folder + '/' + filename

        try:
            os.remove(filename)
        except OSError:
            pass

        self.image.save(filename)

def get_summed_activation_of_feature_map(top_layer_model, f, preprocd_img):
    # Get activations for shortened model
    activation_img = top_layer_model.predict(preprocd_img)
    activation_img = activation_img[0, :, :, f-1]
    activation_img = activation_img.sum(1)
    activation_img = activation_img.sum(0)
    return activation_img

def get_heatmaps(img_id, deconv_base_model):
    assert img_id != None and deconv_base_model != None
    top_layer_model = deconv_base_model.conv_sub_models[13]
    path = get_path_from_id(img_id)
    save_to_folder = '../output'

    # read and process img
    img = image.load_img(path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # pred result
    pred = deconv_base_model.conv_base_model.predict(img)
    ind = pred[0].argmax()
    pred_word = imagenet_id_word.get_word(ind)


    top_layer_output = top_layer_model.predict(img)
    max_filter = image_ops.get_strongest_filter(top_layer_output)
    print("imgae: {} => pred class: {}.{} =>max_filter of top layer: {}".format(img_id, ind, pred_word, max_filter))
    activations = np.zeros((30, 30))
    class_prop = np.zeros((30, 30))

    for x in range(0, 30):
        for y in range(0, 30):
            prep_image = image_ops.grey_square(copy.deepcopy(img), 10 + x * 7, 10 + y * 7)
            activation = get_summed_activation_of_feature_map(top_layer_model, max_filter, prep_image)
            prediction = deconv_base_model.conv_base_model.predict(prep_image)
            activations[x, y] = activation
            class_prop[x, y] = prediction[0][ind]

    fig, ax = plt.subplots()
    cax = ax.imshow(activations, interpolation='nearest', cmap='plasma')
    plt.axis('off')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    plt.savefig('{}/{}-top-layer-activate-mask.jpg'.format(save_to_folder, img_id))
    fig, ax = plt.subplots()
    cax = ax.imshow(class_prop, interpolation='nearest', cmap='plasma')
    plt.axis('off')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    plt.savefig('{}/{}-pred-mask.jpg'.format(save_to_folder, img_id))

if __name__ == '__main__':
    deconv_base_model = DeconvVgg16(vgg.VGG16NET().model)
    for img_id in [11337]:
        project_multiple_layer_filters(img_id=img_id, deconv_base_model=deconv_base_model)
        get_heatmaps(img_id, deconv_base_model=deconv_base_model)