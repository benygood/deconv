from __future__ import absolute_import, division, print_function, unicode_literals


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from base import imagenet_id_word

class VGG16NET():

    conv_layer_names = ['', 'block1_conv1','block1_conv2',
                        'block2_conv1','block2_conv2',
                        'block3_conv1','block3_conv2','block3_conv3',
                        'block4_conv1','block4_conv2','block4_conv3',
                        'block5_conv1','block5_conv2','block5_conv3']


    def __init__(self, highest_layer_num=None, base_model=None):
        self.highest_layer_num = highest_layer_num
        self.base_model = base_model if base_model else VGG16()  # If no base_model, create alexnet_model
        self.model = self._sub_model() if highest_layer_num != None and highest_layer_num < 14 else self.base_model

    def _sub_model(self):
        highest_layer_name = VGG16NET.conv_layer_names[self.highest_layer_num]
        highest_layer = self.base_model.get_layer(highest_layer_name)
        return Model(inputs=self.base_model.input,
                     outputs=highest_layer.output)

    def predict(self, x):
        return self.model.predict(x)

if __name__ == "__main__":

    model = VGG16(weights='imagenet', include_top=True)

    img_path = 'D:\data\imagenet-data\ILSVRC2012_img_val\ILSVRC2012_val_00000014.JPEG'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    pred = features[0].argmax()
    print('pred: {} - {}'.format(pred, imagenet_id_word.get_word(pred)))