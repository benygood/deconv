from __future__ import absolute_import, division, print_function, unicode_literals


from tensorflow.keras.applications.resnet50 import ResNet50
from keras.utils.vis_utils import plot_model

if __name__ == '__main__':
    m = ResNet50()
    plot_model(m, to_file='test_resnet50.png', show_shapes=True)