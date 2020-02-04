'''
https://blog.csdn.net/xys430381_1/article/details/90413169
'''
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.framework import ops
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from base.tf_resnet50 import ResNet50, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
import base.image_ops as image_ops
import cv2

import numpy as np
import sys,os

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
               tf.cast(op.inputs[0] > 0., dtype)
            # return grad * tf.cast(op.inputs[0]  > 0., dtype)
            # return grad * tf.cast(grad > 0., dtype)

#def modify_backprop(model, name):
def modify_backprop(name, ModelType):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
        # #re-instanciate a new model
        #new_model = ResNet50()
        new_model = ModelType()
    return new_model

def compile_saliency_function(model, activation_layer='activation_48', filter_index=None):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = None
    if filter_index is None:
        max_output = K.max(layer_output, axis=3)
    elif filter_index >= 0:
        ### check one filter
        max_output = layer_output[:,:,:,filter_index]
    ### check one point
    # mask = np.zeros([7,7])
    # mask[0][0] = 1
    # max_output *= mask
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

def grad_cam(input_model, image, category_index, layer_name):

    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    #model.summary()
    loss = K.sum(model.output)
    conv_output_layer = [l for l in model.layers if l.name == layer_name]
    conv_output = conv_output_layer[0].output
    grads = K.gradients(loss, conv_output)
    grads = normalize(grads[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0) # equar to relu
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def visualize_cam_by_imgenatid():
    # CurModel = VGG16
    # top_layer = 'block5_conv3'

    CurModel = ResNet50
    top_layer = 'act5c_branch2c'
    # top_layer = 'act4f_branch2c'
    # top_layer = 'act3d_branch2c'
    # top_layer = 'act2c_branch2c'
    # top_layer = 'max_pooling2d'

    model = CurModel()
    register_gradient()
    guided_model = modify_backprop('GuidedBackProp', CurModel)
    saliency_fn = compile_saliency_function(guided_model, activation_layer=top_layer)

    ### imagenet id input
    for id in ['3', '2155', '233', '10587']:
        path = image_ops.get_path_from_id(id)
        fname = os.path.basename(path)
        preprocessed_input = image_ops.load_image(path)
        predictions = model.predict(preprocessed_input)
        top_1 = decode_predictions(predictions)[0][0]

        print('img{}: {} ({}) with probability {:.2f}'.format(fname, top_1[1], top_1[0], top_1[2]))

        predicted_class = np.argmax(predictions)
        cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, top_layer)
        cv2.imwrite("../output/{}_{}_{}_gradcam.jpg".format(fname, top_layer, model.name), cam)

        preprocessed_input = image_ops.load_image(path)
        saliency = saliency_fn([preprocessed_input, 0])
        guided_gradcam = saliency[0] * heatmap[..., np.newaxis]
        #guided_gradcam = saliency[0]
        cv2.imwrite("../output/{}_{}_{}_guided.jpg".format(fname, top_layer, model.name),
                    image_ops.deprocess_image(guided_gradcam))

def visualize_cam_by_path():
    # CurModel = VGG16
    # top_layer = 'block5_conv3'

    CurModel = ResNet50
    top_layer = 'act5c_branch2c'
    # top_layer = 'act4f_branch2c'
    # top_layer = 'act3d_branch2c'
    # top_layer = 'act2c_branch2c'
    # top_layer = 'max_pooling2d'

    model = CurModel()
    register_gradient()
    guided_model = modify_backprop('GuidedBackProp', CurModel)
    saliency_fn = compile_saliency_function(guided_model, activation_layer=top_layer)

    ## single path input
    # path = '../data/monkey1.jpg'
    path = '../data/faceu_20171126184737.jpg'
    fname = os.path.basename(path)
    preprocessed_input = image_ops.load_image(path)
    predictions = model.predict(preprocessed_input)
    top_1 = decode_predictions(predictions)[0][0]

    print('img{}: {} ({}) with probability {:.2f}'.format(fname, top_1[1], top_1[0], top_1[2]))

    predicted_class = np.argmax(predictions)
    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, top_layer)
    cv2.imwrite("../output/{}_{}_{}_gradcam.jpg".format(fname, top_layer, model.name), cam)

    preprocessed_input = image_ops.load_image(path)
    saliency = saliency_fn([preprocessed_input, 0])
    guided_gradcam = saliency[0] * heatmap[..., np.newaxis]
    # guided_gradcam = saliency[0]
    cv2.imwrite("../output/{}_{}_{}_guided.jpg".format(fname, top_layer, model.name),
                image_ops.deprocess_image(guided_gradcam))

if __name__ == '__main__':
    # visualize_cam_by_imgenatid()
    visualize_cam_by_path()