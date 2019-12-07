import numpy as np
import cv2
from scipy.misc import imread, imresize
from tensorflow.keras.preprocessing import image
from keras.applications import imagenet_utils

def get_bounding_box_coordinates(projection):
    combined_channels = np.sum(projection[0], axis=2)

    arg_positions = np.argwhere(combined_channels)
    (xstart, ystart), (xstop, ystop) = arg_positions.min(0), arg_positions.max(0)

    return (ystart, ystop, xstart, xstop)



def draw_bounding_box_cv2(image, bounding_boxes):
    count = 0
    for xstart,xstop,ystart,ystop in bounding_boxes:
        count += 1
        cv2.rectangle(image, (xstart,ystart), (xstop,ystop), (255, 0, 0), 1 )
        cv2.putText(image, "b%d"%(count), (xstart+10,ystart+10), 1, 1, (255, 0, 0), 1)
    return image

def get_strongest_filter(activation_img):

    # Make sure that dimensions 2 and 3 are spacial (Image is square)
    assert activation_img.shape[1] == activation_img.shape[2], "Index ordering incorrect"

    # Find maximum activation for each filter for a given image
    activation_img = np.nanmax(activation_img, axis=2)
    activation_img = np.nanmax(activation_img, axis=1)

    # Remove batch size dimension
    assert activation_img.shape[0] == 1
    activation_img = activation_img.sum(0)

    # Make activations 1-based indexing
    activation_img = np.insert(activation_img, 0, 0.0)

    #  activation_image is now a vector of length equal to number of filters (plus one for one-based indexing)
    #  each entry corresponds to the maximum/summed activation of each filter for a given image

    return activation_img.argmax()

def get_strongest_filters(activation_img, N=2):

    # Make sure that dimensions 2 and 3 are spacial (Image is square)
    assert activation_img.shape[1] == activation_img.shape[2], "Index ordering incorrect"

    # Find maximum activation for each filter for a given image
    # Find maximum activation for each filter for a given image
    activation_img = np.nanmax(activation_img, axis=2)
    activation_img = np.nanmax(activation_img, axis=1)

    # Remove batch size dimension
    assert activation_img.shape[0] == 1
    activation_img = activation_img.sum(0)

    # Make activations 1-based indexing
    activation_img = np.insert(activation_img, 0, 0.0)

    #  activation_image is now a vector of length equal to number of filters (plus one for one-based indexing)
    #  each entry corresponds to the maximum/summed activation of each filter for a given image

    return activation_img.argsort()[-N:][::-1]

def preprocess_image_batch(image_paths, img_size=(256, 256), crop_size=(224, 224), color_mode="rgb", out=None):
    """
    Resize, crop and normalize colors of images
    to make them suitable for alexnet_model (if default parameter values are chosen)

    This function is also from
    https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
    with only some minor changes
    """

    # Make function callable with single image instead of list
    if type(image_paths) is str:
        image_paths = [image_paths]

    img_list = []
    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        img = imresize(img, img_size)

        img = img.astype('float32')
        # Normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        if color_mode == "bgr":
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
            , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                         ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch

def grey_square(img, x, y, radius = 10):

    h = img.shape[1]
    w = img.shape[2]
    assert radius <= x < h - radius
    assert radius <= y < w - radius
    #square size = 2*radius + 1
    img[:, x-radius:x+radius, y-radius:y+radius, :] = 0
    return img


def set_zero_except_maximum(filter, x):
    assert filter >= 1
    # Set other layers to zero
    new_array = np.zeros_like(x)
    new_array[0, :, :, filter - 1] = x[0, :, :, filter - 1]
    # Set other activations in same layer to zero
    max_index_flat = np.nanargmax(new_array)
    max_index = np.unravel_index(max_index_flat, new_array.shape)
    out = np.zeros_like(new_array)
    out[max_index] = new_array[max_index]
    return out

def get_path_from_id(img_id):
    '''
    used for imagenet data
    :param img_id:
    :return:
    '''
    img_id_str = str(img_id)
    while len(img_id_str) < 5:
        img_id_str = '0' + img_id_str
    folder = 'D:\data\imagenet-data\ILSVRC2012_img_val\\'
    file = 'ILSVRC2012_val_000' + img_id_str + '.JPEG'
    path = folder + file
    return path

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)
    return x

def deprocess_image(x, format = 'cl'):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if format == 'cf':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x