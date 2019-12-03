import os
import numpy as np
from PIL import Image
from base import image_ops

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



class DeConv:
    def __init__(self):
        self.conv_model = None
        self.conv_sub_model = None
        self.conv_layers = None
        self.conv_layer_names = None
        self.tops = None
        self.top_names = None
        self.deconv_model = None
        self.before_maxpooling_model = None
        self.before_maxpooling_acts = []


    def preprocess(self, input = ''):
        pass

    def conv_predict(self, input, layer_label=None):
        pass

    def before_max_pooling_predict(self, input):
        pass

    def deconv_predict(self, input, layer_label=None):
        pass

    def project_multiple_layer_filters(self, img_input = None, img_name = '', max_filter = True):
        save_to_folder = '../output'
        projections = []
        box_borders = []

        for layer in self.top_names:
            print('layer: {}'.format(layer))
            x = self.conv_predict(img_input, layer)
            if max_filter:
                f = image_ops.get_strongest_filter(x)
                x = image_ops.set_zero_except_maximum(f, x)
                print("layer {}'s strongest filter No. is: {}".format(layer, f))
            layer_out = self.deconv_predict(x, layer)

            percentile = 99
            max_val = np.percentile(layer_out, percentile)
            #max_val = np.nanmax(layer_out)
            if max_val == 0.0: max_val = 1.0
            layer_out *= (120 / max_val)

            box_borders.append(image_ops.get_bounding_box_coordinates(layer_out))
            projections.append(layer_out)

        superposed_projections = np.maximum.reduce(projections)
        #superposed_projections = np.average(projections,axis=0)
        assert superposed_projections.shape == projections[0].shape
        DeconvOutput(superposed_projections, mode="BGR").save_as(save_to_folder, '{}_activations.JPEG'.format(img_name))
        out_img = DeconvOutput(img_input, mode="BGR")
        out_img.array = image_ops.draw_bounding_box_cv2(out_img.array, box_borders)
        out_img.save_as(save_to_folder, '{}.JPEG'.format(img_name))

if __name__ == '__main__':
    img = np.ones(shape=(30,30,3))
    img255 = img*1000
    img0 = img*0
    img100 = img*100
    out_img255 = Image.fromarray(img255.astype(np.uint8), "RGB")
    out_img255.save('img255.jpeg')
    out_img0 = Image.fromarray(img0.astype(np.uint8), "RGB")
    out_img0.save('img0.jpeg')
    out_img100 = Image.fromarray(img100.astype(np.uint8), "RGB")
    out_img100.save('img100.jpeg')