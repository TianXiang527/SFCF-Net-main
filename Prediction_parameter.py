#----------------------------------------------------#
#  Setting the parameters of prediction of the proposed SFCF-Net
#  @author:  Xiang Tian
#----------------------------------------------------#
import colorsys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.SFCF_Net import SFCF_Net
from utils.utils import cvtColor, preprocess_input, resize_image

#-----------------------------------------------------------------------------------#

class Sfcf_net(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   File path of model parameters
        #-------------------------------------------------------------------#
        "model_path"        : r'.\logs\ep200-loss0.069-val_loss0.067.pth',
        #----------------------------------------#
        #   The number of classes to be distinguished
        #----------------------------------------#
        "num_classes"       : 3,
        # ----------------------------------------#
        #   The number of image channel
        # ----------------------------------------#
        "num_channel": 3,
        #----------------------------------------#
        #   The size of input image
        #----------------------------------------#
        "input_shape"       : [256, 256],
        #----------------------------------------#
        "cuda"     : True,


    }

    #---------------------------------------------------#
    #   SFCF-Net Initialization
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   Color setting
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (125, 255,125),  (255, 116, 0), (255, 20, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()
                    

    def generate(self, onnx=False):
        #-------------------------------#
        #   Load models and weights
        #-------------------------------#
        self.net=SFCF_Net(self.num_channel,self.num_classes,16)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Detection image
    #---------------------------------------------------#
    def detect_image(self, image,image2,image3,image4):

        image       = cvtColor(image)
        image2 = cvtColor(image2)
        image3 = cvtColor(image3)
        image4 = cvtColor(image4)

        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data2, nw, nh = resize_image(image2, (self.input_shape[1], self.input_shape[0]))
        image_data3, nw, nh = resize_image(image3, (self.input_shape[1], self.input_shape[0]))
        image_data4, nw, nh = resize_image(image4, (self.input_shape[1], self.input_shape[0]))

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        image_data2 = np.expand_dims(np.transpose(preprocess_input(np.array(image_data2, np.float32)), (2, 0, 1)), 0)
        image_data3 = np.expand_dims(np.transpose(preprocess_input(np.array(image_data3, np.float32)), (2, 0, 1)), 0)
        image_data4 = np.expand_dims(np.transpose(preprocess_input(np.array(image_data4, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images2 = torch.from_numpy(image_data2)
            images3 = torch.from_numpy(image_data3)
            images4 = torch.from_numpy(image_data4)
            if self.cuda:
                images = images.cuda()
                images2 = images2.cuda()
                images3 = images3.cuda()
                images4 = images4.cuda()

            #---------------------------------------------------#
            #   Image prediction
            #---------------------------------------------------#
            pr = self.net(images, images2, images3, images4)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))

        return image


    
