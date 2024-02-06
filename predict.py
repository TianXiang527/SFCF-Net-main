#----------------------------------------------------#
#   Perform the prediction of EIT images
#   @author:  Xiang Tian
#----------------------------------------------------#

import cv2
import numpy as np
from PIL import Image
from Prediction_parameter import Sfcf_net

if __name__ == "__main__":
    deeplab = Sfcf_net()
    #-------------------------------------------------------------------------#
    #   The mode of prediction
    #   dir_predict: multi-frame prediction
    #   predict: single-frame prediction
    #-------------------------------------------------------------------------#
    mode = "dir_predict"

    #-------------------------------------------------------------------------#
    #   dir_origin_path:     The file path of the images used for prediction.
    #   dir_save_path:       The saved path of the predicted images.
    #-------------------------------------------------------------------------#
    dir_origin_path = "image_input/"
    dir_save_path   = "image_output/"
    #-------------------------------------------------------------------------#

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = deeplab.detect_image(image)
                r_image.show()

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        dir_f1_path = os.path.join(dir_origin_path, "F1")
        img_names = os.listdir(dir_f1_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_f1_path = os.path.join(dir_f1_path, img_name)
                image_f2_path = os.path.join(os.path.join(dir_origin_path, "F2"), img_name)
                image_f3_path = os.path.join(os.path.join(dir_origin_path, "F3"), img_name)
                image_f4_path = os.path.join(os.path.join(dir_origin_path, "F4"), img_name)

                image = Image.open(image_f1_path)
                image2 = Image.open(image_f2_path)
                image3 = Image.open(image_f3_path)
                image4 = Image.open(image_f4_path)

                r_image     = deeplab.detect_image(image,image2,image3,image4)

                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'dir_predict'.")
