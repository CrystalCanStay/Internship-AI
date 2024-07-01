import math
import os
from pprint import pprint
from PIL import Image

import cv2
import numpy as np


#Code to resize images so we can actually use them in our CNN
def bl_resize(original_img, new_h, new_w):
    """
        Resizes an image given a file path, a new height for the image and a new width for the image

        Args:
            original_img (string): Directory of image.
            new_h (int): New size of the height of the image
            new_w (int): New size of the width of the image

        Returns:
            The resized image as an array
    """


    #get dimensions of original image
    old_h, old_w, c = original_img.shape
    #create an array of the desired shape.
    #We will fill-in the values later.
    resized = np.zeros((new_h, new_w, c))
    #Calculate horizontal and vertical scaling factor
    w_scale_factor = (old_w) / (new_w) if new_h != 0 else 0
    h_scale_factor = (old_h) / (new_h) if new_w != 0 else 0
    for i in range(new_h):
        for j in range(new_w):
            #map the coordinates back to the original image
            x = i * h_scale_factor
            y = j * w_scale_factor
            #calculate the coordinate values for 4 surrounding pixels.
            x_floor = math.floor(x)
            x_ceil = min(old_h - 1, math.ceil(x))
            y_floor = math.floor(y)
            y_ceil = min(old_w - 1, math.ceil(y))
            if (x_ceil == x_floor) and (y_ceil == y_floor):
                q = original_img[int(x), int(y), :]
            elif (x_ceil == x_floor):
                q1 = original_img[int(x), int(y_floor), :]
                q2 = original_img[int(x), int(y_ceil), :]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif (y_ceil == y_floor):
                q1 = original_img[int(x_floor), int(y), :]
                q2 = original_img[int(x_ceil), int(y), :]
                q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
            else:
                v1 = original_img[x_floor, y_floor, :]
                v2 = original_img[x_ceil, y_floor, :]
                v3 = original_img[x_floor, y_ceil, :]
                v4 = original_img[x_ceil, y_ceil, :]
                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            resized[i, j, :] = q
    return resized.astype(np.uint8)





def convert_data(source_name, new_h, new_w):
    """
            Resizes all images in a given folder to a new height and width
            May take a few minutes depending on how many images are in the file

            Args:
                source_name (string): Directory of all images
                new_h (int): New size of the height of the image
                new_w (int): New size of the width of the image

            Returns:
                Resized images in the source folder location
    """




    files = []
    for dirname, dirnames, filenames in os.walk(source_name):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            files.append(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            files.append(os.path.join(dirname, filename))
    for i in range(len(files)):
        imagename = files[i]
        image = cv2.imread(imagename)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        new_image = bl_resize(img, new_h, new_w)
        print(new_image.shape)
        final_image = Image.fromarray(new_image)
        final_image.save(imagename, 'JPEG')
