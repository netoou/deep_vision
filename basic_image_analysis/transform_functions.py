import cv2
import numpy as np


def simple_blurring(image:np.array, kernel_size):
    """
    Basic of basic image blurring function, in other words average filter
    :param image: target image
    :param kernel_size: shape of kernel (filter)
    :return: blurred image
    """
    if type(kernel_size) == int:
        kernel = np.ones((kernel_size, kernel_size), np.float) / (kernel_size ** 2)
    elif type(kernel_size) == tuple:
        kernel = np.ones(kernel_size, np.float) / (kernel_size[0] * kernel_size[1])
    else:
        raise TypeError("kernel_size parameter only takes int, tuple type.")

    image = np.array(image)
    image = cv2.filter2D(image, -1, kernel)

    return image

def low_pass_filter_fourier(image_list:list):
    # TODO Complete smoothing function, study what is low pass filter, It's a kind of algorithm for signal processing but it used to computer vision

    return 0

def band_pass_filter(image_list:list):
    # TODO Similar way to above
    return 0

if __name__=='__main__':
    print(1234)