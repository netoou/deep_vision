import cv2
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import convolve2d


def simple_blurring(image:np.array, kernel_size):
    """
    Basic of basic image blurring function, in other words average filter
    :param image: target image()
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

def low_pass_filter_fourier(image:np.array, kernel_size):
    """
    Image blurring
    Original image -> fft -> convolution -> ifft
    :param image: input image, 3 channels image was work well, grayscale image have showed darker in the middle
    :param kernel_size: kernel
    :return: smoothed image
    """
    # Step1. pre-process image
    # Step2. Fourier transform the image
    # Step3. Maybe apply filter to transformed image
    # Step4. Restore image to inverse fourier transform
    # Step5. post-processing
    if type(kernel_size) == int:
        kernel = np.ones((kernel_size, kernel_size), np.float) / (kernel_size ** 2)
    elif type(kernel_size) == tuple:
        kernel = np.ones(kernel_size, np.float) / (kernel_size[0] * kernel_size[1])
    else:
        raise TypeError("kernel_size parameter only takes int, tuple type.")
    # skip step1
    image = np.array(image)
    # step2
    image = fft(image)
    # step3
    for i in range(3):
        image[:, :, i] = convolve2d(image[:, :, i], kernel, 'same')
    # step4
    image = ifft(image)
    image = np.array(image, dtype=np.uint8)
    # skip step5
    # TODO FFT, IFFT consumes much time, need to faster, use pycuda or cufft or torch, tensorflow to speed up
    # Done, complete smoothing function, study what is low pass filter, It's a kind of algorithm for signal processing but it used to computer vision

    return image

def band_pass_filter(image_list:list):
    # TODO Similar way to above
    return 0

if __name__=='__main__':
    ##### opencv idft cannot resotre original matrix
    rimg = np.random.randn(300,400)
    nrimg = np.array(rimg)
    cp = np.array(rimg)
    print(rimg.shape)
    cv2.dft(rimg, rimg)
    print(rimg.shape)
    cv2.idft(rimg, rimg)
    print(rimg.shape)
    print('-'*40)
    print(nrimg.shape)
    nrimg = np.fft.fftn(nrimg)
    print(nrimg.shape)
    nrimg = np.fft.ifftn(nrimg)
    print(nrimg.shape)
    print("true" if np.all(nrimg == cp) else "False")