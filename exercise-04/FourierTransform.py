'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.core.fromnumeric import size
# do not import more modules!



def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center...intensity of a palmprint s creases, see page 10
    :param theta: angle...direction of palmprint s features, see page 10
    :return: y, x
    '''
    # use a polar coordination system, (r,θ), to represent the frequency domain images
    # shape of the image under right-angle coordination system ... divided by 2
    right_angle_y = shape[0]/2
    right_angle_x = shape[1]/2

    #x = int(64 + r * np.cos(theta))
    #y = int(64 + r * np.sin(theta))

    # cartesian coordinate calculation  
    x = int(right_angle_x + r * np.cos(theta))
    y = int(right_angle_y + r * np.sin(theta))
    return y, x


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    :param img:
    :return:
    '''
    # calculate frequency transform which is a complex array...situated at top left corner
    fourier_transform = np.fft.fft2(img)
    shiftet_to_center = np.fft.fftshift(fourier_transform)

    # Make sure to compute the (absolute) magnitude
    magnitude_spectrum = 20 * np.log10(np.abs(shiftet_to_center))

    return magnitude_spectrum

def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''
    # see formula on the page 10
    # theta = 0
    rings = np.zeros(k)

    # energy in each ring-like area inclusive summation
    for i in range(1, k + 1):
        #param theta: angle
        #param r: radius
            for j in range(sampling_steps):
                theta = np.pi * j / (sampling_steps - 1)
                for r in range(k * (i - 1), k * i +1):
                    coordination = polarToKart((magnitude_spectrum.shape[0], magnitude_spectrum.shape[1]), r, theta)
                    rings[i - 1] += magnitude_spectrum[coordination]
    #print(rings)
    return rings

def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area
    :return: feature vector of length k
    """

    fans = np.zeros(k)

    # energy in each fan-like area inclusive summation
    for i in range(1, k + 1):
        #param theta: angle
        #param r: radius
            for j in range(sampling_steps):
                theta = np.pi * j / (sampling_steps - 1)
                for r in range(0, 64):
                    coordination = polarToKart((magnitude_spectrum.shape[0], magnitude_spectrum.shape[1]), r, (theta*np.pi)/k)
                    fans[i - 1] += magnitude_spectrum[coordination]
    #print(fans)
    return fans

def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''

    rings = np.zeros(size(k))
    fans = np.zeros(size(k))
    
    magnitude_spectrum = calculateMagnitudeSpectrum(img)

    rings = extractRingFeatures(magnitude_spectrum, k, sampling_steps)

    fans = extractFanFeatures(magnitude_spectrum, k, sampling_steps)

    return rings, fans

if __name__ == '__main__':
    img1 = cv2.imread('Hand1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('Hand2.jpg', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('Hand3.jpg', cv2.IMREAD_GRAYSCALE)

    magnitude_spectrum = calculateMagnitudeSpectrum(img1)
    #plt.imshow(magnitude_spectrum)
    #plt.show()
    rings = extractRingFeatures(magnitude_spectrum, 6, 10)