from PIL import Image
import numpy as np
from scipy.signal import convolve
from scipy.signal import gaussian
import matplotlib.pyplot as plt
import math


def make_gauss_kernel(ksize, sigma):
    # implement the Gaussian kernel here
    # create matrix for kernel
    kernel = np.zeros((ksize, ksize))
    #print(kernel)
    k = math.floor(ksize/2)
    base = 1 / (2 * math.pi * sigma**2)
    # fill the zero kernel matrix by the values based on the formula
    for a in range(-k, k+1):
        for b in range(-k, k+1):
            base = 1 / (2 * math.pi * sigma**2)
            expo = np.exp(-(a**2+b**2)/(2*sigma**2))
            kernel[a + k, b + k] = base * expo
    #print("Kernel for loop:")
    #print(kernel)
    # to find sum of all values in kernel matrix
    suma = 0
    for e in range(0, ksize):
        for r in range(0, ksize):
            suma = suma + kernel[e][r]
    #print("suma:")
    #print(suma)
    # normalize the kernel matrix
    norma_kernel = np.zeros((ksize, ksize))
    for u in range(0, ksize):
        for t in range(0, ksize):
            norma_kernel[u][t] = kernel[u][t] / suma
    #print("Normalizovaný kernel for loop:")
    #print(norma_kernel)

    return norma_kernel

    # ************************
    # Using gaussian function
"""     #alt + shift + a
    # create 1D kernel matrix
    kernel1D = gaussian(ksize, sigma)
    # convert 1D kernel matrix to 2D kernel matrix
    kernel2D = np.outer(kernel1D, kernel1D)
    # create new matrix
    norm_kernel2D = np.zeros((ksize, ksize))
    height, width = norm_kernel2D.shape
    sum = 0
    for i in range(0, height):
        for j in range(0, width):
            sum = sum + kernel2D[i][j]
    #print(sum)
    # fill the percentage of the sum of each pixel to get normalized kernel matrix
    for a in range(0, height):
        for b in range(0, width):
            norm_kernel2D[a][b] = kernel2D[a][b]/sum
    print("Kernel gaussian:")
    print(kernel2D)
    print("Normalized kernel gaussian")
    print(norm_kernel2D)
    return norm_kernel2D """

def sigma(ksize):
    sig = ksize / 5
    return sig

def slow_convolve(arr, kernel):
    # implement the convolution with padding here

    #print('Input array:')
    #print(arr)
    #print('Input array shape:')
    #print(arr.shape)
    print("Kernel:")
    print(kernel)
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    # new image with the same size as input image
    new_image = np.zeros(arr.shape)
    # size of the border depending on the kernel
    round_height = math.floor(kernel_height / 2)
    round_width = math.floor(kernel_width / 2)

    # Flip the kernel
    #k = np.flipud(np.fliplr(kernel))
    kernel = np.flip(kernel)

    # probably to get 3 layers of 3 dimensions of the image including zero padding
    #  and multiply with kernel ->
    # -> after that sum it and divide by 3 -> to get convolution
    # -> how do I get the 3 layers
    # shape of input image 1 (378, 394, 3)

    # pad input image with zeros to the correct size depending on kernel
    image_array = np.pad(arr, (round_height, round_width), constant_values=(0, 0))
    print('Padded_array:')
    print(image_array)
    # CONVOLUTION
    # loop in the padded image without zeros padding
    for i in range(round_height, image_array.shape[0] - round_height):
        for j in range(round_width, image_array.shape[1] - round_width):
            sum = 0
            # loop in the kernel matrix
            for u in range(kernel_height):
                for v in range(kernel_width):
                    sum = sum + kernel[u, v] * image_array[i+u-round_height, j+v-round_width]
                    #print(kernel[u][v])
                    #print(image_array[i+u-round_height, j+v-round_width])
            new_image[i-1][j-1] = sum
            #print(image_array[i][j])

    #print("New image array:")
    #print(new_image)

    return new_image

    # Using convolve function
    # convolution - mode=valid output consists of non-zero elements
"""     convolved_array = convolve(image_array, kernel, mode='valid', method='direct')
    print('Convolved array:')
    print(convolved_array)
    print('Convolved array shape:')
    print(convolved_array.shape)

    return convolved_array """


if __name__ == '__main__':
    # choose the kernel size for Gaussian filter
    kernel = make_gauss_kernel(3, sigma(3))   # todo: find better parameters
    #print("Kernel 2D array:")
    #print(kernel)
    #print("Kernel shape:")
    #print(kernel.shape)
    #plt.imshow(kernel)
    #plt.show()
    # TODO: chose the image you prefer
    im1 = np.array(Image.open('input1.jpg'))
    #print(im1.shape)
    im1_2d = im1[:, :, 0]
    #print("2D array from 3D array:")
    #print(im1_2d)
    height1, width1, nothing1 = im1.shape
    #print(height1)
    #print(width1)
    #print(nothing1)
    #im2 = np.array(Image.open('input2.jpg'))
    #print(im2.shape)
    #height2, width2, nothing2 = im2.shape
    #print(height2)
    #print(width2)
    #print(nothing2)
    #im3 = np.array(Image.open('input3.jpg'))
    #print(im3.shape)
    #height3, width3, nothing3 = im3.shape
    #print(height3)
    #print(width3)
    #print(nothing3)

    # Initiate a new image with the same size as the input image
    output_image = np.zeros((height1, width1))
    #print(output_image)
    #print(output_image.shape)
 
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result

    # input matrix
    input = np.array([[1, 2, 3], [4, 5, 6], [7, 45, 100], [1, 1, 5]])
    print("Original input image")
    print(input)

    # convolved matrix
    blur = slow_convolve(input, kernel)
    print("Blured picture")
    print(blur)

    # substruct matrix
    substruct = np.zeros(blur.shape)
    for i in range(0, blur.shape[0]):
        for j in range(0, blur.shape[1]):
            substruct[i][j] = input[i][j] - blur[i][j]
    print("Substruct matrix")
    print(substruct)

    # result matrix
    result = np.zeros(blur.shape)
    for i in range(0, blur.shape[0]):
        for j in range(0, blur.shape[1]):
            result[i][j] = input[i][j] + substruct[i][j]
    print("Result matrix")
    print(result)

    # clip the values to the range 0 - 255
    minimum = 0
    maximum = 255
    clipped = result.copy()
    clipped[clipped<minimum] = minimum
    clipped[clipped>maximum] = maximum
    print("Clipped matrix")
    print(clipped)

    # convert data type from float64 into uint8
    output_array = clipped.astype('uint8')
    print(output_array.dtype)
    print("Final array in unit8")
    print(output_array)
    print("Original input array")
    print(input)

    #plt.imshow(input, interpolation='nearest')
    #plt.show()

    #plt.imshow(output_array, interpolation='nearest')
    #plt.show()