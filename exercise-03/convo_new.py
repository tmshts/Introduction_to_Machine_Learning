from PIL import Image
import numpy as np
from scipy.signal import convolve
from scipy.signal import gaussian
import matplotlib.pyplot as plt
import math
import cv2


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
    #print("Kernel:")
    #print(kernel)
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
    #print('Padded_array:')
    #print(image_array)
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
    
    im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    #input = im.transpose()
    ## print(im)
    ## print(imt)

    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]

    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result

    # input matrix
    #input = np.array([[1, 2, 3], [4, 5, 6], [7, 45, 100], [1, 1, 5]])
    #input = np.ndarray(height1, width1)
    #print("Original input image")
    #print(input)

    # convolved matrix
    blurR = slow_convolve(R, kernel)
    print("Blured R picture")
    print(blurR)

    blurG = slow_convolve(G, kernel)
    print("Blured G picture")
    print(blurG)

    blurB = slow_convolve(B, kernel)
    print("Blured B picture")
    print(blurB)

    substructR = np.zeros(blurR.shape)
    substructR = R - blurR

    substructG = np.zeros(blurG.shape)
    substructG = G - blurG

    substructB = np.zeros(blurB.shape)
    substructB = B - blurB

    # substruct matrix
    #substruct = np.zeros(blur.shape)
    #for i in range(0, blur.shape[0]):
    #    for j in range(0, blur.shape[1]):
    #        substruct[i][j] = input[i][j] - blur[i][j]
    #print("Substruct matrix")
    #print(substruct)

    resultR = np.zeros(blurR.shape)
    resultR = R + substructR

    resultG = np.zeros(blurG.shape)
    resultG = G + substructG

    resultB = np.zeros(blurB.shape)
    resultB = B + substructB

    # result matrix
    #result = np.zeros(blur.shape)
    #for i in range(0, blur.shape[0]):
    #    for j in range(0, blur.shape[1]):
    #        result[i][j] = input[i][j] + substruct[i][j]
    #print("Result matrix")
    #print(result)

    # clip the values to the range 0 - 255
    minimum = 0
    maximum = 255
    clippedR = resultR.copy()
    clippedR[clippedR<minimum] = minimum
    clippedR[clippedR>maximum] = maximum
    print("Clipped matrix R")
    print(clippedR)

    clippedG = resultG.copy()
    clippedG[clippedG<minimum] = minimum
    clippedG[clippedG>maximum] = maximum
    print("Clipped matrix G")
    print(clippedG)

    clippedB = resultB.copy()
    clippedB[clippedB<minimum] = minimum
    clippedB[clippedB>maximum] = maximum
    print("Clipped matrix B")
    print(clippedB)

    # convert data type from float64 into uint8
    output_arrayR = clippedR.astype('uint8')
    output_arrayG = clippedG.astype('uint8')
    output_arrayB = clippedB.astype('uint8')

    

    #ultimate = output_arrayR + output_arrayG + output_arrayB
    ultimate = np.array([output_arrayR.T, output_arrayG.T, output_arrayB.T]).T
    #print(output_array.dtype)
    #print("Final array in unit8")
    #print(output_array)
    #print("Original input array")
    #print(input)

    #output = output_array.transpose()

    #interpolation='nearest'
    plt.imshow(im)
    plt.show()

    plt.imshow(ultimate)
    plt.show()