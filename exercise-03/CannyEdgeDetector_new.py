import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import math
from PIL import Image, ImageOps

#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
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

    #print("norma_kernel:")
    #print(norma_kernel)
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    # new image with the same size as input image
    new_image = np.zeros(img_in.shape)
    # size of the border depending on the kernel
    round_height = math.floor(kernel_height / 2)
    round_width = math.floor(kernel_width / 2)

    # Flip the kernel
    #k = np.flipud(np.fliplr(kernel))
    kernel = np.flip(norma_kernel)

    # pad input image with zeros to the correct size depending on kernel
    image_array = np.pad(img_in, (round_height, round_width), constant_values=(0, 0))
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
            new_image[i-round_height][j-round_width] = sum
            #print(image_array[i][j])

    #print("New image array:")
    #print(new_image)
    #print(new_image.dtype)

    # convert data type from float64 into int
    output_array = new_image.astype(np.int_)
    #print(output_array.dtype)
    #print("Final array in int")
    #print(output_array)
    #print("Original input array")
    #print(img_in)

    return kernel, output_array

def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    #print(gx)
    #print(gy)

    height, width = gx.shape
    #heighty, widthy = gy.shape

    # create new images for each direction
    new_sobelx = np.zeros(img_in.shape)
    new_sobely = np.zeros(img_in.shape)
    
    # size of the border depending on the kernel
    round_height = math.floor(height / 2)
    round_width = math.floor(width / 2)

    # Flip the kernel in direction X
    sobelx = np.flip(gx)
    #sobely = np.flip(gy)
    # leave the kernel in direction Y
    sobely = gy

    #print(sobelx)
    #print(sobely)

    # pad input image for x with zeros to the correct size depending on kernel
    image_arrayx = np.pad(img_in, (round_height, round_width), constant_values=(0, 0))
   # pad input image for y with zeros to the correct size depending on kernel
    image_arrayy = np.pad(img_in, (round_height, round_width), constant_values=(0, 0))
    #print('Padded_array:')
    #print(image_arrayy)

    # CONVOLUTION for X direction
    # loop in the padded image without zeros padding
    for i in range(round_height, image_arrayx.shape[0] - round_height):
        for j in range(round_width, image_arrayx.shape[1] - round_width):
            sumx = 0
            # loop in the sobelx matrix
            for u in range(height):
                for v in range(width):
                    sumx = sumx + sobelx[u, v] * image_arrayx[i+u-round_height, j+v-round_width]
                    #print(kernel[u][v])
                    #print(image_array[i+u-round_height, j+v-round_width])
            new_sobelx[i-round_height][j-round_width] = sumx
            #print(image_array[i][j])

    #print("New image array for X direction:")
    #print(new_sobelx)
    
    # CONVOLUTION for Y direction
    # loop in the padded image without zeros padding
    for a in range(round_height, image_arrayy.shape[0] - round_height):
        for b in range(round_width, image_arrayy.shape[1] - round_width):
            sumy = 0
            # loop in the sobely matrix
            for q in range(height):
                for w in range(width):
                    sumy = sumy + sobely[q, w] * image_arrayy[a+q-round_height, b+w-round_width]
            new_sobely[a-round_height][b-round_width] = sumy
            #print(image_array[i][j])

    #print("New image array for Y direction:")
    #print(new_sobely)

    """
    new_sobelx = convolve(image_arrayx, sobelx)
    print('Convolved array:')
    print(new_sobelx)
    
    new_sobely = convolve(image_arrayy, sobely)
    print('Convolved array:')
    print(new_sobely)
    """

    new_sobelx = new_sobelx.astype(np.int_)
    new_sobely = new_sobely.astype(np.int_)

    return new_sobelx, new_sobely


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # create 2D array for gradient magnitude
    g = np.zeros(gx.shape)
    # fill the gradient magnitude based on the formula
    for i in range(0, gx.shape[0]):
        for j in range(0, gx.shape[1]):
            # same shape of gx and gy = same indexing
            g[i][j] = math.sqrt((gx[i][j]**2 + gy[i][j]**2))
    #print(g)

    # create 2D array for gradient theta
    theta = np.zeros(gx.shape)
    # fill the gradient theta based on the formula - to find out DIRECTION
    for a in range(0, gx.shape[0]):
        for b in range(0, gx.shape[1]):
            # same shape of gx and gy = same indexing
            theta[a][b] = np.arctan2(gy[a][b], gx[a][b])
    #print(theta)

    g = g.astype(np.int_)
    #theta = theta.astype(np.int_)

    #print(g.dtype)
    #print("Gradient magnitude:")
    #print(g)
    #print(theta.dtype)
    #print("Theta:")
    #print(theta)

    return g, theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    #convert radians into degree
    degree = angle * 180 / np.pi
    # degree within the interval 0,180
    while degree > 180:
        degree = degree - 180
    while degree < 0:
        degree = degree + 180

    near_degree = 0
    if 0 <= degree < 22.5 or 157.5 <= degree <= 180:
        near_degree = near_degree + 0 
    elif 22.5 <= degree < 67.5:
        near_degree = near_degree + 45
    elif 67.5 <= degree < 112.5:
        near_degree = near_degree + 90
    elif 112.5 <= degree < 157.5:
        near_degree = near_degree + 135

    return near_degree

def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    round_height = 1
    round_width = 1
    # pad input image with zeros
    padded_array = np.pad(g, (round_height, round_width), constant_values=(0, 0))
    #print(padded_array)
    max_sup = np.zeros(g.shape)
    # loop in the padded image without zeros padding
    for i in range(round_height, padded_array.shape[0] - round_height):
        for j in range(round_width, padded_array.shape[1] - round_width):
            # to convert angle in theta to degree
            degree = theta[i-1][j-1] * (180 / np.pi)
            #print(degree)
            # degree within the interval 0,180
            degree = int(degree)
            while degree > 180:
                degree = degree - 180
            while degree < 0:
                degree = degree + 180

            #print(degree)
            # convert degree back to radian
            radian_value = degree * (math.pi / 180)
            # use the convertAngle to find the nearest neighbor in degree
            nearest_neighbor = convertAngle(radian_value)
            # according to the angle find out the local maximum of 3 PIXELS in that direction
            left = 0
            right = 0
            if nearest_neighbor == 0:
                left =padded_array[i][j-1]
                right =padded_array[i][j+1]
            elif nearest_neighbor == 45:
                left = padded_array[i+1][j-1]
                right = padded_array[i-1][j+1]
            elif nearest_neighbor == 90:
                left = padded_array[i+1][j]
                right = padded_array[i-1][j]
            elif nearest_neighbor == 135:
                left = padded_array[i-1][j-1]
                right = padded_array[i+1][j+1]
            
            if (padded_array[i][j] >= left) and (padded_array[i][j] >= right):
                max_sup[i-1][j-1] = padded_array[i][j]
            else:
                max_sup[i-1][j-1] = 0
    return max_sup

def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    threshimg = np.zeros(max_sup.shape)
    for i in range(0, max_sup.shape[0]):
        for j in range(0, max_sup.shape[1]):
            if max_sup[i][j] <= t_low:
                threshimg[i][j] = 0
            elif max_sup[i][j] > t_low and max_sup[i][j] <= t_high:
                threshimg[i][j] = 1
            elif max_sup[i][j] > t_high:
                threshimg[i][j] = 2
    #print("Thresimg:")
    #print(threshimg)
    result = np.zeros(threshimg.shape)
    round_height = 1
    round_width = 1
    # pad thresimg image with zeros
    padded_array = np.pad(threshimg, (round_height, round_width), constant_values=(0, 0))
    for a in range(round_height, padded_array.shape[0] - round_height):
        for b in range(round_width, padded_array.shape[1] - round_width):
            if padded_array[a][b] == 2:
                # attention - different moving indexing
                result[a-round_height][b-round_height] = 255
                if padded_array[a][b-1] == 1:
                    result[a - round_height][b-1 - round_height] = 255
                if padded_array[a][b+1] == 1:
                    result[a - round_height][b+1 - round_height] = 255
                if padded_array[a+1][b-1] == 1:
                    result[a+1 - round_height][b-1 - round_height] = 255
                if padded_array[a-1][b+1] == 1:
                    result[a-1 - round_height][b+1 - round_height] = 255
                if padded_array[a+1][b] == 1:
                    result[a+1 - round_height][b - round_height] = 255
                if padded_array[a-1][b] == 1:
                    result[a-1 - round_height][b - round_height] = 255
                if padded_array[a-1][b-1] == 1:
                    result[a-1 - round_height][b-1 - round_height] = 255
                if padded_array[a+1][b+1] == 1:
                    result[a+1 - round_height][b+1 - round_height] = 255
    #print("Padded array:")
    #print(padded_array)
    #print("result")
    #print(result)

    return result


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result

if __name__ == '__main__':
    """
    img = np.array([[1, 2, 3, 6, 8], [6, 5, 4, 5, 6], [1, 1, 7, 45, 100], [2, 4, 1, 1, 5], [1, 9, 9, 1, 5], [1, 1, 7, 6, 5], [1, 1, 3, 2, 5]])
    print("2D array picture:")
    print(img)
    kernel, gauss = gaussFilter(img, 5, 2)
    print("Kernel:")
    print(kernel)
    print("Gauss:")
    print(gauss)

    gx, gy = sobel(img)
    print("After sobel filter x direction:")
    print(gx)
    print("After sobel filter y direction:")
    print(gy)

    g, theta = gradientAndDirection(gx, gy)

    print("Gradient magnitude:")
    print(g)
    print("Theta:")
    print(theta)

    maxS_img = maxSuppress(g, theta)
    print("Max suppresion:")
    print(maxS_img)

    result = hysteris(maxS_img, 50, 75)
    print("Result:")
    print(result)
    """

    im = np.array(Image.open('contrast.jpg'))

    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]

    #add = R + G + B

    result = canny(R)

    #result = canny(R)
    #result = canny(G)
    #result = canny(B)

