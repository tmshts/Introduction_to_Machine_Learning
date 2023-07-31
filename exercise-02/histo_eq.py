# Implement the histogram equalization in this file
import cv2
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt



def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    #print(img)
    #print(img.shape) # (473, 1095)
    # initialize empty 2D array
    histogram = np.zeros([256], np.int32)
    #print(histogram)
    # iterate through rows and column of the 2D image
    # img[j, k] gives us the pixel value at that position
    # we put that pixel value as the position of histogram
    for j in range(0, img.shape[0]): # max #rows
        for k in range(0, img.shape[1]): # max #columns
            # we count the number of pixel value by + 1
            histogram[img[j][k]] = histogram[img[j][k]] + 1
    #print(histogram.shape)
    #print(histogram)
    return histogram

def cumulative_distribution(histogram):
    count = 0
    for j in range(0, len(histogram)):
        count = count + histogram[j]
    #print(count)
    # create probability distribution
    pX_array = np.zeros([256])
    for k in range(0, len(histogram)):
        pX_array[k] = histogram[k] / count
    #print(pX_array)

    # create cumulative distribution
    cX_array = np.zeros([256])
    for l in range(0, len(pX_array)):
        if l == 0:
            cX_array[l] = pX_array[l]
        else:
            cX_array[l] = cX_array[l-1] + pX_array[l]
    #print(cX_array)

    #print(cX_array)
    #plt.plot(cX_array)
    #plt.show()

    return cX_array

def mapping_array(cX_array):
    # smallest non-zero value
    min_value = np.min(ma.masked_where(cX_array==0, cX_array))
    #print(min_value)

    # mapping from cumulative distribution
    after_mapping_array = np.zeros([256])
    for p in range(0, len(cX_array)):
        after_mapping_array[p] = ((cX_array[p] - min_value)/(1 - min_value)) * 255
    #print(after_mapping_array)

    #plt.plot(after_mapping_array)
    #plt.show()

    return after_mapping_array

if __name__ == '__main__':
    image = cv2.imread('hello.png', cv2.IMREAD_GRAYSCALE)

    histogram = create_greyscale_histogram(image)

    cX_array = cumulative_distribution(histogram)

    mapping = mapping_array(cX_array)

    # mapping array will convert the image
    new_image = mapping[image]

    #print(type(new_image))
    #print(new_image.shape)

    #kitty = cv2.imread('kitty.png', cv2.IMREAD_GRAYSCALE)

    #hist = create_greyscale_histogram(kitty)
    #plt.plot(hist)
    #plt.show()

    #cX_array = cumulative_distribution(hist)
    #plt.plot(cX_array)
    #plt.show()

    # save picture
    cv2.imwrite('kitty.png', new_image)

    plt.imshow(new_image, 'gray')
    plt.show()