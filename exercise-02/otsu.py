import numpy as np
#
# NO OTHER IMPORTS ALLOWED
#

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

def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    #print(img)
    #print(img.shape)
    # iterate through pixel values in img -> the values which are
    # bigger than threshold t are multiplied 255 (white color)
    bin_image = (img > t) * 255
    #print("now binarized image")
    #print(bin_image)
    return bin_image


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    # number of pixels in the image
    count = 0
    for c in range(0, len(hist)):
        count = count + hist[c]
    #print(count)
    # number of background pixels
    p0 = 0
    for zero in range(0, theta + 1):
        p0 = p0 + hist[zero]
    #print(p0)
    # number of foreground (object) pixels
    p1 = 0
    for one in range(theta + 1, len(hist)):
        p1 = p1 + hist[one]
    #print(p1)
    #sum = p0 + p1
    #print("The sum is: ", sum)
    #print(p0, p1)
    # probabilities of background
    #p0 = p0 / count
    # probabilities of foreground
    #p1 = p1 / count
    # p0 + p1 = 1
    #print(p0, p1)
    return [p0, p1]

def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    # class mean for background
    sum_0 = 0
    prob0 = 0
    for zero in range(0, theta+1):
        prob0 = zero * hist[zero]
        sum_0 = sum_0 + prob0
    #print("Sum of p0 is: ", sum_0)
    if p0 == 0:
        mu_0 = 0
    else:
        mu_0 = (1 / p0) * sum_0
    #print("Mu_0 for background: ", mu_0)
    # class mean for foreground
    sum_1 = 0
    prob1 = 0
    for one in range(theta+1, len(hist)):
        prob1 = one * hist[one]
        sum_1 = sum_1 + prob1
    #print("Sum of p1 is: ", sum_1)
    if p1 == 0:
        mu_1 = 0
    else:
        mu_1 = (1 / p1) * sum_1
    #print("Mu_1 for foreground: ", mu_1)
    return [mu_0, mu_1]
    # If I want to call just mu_0 ->
#first = mu_helper(create_greyscale_histogram(img), 124, 0.33, 0.66)[0]
    #print("the first is: ", first)

def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables
    #theta = 0
    #class_variance_max = 0
    class_variance_array = np.zeros([256])

    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1
    # basically I have histogram with absolute values ->
    # -> I convert each absolute value into percentage
    sum = 0
    for index in range(0, len(hist)):
        sum += hist[index]
    #print(sum)
    #print(hist)
    # initialize empty 2D array
    histogram_prob = np.zeros([256])
    #float_formatter = "{:.3f}".format
    #print(histogram_prob)
    for i in range(0, len(hist)):
        histogram_prob[i] = hist[i] / sum
        #histogram_prob[i] = float_formatter(histogram_prob[i])
    #print(histogram_prob)
    #prob = 0
    #for i in range(0, len(histogram_prob)):
    #    prob += histogram_prob[i]
    #print(prob)

    # TODO loop through all possible thetas
    for theta in range(0, len(histogram_prob)):
        # TODO compute p0 and p1 using the helper function
        p0 = p_helper(histogram_prob, theta)[0]
        p1 = p_helper(histogram_prob, theta)[1]
        #print(p0, p1)
        # TODO compute mu and m1 using the helper function
        mu0 = mu_helper(histogram_prob, theta, p0, p1)[0]
        mu1 = mu_helper(histogram_prob, theta, p0, p1)[1]
        #print(mu0, mu1)
        # TODO compute between class variance
        class_variance = p0*p1*((mu1 - mu0)**2)
        class_variance_array[theta] = class_variance
        #print(class_variance)
        # TODO update the threshold
        #if class_variance_max < class_variance:
        #    class_variance_max = class_variance
        #else:
        #    class_variance_max = class_variance_max
    threshold = np.argmax(class_variance_array)
    #print(threshold)
    return threshold


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255] ->
    # -> probably call method def create_greyscale_histogram(img):
    :return: np.ndarray binarized image with values {0, 255} ->
    # I used the above instagram into def calculate_otsu_threshold(hist):
    # -> I get the max threshold which I put into following method
    # -> probably call method def binarize_threshold(img, t):

    '''
    # TODO
    histogram = create_greyscale_histogram(img)
    threshold = calculate_otsu_threshold(histogram)
    image = binarize_threshold(img, threshold)
    #return binarize_threshold(img, calculate_otsu_threshold(create_greyscale_histogram(img)))
    return image