'''
Created on 25.11.2017
Modified on 05.12.2020

@author: Daniel, Max, Charly, Mathias
'''
import cv2
import matplotlib.pyplot as plt
from otsu import binarize_threshold, calculate_otsu_threshold, create_greyscale_histogram, mu_helper, otsu, p_helper

img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)

#create_greyscale_histogram(img)
#binarize_threshold(img, 180)
#p0 = p_helper(create_greyscale_histogram(img), 127)[0]
#print("p0 is: ", p0)
#p1 = p_helper(create_greyscale_histogram(img), 127)[1]
#print("p1 is: ", p1)
#first = mu_helper(create_greyscale_histogram(img), 124, 0.33, 0.66)[0]
#print("the first ist: ", first)
#calculate_otsu_threshold(create_greyscale_histogram(img))

res = otsu(img)

plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.title('Original')
if res is not None:
    plt.subplot(1, 2, 2)
    plt.imshow(res, 'gray')
    plt.title('Otsu\'s - Threshold = 120')
plt.show()