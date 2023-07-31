'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''

    rx = len(Rx)
    ry = len(Ry)
    sum = 0
    for i in range(rx):
        sum = sum + np.absolute(Rx[i] - Ry[i])
    similarity_index = (1/rx) * sum

    return similarity_index

def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''

    thetax = len(Thetax)
    thetay = len(Thetay)
    sumxx = 0
    Ixx = 0
    for i in range(thetax):
        sumxx = sumxx + Thetax[i]
    for j in range(thetax):
        Ixx = Ixx + ((Thetax[j] - (1/thetax * sumxx))**2)

    sumyy = 0
    Iyy = 0
    for a in range(thetay):
        sumyy = sumyy + Thetay[a]
    for b in range(thetax):
        Iyy = Iyy + ((Thetay[b] - (1/thetay * sumyy))**2)

    Ixy = 0
    for k in range(thetax):
        Ixy = Ixy + ((Thetax[k] - (1/thetax * sumxx)) * (Thetay[k] - (1/thetay * sumyy)))

    similarity_index = (1 - ((Ixy * Ixy) / (Ixx * Iyy))) * 100
    
    return similarity_index
