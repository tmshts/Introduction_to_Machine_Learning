import glob
from math import ceil
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.svm import SVC

import glob

# image size
N = 64

# Define the classifier in clf - Try a Support Vector Machine with C = 0.025 and a linear kernel
# DON'T change this!
clf = SVC(kernel="linear", C=0.025)


def create_database_from_folder(path):
    '''
    DON'T CHANGE THIS METHOD.
    If you run the Online Detection, this function will load and reshape the
    images located in the folder. You pass the path of the images and the function returns the labels,
    training data and number of images in the database
    :param path: path of the training images
    :return: labels, training images, number of images
    '''
    labels = list()
    filenames = np.sort(path)
    num_images = len(filenames)
    train = np.zeros((N * N, num_images))
    for n in range(num_images):
        img = cv2.imread(filenames[n], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (N, N))
        assert img.shape == (N, N), 'Image {0} of wrong size'.format(filenames[n])
        train[:, n] = img.reshape((N * N))
        labels.append(filenames[n].split("eigenfaces\\")[1].split("_")[0])
    print('Database contains {0} images'.format(num_images))
    labels = np.asarray(labels)
    return labels, train, num_images


def process_and_train(labels, train, num_images, h, w):
    '''
    Calculate the essentials: the average face image and the eigenfaces.
    Train the classifier on the eigenfaces and the given training labels.
    :param labels: 1D-array
    :param train: training face images, 2D-array with images as row vectors (e.g. 64x64 image ->  4096 vector)
    :param num_images: number of images, int
    :param h: height of an image
    :param w: width of an image
    :return: the eigenfaces as row vectors (2D-array), number of eigenfaces, the average face
    '''

    # Compute the average face --> calculate_average_face()
    average_face_1D = calculate_average_face(train)
    # calculate the maximum number of eigenfaces
    number_of_eigenfaces = num_images - 1
    # calculate the eigenfaces --> calculate_eigenfaces()
    eigenfaces_u = calculate_eigenfaces(train, average_face_1D, number_of_eigenfaces, h, w)
    # calculate the coefficients/features for all images --> get_feature_representation()
    coefficient_array = get_feature_representation(train, eigenfaces_u, average_face_1D, number_of_eigenfaces)
    # train the classifier using the calculated features
    clf.fit(coefficient_array, labels)
    return eigenfaces_u, number_of_eigenfaces, average_face_1D


def calculate_average_face(train):
    '''
    Calculate the average face using all training face images
    :param train: training face images, 2D-array with images as row vectors
    :return: average face, 1D-array shape(#pixels)
    '''
    # array storing the number of pictures - each picture stores number of pixels
    sum_of_all_faces_array = np.zeros_like(train[0])
    # train[number_of_images, dimensions_of_image]
    # one image takes the amount of row as the size of column for this picture
    len_of_number_of_images = len(train[:, 0])

    for i in range(len_of_number_of_images):
        # : represents all the pixels of the image
        sum_of_all_faces_array += train[i, :]

    average_face_1D = (1 / len_of_number_of_images) * sum_of_all_faces_array
    print("Average face 1D:")
    print(average_face_1D)
    
    return average_face_1D
    #return np.mean(train, 0)
    

def calculate_eigenfaces(train, avg, num_eigenfaces, h, w):
    '''
    Calculate the eigenfaces from the given training set using SVD
    :param train: training face images, 2D-array with images as row vectors
    :param avg: average face, 1D-array
    :param num_eigenfaces: number of eigenfaces to return from the computed SVD
    :param h: height of an image in the training set
    :param w: width of an image in the training set
    :return: the eigenfaces as row vectors, 2D-array --> shape(num_eigenfaces, #pixel of an image)
    '''

    # subtract the average face from every training sample
    #X = train - np.matlib.repmat(avg.reshape((N * N, 1)), 1, len(train[0]))
    # 2D array for all the images and their pixels -> train
    X = np.zeros_like(train)
    # same like in the previous method
    len_of_number_of_images = len(train[:, 0])
    for i in range(len_of_number_of_images):
        X[i, :] = train[i, :] - avg

    # compute the eigenfaces using svd
    # You might have to swap the axes so that the images are represented as column vectors
    transposed_X = X.transpose()
    columns_u, array_of_values_s, rows_v = np.linalg.svd(transposed_X)

    # represent your eigenfaces as row vectors in a 2D-matrix & crop it to the requested amount of eigenfaces
    #cov = np.cov(train)
    eigenfaces_u = np.zeros((num_eigenfaces, columns_u.shape[0]))
    for i in range(num_eigenfaces):
        eigenfaces_u[i, :] = columns_u[i, :]

    # plot one eigenface to check whether you're using the right axis
    # comment out when submitting your exercise via studOn
    #plt.imshow(eigenfaces[5].reshape(h, w), cmap='gray')
    #plt.show()

    return eigenfaces_u


def get_feature_representation(images, eigenfaces, avg, num_eigenfaces):
    '''
    For all images, compute their eigenface-coefficients with respect to the given amount of eigenfaces
    :param images: 2D-matrix with a set of images as row vectors, shape (#images, #pixels)
    :param eigenfaces: 2D-array with eigenfaces as row vectors, shape(#pixels, #pixels)
                       -> only use the given number of eigenfaces
    :param avg: average face, 1D-array
    :param num_eigenfaces: number of eigenfaces to compute coefficients for
    :return: coefficients/features of all training images, 2D-matrix (#images, #used eigenfaces)
    '''

    # compute the coefficients for all images and save them in a 2D-matrix
    # 1. iterate through all images (one image per row)
    # 1.1 compute the zero mean image by subtracting the average face
    zero_mean_array = np.zeros_like(images)
    # same like in the previous method
    len_of_number_of_images = len(images[:, 0])
    for i in range(len_of_number_of_images):
        zero_mean_array[i, :] = images[i, :] - avg
    # 1.2 compute the image's coefficients for the expected number of eigenfaces
    # dimension [number of images, number of eigenfaces]
    coefficient_array = np.zeros((images.shape[0], num_eigenfaces))
    # for all images
    for i in range(images.shape[0]):
        # for given amount of eigenfaces
        for j in range(num_eigenfaces):
            coefficient_array[i, j] = np.dot(zero_mean_array[i, :], eigenfaces[j, :])
    return coefficient_array


def reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    '''
    Reconstruct the given image by weighting the eigenfaces according to their coefficients
    :param img: input image to be reconstructed, 1D array
    :param eigenfaces: 2D array with all available eigenfaces as row vectors
    :param avg: the average face image, 1D array
    :param num_eigenfaces: number of eigenfaces used to reconstruct the input image
    :param h: height of a original image
    :param w: width of a original image
    :return: the reconstructed image, 2D array (shape of a original image)
    '''
    # reshape the input image to fit in the feature helper method -> I need 2D array
    reshaped_image = img.reshape(1, h * w)

    # compute the coefficients to weight the eigenfaces --> get_feature_representation()
    coefficient_array = get_feature_representation(reshaped_image, eigenfaces, avg, num_eigenfaces)

    # use the average image as starting point to reconstruct the input image
    #starting_point = np.zeros_like(avg)
    #for i in range(len(starting_point)):
    #    starting_point[i] = avg[i]

    # reconstruct the input image using the coefficients
    for i in range(num_eigenfaces):
        avg = avg + (coefficient_array[0, i] * eigenfaces[i])

    # reshape the reconstructed image back to its original shape
    reshaped_final_image = avg.reshape(h, w)
    
    return reshaped_final_image


def classify_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    '''
    Classify the given input image using the trained classifier
    :param img: input image to be classified, 1D-array
    :param eigenfaces: all given eigenfaces, 2D array with the eigenfaces as row vectors
    :param avg: the average image, 1D array
    :param num_eigenfaces: number of eigenfaces used to extract the features
    :param h: height of a original image
    :param w: width of a original image
    :return: the predicted labels using the classifier, 1D-array (as returned by the classifier)
    '''

    # reshape the input image as an matrix with the image as a row vector
    reshaped_image = img.reshape(1, h * w)
    # extract the features/coefficients for the eigenfaces of this image
    coefficient_array = get_feature_representation(reshaped_image, eigenfaces, avg, num_eigenfaces)
    # predict the label of the given image by feeding its coefficients to the classifier
    label = clf.predict(coefficient_array)
    # train the classifier using the calculated features
    
    return label


if __name__ == '__main__':
    labels, train, num_images = create_database_from_folder(glob.glob('eigenfaces\\*.png'))

    print("Labels: ")
    print(labels)
    print("Train data: ")
    print(train)
    print("Number of images: ")
    print(num_images)

    average_face_1D = calculate_average_face(train)
