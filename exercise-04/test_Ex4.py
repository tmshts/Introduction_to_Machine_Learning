'''
Created on 03.01.2021
@author: Max, Charly
'''

import unittest
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PalmprintAlignmentAutomatic import binarizeAndSmooth, drawLargestContour, getFingerContourIntersections, \
    findKPoints, getCoordinateTransform, palmPrintAlignment
from FourierTransform import polarToKart, calculateMagnitudeSpectrum, extractRingFeatures, extractFanFeatures, \
    calcuateFourierParameters
from DistanceMeasure import calculate_R_Distance, calculate_Theta_Distance


def get_center(M):
    alpha_1 = 1 - M[0, 0]
    beta = M[0, 1]
    # cv2 formal naming: x = first center point coordinate
    cx = np.round(((M[1, 2] * alpha_1) - (beta * M[0, 2])) / (np.power(beta, 2) + np.power(alpha_1, 2)))
    cy = np.round((M[0, 2] + (beta * cx)) / alpha_1)
    return cx, cy


class TestPalmprint(unittest.TestCase):
    def setUp(self) -> None:
        self.img = cv2.imread('Hand1.jpg', cv2.IMREAD_GRAYSCALE)

    def test_binarizeAndSmooth(self):
        test_0 = np.ones((5, 5))
        test_0[2, 2] = 115
        result_0 = binarizeAndSmooth(test_0)
        self.assertFalse(np.abs(result_0[2, 2]) > 0.1, msg='Check your threshold, 115 --> background')

        test_1 = np.ones((5, 5))
        test_1[2, 2] = 116
        result_1 = binarizeAndSmooth(test_1)

        self.assertIsInstance(result_1, np.ndarray, msg='Check which return value you get from the cv2 thresholding')
        self.assertNotEqual(result_1[1, 1], 0, msg='Check your processing order')
        self.assertNotEqual(result_1[0, 0], 0, msg='Check the size of your Gaussian kernel')
        self.assertTrue(np.allclose(result_1[:, :2], (result_1[:, 3:])[:, [1, 0]]),
                        msg='Check the size of your Gaussian kernel')
        self.assertTrue(np.abs(result_1[0, 4] - 4) < 1, msg='Check your Gaussian Filter')

    def test_drawLargestContour(self):
        contour_1 = drawLargestContour(np.eye(19, 22, dtype=np.uint8))
        row = contour_1[0, :]
        self.assertTrue(contour_1.shape == (19, 22), msg='Check your contour_img shape')
        self.assertTrue(row[0] >= row[1] >= row[2] >= row[3], msg='Black background, bright contour')
        self.assertTrue(row[1] > row[3], msg='Check your stroke width')

        circles = np.zeros((25, 25), dtype=np.uint8)
        cv2.circle(circles, (12, 12), 3, 255, 1)
        cv2.circle(circles, (12, 12), 9, 255, 1)

        contour_1 = drawLargestContour(circles.astype('uint8'))
        test_circle = np.zeros((25, 25))
        cv2.circle(test_circle, (12, 12), 9, 255, 1)
        self.assertTrue(np.allclose(np.zeros((7, 7)), contour_1[9:16, 9:16]), msg='Only draw the largest contour')
        self.assertTrue(np.all(contour_1[np.where(test_circle == 255)] > 0), msg='Draw the largest contour')

    def test_getFingerContourIntersections(self):
        img = np.zeros((108, 11))
        img[0:4] = img[13:18] = img[28:33] = img[43:48] = img[58:63] = img[73:78] = img[88:93] = img[103:108] = 255
        y = getFingerContourIntersections(img, 10)
        self.assertIsInstance(y, np.ndarray, 'Return a nd-array')
        self.assertTrue(len(y) == 6, msg='Trace 6 intersections')
        self.assertTrue(y[0] > 4, msg='Do not trace the images border')
        self.assertTrue(y[5] < 93, msg='Do not trace the images border')
        self.assertTrue(12 < y[0] < 18, msg='Did not find first contour')
        self.assertTrue((np.diff(y) > 10).all, msg='Keep the thickness of your contour in mind')

        img_2 = np.zeros((128, 11))
        img_2[0:4] = 255
        img_2[20:128, 0:11] = img
        y_2 = getFingerContourIntersections(img_2, 5)
        self.assertIsInstance(y, np.ndarray, 'Return a nd-array')
        self.assertTrue(len(y_2) == 6, msg='Trace 6 intersections')
        self.assertTrue(19 < y_2[0] < 25, msg='Did not find first contour')
        self.assertTrue(92 < y_2[5] < 123, msg='Do not trace the images border')
        self.assertTrue(np.all(np.diff(y_2) > 9), msg='Keep the thickness of your contour in mind')

    def test_findKPoints(self):
        img_1 = 255 * np.sum(np.array([np.eye(20, 20, k=i) for i in [-1, 0, 1]]), axis=0)
        res_1 = findKPoints(img_1, 12, 1, 13, 4)
        self.assertIsInstance(res_1, tuple, msg='Did not find an intersection point')
        self.assertTrue(np.abs(res_1[0] - 17) < 4, 'Check your intersection point (y) - visualize your computation')
        self.assertTrue(np.abs(res_1[1] - 16) < 4, 'Check your intersection point (x) - visualize your computation')

        img_2 = np.zeros((40, 30))
        cv2.circle(img_2, (20, 15), 7, 255, 2)
        res_2 = findKPoints(img_2, 28, 4, 25, 7)
        self.assertIsInstance(res_2, tuple, msg='Did not find an intersection point - check x & y order')
        self.assertTrue(np.abs(res_2[0] - 19) < 4, 'Check your intersection point (y) - visualize your computation')
        self.assertTrue(np.abs(res_2[1] - 13) < 4, 'Check your intersection point (x) - visualize your computation')

    def test_getCoordinateTransform(self):
        M_1 = getCoordinateTransform((10, 15), (20, 7), (40, 28))
        cx, cy = get_center(M_1)

        self.assertIsInstance(M_1, np.ndarray, msg='Make sure to return a numpy array')
        self.assertEqual((2, 3), np.shape(M_1), msg='Check the shape of your matrix, use cv2 to generate it')
        self.assertTrue(M_1[0, 0] == M_1[1, 1] and -M_1[0, 1] == M_1[1, 0], msg='Check your rotation matrix')
        self.assertTrue(np.abs(cx - 17) < 5, msg='Check the origin (x) of your new coordinate system')
        self.assertTrue(np.abs(cy - 16) < 5, msg='Check the origin (y) of your new coordinate system')
        self.assertFalse(np.abs(M_1[0, 1]) <= 0.1, msg='Check deg & rad for your rotation angle')
        self.assertTrue(np.round(M_1[0, 1] / np.sin(np.deg2rad(-23.429))) == 1, msg='Set scale to 1 when computing the '
                                                                                    'transform')
        self.assertTrue(np.abs(np.arcsin(M_1[0, 1]) - np.deg2rad(-23.429)) < np.deg2rad(10),
                        msg='Check your rotation angle')

        M_2 = getCoordinateTransform((20, 30), (30, 9), (40, 14))
        cx_2, cy_2 = get_center(M_2)

        self.assertEqual((2, 3), np.shape(M_2), msg='Check the shape of your matrix, use cv2 to generate it')
        self.assertTrue(M_2[0, 0] == M_2[1, 1] and -M_2[0, 1] == M_2[1, 0], msg='Check your rotation matrix')
        self.assertFalse(np.abs(cy_2 - 17) < 5, msg='Check the order of the center of point getRotationMatrix')
        self.assertFalse(np.abs(cx_2 - 36) < 5, msg='Check the order of the center of point getRotationMatrix')
        self.assertTrue(np.abs(cy_2 - 36) < 5, msg='Check the origin (y) of your new coordinate system')
        self.assertTrue(np.abs(cx_2 - 17) < 5, msg='Check the origin (x) of your new coordinate system')
        self.assertFalse(np.abs(M_2[0, 1]) <= 0.1, msg='Check deg & rad for your rotation angle')
        self.assertTrue(np.round(M_2[0, 1] / np.sin(np.deg2rad(38.659))) == 1, msg='set scale to 1')
        self.assertTrue(np.abs(np.arcsin(M_2[0, 1]) - np.deg2rad(38.659)) < np.deg2rad(10),
                        msg='Check your rotation angle')

    def test_palmPrintAlignment(self):
        palm = palmPrintAlignment(np.copy(self.img))
        self.assertIsInstance(palm, np.ndarray, msg='Must return an image as a ndarray')
        self.assertTrue(self.img.shape == palm.shape, msg='Rotated palm must have the same dimensions as input')
        self.assertFalse(np.allclose(self.img, palm, atol=5), msg='You must rotate the image')
        self.assertFalse(np.histogram(palm, bins=256, range=[0, 256])[0][-1] > 2000,
                         msg='Return the rotated input image, not the contour')


class TestFourier(unittest.TestCase):
    def setUp(self) -> None:
        magn = np.zeros((100, 90))
        for index, _ in np.ndenumerate(magn):
            magn[index] = np.linalg.norm(index - (np.array(magn.shape) // 2))
        magn *= (255 / np.max(magn))
        self.magnitude = magn

        self.fan = np.zeros((90, 100))
        self.fan[:, :50] = 255

        self.img = cv2.imread('Hand1.jpg', cv2.IMREAD_GRAYSCALE)

    def test_polarToKart(self):
        a = polarToKart((15, 15), 7, 0.5 * np.pi)
        y_1, x_1 = a[0], a[1]
        self.assertIsInstance(a, tuple, msg='Return a tuple')
        self.assertFalse(x_1 > y_1, msg='Check the order of your returned values')

        y_2, x_2 = polarToKart((10, 15), 6, 0.25 * np.pi)
        self.assertFalse(np.abs(y_2 - 5) < 2 and np.abs(x_2 - 13) < 2, msg='Theta is given in radians')

        y_3, x_3 = polarToKart((9, 15), 11, 0)

        self.assertFalse(np.abs(y_3) < 2, msg='Shift your y to the origin')
        self.assertFalse(np.abs(x_3 - 11) < 2, msg='Shift your x to the origin')
        self.assertFalse(np.abs(y_3 - 7) < 2, msg='Check which shape corresponds to y')
        self.assertFalse(np.abs(x_3 - 15) < 2, msg='Check which shape corresponds to x')

        self.assertTrue(np.abs(y_2 - 9) < 2, msg='Check your y output')
        self.assertTrue(np.abs(x_2 - 11) < 2, msg='Check your x output')
        self.assertTrue(np.abs(y_3 - 4) < 2, msg='Check your y output')
        self.assertTrue(np.abs(x_3 - 18) < 2, msg='Check your x output')

    def test_calculateMagnitudeSpectrum(self):
        img = np.zeros((100, 100))
        img[50] = 255
        img[:, 50] = 255
        spectrum = calculateMagnitudeSpectrum(np.copy(img))
        bin_spectrum = np.zeros_like(spectrum)
        bin_spectrum[np.where(spectrum > 60)] = 255
        self.assertIsInstance(spectrum, np.ndarray, 'Return an spectrum matrix')
        self.assertTrue(spectrum.shape == img.shape, msg='Returned spectrum should have same size as input')
        self.assertLess(np.max(spectrum), 256, msg='Make sure to convert in decibel')
        self.assertNotIsInstance(spectrum[0, 0], np.complex128, msg='Make sure to compute the (absolute) magnitude')
        self.assertTrue(np.allclose(img, bin_spectrum, atol=1), msg='Shift the spectrum to the center')

        sin = np.sin(np.arange(90))
        img_2 = np.vstack([sin] * 80)

        spectrum_2 = calculateMagnitudeSpectrum(np.copy(img_2))  # ignore runtime warning -> log(/0)
        self.assertEqual(img_2.shape, spectrum_2.shape, msg='Returned spectrum should have same size as input')
        self.assertNotIsInstance(spectrum_2[0, 0], np.complex128, msg='Make sure to convert in decibel')
        self.assertFalse(np.allclose(img_2, spectrum_2, atol=1), msg='Return the magnitude spectrum')

    def test_extractRingFeatures(self):
        magn_3 = np.zeros((100, 90))
        magn_3[:50] = 255
        R_3 = extractRingFeatures(np.copy(magn_3), 6, 7)
        self.assertEqual((6,), R_3.shape, msg='Check which input variable resembles the number of ring features')
        self.assertIsInstance(R_3[0], float, msg='Return the features as float values')
        self.assertLess(np.max(R_3), 10000, msg='Check you convert the coordinates back to cartesian')
        self.assertLess(np.max(R_3), 0.1, msg='Check theta\'s upper boundary')

        magn_4 = np.copy(magn_3)
        magn_4[50, : 44] = 255
        R_4 = extractRingFeatures(np.copy(magn_4), 5, 4)
        self.assertGreater(np.min(R_4), 10, msg='Check that you reach theta\'s upper boundary (inclusive summation)')

        mag_0 = 255 * np.ones_like(self.magnitude)
        mag_0[50:] = self.magnitude[50:]
        R_0 = extractRingFeatures(mag_0, 6, 18)
        res_0 = np.array([1266.394, 4125.157, 6999.781, 9881.776, 12742.549, 15571.304])
        self.assertEqual((6,), R_0.shape, msg='Check which input variable resembles the number of ring features')
        self.assertIsInstance(R_0[0], float, msg='Return the features as float values')
        self.assertGreater(R_0[0], 200, msg='Check that your radius reaches its upper boundary (check formula)')
        self.assertLess(R_0[-1], 16000.0, msg='Check your theta incrementing')
        self.assertTrue(np.allclose(res_0, R_0, atol=30),
                        msg='There is something wrong with your ring feature extraction')

        R_1 = extractRingFeatures(np.copy(self.magnitude), 5, 8)
        res_1 = np.array([380.716, 1250.537, 2102.192, 2981.219, 3834.599])
        self.assertLess(np.max(R_1), 10000.0, msg='Do not forget to convert from polar to cartesian coordinates')
        self.assertFalse(np.allclose(res_1[0:4], R_1[1:], atol=3), msg='Check where a ring starts and ends')
        self.assertGreater(R_1[0], 340.0, msg='Check that your theta reaches its upper boundary (check formula)')

    def test_extractFanFeatures(self):
        T_1 = extractFanFeatures(np.copy(self.fan), 4, 10)
        self.assertEqual((4,), T_1.shape, msg='Check which input variable resembles the number of ring features')
        self.assertIsInstance(T_1[0], float, msg='Return the features as float values')
        self.assertTrue(np.abs(T_1[0] - T_1[1]) < 3, msg='Check your lower boundary of theta and your coordinates')
        self.assertTrue(T_1[2] > T_1[1], msg='Check the upper boundary of theta')
        self.assertTrue(np.abs(T_1[3] - T_1[2]) < T_1[2], msg='Check your theta sampling')
        self.assertLess(T_1[2], 100000, msg='Check your incrementation steps of theta')

        T_2 = extractFanFeatures(np.rot90(np.rot90(self.fan))[:, :80], 6, 18)
        self.assertEqual((6,), T_2.shape, msg='Check which input variable resembles the number of ring features')
        self.assertTrue(T_2[0] >= T_2[1] >= T_2[2] >= T_2[3],
                        msg='Visualize your sampling by plotting your magnitude coordinates')

    def test_calculateFourierParameters(self):
        res_1 = calcuateFourierParameters(np.copy(self.img), 4, 12)
        self.assertIsInstance(res_1, tuple, msg='Function should a tuple (R,T)')
        R1, T1 = res_1[0], res_1[1]
        self.assertIsInstance(R1, np.ndarray, msg='Check type of R')
        self.assertIsInstance(T1, np.ndarray, msg='Check type of T')
        self.assertFalse(np.max(R1) > np.max(T1), msg='Check the order of your return values')

        res_2 = calcuateFourierParameters(np.copy(self.img).T, 10, 32)
        self.assertIsInstance(res_2, tuple, msg='Function should a tuple (R,T)')
        R2, T2 = res_2[0], res_2[1]
        self.assertIsInstance(R2, np.ndarray, msg='Check type of R')
        self.assertIsInstance(T2, np.ndarray, msg='Check type of T')
        self.assertFalse(np.max(R2) > np.min(T2), msg='Check the order of your return values')

        self.assertTrue(np.max(T1) > np.max(R2) > np.max(R1), msg='Check your values')


class TestDistance(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_calculate_R_Distance(self):
        DR_0 = calculate_R_Distance(np.arange(6), np.arange(6, 0, -1))
        self.assertNotIsInstance(DR_0, np.ndarray, msg='Must return single value')
        self.assertFalse(np.abs(DR_0 - 18) < 4, msg='Must norm your result (check formula)')
        self.assertFalse(np.abs(DR_0 - 2) < 0.5, msg='Must norm using the number of features')
        self.assertTrue(DR_0 > 0, msg='Sum the absolute difference')

        DR_1 = calculate_R_Distance(np.arange(6), np.arange(6))
        self.assertAlmostEqual(0.0, DR_1, 1, msg='Check your R distance computation')

        self.assertTrue(np.abs(DR_0 - 3) < 0.2, msg='Check your R distance computation')

    def test_calculate_Theta_Distance(self):
        T_1 = calculate_Theta_Distance(np.arange(6), np.array([99, 104, 77, 32, 1, 0]))
        self.assertTrue(T_1 > 0, msg='Check your D_Theta computation --> brackets')
        self.assertTrue(T_1 > 1, msg='Check your D_Theta computation')

        T_2 = calculate_Theta_Distance(np.array([5, 3300, 9294, 2222, 5433, 7777]), np.array([99, 104, 77, 32, 1, 0]))
        self.assertFalse(T_2 < 10, msg='Must norm in lxx & lyy')
        self.assertTrue(np.abs(T_2 - 84) < 3, msg='Check your theta distance result')

        T_3 = calculate_Theta_Distance(np.arange(4), np.arange(4))
        self.assertTrue(T_3 < 0.1, msg='Check your Theta distance result')


class TestMatching(unittest.TestCase):
    def setUp(self) -> None:
        self.k = 10
        self.samplingSize = 200
        self.img2 = cv2.imread('Hand2.jpg', cv2.IMREAD_GRAYSCALE)
        self.img3 = cv2.imread('Hand3.jpg', cv2.IMREAD_GRAYSCALE)

    def test_everything_at_once(self):
        # palmprint alignment
        img2_aligned = palmPrintAlignment(self.img2)
        img3_aligned = palmPrintAlignment(self.img3)

        # compute features using Fourier Transform
        R2, Theta2 = calcuateFourierParameters(img2_aligned, self.k, self.samplingSize)
        R3, Theta3 = calcuateFourierParameters(img3_aligned, self.k, self.samplingSize)

        # Compute similarity
        DR_22 = calculate_R_Distance(R2, R2)
        DTheta_22 = calculate_Theta_Distance(Theta2, Theta2)
        self.assertTrue(DR_22 < 0.1, msg='Check DR computation')
        self.assertTrue(DTheta_22 < 0.1, msg='Check DTheta computation')

        DR_32 = calculate_R_Distance(R3, R2)
        DTheta_32 = calculate_Theta_Distance(Theta3, Theta2)

        DR_23 = calculate_R_Distance(R2, R3)
        DTheta_23 = calculate_Theta_Distance(Theta2, Theta3)

        self.assertAlmostEqual(DR_23, DR_32, delta=2, msg='These R should be the same')
        self.assertAlmostEqual(DTheta_23, DTheta_32, delta=2, msg='These Theta should be the same')


if __name__ == '__main__':
    unittest.main()
