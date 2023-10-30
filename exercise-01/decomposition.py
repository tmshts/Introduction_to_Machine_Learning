import numpy as np
from matplotlib import pyplot as plt


def createTriangleSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array or list
    # TODO
    t = np.linspace(0, 1, samples)
    z = 0
    for k in range(0, k_max + 1):
        z += (8 / (np.pi ** 2)) * ((-1)**k) * (np.sin(2 * np.pi * (2*k + 1) * frequency * t) / ((2*k + 1) ** 2))
    plt.plot(t, z)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.title('Triangle Signal')
    plt.show()
    return z


def createSquareSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array or list
    # TODO
    t = np.linspace(0, 1, samples)
    z = 0
    for k in range(1, k_max + 1):
        z += (4 / np.pi) * (np.sin(2 * np.pi * (2*k - 1) * frequency * t) / (2*k - 1))
    plt.plot(t, z)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.title('Square Signal')
    plt.show()
    return z


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    # returns the signal as 1D-array or list
    # TODO
    t = np.linspace(0, 1, samples)
    A = amplitude
    z = (A / 2)
    for k in range(1, k_max + 1):
        z -= (A / np.pi) * (np.sin(2 * np.pi * k * frequency * t) / k)
    plt.plot(t, z)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.title('Sawtooth Signal')
    plt.show()
    return z
