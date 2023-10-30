from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal
from matplotlib import pyplot as plt


# TODO: Test the functions imported in lines 1 and 2 of this file.
if __name__ == '__main__':
    chirp = createChirpSignal(200, 1, 1, 10, False)
    # when observing the wider range of mapping the function it shows amplitude differences
    triangle = createTriangleSignal(300, 5, 10000)
    square = createSquareSignal(300, 5, 10000)
    sawtooth = createSawtoothSignal(300, 5, 10000, 20)
