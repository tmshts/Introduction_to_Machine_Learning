from math import ceil, floor
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.fft import fftfreq
from tqdm import tqdm
from scipy.fftpack import fft, fftfreq

# divide with integral result (discard remainder)
def load_sample(filename, duration=4*44100, offset=44100//10):
    # Complete this function
    data = np.load(filename)
    # position of highest value of signal
    highest = np.argmax(data)
    # start of signal
    start = highest + offset
    # length of signal
    length = start + duration
    # the cut short part of the data array
    short_part = data[start:length]
    #plt.plot(short_part)
    #plt.show()
    return short_part

def compute_frequency(signal, min_freq=20):
    # calculate frequency - axis X
    sampling_rate = 44100
    T = 1 / sampling_rate
    freq = fftfreq(len(signal), T)
    # only positive values
    cut = np.where(freq >= 0)
    # calculate amplitude - axis Y
    signal_fft = fft(signal)
    amplitude = np.abs(signal_fft)
    #print(amplitude)
    #print(freq)
    plt.plot(freq[cut], amplitude[cut])
    plt.xlabel('Frequency in Hertz (Hz) frequency domain')
    plt.ylabel('Amplitude - Magnitude - Strongest, Power')
    plt.title('Fourier signal (frequency domain)')
    plt.grid()
    plt.show()
    #print(type(freq))
    #freq_over = np.array([])
    #print(type(freq_over))
    #new_freq = []
    #for i in freq:
    #    if i > min_freq:
    #        new_freq.append(i)
    #        extended_array = np.append(freq_over, [i])
    #print(new_freq)
    #freq_new = np.array(new_freq)
    #print(type(extended_array))
    #print(len(extended_array))
    amp_freq = np.array([amplitude, freq])
    amp_position = amp_freq[0,:].argmax()
    peak_freq = amp_freq[1, amp_position]

    return peak_freq

if __name__ == '__main__':
    # Implement the code to answer the questions here
    note2 = load_sample('C:/D/fau/5semester/machine_learning/exercises/exercise-01/sounds/Piano.ff.A2.npy')
    #note3 = load_sample('C:/D/fau/5semester/machine_learning/exercises/exercise-01/sounds/Piano.ff.A3.npy')
    #note4 = load_sample('C:/D/fau/5semester/machine_learning/exercises/exercise-01/sounds/Piano.ff.A4.npy')
    #note5 = load_sample('C:/D/fau/5semester/machine_learning/exercises/exercise-01/sounds/Piano.ff.A5.npy')
    #note6 = load_sample('C:/D/fau/5semester/machine_learning/exercises/exercise-01/sounds/Piano.ff.A6.npy')
    #noteXX = load_sample('C:/D/fau/5semester/machine_learning/exercises/exercise-01/sounds/Piano.ff.A7.npy')

    comp_note2 = ceil(compute_frequency(note2))
    #comp_note3 = floor(compute_frequency(note3))
    #comp_note4 = floor(compute_frequency(note4))
    #comp_note5 = floor(compute_frequency(note5))
    #comp_note6 = floor(compute_frequency(note6))
    #comp_noteXX = floor(compute_frequency(noteXX))

    # gives me key number of specific frequency
    key = floor(12*math.log2(comp_note2/440) + 49)
    print("Mysterious note has this key number: ", key)
    # gives me frequency of the key, not necessary to calculate :)
    freq = 2**((key-49)/12) * 440
    print("Mysterious note has this frequency (Hz): ", round(freq, 3))

    print("A2 has following frequency (Hz): ", comp_note2, ", expexted value: 110")
    #print("A3 has following frequency (Hz): ", comp_note3, ", expexted value: 220")
    #print("A4 has following frequency (Hz): ", comp_note4, ", expexted value: 440")
    #print("A5 has following frequency (Hz): ", comp_note5, ", expexted value: 880")
    #print("A6 has following frequency (Hz): ", comp_note6, ", expexted value: 1760")
    #print("XX has following frequency (Hz): ", comp_noteXX, ", expexted value: 1174.659 (Note D6)")

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
