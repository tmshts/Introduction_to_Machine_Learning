import matplotlib.pyplot as plt
import numpy as np
from math import pi

# samples = 200, frequency = 2, kMax = 10000, [amplitude = 1])
# returns the signal as 1D-array (np.ndarray)
samples = 200
frequency = 2
k_max = 10000
A = 1
t = np.linspace(0, frequency, num=samples)
sum_saw = 0
for k in range(0, k_max):
    #saw = saw + np.sin(2*pi*k*t)/k
    #sum_saw = (A/2)-(A/pi)*saw
    sum_saw += ((A/2)-(A/pi)) * (np.sin(2*pi*k*t)/k)
plt.plot(t, sum_saw)
plt.grid(True, which="both")
plt.title("Saw tooth Signal")
plt.xlabel('Time t (sec)')
plt.ylabel('Amplitude = sin(time)')
plt.show()