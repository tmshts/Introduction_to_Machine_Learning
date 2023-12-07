import matplotlib.pyplot as plt
import numpy as np
from math import pi

# samples = 200, frequency = 2, kMax = 10000, [amplitude = 1])
# returns the signal as 1D-array (np.ndarray)
samples = 200
frequency = 2
k_max = 10000
t = np.linspace(0, frequency, num=samples)
sum_squa = 0
# iteration
for k in range(0, k_max):
    #squa = squa + np.sin(2*pi*((2*k) - 1)*t)/((2*k) - 1)
    #sum_squa = (4/pi)*squa
    sum_squa += (4/pi) * (np.sin(2*pi*((2*k) - 1)*t)/((2*k) - 1))
plt.plot(t, sum_squa)
plt.grid(True, which="both")
plt.title("Square Signal")
plt.xlabel('Time t (sec)')
plt.ylabel('Amplitude = sin(time)')
plt.show()