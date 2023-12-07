import matplotlib.pyplot as plt
import numpy as np
from math import pi

# samples = 200, frequency = 2, kMax = 10000, [amplitude = 1])
# returns the signal as 1D-array (np.ndarray)
samples = 200
frequency = 2
k_max = 10000
t = np.linspace(0, frequency, num=samples)
sum_trian = 0
for k in range(0, k_max):
    #trian = trian + (((-1)**k) * ((np.sin(2*pi*(2*k + 1)*t))/((2*k + 1)**2)))
    #sum_trian = (8/pi**2)*trian
    sum_trian += (8/pi**2) * ((-1)**k) * ((np.sin(2*pi*(2*k + 1)*t))/((2*k + 1)**2))
plt.plot(t, sum_trian)
plt.grid(True, which="both")
plt.title("Triangle Signal")
plt.xlabel('Time t (sec)')
plt.ylabel('Amplitude = sin(time)')
plt.show()