import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 30.0
time = np.arange(0, 10, 1/sampling_rate) # 0 -10 in steps of .....
#data = np.sin(2*np.pi*6*time) + np.random.randn(len(time))
data = np.sin(2*np.pi*6*time)
fourier_transform = np.fft.rfft(data)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
plt.plot(frequency, power_spectrum)



# Good Now try an ocean data set

sampling rate = 24 # 24 hours per day
t = np.arange(0, 7, 1/sampling_rate)  # samples per week
x = 2.5 * np.sin(2 * np.pi * t / 12.42)
x += 1.5 * np.sin(2 * np.pi * t / 12.0)
x += 0.3 * np.random.randn(len(t))
fourier_transform = np.fft.rfft(x)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
plt.plot(frequency, power_spectrum)

# Hourly data from ADDA
# Not working

station=8534720
sampling_rate = 1
time=df_hourlyOBS.index.to_list()
data=df_hourlyOBS[station]
fourier_transform = np.fft.rfft(data)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
plt.plot(frequency, power_spectrum)


# differenty 

station=8534720
sampling_rate = 24 # 24 hours per day
time = np.arange(0, 7, 1/sampling_rate)  # samples per week
data=df_hourlyOBS[station]
fourier_transform = np.fft.rfft(data)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
plt.plot(frequency, power_spectrum)

# try messingh up the sampling rate: Gives a completely diff rersulkt
#sampling_rate = 30.0
sampling_rate = 1
#time = np.arange(0, 10, 1/sampling_rate) # 0 -10 in steps of .....
time = np.arange(0, 30*10)
#data = np.sin(2*np.pi*6*time) + np.random.randn(len(time))
data = np.sin(2*np.pi*6*time)
fourier_transform = np.fft.rfft(data)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
plt.plot(frequency, power_spectrum)

# Try again  just monthy

df = df_hourlyOBS[station]['2018-01-01 00:00:00':'2018-01-31 00:00:00']
data = df.to_numpy()
sampling_freq=1
fourier_transform = np.fft.rfft(data)

