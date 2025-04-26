import numpy as np 
import matplotlib.pyplot as plt 


    #load data
SLP = np.loadtxt('SLP.txt', unpack = True)
longitude = np.loadtxt('lon.txt', unpack = True)
times = np.loadtxt('times.txt', unpack = True)

N = len(times)
dt = times[1] - times[0]

M = len(longitude)
dl = longitude[1] - longitude[0]


"""
-------------------------- (a)
"""

    #take FFT along the spatial axis
SLP_FT = np.fft.fft(SLP, axis=0)*dl/M  

    #iterate over indexes to extract only the m=3 and m=5 Fourier wavenumbers
for k in range(len(SLP_FT)):
    if k == 3 or k == 5:
        pass
    else:
        SLP_FT[k] = np.zeros(120)   #set everything else to zero


SLP_fil = np.fft.ifft(SLP_FT, axis=0)*M/dl




fig, ax = plt.subplots()
cs = ax.contourf(SLP, levels = np.linspace(np.min(SLP), np.max(SLP)))
ax.set_xlabel('Times (Num-days)', fontsize=14)
ax.set_ylabel('Longitude (Degrees)',fontsize=14)
ax.set_title('SLP Unfiltered Contour Plot', fontsize=16)
fig.colorbar(cs)
plt.text(140, 40, 'Pressure Level (hPa)', rotation = 'vertical',  fontsize=14)
plt.show()




fig, ax = plt.subplots()
# cs = ax.contourf(SLP_fil, levels = np.linspace(np.min(SLP), np.max(SLP)))
cs = ax.contourf(SLP_fil, levels = np.linspace(np.min(SLP_fil), np.max(SLP_fil)))
ax.set_xlabel('Times (Num-days)', fontsize=14)
ax.set_ylabel('Longitude (Degrees)',fontsize=14)
ax.set_title('SLP Filtered Contour Plot', fontsize=16)
fig.colorbar(cs)
plt.text(140, 40, 'Pressure Level (hPa)', rotation = 'vertical',  fontsize=14)
plt.show()






