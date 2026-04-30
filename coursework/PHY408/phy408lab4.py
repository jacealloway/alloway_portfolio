# %matplotlib inline
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

    #latex encoding
plt.rcParams['text.usetex'] = True
    #font changing
tfont = {'fontname':'DejaVu Sans', 'size':'20'}
afont = {'fontname':'Helvetica', 'size':'18'}

i = 1j  #def of complex number
def abs_square(x):      
    """
    x: complex number / array
    """
    squared = np.zeros(len(x))
    for i in range(len(x)):
        squared[i] = ((x[i]).real)**2+((x[i]).imag)**2
    return squared

def df_align(input):
    output=[]
    for i in range(len(input)):
        output = np.append(output, input[i])
    return output

def hann(input):
    N = len(input)
    out=np.zeros(N)
    for n in range(N):
        out[n] = (1 - np.cos(2*np.pi*n/N)) * input[n]
    return out




#(1.1) 
df_mlac=np.genfromtxt('MLAC_data.txt')
mlac = df_mlac.flatten()
df_phl = np.genfromtxt('PHL_data.txt')
phl = df_phl.flatten()

dt=1
time = np.arange(-12*60*60, 12*60*60, dt)    #dt=1s, 24 hours

def cross_correlate(f, g, dt):
    N = len(f)
    f = np.pad(f, (0, N-1), 'constant') #pad arrays
    g = np.pad(g, (0, N-1), 'constant')
    f_fft = np.fft.fft(f)*dt    #fft arrays into frequency domain
    g_fft = np.fft.fft(g)*dt
    c = np.conjugate(g_fft) * f_fft #multiply arrays
    out = np.fft.ifftshift(np.fft.ifft(c)*dt)   #shift output
    lag_axis = np.arange(-len(out)/2, len(out)/2, dt)   #define axis
    return (lag_axis, out)


lag = cross_correlate(phl, mlac, 1)[0]  #shifted time axis

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(lag, cross_correlate(phl, mlac, 1)[1], label='Manual FFT Correlation', color='blue', lw=1)
ax2.plot(time, np.correlate(phl, mlac, mode='same'), label='Numpy Correlation', color='red', lw=1)
ax1.set_xlabel('LAG Axis (s)', **afont)
ax1.set_ylabel('Correlation Amplitude', **afont)
ax2.set_xlabel('LAG Axis (s)', **afont)
ax2.set_ylabel('Correlation Amplitude', **afont)
ax1.set_xlim(-250, 250)
ax2.set_xlim(-250, 250)
ax1.legend(loc='best', fontsize=16)
ax2.legend(loc='best', fontsize=16)
ax1.set_title('Cross Correlation using FFT and np.correlate Between Two Seismic Stations', **tfont)
plt.show()




#(1.2) 
bit_phl = np.sign(phl)
bit_mlac = np.sign(mlac)
lag = cross_correlate(bit_phl, bit_mlac, 1)[0]  #re-define lag axis for bit conversion


fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(lag, cross_correlate(bit_phl, bit_mlac, 1)[1], label='Manual FFT Correlation', color='blue', lw=1)
ax2.plot(time, np.correlate(bit_phl, bit_mlac, mode='same'), label='Numpy Correlation', color='red', lw=1)
ax1.set_xlabel('LAG Axis (s)', **afont)
ax1.set_ylabel('Correlation Amplitude', **afont)
ax2.set_xlabel('LAG Axis (s)', **afont)
ax2.set_ylabel('Correlation Amplitude', **afont)
ax1.set_xlim(-250, 250)
ax2.set_xlim(-250, 250)
ax1.legend(loc='best', fontsize=16)
ax2.legend(loc='best', fontsize=16)
ax1.set_title('Bit-Cross Correlation using FFT and np.correlate Between Two Seismic Stations', **tfont)
plt.show()



        #bit correlation 
plt.plot(lag, cross_correlate(bit_phl, bit_mlac, 1)[1], label='Bit Conversion CC', color='blue')
plt.plot(lag, cross_correlate(phl, mlac, 1)[1]*10**(13), label='Regular CC', alpha=0.8, color='orange')
plt.xlim(-250, 250)
plt.legend(loc='best', fontsize=16)
plt.xlabel('LAG Axis (s)', **afont)
plt.ylabel('Correlation Amplitude', **afont)
plt.title('Comparison Between Regular and Bit-Converted Correlations', **tfont)
plt.show()





#(2.1)
df = np.loadtxt('nwao.vh1', unpack=True)
dt=10
time = df[0] / (dt*6*60)
velocity = df[1]
N = len(time)



plt.plot(time, velocity, label='Raw Data', color='k')
plt.xlabel('Time (hrs)', **afont)
plt.ylabel('Measured Magnitude', **afont)
plt.title('NWAO Earthquake Seismograph Measurement Over 3 Days', **tfont)
plt.legend(loc='best', fontsize=16)
plt.show()


#(2.2)
freq_axis = np.fft.fftshift(np.fft.fftfreq(len(time)))/dt *1000 #1hz = 1000mHz
df_fft = np.fft.fftshift(np.fft.fft(velocity))
df_pspec = np.conjugate(df_fft)*df_fft

plt.plot(freq_axis, (1/N)*df_pspec, label='FFT', color='k')
plt.xlabel('Frequency (mHz)', **afont)
plt.ylabel('Frequency Presence Amplitude', **afont)
plt.title('Power Spectrum of NWAO Raw Seismic Data', **tfont)
plt.legend(loc='best', fontsize=16)
plt.xlim(0, 50)
plt.show()




#(2.3)
slope, y_int = np.polyfit(time, velocity, 1)    #remove trend
line_trend = slope*time + y_int
detrended = velocity - line_trend
windowed = hann(detrended)  #apply the window function to the data to remove spectral leakage

win_fft = np.fft.fftshift(np.fft.fft(windowed))
win_pspec = np.conjugate(win_fft)*win_fft

plt.plot(freq_axis, (1/N)*win_pspec, label='Windowed Data', color='k')
plt.title('Power Spectrum of NWAO Data Under a Hann Window', **tfont)
plt.xlabel('Frequency (mHz)', **afont)
plt.ylabel('Frequency Presence Amplitude', **afont)
plt.legend(loc='best', fontsize=16)
plt.xlim(0.1, 50)
plt.show()

#(2.4)
plt.plot(freq_axis, win_pspec, label='Windowed', color='blue')
plt.plot(freq_axis, df_pspec, label='Unwindowed', color='red')
plt.title('Comparison of Power Spectra of Windowed and Unwindowed Data', **tfont)
plt.xlabel('Frequency (mHz)', **afont)
plt.ylabel('Frequency Presence Amplitude', **afont)
plt.legend(loc='best', fontsize=16)
plt.xlim(0.1, 2.6)
plt.show()


#(2.5)
modes = ('0.587 $ _0T_2$',
        '0.766 $_0 T_4$',
        '0.840 $_0 S_5$', 
        '----- 0.944 $_3 S_1$', 
        '1.037 $_0 S_6$', 
        '1.079 $_0 T_6$', 
        '1.106 $_3 S_2$', 
        '----- 1.223 $_0 T_7$ (?)', 
        '1.236 $_1 T_1$', 
        '1.412 $_0 S_8$', 
        '1.515 $_2 S_5$', 
        '1.487 $_0 T_9$', 
        '1.577 $_0 S_9$', 
        '1.614 $_0 T_{10}$', 
        '1.681 $_2 S_6$', 
        '1.725 $_0 S_10$', 
        '1.750 $_1 T_5$ (?)', 
        '1.857 $_0 T_{12}$', 
        '1.990 $_0 S_{12}$',                                                                                                                                 
        '2.096/2.103/2.111 $_0 T_{14} / _1 T_7/ _0 S_{13}$', 
        '2.168 $_5 S_3$', 
        '2.228 $_0 S_9$', 
        '2.234 $_3 S_5$', 
        '2.281 $_1 T_8$', 
        '2.410/2.411 $_6 S_2 / _4 S_5$', 
        '2.441 $_0 T_{17}$', 
        '2.344 $_0 S_{15}$', 
        '2.485 $_2 T_5$ (?)', 
        '2.517 $_7 S_2$ (?)', 
        '----- 2.549/2.554/2.555 $_3 S_6/ _0 T_{18}/ _1 S_{12}$', 
        '2.572 $_2 S_{11}$',             
        '0.679 $_1S_2$ (?)', 
        '0.931 $_0 T_5$', 
        '1.320 $_1 T_2$', 
        '1.356/1.370 $_0 T_8 / _1 S_5$ (?)', 
        '1.379 $_2 S_4$ (?)', 
        '1.925 $_1 T_6$ (?)', 
        '2.048 $_4 S_3$ (?)', 
        '2.212 $_0 T_{15}$ (?)', 
        '2.294 $_2 T_3$ (?)', 
        '--------- 2.327 $_0 T_{16}$ (?)', 
        '----- 2.379 $_5 S_4$ (?)',
        '1.172 $_1 S_4$ (?)')
xy = np.array([(0.587, 1e13), 
    (0.766, 2e13), 
    (0.840, 6e12), 
    (0.944, 7e12), 
    (1.035, 1.2e13), 
    (1.072, 1.25e13), 
    (1.1, 1.1e13), 
    (1.21, 4.8e13), 
    (1.237, 2e13), 
    (1.412, 2e13), 
    (1.511, 1.7e13), 
    (1.483, 3.7e13), 
    (1.57, 6e13), 
    (1.61, 5.5e13), 
    (1.676, 2.5e13), 
    (1.72, 3.8e13), 
    (1.746, 4.5e13), 
    (1.85, 4.3e13), 
    (1.996, 4.2e13), 
    (2.1, 7e13), 
    (2.15, 9.5e12), 
    (2.21, 9.3e13), 
    (2.23, 7.7e13), 
    (2.275, 1.3e13), 
    (2.41, 2e13), 
    (2.446, 6.6e13), 
    (2.35, 2.5e13), 
    (2.48, 2e13), 
    (2.51, 8.5e12), 
    (2.545, 1.2e13), 
    (2.574, 5e13), 
    (0.674, 1.5e13), 
    (0.918, 3.7e13), 
    (1.311, 5.5e12), 
    (1.35, 2.1e13), 
    (1.375, 2e13), 
    (1.92, 6.5e12), 
    (2.04, 6.5e12), 
    (2.19, 3.3e13), 
    (2.3, 1.1e13), 
    (2.33, 1.5e13), 
    (2.38, 5.5e12),
    (1.165, 1.3e13)])

plt.plot(freq_axis, win_pspec, label='Windowed', color='blue', lw=3)
for i in range(len(modes)):
    plt.annotate(modes[i], xy[i], color='k', rotation=90, fontsize=11)
plt.legend(loc='best', fontsize=16)
plt.xlim(0.5, 2.6)
plt.xlabel('Frequency (mHz)', **afont)
plt.ylabel('Frequency Magnitude', **afont)
plt.title('Normal Modes of Earth from Windowed Seismograph Data', **tfont)
plt.show()










