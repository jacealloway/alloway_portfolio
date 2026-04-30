import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile







    #load the wav file
samplerate, data = wavfile.read('GraviteaTime.wav')
    #left and right audio channels
left = data[:, 0]       
right =data[:, 1]
    #number of samples
N = len(left)
    #sample spacing 
dt = 1/samplerate

    #create empty output writing arrays 
output = np.empty(data.shape, dtype = data.dtype)
left_out = output[:, 0]
right_out = output[:, 1]





"""
-------------------------- (a)
"""
    #create a time array ( samplerate = samples / second )
time = np.arange(0, N*dt, dt) 

    #plot using subplots
fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
plt.subplots_adjust(wspace = 0.05)
ax1.plot(time, left)
ax2.plot(time, right)
    #labels + grid
ax1.set_xlabel('Time (s)', fontsize=14)
ax1.set_ylabel('Amplitude (arbitrary units)', fontsize=14)
ax1.set_title('GraviteaTime.wav Left Channel', fontsize=16)
ax1.grid()
ax2.set_xlabel('Time (s)', fontsize=14)
ax2.set_title('GraviteaTime.wav Right Channel', fontsize=16)
ax2.grid()
plt.show()





"""
-------------------------- (b)
"""
    #get left and right FT values
leftFT = np.fft.fft(left)*dt/N      #fourier coefficients are gamma_k = c_k / N = np.fft.fft(input)*dt / N
rightFT = np.fft.fft(right)*dt/N   
    #obtain the frequency axis 
freqAxis = np.fft.fftfreq(N, dt)

    #copy arrays to filter them
leftFT_fil = np.copy(leftFT)
rightFT_fil = np.copy(rightFT)
    #set the frequency values above 880hz to zero using a for loop
for k in range(N):
    if np.abs(freqAxis[k]) > 880:
        leftFT_fil[k] = 0
        rightFT_fil[k] = 0


    #invert the filtered FT's
left_fil = np.fft.ifft(leftFT_fil)*N/dt         #invert the magnitudes by multiplying by N/dt (see above)
right_fil = np.fft.ifft(rightFT_fil)*N/dt




    #plot all of 'em
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True, sharex = True)
plt.subplots_adjust(wspace = 0.05)
    #unfiltered
ax1.plot(freqAxis, leftFT)
ax2.plot(freqAxis, rightFT)
    #labels + grid
ax1.set_ylabel('Coefficients (arbitrary units)', fontsize=14)
ax1.set_title('Left Channel FT', fontsize=16)
ax2.set_title('Right Channel FT', fontsize=16)
ax1.grid()
ax2.grid()

    #filtered
ax3.plot(freqAxis, leftFT_fil)
ax4.plot(freqAxis, rightFT_fil)
    #labels + grids
ax3.set_title('Left Filtered', fontsize=16)
ax3.set_xlabel('Frequency (Hz)', fontsize=14)
ax3.set_ylabel('Coefficients (arbitrary units)', fontsize=14)
ax4.set_title('Right Filtered', fontsize=16)
ax4.set_xlabel('Frequency (Hz)', fontsize=14)
ax3.grid()
ax4.grid()

plt.show()




    #repeat with the time-series filtering
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True, sharex = True)
plt.subplots_adjust(wspace = 0.05)
    #unfiltered
ax1.plot(time, left)
ax2.plot(time, right)
    #labels
ax1.set_ylabel('Amplitude', fontsize=14)
ax1.set_title('Unfiltered Left', fontsize=16)
ax2.set_title('Unfiltered Right', fontsize=16)
ax1.grid()
ax2.grid()

    #filtered
ax3.plot(time, left_fil)
ax4.plot(time, right_fil)
    #labels
ax3.set_ylabel('Amplitude', fontsize=14)
ax3.set_xlabel('Time (s)', fontsize=14)
ax3.set_title('Filtered Left', fontsize=16)
ax4.set_xlabel('Time (s)', fontsize=14)
ax4.set_title('Filtered Left', fontsize=16)
ax3.grid()
ax4.grid()

plt.show()







"""
-------------------------- (c)
"""
    #get indexes of 0.03s and 0.06s to define a slice over that time region 
t1 = int(0.03/dt)
t2 = int(0.06/dt)


fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
plt.subplots_adjust(wspace = 0.05)
    #plots over the slice 
    #unfiltered
ax1.plot(time[t1:t2], left[t1:t2], label='Left')
ax1.plot(time[t1:t2], right[t1:t2], label='Right', alpha=0.8)
    #filtered
ax2.plot(time[t1:t2], left_fil[t1:t2], label='Left')
ax2.plot(time[t1:t2], right_fil[t1:t2], label='Right', alpha=0.8)

    #labels and grids
ax1.set_ylabel('Amplitude', fontsize=14)
ax1.set_xlabel('Time (s)', fontsize=14)
ax1.legend(loc='best', fontsize=14)
ax1.set_title('Unfiltered, Zoomed', fontsize=16)
ax2.set_xlabel('Time (s)', fontsize=14)
ax2.set_title('Filtered, Zoomed', fontsize=16)
ax2.legend(loc='best', fontsize=14)
ax1.grid()
ax2.grid()

plt.show()








"""
-------------------------- (d)
"""

    #write the left and right channels
    #all components are real (can show this by printing left_fil.imag / right_fil.imag, you'll get all zeros: 0.0j)
    #hence only plot real components
output[:, 0] = left_fil.real
output[:, 1] = right_fil.real
    #write the file
wavfile.write('GraviteaTime_filtered.wav', samplerate, output)


