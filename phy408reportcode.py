#%matplotlib inline
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import sounddevice as sd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import cbook, cm
from matplotlib.colors import LightSource
'''####Data : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GEVGRO'''
'''####Paper : https://arxiv.org/pdf/2209.02673.pdf'''


    #latex encoding
plt.rcParams['text.usetex'] = True
    #font changing
tfont = {'fontname':'DejaVu Sans', 'size':'20'}
afont = {'fontname':'Helvetica', 'size':'18'}
    #scale output plots
plt.rcParams['figure.figsize'] = (15, 7)


    #possibly useful functions
def integrate(a, dx):
        val = np.sum(a)
        return val * dx

i = 1j  #def of complex number
def abs_square(x):      
    """
    x: complex number / array
    """
    squared = np.zeros(len(x))
    for i in range(len(x)):
        squared[i] = ((x[i]).real)**2+((x[i]).imag)**2
    return squared

def gauss(t, width):
        return 1/(np.sqrt(np.pi)*width)*np.exp(-(t/width)**2)

def boxcar(t, T):
    output=np.zeros(len(t))
    for i in range(len(t)):
        if 0 <= t[i] <=T:
            output[i]=1
        else:
            output[i]= 0
    return output

def hann(t, T):
    output = np.zeros(len(t))
    for i in range(len(t)):
        if 0 <= t[i] <= T:
            output[i] = 0.5*(1-np.cos(2*np.pi*t[i]/T))
        else:
            output[i]=0
    return output

def ratFilter(N, D, x):
    """
    N: Numerator Array  
    D: Denominator Array with D[0]=1
    x: Input Time Series
    """
    output=np.zeros(len(x))
    for i in range(len(x)):
        output[i] = 1/D[0] * (np.convolve(N, x)[i] - np.convolve(D, output)[i])
    return output


                                            #convolution and correlation functions are equivalent to np.convolve() and np.correlate()
                                            #but I'll include them in here just in case
# def myconv(A, B, DELTA):                
#     N=len(A)               
#     M=len(B)               
#     K=N+M-1                
#     conv = np.zeros(K)     
#     for k in np.arange(K):
#         termval=0           
#         for i in np.arange(N):
#             if 0 <= k-i <= M-1:         
#                 termval += A[i] * B[k-i]    
#         conv[k] = termval*DELTA      
#     return conv    

# def cross_correlate(f, g, dt):  
#     N = len(f)
#     f = np.pad(f, (0, N-1), 'constant') #pad arrays
#     g = np.pad(g, (0, N-1), 'constant')
#     f_fft = np.fft.fft(f)*dt    #fft arrays into frequency domain
#     g_fft = np.fft.fft(g)*dt
#     c = np.conjugate(g_fft) * f_fft #multiply arrays
#     out = np.fft.ifftshift(np.fft.ifft(c)*dt)   #shift output
#     lag_axis = np.arange(-len(out)/2, len(out)/2, dt)   #define axis
#     return (lag_axis, out)





    """-------------------------------------------------------------------------"""
    ###-------------------------------------------------------------------------###
    """-------------------------------------------------------------------------"""

audio = False

    #load data
with open('validation-6s-merger-wnoise410Mpc-waveform_data.pkl', 'rb') as f:
    df = pickle.load(f)

# print(df.columns)     ###COLUMNS   'Waveform', 'm1', 'm2', 'Merger Position', 'spin1z', 'spin2z',  'Distance', 'PSNAR'
n = 178       #data file number: 0 to 1697
print(np.array(df)[n])
waveform = (np.array(df)[n])[0]
N = len(waveform)
# print(N)
dt = 6/N  *10
time = np.arange(0, N*dt, dt)


if audio: 
    waveform = waveform*(10**10)*(10**11)
    sd.play(waveform, N/6)

    #specify frequency domain
freq = np.abs(np.fft.fftshift(np.fft.fft(waveform)))
freq_axis = np.fft.fftshift(np.fft.fftfreq(N))/dt


    #plot FFT - not sure this is useful as it's included later
# plt.plot(freq_axis, freq)
# plt.xlim(0, )   #eliminate aliasing
# plt.show()

    # raw data plot
plt.plot(time, waveform, color='k')
plt.title('Raw Binary Black Hole Merger Signal', **tfont)
plt.xlabel(r'Time $(s)$', **afont)
plt.ylabel('Amplitude', **afont)
plt.xticks((0, 10, 20, 30, 40, 50, 60), labels=(0, 1, 2, 3, 4, 5, 6))
plt.show()



    #design bandpass filter 
min_freq = 0
max_freq = np.max(freq_axis)


epsilon=0.125
M= epsilon**2/(1 + epsilon)**2  #normalization thing for some reason I stumbled upon this 
f_0 = 0
delta_func = np.zeros(len(time))
delta_func[0] = 1

Num = [M]
Den = [1, -2/(1+epsilon)*np.cos(2*np.pi*f_0*dt), 1/((1+epsilon)**2)]


power_spect =np.fft.fftshift(np.fft.fft(ratFilter(Num, Den, delta_func)))
plt.plot(freq_axis, np.abs(power_spect), color='k', label='Delta Response FFT')
plt.title('Band Filter Power Spectrum, Normalized', **tfont)
plt.xlabel('Frequency ($s^{-1}$)', **afont)
plt.ylabel('Amplitude', **afont)
plt.legend(loc='best', fontsize=16)
plt.show()

filtered =(ratFilter(Num, Den, waveform))

if audio: 
    filtered = filtered*(10**10)*(10**11)
    sd.play(filtered, N/6)


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(time, filtered , color='k', label='Filtered')
ax1.plot(time, waveform , color='r', alpha=0.6, label='Raw')
ax1.set_xlim(0,)
ax1.set_title('Filtered Merger Comparison', **tfont)
ax1.set_xlabel(r'Time ($s$)', **afont)
ax1.set_ylabel('Amplitude', **afont)
ax1.legend(loc='best',fontsize=16)

ax2.plot(freq_axis, np.abs(np.fft.fftshift(np.fft.fft(filtered))), color='k', label='Filtered')
ax2.plot(freq_axis, freq, color='r', alpha=0.6, label='Raw')
ax2.set_xlim(0,)
ax2.set_title('Frequency Spectrum Filter Comparison', **afont)
ax2.set_xlabel(r'Frequency ($s^{-1}$)', **afont)
ax2.set_ylabel('Amplitude', **afont)
ax2.legend(loc='best',fontsize=16)
plt.show()




    #Linear Chirp Transformation

    #begin by applying a hanning window to the filtered data to reduce edge effects and the tailend-noise
    #remove tail-end noise from the sample  -- experimentally estimated that the sample ended at 58.074s, at a sample time of 11892
filtered = filtered[0:5166]
time = time[0:5166]
dt = 6/N
n=len(time) #tiny n is length of sliced timescale
sliced_freq = np.fft.fftfreq(len(time))/dt
windowed = hann(time, 25)*filtered[0:5166]      #let's try it as if it wasn't filtered, but windowed

plt.plot(time, windowed, color='k')
plt.title('BBH Pre-Merge Windowed Signal', **tfont)
plt.xlabel(r'Time ($s$)', **afont)
plt.ylabel(r'Amplitude', **afont)
plt.xticks((0, 10, 20, 30, 40, 50, 60), labels=(0, 1, 2, 3, 4, 5, 6))
plt.xlim(0, 25)
plt.show()


    #define linear chirp transformation function
def LCT(input, dt, rate_axis):
    N = len(input)
    M = len(rate_axis)
    grid = np.zeros((M, N), dtype = 'complex_')
    time = np.arange(0, N*dt, dt)
    input_FT = np.fft.fft(input)
    for n in range(N):
        for m in range(M):
            grid[m, n] = np.exp(-2*np.pi*1j*rate_axis[m]*(time[n]**2))
    rate_term = np.fft.fft(grid, axis=1)
    # print(rate_term.shape, input_FT.shape)
    output=np.zeros((M, N), dtype = 'complex_')
    for m in range(M):
        output[m, ] = output[m, ] + np.convolve(input_FT, rate_term[m, ], mode='same')*dt
    return np.abs(output)



test_rate = np.arange(0, 10, 0.01)  #make chirp rate test array
A = LCT(windowed, dt, test_rate)  #make data
    #maximum location and value of A
maxindex= A.argmax()
C = np.unravel_index(maxindex, A.shape)
maxvalue = A[C]

fig, ax = plt.subplots()
im = ax.imshow(A, cmap='viridis_r')

ax.set_yticks(np.arange(0, 1000, 100))
ax.set_yticklabels(np.arange(0, 10, 1))
ax.set_xticks(np.arange(0, n, 100))
ax.set_xticklabels(np.around(np.arange(-n/2, n/2, 100)/(n*dt), 1))

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', format='%.0e')

ax.set_xlabel(r'Frequency $(s^{-1})$', **afont)
ax.set_ylabel(r'Chirp Rate $(s^{-2})$', **afont)
ax.set_title('Linear Chirp Transformation Spectogram of BBH Merger', **tfont)

plt.show()



fig, (ax1, ax2) = plt.subplots(1,2)

ax1.plot(A.T[C[1], ], color='k')
ax1.set_title('LCT Merger: Chirp Rate Cross Section Maximum', **tfont)
ax1.set_xticks(np.arange(0, 1000, 100))
ax1.set_xticklabels(np.arange(0, 10, 1))
ax1.set_xlabel(r'Chirp Rate $(s^{-2})$', **afont)
ax1.set_ylabel(r'Magnitude', **afont)

ax2.plot(A[C[0], ], color='k')
ax2.set_title('LCT Merger: Frequency Cross Section Maximum', **tfont)
ax2.set_xticks(np.arange(0, n, 100))
ax2.set_xticklabels(np.around(np.arange(-n/2, n/2, 100)/(n*dt), 1))
ax2.set_xlabel(r'Frequency $(s^{-1})$', **afont)
ax2.set_ylabel(r'Magnitude', **afont)
plt.show()

adjusted_freq_axis = np.arange(-n/2, n/2)/(n*dt)
basefreq = adjusted_freq_axis[C[1]]
chirpvalue=test_rate[C[0]]

time = np.arange(0, N*dt, dt)
plt.plot(time, 50*np.cos(basefreq*time + chirpvalue*np.square(time)) + 200, color='k', alpha=0.6, label=r'Re($e^{-2\pi i (\omega t + \beta t^2)}$)')
plt.plot(time, basefreq*time + chirpvalue*np.square(time), color='r', label=r'$\omega t + \beta t^2$')
plt.plot(time, basefreq + 2*chirpvalue*time, color='orange', label=r'$\omega+ 2\beta t$')
plt.title('Merger Frequency Evolution',**tfont)
plt.xlabel(r'Time ($s$)', **afont)
plt.ylabel(r'Frequency ($s^{-1}$)', **afont)
plt.legend(loc='upper left', fontsize=16)
plt.show()













"""
        #all the stuff that my computer can't run 
        # that I spent days working on 
        

# def LCT_2(input, rate, freq):
#     N = len(input)
#     output=0
#     for n in range(N):
#         output += input[n] * np.exp(-2*np.pi*1j/N *(freq*n + rate*n**2))
#     return output * 1/np.sqrt(N)




# def LCT(input, dt, M):
#     N = len(input)
#     t1 = np.arange(0, N*dt, dt)
#     t2 = np.arange(0, N*dt, M)
#     windowed = np.zeros((N, N))
#     index_max = 0
#     freq = np.fft.fftshift(np.fft.fftfreq(len(t2)))/M
#     output = np.zeros(len(t2))
#     for j in range(len(t2)):
#         for i in range(N):
#             windowed[j, i] = input[i]*gauss(t1[i]-t2[j], 0.1)
#         index_max = np.argmax(np.abs(np.fft.fft(windowed[j])*dt)) 
#         output[j] = freq[index_max]
#     return output

# plt.plot(time, LCT(filtered, dt, 0.1))
# plt.show()


# def LCT(input, dt, rate, freq):
#     N = len(input)
#     M = len(freq)
#     K = len(rate)
#     time = np.arange(0, N*dt, dt)
#     integrand = np.zeros((M, K))
#     for m in range(M):
#         for k in range(K):
#             for u in range(N):
#                 integrand[m, k] += input[u]*np.exp(-2*np.pi*1j*(freq[m]*time[u] + rate[k]*(time[u]**2)))
#     return integrand*dt

# time = np.arange(0, 0, 0.1)
# rate_range = np.arange(0, 10, 0.5)
# freq_range = np.arange(1, 30, 0.5)
# test = LCT(filtered, 10*dt, rate_range, freq_range)
# print(test)



# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# for i in range(len(freq_range)):
#     for j in range(len(rate_range)):
#         ax.scatter(freq_range[i], rate_range[j], test[i, j],marker='o')
# plt.show()



# def chirp_window(time, width, rate):
#     '''
#     time = actual time - shifted time      [axis - but this doesn't really matter as it's relative to start time point]
#     width = window width [scalar] 
#     rate = chirp rate   [scalar]
#     '''
#     N=len(time)  #needs to match len(input)
#     norm = np.zeros(N)
#     norm = np.sqrt( (4*width**2 + 16*np.pi*rate*1j) / (2*np.pi))   
#     expo = np.exp(-np.square(time)*(0.5*(width**2)) - (2*np.pi*np.square(time)*rate*1j)) 
#     return  norm * expo  


# def JCRTFT(input, dt, tau, omega_axis, rate):
#     '''
#     returns output[freq_value, rate_value, time_value]
#     input = input [array]
#     dt = timestep [scalar]
#     tau = input time axis [array]
#     omega_axis = possible frequency values [array]
#     rate = possible chirp values to convolve [array]
#     '''
#     N = len(tau)  #needs to match len(input)
#     M = len(omega_axis)
#     K = len(rate)
#     output = np.zeros((M, K, N))
#     time = np.arange(0, N*dt, dt) #real time axis
#     for m in range(M):
#         for k in range(K):
#             window = chirp_window(time, omega_axis[m], rate[k])
#             pre_ft = input * window * np.exp(2*np.pi*omega_axis[m]*time*1j)
#             FT = (np.fft.fftshift(np.fft.fft(pre_ft)*dt))
#             output[m, k] = FT[0:N]
            
#     return output




# rate_range = np.arange(0, 10, 0.1)
# freq_range = np.arange(1, 30, 0.1)
# time = np.arange(0, 10, 0.1)
# test = JCRTFT(filtered[0:100], dt, time, freq_range, rate_range)

# chirp_sum = np.sum(test, axis=1)





# fig = plt.figure()
# ax = fig.add_subplot()
# for i in range(len(freq_range)):
#     for j in range(len(time)):
#         ax.scatter(freq_range[i], time[j], c=chirp_sum[i, j], cmap='viridis_r')
# ax.set_xlim(0, 30)
# ax.set_ylim(0,)
# ax.set_ylabel('Time (s)')
# ax.set_xlabel('Frequency (Hz)')
# plt.colorbar()
# plt.show()
"""










