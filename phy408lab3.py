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



# if '__name__' == '__main__':

onetwo=False
if onetwo:
        #(1.1)
    def H(input_spectrum, M, f_s, e, f_0):
        Delta = 1/f_s
        q = np.exp(-2*np.pi*i*f_0*Delta)
        p = (1+e)*q
        z = np.exp(-2*np.pi*i*Delta*input_spectrum)
        F = (z - q)/(z-p)
        F_bar = (z - np.conjugate(q))/(z-np.conjugate(p))
        return M*F*F_bar



        #(1.2)
    f_s = 12    #this implies a sampling rate of 1/12
    f_nyquist = f_s/2 
    f_spectrum = np.linspace(-f_nyquist, f_nyquist, 1000)
    epsilon = 0.05
    M = 1.05
    f_0 = 1


    output = H(f_spectrum, M, f_s, epsilon, f_0)
    power_spectrum = abs_square(output)

    plt.plot(f_spectrum[499:], power_spectrum[499:], color='blue', lw=2, label='Output Frequency Filter')
    plt.plot(f_spectrum[:500], power_spectrum[:500], color='blue', ls='--', lw=2, label='Aliasing Frequencies')
    plt.xlabel('Frequency Spectrum (Cycles/Year)', **afont)
    plt.ylabel('Power Spectrum', **afont)
    plt.title('Power Spectrum of Digital Notch Filter', **tfont)
    #plt.xlim(0, 1/(2*D))                            #this eliminates aliasing 
    plt.legend(loc='lower right', fontsize=16)
    plt.show()


        #(1.3)
    f_spectrum_cut = f_spectrum[499:]
    power_spectrum_cut = power_spectrum[499:]       #slice to elimiate aliasing when determining FWHM
    half_maxes = [ ]
    for i in range(len(power_spectrum_cut)):
        if 0.45 < power_spectrum_cut[i] < 0.55:
            half_maxes = np.append(half_maxes, f_spectrum_cut[i])
            FHWM = np.max(half_maxes) - np.min(half_maxes)
    print(f'the FWHM is {FHWM}')    #print calculated value








        #(2.1) -- on paper
        #(2.2)
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



        #(2.3)
    dt = 1/f_s
    time = np.arange(-10, 80, dt)
    f_0=1



    input = np.zeros(len(time))
    input[120] = 1



    Num = [M/((1+epsilon)**2), -2*M / ((1+epsilon)**2) * np.cos(2*np.pi*f_0/f_s), M /((1+epsilon)**2)]
    Den = [1, -2/(1+epsilon)*np.cos(2*np.pi*f_0/f_s), 1/((1+epsilon)**2)]

    plt.plot(time, input, label='Delta Function (Input)', ls='--', alpha=0.7, color='orange')
    plt.plot(time, ratFilter(Num, Den, input), label='Filter Response', color='red')
    plt.legend(loc='best', fontsize=16)
    plt.xlim(-1, 6)
    plt.xlabel('Time (years)', **afont)
    plt.ylabel('Time Series Response (Arbitrary Units)', **afont)
    plt.title('Notch Filter Impulse Response', **tfont)
    plt.show()




        #(2.4)
    #we have an impulse response calculated from above, and also from W(f) calculated in part (a)
    #FFT of delta function is 1
    freq = np.fft.fftshift(np.fft.fftfreq(len(time)))/dt
    impulse_out = H(freq, M, f_s, epsilon, f_0)
    fft_data = np.fft.fftshift(np.fft.fft(ratFilter(Num, Den, input)))

    plt.plot(f_spectrum, power_spectrum, color='red', ls='--', lw=2, label='Power Spectrum')
    plt.plot(freq, np.square(np.abs(fft_data)), color='blue', label='Frequency Response')
    plt.legend(loc='best', fontsize=16)
    plt.xlabel(r'Frequency (Cycles / Year)', **afont)
    plt.ylabel('Frequency Magnitude', **afont)
    plt.title('Impulse Response Power Spectrum Comparison', **tfont)
    plt.xlim(0, 6)
    plt.show()





    #delete later 
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







    #(3.1)
df = pd.read_csv("methane_global.csv", sep=',', skipinitialspace=True, usecols=['year', 'month', 'decimal', 'average', 'average_unc', 'trend', 'trend_unc'])
time = df['decimal']
average = df['average']
average_unc=df['average_unc']
trend = df['trend']
trend_unc = df['trend_unc']

slope, y_int = np.polyfit(time, average, 1)
line_trend = slope*time + y_int
detrended = average - line_trend


fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(time, average, label='Average Data', color='blue')
ax1.errorbar(time, average, average_unc, ls='none', capsize=1.5, label='Uncertainty', color='blue', alpha=0.6)
ax2.plot(time, detrended, label='Detrended Data', color='red')
ax2.errorbar(time, detrended, average_unc, ls='none', capsize=1.5, label='Uncertainty', color='red', alpha=0.6)
ax1.set_xlabel('Time (years)', **afont)
ax1.set_ylabel('Average Value (ppb)', **afont)
ax1.set_title('Global Averaged Methane Values', **tfont)
ax2.set_xlabel('Time (years)', **afont)
ax2.set_ylabel('Average Value (ppb)', **afont)
ax2.set_title('Global Averaged Methane Values, Detrended', **tfont)
ax1.legend(loc='best', fontsize=16)
ax2.legend(loc='best', fontsize=16)
plt.show()
    


    #(3.2)
f_s = 12    #this implies a sampling rate of 1/12
f_nyquist = f_s/2 
f_spectrum = np.linspace(-f_nyquist, f_nyquist, 1000)
M = 1.05

f_0s = (1, 2, 3, 4) #filter freqs
eps = (0.05, 0.05, 0.05, 0.05)  #Q adjust
numerators = []
denominators = []
for i in range(len(f_0s)):
    num = [M/((1+eps[i])**2), -2*M / ((1+eps[i])**2) * np.cos(2*np.pi*f_0s[i]/f_s), M /((1+eps[i])**2)]
    den = [1, -2/(1+eps[i])*np.cos(2*np.pi*f_0s[i]/f_s), 1/((1+eps[i])**2)]
    numerators.append(num)
    denominators.append(den)

filtering = detrended
for i in range(len(f_0s)):
    filtering = ratFilter(numerators[i], denominators[i], filtering)

retrended = filtering + line_trend


plt.plot(time, retrended, label='Filtered Data, Trend Inclusive', color='k', ls='-')
plt.xlabel('Time (years)', **afont)
plt.ylabel('Average Value (ppb)', **afont)
plt.title('Global Averaged Methane Values, Filtered', **tfont)
plt.legend(loc='best', fontsize=16)
plt.show()




    #(3.3)
N = len(time)
dt = 1/f_s
ft = np.fft.fftshift(np.fft.fft(detrended))
f_axis = np.fft.fftshift(np.fft.fftfreq(len(time)))/(dt)
detrended_amp = np.abs(ft)
detrended_ph = np.angle(ft)

    #filter the data with a ""gaussian""
gauss_filt = ft * (lambda x, t: np.exp(-x**2 / (t**2)))(f_axis, 0.4)
manual_filt = np.fft.ifft(np.fft.ifftshift(gauss_filt))



fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(f_axis, detrended_amp, label='FFT Spectrum', color='blue')
ax1.plot(f_axis, gauss_filt, label='Gaussian Filtered', color='cyan')
ax2.plot(f_axis, detrended_ph, label='Phase Spectrum', color='red')

ax1.set_title('FFT Amplitude', **tfont)
ax1.set_xlabel(r'Frequency (Cycles / Year)', **afont)
ax1.set_ylabel('Frequency Magnitude', **afont)
ax2.set_title('FFT Phase Spectrum', **tfont)
ax2.set_xlabel(r'Frequency (Cycles / Year)', **afont)
ax2.set_ylabel('Phase Angle (rad)', **afont)
ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_xlim(0, 2.5)
ax2.set_xlim(0, 2.5)
ax1.legend(loc='best', fontsize=16)
ax2.legend(loc='best', fontsize=16)
plt.show()



    #(3.4)
plt.plot(time, average, label='Unfiltered Data', color='blue')
plt.errorbar(time, average, average_unc, ls='none', label='Average Uncertainty', color='blue', alpha=0.6, capsize=1.5)
plt.plot(time, retrended, label='Notch Filtered Data', color='magenta')
plt.plot(time, manual_filt+line_trend, label='FFT Filtered Data', color='orange')
plt.legend(loc='best', fontsize=16)
plt.xlabel('Time (years)', **afont)
plt.ylabel('Concentration (ppb)', **afont)
plt.title('Filtered Data Comparison', **tfont)
plt.show()




    #(3.5) 
f_0s = (1, 2, 3, 4) #filter freqs
eps = (0.05, 0.05, 0.05, 0.05)  #Q adjust
numerators = []
denominators = []
for i in range(len(f_0s)):
    num = [M/((1+eps[i])**2), -2*M / ((1+eps[i])**2) * np.cos(2*np.pi*f_0s[i]/f_s), M /((1+eps[i])**2)]
    den = [1, -2/(1+eps[i])*np.cos(2*np.pi*f_0s[i]/f_s), 1/((1+eps[i])**2)]
    numerators.append(num)
    denominators.append(den)
filtering = average
for i in range(len(f_0s)):
    filtering = ratFilter(numerators[i], denominators[i], filtering)


N = len(time)
dt = 1/f_s
ft = np.fft.fftshift(np.fft.fft(average))
f_axis = np.fft.fftshift(np.fft.fftfreq(len(time)))/(dt)
average_amp = np.abs(ft)
average_pph = np.angle(ft)

gauss_filt = ft * (lambda x, t: np.exp(-x**2 / (t**2)))(f_axis, 0.2)
manual_filt = np.fft.ifft(np.fft.ifftshift(gauss_filt))


plt.plot(time, average, label='Unfiltered Data', color='blue')
plt.errorbar(time, average, average_unc, ls='none', label='Average Uncertainty', color='blue', alpha=0.6, capsize=1.5)
plt.plot(time, filtering, label='Notch Filtered Data', color='magenta')
plt.plot(time, manual_filt, label='FFT Filtered Data', color='orange')
plt.legend(loc='best', fontsize=16)
plt.xlabel('Time (years)', **afont)
plt.ylabel('Concentration (ppb)', **afont)
plt.title('Filtered Data Comparison, Trend Inclusive', **tfont)
plt.show()


plt.plot(f_axis, average_amp)
plt.show()
