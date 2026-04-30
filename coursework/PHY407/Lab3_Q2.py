import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

    #load data
data = pd.read_csv('sp500.csv')



"""
-------------------------- (a)
"""
    #write array for business days, extract opening value array 
N = len(data.Date)
busDay = np.arange(0, N)
openVal = data.Open


    #plot against business day 
plt.plot(busDay, openVal)
    #labels
plt.grid()
plt.xlabel('Business Days from 09/02/2014', fontsize=14)
plt.ylabel('Amount ($)', fontsize=14)
plt.title('Opening Value of S&P 500 Stock from 09/02/2014', fontsize=16)
plt.show()








"""
-------------------------- (b)
"""


dt = 1  #1 bus. day is seperation
    #perform FFT
openVal_FT = np.fft.rfft(openVal, N)*dt/N 
freq_axis = np.fft.rfftfreq(N, dt)
    #invert the data to bring back the original data 
original = np.fft.irfft(openVal_FT, N)*N/dt


    #plot
plt.plot(busDay, openVal - original)
    #labels
plt.grid()
plt.xlabel('Business Days from 09/02/2014', fontsize=14)
plt.ylabel('Amount Difference ($)', fontsize=14)
plt.title('OpenVal Difference of S&P 500 Stock Between FT/IFT and Initial Data', fontsize=16)
plt.show()





"""
-------------------------- (c)
"""
six_month_freq = 1/(21*6) # we'll assume, on average, that every month is around 21 business days 
openVal_FT_fil = np.copy(openVal_FT)

for k in range(len(freq_axis)):
    if freq_axis[k] > six_month_freq:
        openVal_FT_fil[k] = 0

    #invert the filtered data 
filtered = np.fft.irfft(openVal_FT_fil, N)*N/dt


    #plot
plt.plot(busDay, openVal, color='orange', lw=2, alpha=0.7, label='Original Data')
plt.plot(busDay, filtered, color='blue', label='filtered Data')
    #labels
plt.grid()
plt.xlabel('Business Days from 09/02/2014', fontsize=14)
plt.ylabel('Amount Difference ($)', fontsize=14)
plt.title('OpenVal Difference of S&P 500 Stock Between FT/IFT and Initial Data', fontsize=16)
plt.legend(loc='best', fontsize=14)
plt.show()



