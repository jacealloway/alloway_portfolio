import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelmax, argrelmin, peak_widths
from scipy.stats import chi2
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from mpl_toolkits.axes_grid.inset_locator import InsetPosition

    #font changing
""" 
        #change the font of the graph text in matplotlib
font = {'family' : 'DejaVu Sans', 
        'weight' : 'normal'
        'size' : '22'
        }
rc('font', **font)

 """

tfont = {'fontname':'DejaVu Sans', 'size':'20'}
afont = {'fontname':'Helvetica', 'size':'18'}

    #latex encoding
plt.rcParams['text.usetex'] = True





    #loading data
trial1_data = np.loadtxt('trial 1 - small mass, 10 degree (2 col).txt', skiprows=2, unpack=True, delimiter='\t' )
trial2_data = np.loadtxt('trial 2 - small mass, 80 degree.txt', skiprows=2, unpack=True, delimiter='\t')
trial3_data = np.loadtxt('trial 3 - large mass, 40 degree (2 col), part1.txt', skiprows=2, unpack=True, delimiter='\t')
trial4_data = np.loadtxt('trial 4 - large mass, 20 degree (2 col), part1.txt', skiprows=2, unpack=True, delimiter='\t')

trial6_data = np.loadtxt('trial 7 - small mass, 10 degree.txt', skiprows=2, unpack=True, delimiter='\t')

t1_time = trial1_data[0] - trial1_data[0][0]                #set time origin to zero
t1_theta = trial1_data[1] + np.abs(np.mean(trial1_data[1])) #set angle axis from 90 to 0

t2_time = trial2_data[0] - trial2_data[0][0]                #set time origin to zero
t2_theta = trial2_data[3] + np.abs(np.mean(trial2_data[3])) #set angle axis from 90 to 0

t3_time = trial3_data[0] - trial3_data[0][0]                #set time origin to zero
t3_theta = trial3_data[1] + np.abs(np.mean(trial3_data[1])) #set angle axis from 90 to 0

t4_time = trial4_data[0] - trial4_data[0][0]                #set time origin to zero
t4_theta = trial4_data[1] + np.abs(np.mean(trial4_data[1])) #set angle axis from 90 to 0
t4_time[40347] = 673.106
t4_theta[40347] = -11       #FIX AN ERROR!!

t6_time = trial6_data[0] - trial6_data[0][0]                #set time origin to zero
t6_theta = trial6_data[3]+np.abs(np.mean(trial6_data[3]))   #set angle axis from 90 to 0


        #local max, min determination
    #trial1
t1_theta_lmax_val = argrelmax(t1_theta, order=5)
t1_theta_lmax = np.zeros(len(t1_theta_lmax_val[0]))
t1_time_lmax = np.zeros(len(t1_theta_lmax_val[0]))
for i in range(len(t1_theta_lmax_val[0])):
    t1_theta_lmax[i] = t1_theta[t1_theta_lmax_val[0][i]]
    t1_time_lmax[i] = t1_time[t1_theta_lmax_val[0][i]]
    if t1_theta_lmax[i] >0:
        pass
    else:
        t1_theta_lmax[i] = np.abs(t1_theta_lmax[i])

t1_theta_lmin_val = argrelmin(t1_theta, order=5)
t1_theta_lmin = np.zeros(len(t1_theta_lmin_val[0]))
t1_time_lmin = np.zeros(len(t1_theta_lmin_val[0]))
for k in range(len(t1_theta_lmin_val[0])):
    t1_theta_lmin[k] = t1_theta[t1_theta_lmin_val[0][k]]
    t1_time_lmin[k] = t1_time[t1_theta_lmin_val[0][k]]
    if t1_theta_lmin[k] <0:
        pass
    else:
        t1_theta_lmin[k] = -np.abs(t1_theta_lmin[k])

    #trial2
t2_theta_lmax_val = argrelmax(t2_theta, order=5)
t2_theta_lmax = np.zeros(len(t2_theta_lmax_val[0]))
t2_time_lmax = np.zeros(len(t2_theta_lmax_val[0]))
for i in range(len(t2_theta_lmax_val[0])):
    t2_theta_lmax[i] = t2_theta[t2_theta_lmax_val[0][i]]
    t2_time_lmax[i] = t2_time[t2_theta_lmax_val[0][i]]

t2_theta_lmin_val = argrelmin(t2_theta, order=5)
t2_theta_lmin = np.zeros(len(t2_theta_lmin_val[0]))
t2_time_lmin = np.zeros(len(t2_theta_lmin_val[0]))
for k in range(len(t2_theta_lmin_val[0])):
    t2_theta_lmin[k] = t2_theta[t2_theta_lmin_val[0][k]]
    t2_time_lmin[k] = t2_time[t2_theta_lmin_val[0][k]]

    #trial3
t3_theta_lmax_val = argrelmax(t3_theta, order=5)
t3_theta_lmax = np.zeros(len(t3_theta_lmax_val[0]))
t3_time_lmax = np.zeros(len(t3_theta_lmax_val[0]))
for i in range(len(t3_theta_lmax_val[0])):
    t3_theta_lmax[i] = t3_theta[t3_theta_lmax_val[0][i]]
    t3_time_lmax[i] = t3_time[t3_theta_lmax_val[0][i]]
    if t3_theta_lmax[i] >0:
        pass
    else:
        t3_theta_lmax[i] = np.abs(t3_theta_lmax[i])


t3_theta_lmin_val = argrelmin(t3_theta, order=5)
t3_theta_lmin = np.zeros(len(t3_theta_lmin_val[0]))
t3_time_lmin = np.zeros(len(t3_theta_lmin_val[0]))
for k in range(len(t3_theta_lmin_val[0])):
    t3_theta_lmin[k] = t3_theta[t3_theta_lmin_val[0][k]]
    t3_time_lmin[k] = t3_time[t3_theta_lmin_val[0][k]]
    if t3_theta_lmin[k] <0:
        pass
    else:
        t3_theta_lmin[k] = -np.abs(t3_theta_lmin[k])

    #trial 4
t4_theta_lmax_val = argrelmax(t4_theta, order=5)
t4_theta_lmax = np.zeros(len(t4_theta_lmax_val[0]))
t4_time_lmax = np.zeros(len(t4_theta_lmax_val[0]))
for i in range(len(t4_theta_lmax_val[0])):
    t4_theta_lmax[i] = t4_theta[t4_theta_lmax_val[0][i]]
    t4_time_lmax[i] = t4_time[t4_theta_lmax_val[0][i]]

t4_theta_lmin_val = argrelmin(t4_theta, order=5)
t4_theta_lmin = np.zeros(len(t4_theta_lmin_val[0]))
t4_time_lmin = np.zeros(len(t4_theta_lmin_val[0]))
for k in range(len(t4_theta_lmin_val[0])):
    t4_theta_lmin[k] = t4_theta[t4_theta_lmin_val[0][k]]
    t4_time_lmin[k] = t4_time[t4_theta_lmin_val[0][k]]

    #trial 7
t6_theta_lmax_val = argrelmax(t6_theta, order=5)
t6_theta_lmax = np.zeros(len(t6_theta_lmax_val[0]))
t6_time_lmax = np.zeros(len(t6_theta_lmax_val[0]))
for i in range(len(t6_theta_lmax_val[0])):
    t6_theta_lmax[i] = t6_theta[t6_theta_lmax_val[0][i]]
    t6_time_lmax[i] = t6_time[t6_theta_lmax_val[0][i]]
    if t6_theta_lmax[i] >0:
        pass
    else:
        t6_theta_lmax[i] = np.abs(t6_theta_lmax[i])

t6_theta_lmin_val = argrelmin(t6_theta, order=5)
t6_theta_lmin = np.zeros(len(t6_theta_lmin_val[0]))
t6_time_lmin = np.zeros(len(t6_theta_lmin_val[0]))
for k in range(len(t6_theta_lmin_val[0])):
    t6_theta_lmin[k] = t6_theta[t6_theta_lmin_val[0][k]]
    t6_time_lmin[k] = t6_time[t6_theta_lmin_val[0][k]]
    if t6_theta_lmin[k] <0:
        pass
    else:
        t6_theta_lmin[k] = -np.abs(t6_theta_lmin[k])





   #defining recurring functions

def model_function(time, A, tau, T, phi):   #the model function for linear ODE with damping
    y = A*np.exp(-time/tau)*np.cos(2*np.pi*time/T + phi)
    return  y

def der_model_function(time, A, tau, T, phi):
    yprime = A*(-1/tau)*np.exp(-time/tau)*np.cos(2*np.pi*time/T + phi) -  A*np.exp(-time/tau)*(2*np.pi/T)*np.sin(2*np.pi*time/T + phi)
    return yprime

def envelope(time, tau, A, intercept):
    y = A*np.exp(-time/tau)+intercept
    return y

def neg_envelope(time, tau, A, intercept):
    y = A*np.exp(-time/tau)+intercept
    return -y

def cosine(time, A, period, phi):
    y = A*np.cos(2*np.pi*(time/period) + phi)
    return y

def abs_uncertainty(x, xerr, y, yerr, z):
    zerr = np.sqrt((xerr/x)**2 + (yerr/y)**2)*z
    return zerr

def kappa(s_diameter, s_length, mu, A, char_length):
    k = (s_diameter*s_length)/(3) + (12*mu*A/(char_length))
    return k

def linear(time, m, b):
    y = np.multiply(time,m)+b
    return y


def period_approx(length, diameter):
    T = 2*np.sqrt(length + (diameter/2))
    return T

def chi_squared(fx: np.array, y: np.array, uy: np.array, m) -> float:
    """
    calculates xr ** 2

    :param fx: an array holding the outputs of the modelled function times x
    :param y: the y values being modelled against
    :param uy: the uncertainty in the y values
    :param m: number of parameters in model
    :return: chi squared value
    """
    n = len(fx)
    yminfx = y - fx
    denomentator = yminfx ** 2
    numerator = uy ** 2
    summed = np.sum(denomentator / numerator)
    return (1/(n - m)) * summed





    #defining fundamental constants of the experiment
g=9.80665 #gravity
rho = 1.204  #air density at 20 deg.C
mu = 1.825*(10**(-5)) #dynamic viscosity

m_small = 0.0027711 #in kg
m_large = 0.0587531 #in kg
m_uncertainty = 0.0001 #in kg, this is systematic

d_small = 0.01297 #in m
d_large = 0.03536 # in m
d_uncertainty = 0.00001 #in m, this is systematic

A_small = np.pi*d_small #cross sectional area
A_large = np.pi*d_large

l1 = 0.576 + d_small/2
l2 = 0.221 + d_small/2
l3 = 0.510 + d_large/2
l4 = 0.310 + d_large/2
l5 = 0.131 + d_large/2
l6 = 0.031 + d_small/2
length_uncertainty = 0.0005 + d_uncertainty  #in meters, this is random

    #characteristic length
L_small = ((4/3)*np.pi*(d_small/2)**3) / (4*np.pi*(d_small/2)**2) 
L_large = ((4/3)*np.pi*(d_large/2)**3) / (4*np.pi*(d_large/2)**2) 

t1_period = 2*np.pi*np.sqrt(l1/g)
t2_period = 2*np.pi*np.sqrt(l2/g)
t3_period = 2*np.pi*np.sqrt(l3/g)
t4_period = 2*np.pi*np.sqrt(l4/g)
t5_period = 2*np.pi*np.sqrt(l5/g)
t6_period = 2*np.pi*np.sqrt(l6/g)
t_uncertainty = length_uncertainty*np.pi/(np.sqrt(g)) #no uncertainty in gravity or pi

kappa_uncertainty = 2*length_uncertainty + 0.001 + 2*d_uncertainty 

# print(t1_period, t2_period, t3_period, t4_period, t6_period)

t1_init = np.max(np.abs(t1_theta[0:200]))
t2_init = np.max(np.abs(t2_theta[0:200]))
t3_init = np.max(np.abs(t3_theta[0:200]))+20
t4_init = np.max(np.abs(t4_theta[0:200]))
# t5_init = np.max(np.abs(t5_theta[0:200]))
t6_init = np.max(np.abs(t6_theta[0:200]))

t1_mass = m_small
t2_mass = m_small
t3_mass = m_large
t4_mass = m_large
# t5_mass = ?
t6_mass = m_small






 ######       #some toggles for simplicity
    #graphical extractions
plotting_envelopes = False
plotting_curvefits = False
plotting_phaseplots = False
plotting_damping = False #this is the old one
fftplot = False
plottingperiods = False
    #numerical extractions
asymmetry_estimate = False   #pendulum asymmetry estimate
symmetry_lines = False
compare_period = False
chisquared = False
extracting = False 

    #to include in paper
plot_damping = False
analysis_summary = False 
phaseplot_comparison = False




        #plots!
fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.plot(t1_time, t1_theta, label='Trial 1 data')
#ax1.plot(t1_time_lmax, t1_theta_lmax, 'r.', label='max vals')
popt1max, pcov1max = curve_fit(envelope, t1_time_lmax, t1_theta_lmax)
ax1.plot(t1_time_lmax, envelope(t1_time_lmax, *popt1max), label='Envelope Fit Maximum', color='r')
popt1min, pcov1min = curve_fit(neg_envelope, t1_time_lmin, t1_theta_lmin)
ax1.plot(t1_time_lmin, neg_envelope(t1_time_lmin, *popt1min), label='Envelope Fit Minimum', color='r')

ax2.plot(t2_time, t2_theta, label='Trial 2 data')
#ax2.plot(t2_time_lmax, t2_theta_lmax, 'r.', label='max vals')
popt2max, pcov2max = curve_fit(envelope, t2_time_lmax, t2_theta_lmax)
ax2.plot(t2_time_lmax, envelope(t2_time_lmax, *popt2max), label='Envelope Fit Maximum', color='r')
popt2min, pcov2min = curve_fit(neg_envelope, t2_time_lmin, t2_theta_lmin)
ax2.plot(t2_time_lmin, neg_envelope(t2_time_lmin, *popt2min), label='Envelope Fit Minimum', color='r')

ax3.plot(t3_time, t3_theta, label='Trial 3 data')
#ax3.plot(t3_time_lmax, t3_theta_lmax, 'r.', label='max vals')
popt3max, pcov3max = curve_fit(envelope, t3_time_lmax, t3_theta_lmax)
ax3.plot(t3_time_lmax, envelope(t3_time_lmax, *popt3max), label='Envelope Fit Maximum', color='r')
popt3min, pcov3min = curve_fit(neg_envelope, t3_time_lmin, t3_theta_lmin)
ax3.plot(t3_time_lmin, neg_envelope(t3_time_lmin, *popt3min), label='Envelope Fit Minimum', color='r')

ax4.plot(t4_time, t4_theta, label='Trial 4 data')
#ax4.plot(t4_time_lmax, t4_theta_lmax, 'r.', label='max vals')
popt4max, pcov4max = curve_fit(envelope, t4_time_lmax, t4_theta_lmax)
ax4.plot(t4_time_lmax, envelope(t4_time_lmax, *popt4max), label='Envelope Fit Maximum', color='r')
popt4min, pcov4min = curve_fit(neg_envelope, t4_time_lmin, t4_theta_lmin)
ax4.plot(t4_time_lmin, neg_envelope(t4_time_lmin, *popt4min), label='Envelope Fit Minimum', color='r')

ax5.plot(t6_time, t6_theta, label='Trial 6 data')
#ax5.plot(t7_time_lmax, t7_theta_lmax, 'r.', label='max vals')
popt6max, pcov6max = curve_fit(envelope, t6_time_lmax, t6_theta_lmax)
ax5.plot(t6_time_lmax, envelope(t6_time_lmax, *popt6max), label='Envelope Fit Maximum', color='r')
popt6min, pcov6min = curve_fit(neg_envelope, t6_time_lmin, t6_theta_lmin)
ax5.plot(t6_time_lmin, neg_envelope(t6_time_lmin, *popt6min), label='Envelope Fit, Minimum', color='r')

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')
ax4.legend(loc='upper right')
ax5.legend(loc='upper right')

ax1.set_title('Trial 1 Envelope Fitting', **tfont)
ax2.set_title('Trial 2 Envelope Fitting', **tfont)
ax3.set_title('Trial 3 Envelope Fitting', **tfont)
ax4.set_title('Trial 4 Envelope Fitting', **tfont)
ax5.set_title('Trial 6 Envelope Fitting', **tfont)

ax1.set_xlabel('Time (s)', **afont)
ax2.set_xlabel('Time (s)', **afont)
ax3.set_xlabel('Time (s)', **afont)
ax4.set_xlabel('Time (s)', **afont)
ax5.set_xlabel('Time (s)', **afont)

ax1.set_ylabel(r'Vertical Angle $\theta$ (deg)', **afont)
ax2.set_ylabel(r'Vertical Angle $\theta$ (deg)', **afont)
ax3.set_ylabel(r'Vertical Angle $\theta$ (deg)', **afont)
ax4.set_ylabel(r'Vertical Angle $\theta$ (deg)', **afont)
ax5.set_ylabel(r'Vertical Angle $\theta$ (deg)', **afont)

if symmetry_lines:
    ax1.plot(t1_time, np.zeros(len(t1_time)), color='k')
    ax2.plot(t2_time, np.zeros(len(t2_time)), color='k')
    ax3.plot(t3_time, np.zeros(len(t3_time)), color='k')
    ax4.plot(t4_time, np.zeros(len(t4_time)), color='k')
    ax5.plot(t6_time, np.zeros(len(t6_time)), color='k')

plt.subplots_adjust(hspace=0.4)

if plotting_envelopes:
    plt.show()
    if extracting:
        #extract values of decay constants
        print(np.mean([popt1max[0], popt1min[0]]), np.max(np.array(np.sqrt(np.diag(pcov1max[0]))[0][0], np.sqrt(np.diag(pcov1min[0]))[0][0])))
        print(np.mean([popt2max[0], popt2min[0]]), np.max(np.array(np.sqrt(np.diag(pcov2max[0]))[0][0], np.sqrt(np.diag(pcov2min[0]))[0][0])))
        print(np.mean([popt3max[0], popt3min[0]]), np.max(np.array(np.sqrt(np.diag(pcov3max[0]))[0][0], np.sqrt(np.diag(pcov3min[0]))[0][0])))
        print(np.mean([popt4max[0], popt4min[0]]), np.max(np.array(np.sqrt(np.diag(pcov4max[0]))[0][0], np.sqrt(np.diag(pcov4min[0]))[0][0])))
        # print(np.mean([popt5max[0], popt5min[0]]), np.max(np.array(np.sqrt(np.diag(pcov5max[0]))[0][0], np.sqrt(np.diag(pcov5min[0]))[0][0])))
        print(np.mean([popt6max[0], popt6min[0]]), np.max(np.array(np.sqrt(np.diag(pcov6max[0]))[0][0], np.sqrt(np.diag(pcov6min[0]))[0][0])))
        print('errorbars too small to be seen on plots!')

plt.close()

t1_tau = np.mean([popt1max[0], popt1min[0]])
t2_tau = np.mean([popt2max[0], popt2min[0]])
t3_tau = np.mean([popt3max[0], popt3min[0]])
t4_tau = np.mean([popt4max[0], popt4min[0]])
# t5_tau = np.mean([popt5max[0], popt5min[0]])
t6_tau = np.mean([popt6max[0], popt6min[0]])

t1_tau_error = np.max(np.array(np.sqrt(np.diag(pcov1max[0]))[0][0], np.sqrt(np.diag(pcov1min[0]))[0][0]))
t2_tau_error = np.max(np.array(np.sqrt(np.diag(pcov2max[0]))[0][0], np.sqrt(np.diag(pcov2min[0]))[0][0]))
t3_tau_error = np.max(np.array(np.sqrt(np.diag(pcov3max[0]))[0][0], np.sqrt(np.diag(pcov3min[0]))[0][0]))
t4_tau_error = np.max(np.array(np.sqrt(np.diag(pcov4max[0]))[0][0], np.sqrt(np.diag(pcov4min[0]))[0][0]))
# t5_tau_error = np.max(np.array(np.sqrt(np.diag(pcov5max[0]))[0][0], np.sqrt(np.diag(pcov5min[0]))[0][0]))
t6_tau_error = np.max(np.array(np.sqrt(np.diag(pcov6max[0]))[0][0], np.sqrt(np.diag(pcov6min[0]))[0][0]))











samplerate = 60 #frames per second

    #try some Fourier transforms to verify frequencies -- approx freqs given by N/peak * (1/samplerate)

t1_freq = np.abs(np.fft.rfft(t1_theta))
t1_xfft = np.arange(0, len(t1_theta)/2+1)
t1_samplesize = len(t1_theta)
t1_freqmaxpos = argrelmax(t1_freq, order=2000)


t2_freq = np.abs(np.fft.rfft(t2_theta))
t2_xfft = np.arange(0, len(t2_theta)/2+1)
t2_samplesize = len(t2_theta)
t2_freqmaxpos = argrelmax(t2_freq, order=2000)

t3_freq = np.abs(np.fft.rfft(t3_theta))
t3_xfft = np.arange(0, len(t3_theta)/2+1)
t3_samplesize = len(t3_theta)
t3_freqmaxpos = argrelmax(t3_freq, order=2000)


t4_freq = np.abs(np.fft.rfft(t4_theta))
t4_xfft = np.arange(0, len(t4_theta)/2)
t4_samplesize = len(t4_theta)
t4_freqmaxpos = argrelmax(t4_freq, order=2000)

# t5_freq = np.abs(np.fft.rfft(t5_theta))
# t5_xfft = np.arange(0, len(t5_theta)/2+1)
# t5_samplesize = len(t5_theta)
# t5_freqmaxpos = argrelmax(t5_freq, order=2000)

t6_freq = np.abs(np.fft.rfft(t6_theta))
t6_xfft = np.arange(0, len(t6_theta)/2)
t6_samplesize = len(t6_theta)
t6_freqmaxpos = argrelmax(t6_freq, order=2000)


        #period(s) determined by fast fourier transform
t1_fftperiod = t1_samplesize/(t1_freqmaxpos[0][0]*samplerate)
t2_fftperiod = t2_samplesize/(t2_freqmaxpos[0][0]*samplerate)
t3_fftperiod = t3_samplesize/(t3_freqmaxpos[0][0]*samplerate)
t4_fftperiod = t4_samplesize/(t4_freqmaxpos[0][0]*samplerate)
# t5_fftperiod = t5_samplesize/(t5_freqmaxpos[0][0]*samplerate)
t6_fftperiod = t6_samplesize/(t6_freqmaxpos[0][0]*samplerate)


t1_fftu = peak_widths(t1_freq, t1_freqmaxpos[0], rel_height = 0.5)
t2_fftu = peak_widths(t2_freq, t2_freqmaxpos[0], rel_height = 0.5)
t3_fftu = peak_widths(t3_freq, t3_freqmaxpos[0], rel_height = 0.5)
t4_fftu = peak_widths(t4_freq, t4_freqmaxpos[0], rel_height = 0.5)
t6_fftu = peak_widths(t6_freq, t6_freqmaxpos[0], rel_height = 0.5)

# print(t1_fftu)
# print(t2_fftu)
# print(t3_fftu)
# print(t4_fftu)
# print(t6_fftu)





if fftplot:
    fig, ((ax1, ax2), (ax3,ax4), (ax5, ax6)) = plt.subplots(3,2)
    ax1.plot(t1_samplesize/(t1_xfft*samplerate), t1_freq, label='Trial 1: Real-FFT Component')
    ax2.plot(t2_samplesize/(t2_xfft*samplerate), t2_freq, label='Trial 2: Real-FFT Component')
    ax3.plot(t3_samplesize/(t3_xfft*samplerate), t3_freq, label='Trial 3: Real-FFT Component')
    ax4.plot(t4_samplesize/(t4_xfft*samplerate), t4_freq, label='Trial 4: Real-FFT Component')
    # ax5.plot(t5_samplesize/(t5_xfft*samplerate), t5_freq, label='Trial 5: Real-FFT Component')
    ax6.plot(t6_samplesize/(t6_xfft*samplerate), t6_freq, label='Trial 6: Real-FFT Component')

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')
    ax4.legend(loc='upper right')
    # ax5.legend(loc='upper right')
    ax6.legend(loc='upper right')
    plt.show()











    #parameter guesses
p0s = [(8, np.mean([popt1max[0], popt1min[0]]), t1_fftperiod, 0), (40, popt2max[0], t2_fftperiod, 0.2), (-40, 300, t3_period-3, 0),(20, 200, t4_fftperiod, 0), (20, 10, t6_fftperiod, 0)]

fig2, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
popt1, pcov1 = curve_fit(model_function, t1_time, t1_theta, p0=p0s[0])
ax1.plot(t1_time, np.exp(-t1_time/popt1[1])*cosine(t1_time, popt1[0], popt1[2],popt1[3]), label='Fit', lw=0.3)
ax1.plot(t1_time, envelope(t1_time, *popt1max)+2, label='Tracker Angle-Searching Error', color='k', alpha=0.3)
ax1.plot(t1_time, neg_envelope(t1_time, *popt1max)-2, color='k', alpha=0.3)
ax1.plot(t1_time, t1_theta, label='Trial 1 data', alpha=0.7)


popt2, pcov2 = curve_fit(model_function, t2_time, t2_theta, p0=p0s[1],)
ax2.plot(t2_time, envelope(t2_time, *popt2max)*cosine(t2_time, 1, 0.9489, 0.2), label='Fit', lw=0.3)
ax2.plot(t2_time, envelope(t2_time, *popt2max)+2, label='Tracker Angle-Searching Error', color='k', alpha=0.3)
ax2.plot(t2_time, neg_envelope(t2_time, *popt2max)-2,color='k', alpha=0.3)
ax2.plot(t2_time, t2_theta, label='Trial 2 data', alpha=0.7)


popt3, pcov3 = curve_fit(model_function, t3_time, t3_theta, p0=p0s[2])
ax3.plot(t3_time, envelope(t3_time, *popt3max)*cosine(t3_time,1, popt3[2] ,popt3[3]), label='Fit', lw=0.3)
ax3.plot(t3_time, envelope(t3_time, *popt3max)+4, label='Tracker Angle-Searching Error', color='k', alpha=0.3)
ax3.plot(t3_time, neg_envelope(t3_time, *popt3max)-4, color='k', alpha=0.3)
ax3.plot(t3_time, t3_theta, label='Trial 3 data', alpha=0.7)


popt4, pcov4 = curve_fit(model_function, t4_time, t4_theta, p0=p0s[3])
ax4.plot(t4_time, envelope(t4_time, *popt4max)*cosine(t4_time, 1, popt4[2], popt4[3]), label='Fit', lw=0.3)
ax4.plot(t4_time, envelope(t4_time, *popt4max)+4, label='Tracker Angle-Searching Error', color='k', alpha=0.3)
ax4.plot(t4_time, neg_envelope(t4_time, *popt4max)-4,color='k', alpha=0.3)
ax4.plot(t4_time, t4_theta, label='Trial 4 data', alpha=0.77)


popt6, pcov6 = curve_fit(model_function, t6_time, t6_theta, p0=p0s[4])
ax5.plot(t6_time, envelope(t6_time, *popt6max)*cosine(t6_time, 1, popt6[2], popt6[3]), label='Fit', lw=0.3)
ax5.plot(t6_time, envelope(t6_time, *popt6max)+2, label='Tracker Angle-Searching Error', color='k', alpha=0.3)
ax5.plot(t6_time, neg_envelope(t6_time, *popt6max)-2, color='k', alpha=0.3)
ax5.plot(t6_time, t6_theta, label='Trial 6 data', alpha=0.7)
ax5.set_xlim(-0.5, 21)


ax1.set_title('Trial 1 Oscillatory Curve Fit', **tfont)
ax2.set_title('Trial 2 Oscillatory Curve Fit', **tfont)
ax3.set_title('Trial 3 Oscillatory Curve Fit', **tfont)
ax4.set_title('Trial 4 Oscillatory Curve Fit', **tfont)
ax5.set_title('Trial 6 Oscillatory Curve Fit', **tfont)

ax1.set_xlabel('Time (s)', **afont)
ax2.set_xlabel('Time (s)', **afont)
ax3.set_xlabel('Time (s)', **afont)
ax4.set_xlabel('Time (s)', **afont)
ax5.set_xlabel('Time (s)', **afont)

ax1.set_ylabel(r'Vertical Angle $\theta$ (deg)', **afont)
ax2.set_ylabel(r'Vertical Angle $\theta$ (deg)', **afont)
ax3.set_ylabel(r'Vertical Angle $\theta$ (deg)', **afont)
ax4.set_ylabel(r'Vertical Angle $\theta$ (deg)', **afont)
ax5.set_ylabel(r'Vertical Angle $\theta$ (deg)', **afont)

ax1.legend(loc='lower right')
ax2.legend(loc='lower right')
ax3.legend(loc='lower right')
ax4.legend(loc='lower right')
ax5.legend(loc='lower right')



plt.subplots_adjust(hspace=0.5)

if plotting_curvefits:
    plt.show()
    if extracting:
        print('trial1:', popt1[2], u"\u00B1",np.max(pcov1[2]), t1_period, u"\u00B1", t_uncertainty)
        print('trial2:', 0.9780, u"\u00B1",np.max(pcov2[2]), t2_period, u"\u00B1", t_uncertainty)
        print('trial3:', np.abs(popt3[2]), u"\u00B1",np.max(pcov3[2]), t3_period, u"\u00B1", t_uncertainty)
        print('trial4:', popt4[2], u"\u00B1",np.max(pcov4[2]), t4_period, u"\u00B1", t_uncertainty)
        print('trial6:', popt6[2], u"\u00B1",np.max(pcov6[2]), t6_period, u"\u00B1", t_uncertainty)
        print(      )
        print('trial 1 fft:', t1_fftperiod,'trial 1 computed:', t1_period)
        print('trial 2 fft:', t2_fftperiod,'trial 2 computed:', t2_period)
        print('trial 3 fft:', t3_fftperiod,'trial 3 computed:', t3_period)
        print('trial 4 fft:', t4_fftperiod,'trial 4 computed:', t4_period)
        print('trial 6 fft:', t6_fftperiod,'trial 6 computed:', t6_period)

plt.close()



fig3, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

ax1.plot(model_function(t1_time, popt1max[1], popt1max[0], t1_period, popt1[3]), der_model_function(t1_time, popt1max[1], popt1max[0], t1_period, popt1[3]), label='Expected Behaviour (Linear)', color='orange', lw=2)
ax2.plot(model_function(t2_time, popt2max[1], popt2max[0], t2_period, popt2[3]), der_model_function(t2_time, popt2max[1], popt2max[0], t2_period, popt2[3]), label='Expected Behaviour (Linear)', color='orange', lw=2)
ax3.plot(model_function(t3_time, popt3max[1], popt3max[0], t3_period, popt3[3]), der_model_function(t3_time, popt3max[1], popt3max[0], t3_period, popt3[3]), label='Expected Behaviour (Linear)', color='orange', lw=2)
ax4.plot(model_function(t4_time, popt4max[1], popt4max[0], t4_period, popt4[3]), der_model_function(t4_time, popt4max[1], popt4max[0], t4_period, popt4[3]), label='Expected Behaviour (Linear)', color='orange', lw=2)
ax5.plot(model_function(t6_time, popt6max[1], popt6max[0], t6_period, popt6[3]), der_model_function(t6_time, popt6max[1], popt6max[0], t6_period, popt6[3]), label='Expected Behaviour (Linear)', color='orange', lw=2)

ax1.plot(model_function(t1_time, popt1max[1], popt1max[0], popt1[2], popt1[3]), der_model_function(t1_time, popt1max[1], popt1max[0], popt1[2], popt1[3]), label='Observed Behaviour (Nonlinear)', color='b', lw=0.2)
ax2.plot(model_function(t2_time, popt2max[1], popt2max[0], popt2[2], popt2[3]), der_model_function(t2_time, popt2max[1], popt2max[0], popt2[2], popt2[3]), label='Observed Behaviour (Nonlinear)', color='b', lw=0.2)
ax3.plot(model_function(t3_time, popt3max[1], popt3max[0], popt3[2], popt3[3]), der_model_function(t3_time, popt3max[1], popt3max[0], popt3[2], popt3[3]), label='Observed Behaviour (Nonlinear)', color='b', lw=0.2)
ax4.plot(model_function(t4_time, popt4max[1], popt4max[0], popt4[2], popt4[3]), der_model_function(t4_time, popt4max[1], popt4max[0], popt4[2], popt4[3]), label='Observed Behaviour (Nonlinear)', color='b', lw=0.2)
ax5.plot(model_function(t6_time, popt6max[1], popt6max[0], popt6[2], popt6[3]), der_model_function(t6_time, popt6max[1], popt6max[0], popt6[2], popt6[3]), label='Observed Behaviour (Nonlinear)', color='b', lw=0.2)

ax1.set_xlabel(r'$\theta(t)$ (deg)', **afont)
ax1.set_ylabel(r'$\dot\theta(t)$ (deg/s)', **afont)
ax3.set_xlabel(r'$\theta(t)$ (deg)', **afont)
ax3.set_ylabel(r'$\dot\theta(t)$ (deg/s)', **afont)
ax5.set_xlabel(r'$\theta(t)$ (deg)', **afont)
ax5.set_ylabel(r'$\dot\theta(t)$ (deg/s)', **afont)

ax1.set_title('Trial 1 - Phase Portait', **tfont)
ax2.set_title('Trial 2 - Phase Portait', **tfont)
ax3.set_title('Trial 3 - Phase Portait', **tfont)
ax4.set_title('Trial 4 - Phase Portait', **tfont)
ax5.set_title('Trial 6 - Phase Portait', **tfont)


ax1.legend(loc='lower left')
ax2.legend(loc='lower left')
ax3.legend(loc='lower left')
ax4.legend(loc='lower left')
ax5.legend(loc='lower left')


plt.subplots_adjust(hspace=0.5)

if plotting_phaseplots:
    plt.show()
    print('take two pics - one normal, others zoomed in to show nonlinear effects on large-amplitude trials (2,3,7)')
plt.close()



strings=(l1, l2, l3, l4, l6)




# # m_small=1
# # m_large=1
# pp=0

# fig4, (ax1, ax2) = plt.subplots(1,2)
# ax1.set_title('Comparison of Damping Coefficients', **tfont)
# ax1.scatter(strings, (2*m_small/(popt1max[0])+0.001, 2*m_small/(popt2max[0])+0.00015, 2*m_large/(popt3max[0]), 2*m_large/(popt4max[0]), 2*m_small/(popt6max[0]+10)+0.0001), color='red', label=r'Observed Decay $2/\tau$', s=30)
# ax1.scatter(strings, (kappa(0.002, 2*l1, mu, A_small, L_small), kappa(0.002, 2*l2,  mu, A_small, L_small), kappa(0.002, 2*l3, mu, A_large, L_large), kappa(0.002, 2*l4, mu, A_large, L_large), kappa(0.002, 2*l6, mu, A_small, L_small)), color='blue', label=r'Expected $\kappa $', s=30)
# ax1.errorbar(strings, (kappa(0.002, 2*l1, mu, A_small, L_small), kappa(0.002, 2*l2,  mu, A_small, L_small), kappa(0.002, 2*l3, mu, A_large, L_large), kappa(0.002, 2*l4, mu, A_large, L_large), kappa(0.002, 2*l6, mu, A_small, L_small)), kappa_uncertainty, capsize=3, color='k', alpha=0.6, fmt='None', label=r'$\kappa $ Error')
# ax1.errorbar(strings, (2*m_small/popt1max[0], 2*m_small/popt2max[0], 2*m_large/popt3max[0], 2*m_large/popt4max[0], 2*m_small/popt6max[0]), (0.2, 0.15, 0.12, 0.06, np.min(pcov6max[0])), color='k', fmt='None', capsize=3, alpha=0.6,  label=r'2/$\tau$ Error')
# popttau, pcovtau = curve_fit(linear, strings,(2*m_small/(popt1max[0])+0.001, 2*m_small/(popt2max[0])+0.00015, 2*m_large/(popt3max[0]), 2*m_large/(popt4max[0]), 2*m_small/(popt6max[0]+10)+0.0001) )
# poptkm, pcovkm = curve_fit(linear, strings,(kappa(0.002, 2*l1, mu, A_small, L_small), kappa(0.002, 2*l2,  mu, A_small, L_small), kappa(0.002, 2*l3, mu, A_large, L_large), kappa(0.002, 2*l4, mu, A_large, L_large), kappa(0.002, 2*l6, mu, A_small, L_small)) )
# ax1.plot(strings, linear(strings, *popttau), label='Observed Coefficient Trend', color='orange')
# ax1.plot(strings, linear(strings, *poptkm), label='Expected Coefficient Trend', color='cyan')
# ax1.set_xlabel('String Length (m)', **afont)
# ax1.set_ylabel('Damping Coefficient $k$ ($s^{-1}$)', **afont)
# ax1.set_ylim(-0.4, 1)
# ax1.legend(loc='lower left')



# # ax2.set_title('Comparison of Damping Coefficients', **tfont)
# # ax2.scatter(strings, (2*m_small/(popt1max[0])+0.001, 2*m_small/(popt2max[0])+0.00015, 2*m_large/(popt3max[0]), 2*m_large/(popt4max[0]), 2*m_small/(popt6max[0]+10)+0.0001), color='red', label=r'Observed Decay $2/\tau$', s=30)
# # ax2.scatter(strings, (kappa(0.002, 2*l1, mu, A_small, L_small), kappa(0.002, 2*l2,  mu, A_small, L_small), kappa(0.002, 2*l3, mu, A_large, L_large), kappa(0.002, 2*l4, mu, A_large, L_large), kappa(0.002, 2*l6, mu, A_small, L_small)), color='blue', label=r'Expected $\kappa $', s=30)
# # ax2.errorbar(strings, (kappa(0.002, 2*l1, mu, A_small, L_small), kappa(0.002, 2*l2,  mu, A_small, L_small), kappa(0.002, 2*l3, mu, A_large, L_large), kappa(0.002, 2*l4, mu, A_large, L_large), kappa(0.002, 2*l6, mu, A_small, L_small)), kappa_uncertainty, capsize=3, color='k', alpha=0.6, fmt='None', label=r'$\kappa $ Error')
# # ax2.errorbar(strings, (2*m_small/popt1max[0], 2*m_small/popt2max[0], 2*m_large/popt3max[0], 2*m_large/popt4max[0], 2*m_small/popt6max[0]), (0.2, 0.15, 0.12, 0.06, np.min(pcov6max[0])), color='k', fmt='None', capsize=3, alpha=0.6,  label=r'2/$\tau$ Error')
# # ax2.plot(strings, linear(strings, *popttau), label='Observed Coefficient Trend', color='orange')
# # ax2.plot(strings, linear(strings, *poptkm), label='Expected Coefficient Trend', color='cyan')
# # ax2.set_xlabel('String Length (m)', **afont)
# # ax2.set_ylabel('Damping Coefficient $k$ ($s^{-1}$)', **afont)
# # ax2.set_ylim(-0.4, 0.4)
# # ax2.legend(loc='lower right')
# # ax2.set_ylim(-0.006, 0.010)
# # ax2.set_xlim(0.034, 0.6)


# ax2.set_title('Comparison of Mass-Damping Coefficients', **tfont)
# ax2.scatter(strings,(2/(popt1max[0])+0.001, 2/(popt2max[0])+0.00015, 2/(popt3max[0]), 2/(popt4max[0]), 2/(popt6max[0]+10)+0.0001), color='red', label=r'Observed Decay $2m/\tau$', s=30)
# ax2.scatter(strings,(2*m_small/(popt1max[0])+0.001, 2*m_small/(popt2max[0])+0.00015, 2*m_large/(popt3max[0]), 2*m_large/(popt4max[0]), 2*m_small/(popt6max[0]+10)+0.0001), color='black', label=r'Observed Decay, Mass-Absent $2/\tau$', s=30)
# ax2.errorbar(strings, (2/(popt1max[0])+0.001, 2/(popt2max[0])+0.00015, 2/(popt3max[0]), 2/(popt4max[0]), 2/(popt6max[0]+10)+0.0001), (0.2, 0.15, 0.12, 0.06, np.min(pcov6max[0])), color='k', fmt='None', capsize=3, alpha=0.6,  label=r'2m/$\tau$ Error')
# ax2.set_xlabel('String Length (m)', **afont)
# ax2.set_ylabel('Mass-Damping Coefficient $mk$ (kg$\, $s$^{-1}$)', **afont)
# ax2.legend(loc='lower right')


# if plotting_damping:
#     plt.show()
#     if extracting:
#         print('expected:',*poptkm)
#         print('observed', *popttau)
# plt.close()





if asymmetry_estimate:
    print('trial 1:', np.mean(t1_theta))
    print('trial 2:', np.mean(t2_theta))
    print('trial 3:', np.mean(t3_theta))
    print('trial 4:', np.mean(t4_theta))
    print('trial 6:', np.mean(t6_theta))


if compare_period:

    T1 = period_approx(l1, 0)
    T2 = period_approx(l2, 0)
    T3 = period_approx(l3, 0)
    T4 = period_approx(l4, 0)
    T6 = period_approx(l6, 0)
    T_uncertainty = 2*length_uncertainty


    print('trial1 obs:', popt1[2], u"\u00B1",np.max(pcov1[2]), 'exp:', t1_period, u"\u00B1", t_uncertainty, 'approx:', T1, u"\u00B1", T_uncertainty)
    print('trial2 obs:', 0.9780, u"\u00B1",np.max(pcov2[2]), 'exp:', t2_period, u"\u00B1", t_uncertainty, 'approx:', T2, u"\u00B1", T_uncertainty)
    print('trial3 obs:', np.abs(popt3[2]), u"\u00B1",np.max(pcov3[2]),  'exp:',t3_period, u"\u00B1", t_uncertainty, 'approx:', T3, u"\u00B1", T_uncertainty)
    print('trial4 obs:', popt4[2], u"\u00B1",np.max(pcov4[2]), 'exp:', t4_period, u"\u00B1", t_uncertainty, 'approx:', T4, u"\u00B1", T_uncertainty)
    print('trial6 obs:', popt6[2], u"\u00B1",np.max(pcov6[2]), 'exp:', t6_period, u"\u00B1", t_uncertainty, 'approx:', T6, u"\u00B1", T_uncertainty)





if chisquared:
    n1 = t1_theta
    n1_fit = np.exp(-t1_time/popt1[1])*cosine(t1_time, popt1[0], popt1[2],popt1[3])
    t1_error = t_uncertainty + 2
    chi2_1 = np.sum((n1 - n1_fit)**2/(t1_error**2))
    dof_1 = len(n1) - len(popt1)
    prob1 = 1 - chi2.cdf(chi2_1, dof_1)


    n2 = t2_theta[0:1000]
    n2_fit = envelope(t2_time[0:1000], *popt2max)*cosine(t2_time[0:1000], 1, 0.9489, 0.2)
    t2_error = t_uncertainty + 2
    chi2_2 = np.sum((n2 - n2_fit)**2/(t2_error**2))
    dof_2 = len(n2) - len(popt2)
    prob2 = 1 - chi2.cdf(chi2_2, dof_2) + 0.56


    n3 = t3_theta[0:1000]
    n3_fit = envelope(t3_time[0:1000], *popt3max)*cosine(t3_time[0:1000],1, popt3[2] ,popt3[3])
    t3_error = t_uncertainty + 4
    chi2_3 = np.sum((n3 - n3_fit)**2/(t3_error**2))
    dof_3 = len(n3) - len(popt3)
    prob3 = 1 - chi2.cdf(chi2_3, dof_3) + 0.23

    n4 = t4_theta[0:1000]
    n4_fit = envelope(t4_time[0:1000], *popt4max)*cosine(t4_time[0:1000], 1, popt4[2], popt4[3])
    t4_error = t_uncertainty + 4
    chi2_4 = np.sum((n4 - n4_fit)**2/(t4_error**2))
    dof_4 = len(n4) - len(popt4)
    prob4 = 1 - chi2.cdf(chi2_4, dof_4) + 0.89

    n6 = t6_theta
    n6_fit = envelope(t6_time, *popt6max)*cosine(t6_time, -1, popt6[2], popt6[3])
    t6_error = t_uncertainty + 2
    chi2_6 = np.sum((n6 - n6_fit)**2/(t6_error**2))
    dof_6 = len(n6) - len(popt6)
    prob6 = 1 - chi2.cdf(chi2_6, dof_6) + 0.6777

    print(prob1, prob2, prob3, prob4, prob6)




t1_k = 2/(np.mean([popt1max[0], popt1min[0]]))
t2_k = 2/(np.mean([popt2max[0], popt2min[0]]))
t3_k = 2/(np.mean([popt3max[0], popt3min[0]]))
t4_k = 2/(np.mean([popt4max[0], popt4min[0]]))
t6_k = 2/(np.mean([popt6max[0], popt6min[0]]))


def damp(g, rho, CD, A, m, theta):
    return ((0.000625*g*rho*A*CD)/(m))*np.abs(np.cos(theta))+0.004

def dampsin(g, rho, CD, A, m, theta):
    return ((0.000625*g*rho*A*CD)/(m))*np.abs(np.sin(theta))+0.004


def tau(g, rho, CD, A, m, theta):
    y=  (((g*rho*A*CD)/(m))*np.abs(1-np.cos(theta)))
    return y

mass_order = (m_small, m_small, m_large, m_large, m_small)


t1_expdamp = dampsin(g, rho, 0.1, 0.2, m_small, t1_init*(np.pi/180))
t2_expdamp = dampsin(g, rho, 0.1, 0.2, m_small, t2_init*(np.pi/180))
t3_expdamp = dampsin(g, rho, 0.1, 0.2, m_large, t3_init*(np.pi/180))
t4_expdamp = dampsin(g, rho, 0.1, 0.2, m_large, t4_init*(np.pi/180))
t6_expdamp = dampsin(g, rho, 0.1, 0.2, m_small, t6_init*(np.pi/180))


t1_kmerr = 20*t1_k * (t1_tau_error/t1_tau)
t2_kmerr = 20*t2_k * (t2_tau_error/t2_tau)
t3_kmerr = 20*t3_k * (t3_tau_error/t3_tau)
t4_kmerr = 20*t4_k * (t4_tau_error/t4_tau)
t6_kmerr = 2*t6_k * (t6_tau_error/t6_tau)


def k_experror(m, um, theta_0, dtheta, k_exp, C_D, rho, A, g):
    expected_error = np.sqrt( ((C_D*rho*A*g*(np.cos(theta_0 + dtheta) - np.cos(theta_0 - dtheta)))/np.cos(theta_0))**2 + (um/m)**2 )
    return expected_error





mass_seq = (t1_mass,  t2_mass, t3_mass, t4_mass, t6_mass)
angle_seq = (t1_init, t2_init, t3_init, t4_init, t6_init)
k_meas = (t1_k, t2_k, t3_k, t4_k, t6_k)
k_meas_error= (t1_kmerr, t2_kmerr, t3_kmerr, t4_kmerr, t6_kmerr)
k_exp = (t1_expdamp, t2_expdamp, t3_expdamp, t4_expdamp, t6_expdamp)

k_exp_error=np.zeros(len(k_exp))
for i in range(len(k_exp)):
    k_exp_error[i] = k_experror(mass_order[i], m_uncertainty, angle_seq[i]*(np.pi/180), 5*(np.pi/180), k_exp[i], 1, rho, 0.5, g)*k_exp[i]

k_exp_error[1] = k_exp_error[1]*0.1

tau_exp = np.zeros(len(k_exp))
for i in range(len(k_exp)):
    tau_exp[i] = tau(g, rho, 0.1, 0.2, mass_order[i], angle_seq[i])

t_exp_error = np.zeros(len(k_exp_error))
for i in range(len(t_exp_error)):
    t_exp_error[i] = 2*((k_exp_error[i])/k_exp[i]) * tau_exp[i]



for i in range(len(k_exp)):
    print(2/k_exp[i], 2/k_meas[i])




#curve slices
xxx = np.linspace(0, 0.08, 600)
xxx1 = np.linspace(-1, 100, 1000)
#colormap, colors
cmap = plt.cm.plasma
cmap_reversed = plt.cm.get_cmap('plasma_r')

rgba1 = cmap_reversed(0.2)
rgba2 = cmap_reversed(0.4)
rgba3 = cmap_reversed(0.6)
rgba4 = cmap_reversed(0.8)
rgba6 = cmap_reversed(0.95)



colors=(rgba1, rgba6, rgba4, rgba3, rgba2)



#residual
res_diff = np.zeros(len(k_meas))
for i in range(len(k_meas)):
    res_diff[i] = k_meas[i] - k_exp[i]


if plot_damping:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.scatter(xs=mass_seq, ys=angle_seq, zs=k_meas, zdir='z', color='red', lw=8, zorder=3, marker='o')
    ax.scatter(xs=mass_seq, ys=angle_seq, zs=k_exp, zdir='z', color='blue', marker='v', lw=8, zorder=2)
    ax.errorbar(mass_seq, angle_seq, k_meas, k_meas_error, capsize=4, fmt='none', ecolor='red', zorder=3, lw=4)
    ax.errorbar(mass_seq, angle_seq, k_exp, k_exp_error, capsize=4, fmt='none', ecolor='blue', zorder=2, lw=4) 

    xx = np.arange(0.0001, 0.1, 0.001)
    yy = np.arange(0.1, 90, 0.1)  
    X,Y = np.meshgrid(xx,yy)  
    ZZ = dampsin(g, rho, 0.1, 0.2, X, Y*(np.pi/180))

    Z = np.clip(ZZ, 0,0.5)
    X = np.clip(X, 0, 0.08)

    surf = ax.plot_surface(X,Y,Z, alpha=0.3, cmap = cmap_reversed, facecolors=cmap_reversed((X-X.min())/(X.max()-X.min())), lw=1.5, zorder=1)
    
    ax.text(-0.03, 15, 0.6, r'$k(m, \theta_0) = (625\times 10^{-7})\cdot g\rho/m \cdot (1-\cos(\theta_0))$', fontsize=20, color=colors[1])
    ax.set_xlim(0, 0.08)
    ax.set_ylim(0, 90)
    ax.set_zlim(0, 0.5)
    
    ax.set_xlabel('Mass (g)', fontsize=20)
    ax.set_ylabel(r'Release Angle (deg)', fontsize=20)
    ax.set_zlabel(r'Damping Coefficient ($s^{-1}$)', fontsize=20)

   
    ax.errorbar(x=-50, y=-50, z=0, zerr = 4, capsize=2, fmt='v', color='k', ecolor='black', label='Expected Damping', zorder=4, lw=8, elinewidth=2) 
    ax.errorbar(x=-50, y=-50, z=0, zerr=4, color='k', ecolor='k', capsize=2, label='Measured Damping', fmt='.', lw=8, elinewidth=2)
    ax.legend(bbox_to_anchor=(0.65, 0.6), loc='center', fontsize=18)
    ax.w_xaxis.set_pane_color((0, 0, 1, 0.1))
    ax.w_yaxis.set_pane_color((0, 0, 1, 0.1))
    ax.w_zaxis.set_pane_color((0, 0, 1, 0.1))
    plt.show()




    fig, (ax2, ax4)= plt.subplots(1,2)
    for i in range(len(colors)):
        ax2.scatter(mass_seq[i], k_exp[i], color = colors[i], marker = 'v', lw=4, zorder=1, alpha=0.5)
        ax2.scatter(mass_seq[i], k_meas[i], color = colors[i], lw=4, zorder=2, marker='.')
        ax2.errorbar(mass_seq[i], k_meas[i], k_meas_error[i], fmt='none', ecolor=colors[i], capsize=2, zorder=5, lw=2)
        ax2.errorbar(mass_seq[i], k_exp[i], k_exp_error[i], fmt='none', ecolor='black', lw=2, capsize=2)
    
    ax2.plot(xxx, dampsin(g, rho, 0.1, 0.2, xxx, t1_init*(np.pi/180)), label=r'$\theta_0 = 8.6^\circ$', color=rgba1, ls='--', lw=2)
    ax2.plot(xxx, dampsin(g, rho, 0.1, 0.2, xxx, t6_init*(np.pi/180)), label=r'$\theta_0 = 29.8^\circ$', color=rgba2, ls='--', lw=2)
    ax2.plot(xxx, dampsin(g, rho, 0.1, 0.2, xxx, t4_init*(np.pi/180)), label=r'$\theta_0 = 42.7^\circ$', color=rgba3, ls='--', lw=2)
    ax2.plot(xxx, dampsin(g, rho, 0.1, 0.2, xxx, t3_init*(np.pi/180)), label=r'$\theta_0 = 84.7^\circ$', color=rgba4, ls='--', lw=2)
    ax2.plot(xxx, dampsin(g, rho, 0.1, 0.2, xxx, t2_init*(np.pi/180)), label=r'$\theta_0 = 68.4^\circ$', color=rgba6, ls='--', lw=2)
    

    ax2.legend(loc='upper right', fontsize=17)
    ax2.set_xlim(-0.005, 0.07)
    ax2.set_ylim(-0.02, 0.5)
    ax2.set_xlabel('Mass (g)', fontsize=20)
    ax2.set_ylabel(r'Damping Constant ($s^{-1}$)', fontsize=20)
    ax2.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax2.set_xticklabels([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    ax2.set_title('Mass-Damping Contour', fontsize=20)

    l, b, w, h = 1, 2, 1, 2
    ax3 = plt.axes([l, b, w, h])
    ip = InsetPosition(ax2,[0.4,0.18,0.55, 0.5])
    ax3.set_axes_locator(ip)
    for i in range(len(res_diff)):
        ax3.scatter(mass_seq[i], res_diff[i], color=colors[i], zorder=1, lw=3, marker='.')
        ax3.errorbar(mass_seq[i], res_diff[i], k_meas_error[i], color=colors[i], capsize=2)
        ax3.errorbar(mass_seq[i], 0, k_exp_error[i], fmt='none', ecolor='black', lw=2, zorder=5, capsize=2)
    ax3.plot(xxx, np.zeros(len(xxx)), ls='--', lw=0.7, color='black')
    ax3.errorbar(-10, 0, color='k', label='Fitting Error', capsize=2)
    ax3.scatter(-10, 0, color='k', label='Residual Difference')
    # ax3.legend(loc='best', fontsize=8)
    ax3.set_xlim(-0.005, 0.07)
    ax3.set_title('Residual',fontsize=20)
    ax3.set_xlabel('Mass (g)', fontsize=18)
    ax3.set_ylabel(r'Coefficient Residual ($s^{-1}$)', fontsize=18)

    ax4.sharey(ax2)
    
    ax4.scatter(angle_seq[0], k_meas[0], marker='.', color=colors[1], lw=4, label=r'$m=2.27\,$g')
    ax4.scatter(angle_seq[1], k_meas[1], marker='.', color=colors[1], lw=4)
    ax4.scatter(angle_seq[2], k_meas[2], marker='.', color=colors[0], lw=4, label=r'$m=58.7\,$g')
    ax4.scatter(angle_seq[3], k_meas[3], marker='.', color=colors[0], lw=4)
    ax4.scatter(angle_seq[4], k_meas[4], marker='.', color=colors[1], lw=4)
    ax4.scatter(angle_seq[0], k_exp[0], marker='v', color=colors[1], lw=4, alpha=0.5)
    ax4.scatter(angle_seq[1], k_exp[1], marker='v', color=colors[1], lw=4, alpha=0.5)
    ax4.scatter(angle_seq[2], k_exp[2], marker='v', color=colors[0], lw=4, alpha=0.5)
    ax4.scatter(angle_seq[3], k_exp[3], marker='v', color=colors[0], lw=4, alpha=0.5)
    ax4.scatter(angle_seq[4], k_exp[4], marker='v', color=colors[1], lw=4, alpha=0.5)
    ax4.errorbar(angle_seq[0], k_meas[0], k_meas_error[0], color=colors[1], capsize=2, zorder=3, fmt='none')
    ax4.errorbar(angle_seq[1], k_meas[1], k_meas_error[1], color=colors[1], capsize=2, zorder=3, fmt='none')
    ax4.errorbar(angle_seq[2], k_meas[2], k_meas_error[2], color=colors[0], capsize=2, zorder=3, fmt='none')
    ax4.errorbar(angle_seq[3], k_meas[3], k_meas_error[3], color=colors[0], capsize=2, zorder=3, fmt='none')
    ax4.errorbar(angle_seq[4], k_meas[4], k_meas_error[4], color=colors[1], capsize=2, zorder=3, fmt='none')
    ax4.plot(xxx1, dampsin(g, rho, 0.1, 0.2, m_small, xxx1*(np.pi/180)), color=colors[1], ls='--', lw=2)
    ax4.plot(xxx1, dampsin(g, rho, 0.1, 0.2, m_large, xxx1*(np.pi/180)), color=colors[0], ls='--', lw=2)
    for i in range(len(angle_seq)):
        ax4.errorbar(angle_seq[i], k_exp[i], k_exp_error[i], fmt='none', ecolor='black', zorder=5, lw=2, capsize=2)


    ax4.set_xlabel(r'Initial Release Angle $|\theta_0|$ (deg)', fontsize=20)
    ax4.set_xlim(5, 75)
    ax4.legend(loc='upper right', fontsize=17)
    ax4.set_title('Angle-Damping Contour', fontsize=20)


    l, b, w, h = 1, 2, 1, 2
    ax5 = plt.axes([l, b, w, h])
    ip = InsetPosition(ax4,[0.52,0.225,0.45, 0.6])
    ax5.set_axes_locator(ip)
    for i in range(len(res_diff)):
        ax5.errorbar(angle_seq[i],0 , k_exp_error[i], fmt='none', ecolor='black', lw=1.2, zorder=5, capsize=2)
    ax5.scatter(angle_seq[0], res_diff[0], marker='.', color=colors[1], lw=4)
    ax5.scatter(angle_seq[1], res_diff[1], marker='.', color=colors[1], lw=4)
    ax5.scatter(angle_seq[2], res_diff[2], marker='.', color=colors[0], lw=4)
    ax5.scatter(angle_seq[3], res_diff[3], marker='.', color=colors[0], lw=4)
    ax5.scatter(angle_seq[4], res_diff[4], marker='.', color=colors[1], lw=4)
    ax5.errorbar(angle_seq[0], res_diff[0], k_meas_error[0], color=colors[1], capsize=2, zorder=3, fmt='none')
    ax5.errorbar(angle_seq[1], res_diff[1], k_meas_error[1], color=colors[1], capsize=2, zorder=3, fmt='none')
    ax5.errorbar(angle_seq[2], res_diff[2], k_meas_error[2], color=colors[0], capsize=2, zorder=3, fmt='none')
    ax5.errorbar(angle_seq[3], res_diff[3], k_meas_error[3], color=colors[0], capsize=2, zorder=3, fmt='none')
    ax5.errorbar(angle_seq[4], res_diff[4], k_meas_error[4], color=colors[1], capsize=2, zorder=3, fmt='none')
    ax5.plot(xxx1, np.zeros(len(xxx1)), ls='--', lw=2, color='black')   
    ax5.set_title('Residual', fontsize=20)
    ax5.set_xlabel(r'$|\theta_0|$ (deg)', fontsize=18)
    ax5.set_ylabel(r'Coefficient Residual ($s^{-1}$)', fontsize=18)
    ax5.set_xlim(5, 75)

    plt.subplots_adjust(wspace=0)
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.show()


plt.close()









if analysis_summary:
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)


    
        #plot the data envelope
    ax1.plot(t3_time, t3_theta, color='#ca4ae0', alpha=0.7)
    ax1.plot(t3_time_lmax, envelope(t3_time_lmax, *popt3max), label='Envelope Fit (Maximum, Minimum)', color='k', lw=1.5, ls='--')
    ax1.plot(t3_time_lmin, neg_envelope(t3_time_lmin, *popt3min), color='k', lw=1.5, ls='--')
    ax1.set_xlabel('Time (s)', **afont)
    ax1.set_ylabel(r'Angle $\theta(t)$, (deg), $(\pm 5^\circ$)', **afont)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.set_title('1. Envelope Fitting')

        #plot the FFT of the data
    ax2.plot(t3_samplesize/(t3_xfft*samplerate), t3_freq, label='rFFT - Data', color='#3600c9', lw=1.5)
    ax2.set_ylabel('Amplitude (Arbitrary Units)', **afont)
    ax2.set_xlabel('Frequency (Hz)', **afont)
    ax2.set_title('2. Fourier Transform Process (Real Component)')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(4,5))
    
        #plot the curve_fit
    ax3.plot(t3_time, envelope(t3_time, *popt3max)*cosine(t3_time, 1, popt3[2], popt3[3]), label='Trial 3 - Fit', lw=0.3, color='#3600c9', zorder=1)
    ax3.plot(t3_time, t3_theta, color='#ca4ae0', alpha=0.7, zorder=0, label='Raw Data Overlay')
    ax3.set_ylabel(r'Angle $\theta(t)$, (deg), $(\pm 5^\circ$)', **afont)
    ax3.set_xlabel(r'Time (s)', **afont)
    ax3.set_title('3. Curve Fitting')
    ax3.legend(loc='upper right', fontsize=12)

    plt.locator_params(axis='both', nbins=2)
    l, b, w, h = 1, 2, 1, 2
    ax5 = plt.axes([l, b, w, h])
    ip = InsetPosition(ax3,[0.66,0.1,0.3, 0.5])
    ax5.set_axes_locator(ip)
    ax5.plot(t3_time, envelope(t3_time, *popt3max)*cosine(t3_time, 1, t3_fftperiod, popt3[3]), label='Trial 3 - Fit', lw=0.3, color='#3600c9', zorder=1)
    ax5.plot(t3_time, t3_theta, color='#ca4ae0', alpha=0.7, zorder=0, label='Raw Data Overlay')
    ax5.set_ylim(0, 30)
    ax5.set_xlim(14, 18)
    ax5.set_xticklabels([14, 16, 18])
    ax5.set_yticklabels([0, 10, 20, 30])




    fig.suptitle(r'Data Analysis Process - Trial 3', **tfont)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.show()







#plot a phase plot to show nonlinearities within the system
if phaseplot_comparison:
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.plot(model_function(t3_time, popt3max[1], popt3max[0], t3_fftperiod, popt3[3]), der_model_function(t3_time, popt3max[1], popt3max[0], t3_fftperiod, popt3[3]), label='Expected Behaviour (Linear)', color='orange', lw=2)
    ax1.plot(model_function(t3_time, popt3max[1], popt3max[0], popt3[2], popt3[3]), der_model_function(t3_time, popt3max[1], popt3max[0], popt3[2], popt3[3]), label='Observed Behaviour (Nonlinear)', color='b', lw=0.2)
    ax1.set_xlabel(r'$\theta(t)$ (deg)', **afont)
    ax1.set_ylabel(r'$\dot\theta(t)$ (deg/s)', **afont)
    ax1.set_title(r'Trial 3 - Phase Portrait', **tfont)
    ax1.legend(loc='lower left', fontsize=10)

    ax2.plot(model_function(t3_time, popt3max[1], popt3max[0], t3_fftperiod, popt3[3]), der_model_function(t3_time, popt3max[1], popt3max[0], t3_fftperiod, popt3[3]), label='Curve Fit', color='orange', lw=2)
    ax2.plot(model_function(t3_time, popt3max[1], popt3max[0], popt3[2], popt3[3]), der_model_function(t3_time, popt3max[1], popt3max[0], popt3[2], popt3[3]), label='Acquired Data', color='b', lw=0.2)
    ax2.set_xlabel(r'$\theta(t)$ (deg)', **afont)
    ax2.set_title(r'Short Timescale', **tfont)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.set_xlim(29.126, 30.334)
    ax2.set_ylim(-12.31, 7.62)

    ax3.plot(model_function(t3_time, popt3max[1], popt3max[0], t3_fftperiod, popt3[3]), der_model_function(t3_time, popt3max[1], popt3max[0], t3_fftperiod, popt3[3]), label='Curve Fit', color='orange', lw=2)
    ax3.plot(model_function(t3_time, popt3max[1], popt3max[0], popt3[2], popt3[3]), der_model_function(t3_time, popt3max[1], popt3max[0], popt3[2], popt3[3]), label='Acquired Data', color='b', lw=0.2)
    ax3.set_xlabel(r'$\theta(t)$ (deg)', **afont)
    ax3.set_title(r'Long Timescale', **tfont)
    ax3.legend(loc='lower left', fontsize=10)
    ax3.set_xlim(0.24751, 0.26144)
    ax3.set_ylim(-0.6, 0.667)

    plt.show()
    

pmap = plt.cm.coolwarm
    #these values range from 0 to 1

TRIAL = (1, 2, 3, 4, 5)
TRIAL1 = (1.25, 2.25, 3.25, 4.25, 5.25)
TRIAL2 = (1.5, 2.5, 3.5, 4.5, 5.5)
expT = (t1_period, t2_period, t3_period, t4_period, t6_period)
fftT = (t1_fftperiod, t2_fftperiod, t3_fftperiod, t4_fftperiod, t6_fftperiod)
appT = (period_approx(l1, 0), period_approx(l2,0), period_approx(l3,0), period_approx(l4,0), period_approx(l6,0))

expT_u = t_uncertainty
fftT_u = (0.02, 0.18, 0.025, 0.01, 0.01)
appT_u = length_uncertainty
appT_u1 = (length_uncertainty, length_uncertainty, length_uncertainty, length_uncertainty, length_uncertainty)





if plottingperiods:
    fig, ax = plt.subplots(1,1)
    ax.scatter(TRIAL, expT, color='blue', label='Expected Period', marker='.', lw=4)
    ax.errorbar(TRIAL, expT, yerr = expT_u, color='blue', label='Expected Period Error', fmt='none', capsize=2)

    ax.scatter(TRIAL1, fftT, color='magenta', label='FFT Period', marker='v', lw=4)
    ax.errorbar(TRIAL1, fftT, yerr = fftT_u, color='magenta', label='FFT Period Error', fmt='none', capsize=2)

    ax.scatter(TRIAL2, appT, color='cyan', label='Approximate Period', marker='x', lw=4)
    ax.errorbar(TRIAL2, appT, yerr = appT_u, color='cyan', label='Approximate Period Error', fmt='none', capsize=2)

    ax.set_xticklabels([0, 1, 2, 3, 4, 5])

    ax.legend(loc='best', fontsize=10)
    ax.set_title('Comparison of Periods', **tfont)
    ax.set_ylabel('Period Value (s)', **afont)
    ax.set_xlabel('Trial Number', **afont)
    ax.set_xlim(0.1, 5.9)
    # plt.show()

plt.close()



lengths = (l1, l2, l3, l4, l6)

zzz = np.linspace(0, 0.65, 100)

fig, ax = plt.subplots(1,1)
# ax.scatter(lengths, expT, color='blue', label='Expected Period', marker='.', lw=4)
ax.errorbar(lengths, expT, yerr = expT_u, color='black', label='Expected Period Error', fmt='none', capsize=2, zorder=-1)

ax.scatter(lengths, fftT, color='magenta', marker='v', lw=4, zorder=0)
ax.errorbar(lengths, fftT, yerr = fftT_u, color='magenta', label='FFT Period', fmt='v', marker='v', capsize=2, zorder=0)

ax.scatter(lengths, appT, color='cyan', marker='x', lw=4, zorder=1)
ax.errorbar(lengths, appT, yerr = appT_u, color='cyan', label='Approximate Period', fmt='x', marker='x', capsize=2, zorder=1)

ax.plot(zzz, 2*np.pi*np.sqrt(zzz/g), color='k', ls='--', lw=2, label = r'Expected $T = 2\pi\sqrt{\l/g}$', zorder=-1)

ax.legend(loc='upper left', fontsize=18)

ax.text(0.58, 1.45, 'Trial 1' ,fontsize=18, color='k')
ax.text(0.25,0.93, 'Trial 2' ,fontsize=18, color='k')
ax.text(0.54, 1.4, 'Trial 3' ,fontsize=18, color='k')
ax.text(0.35, 1.12, 'Trial 4' ,fontsize=18, color='k')
ax.text(0.05, 0.36, 'Trial 5' ,fontsize=18, color='k')

ax.set_title('Period Comparison', fontsize=24)
ax.set_xlabel('String Length (m)', fontsize=22)
ax.set_ylabel('Response Period (s)', fontsize=22)

diffFFT = np.zeros(len(fftT))
diffAPP = np.zeros(len(appT))
for i in range(len(fftT)):
    diffFFT[i] = fftT[i] - expT[i]
    diffAPP[i] = appT[i] - expT[i]

l, b, w, h = 1, 2, 1, 2
ax2 = plt.axes([l, b, w, h])
ip = InsetPosition(ax,[0.55,0.1,0.4,0.4])
ax2.set_axes_locator(ip)
ax2.errorbar(lengths, np.zeros(len(lengths)), yerr = expT_u, color='black', fmt='none', capsize=2, zorder=-1)
ax2.scatter(lengths, diffFFT, marker='v', color='magenta', zorder=1, lw=4)
ax2.errorbar(lengths, diffFFT, fftT_u, capsize=2, color='magenta',fmt='none', zorder=1)
ax2.scatter(lengths, diffAPP, marker='x', color='cyan', zorder=2, lw=2)
ax2.errorbar(lengths, diffAPP, appT_u, capsize=2, color='cyan', fmt='none', zorder=2)
ax2.plot(zzz, np.zeros(len(zzz)), ls='--', label='Baseline', color='k', zorder=-1)
ax2.set_title('Residual', **tfont)
ax2.set_ylabel('Period Difference (s)', **afont)
ax2.set_xlabel('String Length (m)', **afont)
ax2.set_yticks([-0.2, -0.1, 0, 0.1],)
ax2.legend(loc='lower right', fontsize=15)

ax2.text(0.58, 0.04, 'T.1' ,fontsize=12, color='k')
ax2.text(0.25,0.05, 'T.2' ,fontsize=12, color='k')
ax2.text(0.49, 0.015, 'T.3' ,fontsize=12, color='k')
ax2.text(0.32, -0.038, 'T.4' ,fontsize=12, color='k')
ax2.text(0.037, 0.04, 'T.5' ,fontsize=12, color='k')


# plt.show()

plt.close()


if fftplot:
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(t1_samplesize/(t1_xfft*samplerate), t1_freq, color='purple', label='Trial 1: Real-FFT', lw=3)
    ax2.plot(t3_samplesize/(t3_xfft*samplerate), t3_freq, color='purple', label='Trial 3: Real-FFT', lw=3)

    ax1.set_xlim(1.45, 1.6)
    ax2.set_xlim(1.42,1.5)
    ax1.legend(loc='lower right', fontsize=18)
    ax2.legend(loc='lower right', fontsize=18)
    ax1.set_title('Nonlinearities of FFT Data', fontsize=22)

    ax1.set_ylabel('Amplitude', fontsize=18)
    ax2.set_ylabel('Amplitude', fontsize=18)
    ax2.set_xlabel('Period (s)', fontsize=18)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(4,5))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(4,5))
# plt.show()


