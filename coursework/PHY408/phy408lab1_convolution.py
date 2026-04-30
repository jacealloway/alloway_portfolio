#%matplotlib inline
import numpy as np 
import matplotlib.pyplot as plt 
import time as tm

    #latex encoding
plt.rcParams['text.usetex'] = True
    #font changing
tfont = {'fontname':'DejaVu Sans', 'size':'20'}
afont = {'fontname':'Helvetica', 'size':'18'}


#toggles for problems 
Q1 = False
Q2 = False
Q3 = True





    #question 1 
if Q1:
    def myconv(A, B, DELTA):           #a, b are arrays
        N=len(A)                #   i index is in 0 and N-1
        M=len(B)                #   j index is  in 0 and M-1
        K=N+M-1                 #   condition is k=i+j which implies k is in 0 to N+M-2
                                #   this then implies that  the convolved array length is N+M-1

        conv = np.zeros(K)      #set length of array to avoid appending it each time make sure it stays within the length it needs to
        for k in np.arange(K):
            termval=0           #reset the value to prevent it from adding previous terms
            for i in np.arange(N):
                if 0 <= k-i <= M-1:         #condition on the b[j] index
                    termval += A[i] * B[k-i]    
            conv[k] = termval*DELTA       #set the convolution array  value

        return conv     #complete the function
        

    DELTA=1 #fixed sampling interval


    #lets generate two random arrays for elements between [0,1]
    #of lengths N=50, M=100
    N=50
    M=100
        #two arrays with uniform distribution between 0 and 1
    f = np.random.uniform(0, 1, N)
    w = np.random.uniform(0, 1, M)


        #show that my function agrees with np.convolve
    difference = np.round(myconv(f, w, DELTA) - np.convolve(f,w)*DELTA, 10)

    plt.plot(difference, 'o', color='blue', markersize=1, label='Difference Value')
    plt.title('Differences in Convolution between myConv and np.convolve', **tfont)
    plt.xlabel('Array Index Value', **afont)
    plt.ylabel('Difference', **afont)
    plt.legend(loc='best', fontsize=16) 
    plt.show()      



    #proceed by testing the timing values for each array
    timing=True            #create toggle (it takes up to 3 minutes :(  )
    
    if timing:
        N = (10, 100, 1000, 10000)      #array of conv lengths
        for i in np.arange(len(N)):         #loop to produce randomly generated arrays of length N
            f = np.random.uniform(0, 1, N[i])
            w = np.random.uniform(0, 1, N[i])

            t1=tm.time()
            myconv(f,w, DELTA)     #convolve the arrays and time them
            t2=tm.time()
            np.convolve(f,w)*DELTA
            t3=tm.time()

            print('myconv Time', t2-t1,'Numpy Time', t3-t2, 'for N=%i samples'%N[i])     #print the times

        
            plt.plot(N[i], t2-t1, color='blue', marker='o', markersize=6)
            plt.plot(N[i], t3-t2, color='k', marker='+', markersize=6)
    


        plt.plot(-2000,0, label='myConv Time', color='blue', marker='o', markersize=6)      #these are just here for the label in legend, since I dont want to keep looping and have 4 labels for the same points
        plt.plot(-2000, 0, label='np.convolve Time', color='k', marker='+', markersize=6)
        plt.title('Comparison of Times of Convolution Functions', **tfont)
        plt.xlabel('Lengths of Arrays to be Convolved (Entries)', **afont)
        plt.ylabel('Time (s)', **afont)
        plt.legend(loc='best', fontsize=16)
        plt.xscale('log', base=10)
        plt.xlim([1,10**5])
        plt.show()









    #question 2
if Q2:
    #discretize the R(t)  function such that H(t) = [0.5, 1, 1, ...] and d(t) = [1/dt, 0, 0, ...]
        #begin by defining the sampling interval 
    timelength = 0.020        #define the time length of the sample
    dt = 0.000015              #sampling interval
    time = np.arange(0, timelength, dt) #time array

    H = np.ones(len(time))
    H[0] = 0.5
    D = np.zeros(len(time))
    D[0] = 1/dt

        #impulse response function, discretized
    def impulse(R, L):
        R_t = D - (R/L)*np.exp(-R*time/L)*H 
        return R_t

    def RLresponse(R, L, V_in, dt):     #define time evolution; timelength should be previously defined by specifying the lengths of H, D
        conv = np.convolve(np.exp(-R*time/L), V_in)*dt
        V_out = V_in - (R/L)*conv [0:len(V_in)] 
        return V_out
    
    R=950
    L=4

        #plot for H_n
    fig, ax=plt.subplots(1,1)
    ax.plot(RLresponse(R, L, H, dt)[0:200], 'o', color='red', markersize=1.7, label='Reponse Function')
    ax.plot((H*np.exp(-R*time/L))[0:200], color='k', lw=1.5, label=r'$S(t)$ Theoretical Decay')
    ax.legend(loc='best', fontsize=16)
    ax.set_xlabel('Time (ms)', **afont)
    ax.set_xticklabels([0, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
    ax.set_ylabel('Voltage Response (V)', **afont)
    ax.set_title('Comparison of Voltage Responses', **tfont)
    plt.show()

        #plot of D_n
    fig, ax=plt.subplots(1,1)
    ax.plot(RLresponse(R, L, D, dt)[0:200], 'o', color='red', markersize=1.7, label='Reponse Function')
    ax.plot((impulse(R, L))[0:200], color='k', lw=1.5, label=r'$R(t)$ Theoretical Decay')
    ax.legend(loc='best', fontsize=16)
    ax.set_xlabel('Time (ms)', **afont)
    ax.set_xticklabels([0, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
    ax.set_ylabel('Voltage Response (V)', **afont)
    ax.set_title('Comparison of Voltage Responses', **tfont)
    ax.set_ylim(-260, 100)
    plt.show()











    #question 3
if Q3:
    sem = np.loadtxt('RAYN.II.LHZ.sem', unpack=True)    #load data
    semtime = sem[0] #in seconds
    semdis = sem[1] #in meters

        #define the Gaussian function
    def gauss(t, t_H):
        G = (1/(np.sqrt(np.pi)*t_H))*np.exp(-(t/t_H)**2)
        return G

    t_H=(10, 20)                    #variable lengths as an array
    dt = semtime[10] - semtime[9]    #timestep interval separation the same as the seismograph time  (I just took random array points 10-9)
    def time(t_H, dt):
        return np.arange(-3*t_H, 3*t_H, dt)     #define a time function



        #part (a); two plots

    plt.plot(semtime, semdis, label='Seismograph Time Series')
    plt.xlim(0, 800)
    plt.xlabel('Time (s)', **afont)
    plt.ylabel('Displacement (m)', **afont)
    plt.title('Time Series of Raw Seismograph Data from RAYN', **tfont)
    plt.legend(loc='best', fontsize=16)
    plt.show()



    plt.plot(gauss(time(10, dt), 10), label=r'$t_H=10$s')
    plt.plot(gauss(time(20, dt), 20), label=r'$t_H=20$s')
    plt.legend(loc = 'best', fontsize=16)
    plt.xlabel('Time (s)', **afont)
    plt.ylabel('Amplitude', **afont)
    plt.title('Gaussian Distributions of Various Widths', **tfont)
    plt.show()

            #determine the index value of the t=800 sample time since timesteps are different
    # for i in range(len(semtime)):
    #     if 799 < semtime[i] < 801:
    #         print(i)
    #     #I'll take the  i=4962 to be the t=800s sample time

        #plot the array
    for k in t_H:
        convolution = np.convolve(gauss(time(k, dt), k), semdis[0:4962], mode='same')*dt
        plt.plot(convolution,  label='Seismic Convolution for t_H = %is'%k)
        
    plt.legend(loc='best', fontsize=16)
    plt.xlabel('Time (s)', **afont)
    plt.ylabel('Displacement (m)', **afont)
    plt.title('Convolution of Seismograph Data with Various Gaussian Distributions', **tfont)
    plt.show()


