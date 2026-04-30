import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams.update({"text.usetex": True})  #use latex



"""
--------------------------------------------------- Q2 (a)
"""


def f(x, y):        #define energy function 
    return x**2 - np.cos(4*np.pi*x) + (y-1)**2



def generate_step(i, j):        #write function to move to the next step by drawing from a normal distribution in x and y indep.
    rand1 = np.random.normal(0, 1)
    rand2 = np.random.normal(0, 1)

    return i + rand1, j + rand2

def cooltime(T0, Tf, tau):      #write function to compute total amount of time, based on temperature/tau paramters
    return tau*np.log(T0/Tf)        #total amount of time 

def compute_temp(time, tau, T0):        #re-write the temperature cooling schedule with computed time array 
    return T0*np.exp(-time/tau)



def acceptance_probability(Ei, Ej, temp):       #write function for acceptance probability 
    beta = 1/temp   #set kB = 1
    if Ei >= Ej:        #if the 2nd energy is less than the 1st, it is accepted 
        return 1
    
    elif Ei < Ej:
        return np.exp(-beta * (Ej - Ei))    #if 2nd energy greater than 1st, return new probability based on boltzmann statistics



        #set initial params
T0 = 1
Tf = 0.00001
tau = 1000
time_arr = np.linspace(0, cooltime(T0, Tf, tau), tau)  #timescale with 1000 points 

x0 = 2      #initial conditions 
y0 = 2
x = x0
y = y0

xhist = []      #history lists 
yhist = []

for k in range(len(time_arr)):      #for each time array point 
    xhist.append(x)     #add to x list 
    yhist.append(y)     #add to y list 

    E1 = f(x, y)        #find E1

    x_next, y_next = generate_step(x, y)    #generate next step 

    E2 = f(x_next, y_next)      #find E2 with next step 

    Temp = compute_temp(time_arr[k], tau, T0)       #compute temperature from cooling schedule at time t[k]

    prob = acceptance_probability(E1, E2, Temp)     #compute acceptance probability 

    RANDOM_NUMBER = np.random.random()  #draw random number

    if prob >= RANDOM_NUMBER:       #compare random number with acceptance probability 
        x = x_next          #move forward if accepted
        y = y_next




print('Q2a returned point:', (x, y))        #print point 

    #plot x and y histories vs time 
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(time_arr, xhist, marker='o', ls='')
ax1.set_title(r'History of $x$-steps', fontsize=18)
ax1.set_xlabel(r'Time Array ($s$ | num iter.)', fontsize=16)
ax1.set_ylabel(r'$x$ Position', fontsize=16)
ax2.plot(time_arr, yhist, marker='o', ls='')
ax2.set_title(r'History of $y$-steps', fontsize=18)
ax2.set_xlabel(r'Time Array ($s$ | num iter.)', fontsize=16)
ax2.set_ylabel(r'$y$ Position', fontsize=16)

plt.show()

    #plot x and y positions over time, with initial and actual minima points 
plt.plot(xhist, yhist, marker='o', label='steps and path')
plt.plot(0, 1, marker='o', label='actual minimum', color='red')
plt.plot(2, 2, marker='o', label='init point', color='k')
plt.legend(loc='best', fontsize=16)
plt.title(r'Simulated Annealing Process for $f(x, y)$', fontsize=18)
plt.xlabel(r'$x$ Position', fontsize=16)
plt.ylabel(r'$y$ Position', fontsize=16)

plt.show()









"""
--------------------------------------------------- Q2 (b)
"""

def g(x, y):        #define more complicated function 
    return np.cos(x) + np.cos(x*np.sqrt(2)) + np.cos(x*np.sqrt(3)) + (y-1)**2



        #perform a search for the best T0 and tau values ... because everything else I've tried just doesn't work for the x components. 
        # y components are find, just as in q2(a)

T0_vals = np.linspace(1, 20, 50)        #range of T0 and tau values
tauvals = np.linspace(1, 200, 100)

        #initial conditions - I picked something between 2 and 16 (nearby local and global minima, respectively)
x0 = 12
y0 = 2

   
T0 = 2
tau = 1000000


time_arr = np.linspace(0, cooltime(T0, Tf, tau), 100000)       #same computations as in part (a)

x = x0
y = y0

xhist = []
yhist = []          #history 

for k in range(len(time_arr)):      #loop over time vals for given T0, tau 
    xhist.append(x)
    yhist.append(y)

    E1 = g(x, y)            #energy calculations, etc

    x_next, y_next = generate_step(x, y)

    if (x_next >= 50) or (x_next <= 0) or (np.abs(y_next) >= 20):
            #reject the value return to top of loop
        continue

    E2 = g(x_next, y_next)

    Temp = compute_temp(time_arr[k], tau, T0)

    prob = acceptance_probability(E1, E2, Temp)

    RANDOM_NUMBER = np.random.random()

    if prob >= RANDOM_NUMBER:       #compare 
        x = x_next
        y = y_next



print('Q2b init = {}, returned = {}'.format((x0, y0), (x, y)))  #printing results 

    #plots. literally the same as in part (a)
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(time_arr, xhist, marker='o', ls='')
ax1.set_title(r'History of $x$-steps', fontsize=18)
ax1.set_xlabel(r'Time Array ($s$ | num iter.)', fontsize=16)
ax1.set_ylabel(r'$x$ Position', fontsize=16)
ax2.plot(time_arr, yhist, marker='o', ls='')
ax2.set_title(r'History of $y$-steps', fontsize=18)
ax2.set_xlabel(r'Time Array ($s$ | num iter.)', fontsize=16)
ax2.set_ylabel(r'$y$ Position', fontsize=16)

plt.show()


plt.plot(xhist, yhist, marker='o', label='steps and path')
plt.plot((2, 42), (1, 1), marker='o', ls='', label='other minimum', color='orange')
plt.plot(16, 1, marker='o', label='actual minimum', color='red')
plt.plot(x0, y0, marker='o', label='init point', color='k')
plt.legend(loc='best', fontsize=16)
plt.title(r'Simulated Annealing Process for $g(x, y)$', fontsize=18)
plt.xlabel(r'$x$ Position', fontsize=16)
plt.ylabel(r'$y$ Position', fontsize=16)
plt.xlim(0, 50)
plt.ylim(-10, 10)

plt.show()

"""
some observations from my 5 hours of trying to find T0, tau, Tf values:
    - very sensitive to the initial conditions (x0, y0)
    - for (7, 4) initial, there was no T0 or tau value which consistenly gave (~15.9, ~1)
    - for (12, 2) initial, happened rarely 
    - felt like I was cheating if I set initial condition too close, or used smaller number of time pts 
    - Tf value didn't matter too much, as long as it was quite close to 0 it worked 
    - the number of time points in np.linspace matters. more points, shorter stepsize. less points, larger stepsize. 
        not sure if there is a way to control that using the temp parameters in cooling schedule; I don't think there is,
        since any time you try to generate a time array, a default of 50 points is set. so I just chose 1000 for (a) and 200 for (b)

"""
