import numpy as np 
import matplotlib.pyplot as plt 

plt.rcParams.update({"text.usetex": True})  #use latex





"""
--------------------------------------------------- Q3 (a)
"""

def f(x):           #define the function 
    return 1/(np.sqrt(x)*(1+np.exp(x)))


iterations = 1000       #set initial number of iterations and sampling points 
N = 10000

a = 0       #define integral bounds 
b = 1

I_arrayA = []       #empty array to append 

for j in range(iterations):     #loop thru iteraitons 
    counter = 0
    I = 0       #reset counter and integral value at the start of each iteration
    while counter < N+1:        #loop through sampling points 

        xi = np.random.uniform(0, 1)        #draw uniform random number 

        I += ((b-a)/N)*f(xi)        #compute integral with mean value method 

        counter += 1 

    I_arrayA.append(I)      #append I to integral history list 

print('Q3a int value:', np.mean(I_arrayA))      #print and compute mean 








"""
--------------------------------------------------- Q3 (b)
"""

    #same iterations as in part (a)
# iterations = 1000
# N = 10000

axis = np.linspace(0, 1, N+1)[1:] #x axis, not including zero or else we may compute a divide by zero error in weighted function


def weighted(x):        #define the weighted function 
    return x**(-0.5)

def nonuniform_mapping(number): 
    #a uniformly drawn number will follow the x**2 nonuniform cumulative distribution function
    return number**2

I_arrayB = []       #array to append 

for j in range(iterations):     #loop thru iterations 
    counter = 0         #reset counter and int value at beginning of each iteration 
    I = 0
    while counter < N+1:

        A = np.random.uniform(axis[0], axis[N-1])       #generate random number from uniform distribution, using ends of axis array (if we draw 0, we will get an undefined value )

        xi = A**2           #map the random number to a new value by squaring it, now nonuniform 

        I += (1/N) * f(xi)/weighted(xi) * 2     #compute using weighted importance sampling 

        counter += 1    #up counter 

    I_arrayB.append(I)      #append array value for iteration num             


print('Q3b int value:', np.mean(I_arrayB))      #Print and compute mean 











"""
--------------------------------------------------- Q3 (c)
"""

        #create histogram 
        #plot 
plt.hist(I_arrayB, bins = 100, range = (0.7, 1), color='orange', label = 'Importance Sampling')     
plt.hist(I_arrayA, bins = 100, range = (0.7, 1), alpha = 0.6, label='Mean Value Method')
    #labels
plt.legend(loc='best', fontsize=16)
plt.xlabel(r'$x$ Values', fontsize=16)
plt.ylabel(r'Num. of Occurances', fontsize=16)
plt.title(r'Histogram of Integral Computations Using Two Methods', fontsize=18)
plt.show()










"""
--------------------------------------------------- Q3 (d)
"""

    #same iterations as in part (a)
# iterations = 1000
# N = 10000

upper = 10      #set new bounds for function h 
lower = 0

axis = np.linspace(lower, upper, N)     #define a new axis over [0, 10] using N points 



def h(x):
    return np.exp(-2*np.abs(x-5))       #define integrand 

def h_weight(x):        #define weighted function, will take as probability distribution 
    return (1/np.sqrt(2*np.pi) * np.exp(-0.5*(x-5)**2))


        #empty array to append 
I_arrayD = []

for j in range(iterations):     #loop through iterations 
    counter = 0
    I = 0
    while counter < N+1:

        A = np.random.normal(5, 1)       #draw random number from normal distribution, since we don't need any mapping from a uniform to CDF interval 

        xi = A      #take this as the weighted value, no need to find inverse to map it 

        I += (1/N) * h(xi)/h_weight(xi) * 1     #compute weighted function, taking the integral of weighted function to be 1 (normalized )

        counter += 1 

    I_arrayD.append(I)      #append to array of integrals 


print('Q3d int value:', np.mean(I_arrayD))      #print and compute mean 


        #histogram plot 
plt.hist(I_arrayD, bins = 100, range = (0.9, 1.1))
    #labels
plt.xlabel(r'$x$ Values', fontsize=16)
plt.ylabel(r'Num. of Occurances', fontsize=16)
plt.title(r'Histogram of Integral Computations from Gaussian', fontsize=18)
plt.show()