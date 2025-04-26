import numpy as np 
from math import factorial
import matplotlib.pyplot as plt 

plt.rcParams.update({"text.usetex" : True}) #use latex in plots


"""
---------------------------------- Q1 - (a)
"""
def Hermite(
        n: int,
        x: list[int|float]
        ) -> int|float:
    """
    *n*: integer

    *x*: integer or float in list

    
    **Returns:** The n-th order physicist's Hermite polynomial at x.
    """
    if type(n) == float:            #make sure n is an integer
        raise ValueError('n must be of integer type!')
    if n < 0:                       #make sure n is >= 0
        raise ValueError('Input integer n must be non-negative.')
    
    M = len(x)                          #length of x input array 
    output = np.zeros(M)                #set length of output array - input array axis x must be list   
    H = np.zeros((n+1, M))              #set initial parameters on H
    for m in range(M):
        H[0, m] = 1            
        if n > 0:
            H[1, m] = 2*x[m]                  #only overwrite H[1] if n is not zero

        if n == 0 or 1:                 #first two initial conditions
            output[m] = H[n, m]               #write output as 1 or 2*x, depending on n
            pass                        #close the if statement and return output


        countlist = np.arange(2, n+1)   #list of k values to iterate over to determine the next H term  
        for k in countlist:             #evalute the recursion relation                                 
            H[k, m] = 2*x[m]*H[k-1, m] - 2*(k-1)*H[k-2, m]
        output[m] = H[n, m]                   #overwrite output as the last n-th term calculated
        
    return output                       #return array 
    




"""
---------------------------------- Q1 - (b)
"""

def QHO_psi(                    
        n: int,
        x: int|float
        ) -> int|float:
    """
    **Returns:** The n-th level quantum harmonic oscilator wavefunction value at x.
    """
    if type(n) == float:            #make sure n is an integer
        raise ValueError('n must be of integer type!')
    if n < 0:                       #make sure n is >= 0
        raise ValueError('Input integer n must be non-negative.')
    
    f1 = ( 1 / ( np.sqrt((2**n)*factorial(n)*np.sqrt(np.pi)) ) ) * np.exp(-(x**2)/2)    #write the function and return it 
    return f1 * Hermite(n, x)
    



nrange = np.arange(0, 3+1)  #range of n values 
axis = np.linspace(-4, 4, 100)  #x axis
for m in nrange:
    plt.plot(axis, QHO_psi(m, axis), label=r'$\psi_%i (x)$'%m)

plt.title('Quantum Harmonic Oscillator Wavefunctions for $n=0, ..., 3$', fontsize=16)
plt.xlabel('$x$-position', fontsize=14)
plt.ylabel('Wavefunction Value', fontsize=14)
plt.legend(loc='best', fontsize=14)
plt.show()    




"""
---------------------------------- Q1 - (c)
"""
def gaussxw(N):                 #import gaussxw

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)
    return x,w

def gaussxwab(N,a,b):           #import gaussxwab
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w



def u(                          #define the change of variables to integrate over an infinite region
        z: list[int|float]
    ) -> list[int|float]:
    """
    **Returns:** Change of variables for infinite range (0, infty) over an integrand to (0, 1). u = z/(1-z).
    """
    return z/(1-z)


nvals = np.arange(0, 10+1)      #define range of n values 0 to 10
N = 100                         #set gaussian quadrature num_points
roots, weights = gaussxwab(N, 0, 1) #compute the roots and weights for N between 0 and 1 (after var change)



def PE_integrand(                  #define the function we are integrating for potential energy
        n: int, 
        z: list[int|float]
        ) -> list[int|float]:
    """
    Compute the integrand for the QHO potential energy.
    """
    if type(n) == float:            #make sure n is an integer
        raise ValueError('n must be of integer type!')
    if n < 0:                       #make sure n is >= 0
        raise ValueError('Input integer n must be non-negative.')
    
    x = u(z) #compute variable change

    return (x**2)*(np.abs(QHO_psi(n, x))**2)*(1/((1-z)**2)) #return the function:   d(u(z)) * x^2 * |psi_n(x)|^2 



        #compute the integral 
values = np.zeros(len(nvals))
for n in nvals:
    values[n] = np.sum(weights*PE_integrand(n, roots)) #sum over the roots and weights respectively with dot product, multiply by 2 for both sides of integral

print('The average potential energy values of QHO are {} for n = 0, ..., 10.'.format(values)) 
