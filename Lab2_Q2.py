import numpy as np 
from math import factorial
import matplotlib.pyplot as plt 

plt.rcParams.update({"text.usetex" : True}) #use latex in plots
C = 10**(-16) #define the round error constant


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


"""
---------------------------------- Q2 - (a)
"""
m = 1 #kg         #define variables
k = 12 #n/m
x0 = 0.01 #m
c = 2.99e8 #speed of light m/s

def g(                  #define g with x0 and x inputs
        x0: int|float,
        x: int|float
    ) -> list[int|float]:
    """
    **Returns:** Relativistic velocity value as a function of position and fractional error. 
    """
    numerator = (k) * (x0**2 - x**2) * ( 2*m*c**2 + 0.5*k*(x0**2 - x**2) )      #calculate numerator
    denominator = 2 * ((m*c**2) + 0.5*k*(x0**2 - x**2))**2                      #calculate denominator
    return c*np.sqrt(numerator/denominator)                                     #return value



roots8, weights8 = gaussxwab(8, 0, x0)      #calculate weights for N=8
roots16, weights16 = gaussxwab(16, 0, x0)   #calculate weights for N=16
roots32, weights32 = gaussxwab(32, 0, x0)   #calculaute weights for N=32 for error estimate of N=16 


value8 = 4*np.sum(weights8*(1/g(x0, roots8)))   #calculate the value of the integrals for each of the corresonding N
value16 = 4*np.sum(weights16*(1/g(x0, roots16)))
value32 = 4*np.sum(weights32*(1/g(x0, roots32)))

error8 = value16 - value8       #estimate error
error16 = value32 - value16     #estimate error 


print('For N=8, T = {} with a fractional error estimate of {} %'.format(value8, error8/value8))     #print answers
print('For N=16, T = {} with a fractional error estimate of {} %'.format(value16, error16/value16))






"""
---------------------------------- Q2 - (b)
"""

fig, (ax1, ax2) = plt.subplots(1, 2)                    #create figure
ax1.plot(4/g(x0, roots8), marker='o', label=r'$N=8$, $4/g_k$')      #plot integrand values on same graph
ax1.plot(4/g(x0, roots16), marker='x', label=r'$N=16$, $4/g_k$')
ax1.legend(loc='best', fontsize=14)             #labels 
ax1.set_xlabel('Sample Number [ $k$ ]', fontsize=14)
ax1.set_ylabel('Integrand Value ($s$)', fontsize=14)
ax1.set_title('Plot of Integrand Values for $4/g_k$ for G.Q. $N=8, 16$', fontsize=16)


ax2.plot(4*weights8/g(x0, roots8), marker='o', label = r'$N=8$, $4w_k/g_k$')        #plot weighted values on same graph
ax2.plot(4*weights16/g(x0, roots16), marker='x', label = r'$N=16$, $4w_k/g_k$')
ax2.legend(loc='best', fontsize=14)             #labels 
ax2.set_xlabel('Sample Number [ $k$ ]', fontsize=14)
ax2.set_ylabel('Weighted Value ($s$)', fontsize=14)
ax2.set_title('Plot of Weighted Values for $4 w_k/g_k$ for G.Q. $N=8, 16$', fontsize=16)

plt.show()


            #reasoning 
"""
As int approaches x0 limit:
    ** integrand values 4/g_k -> 0; these values get 'less important' closer to upper bound of integral 
    ** weighted values 4w_k/g_k -> 0; these values are the values which are summed, also get less important in upper bound
                                    hence you're accumulating error and rounding error by continuously adding smaller values 

"""





"""
---------------------------------- Q2 - (c)
"""



xc = c*np.sqrt(m/k)     #define x_c

x_range = np.linspace(1.1, 10*xc-0.1, 40)       #set range of x values to integrate over, take 40 points-ish 
N = 16
T = np.zeros(len(x_range))          #blank period array 
for i in range(len(x_range)):
    roots, weights = gaussxwab(16, 0, x_range[i])       #calculate weights for i-th entry 
    T[i] = 4*np.sum(weights*(1/g(x_range[i], roots)))   #sum to determine period for i-th entry 




fig = plt.figure()          #plot 
plt.plot(x_range, T, label='$(x_0, T)$ Computed', marker='o')           #plot points calculated 
plt.plot(x_range, 4*x_range / c, ls = '--', label=r'$4x_0 / c$ Relativistic Limit')     #plot relativistic limit relation
plt.plot(x_range, 2*np.pi*np.sqrt(m/k)*np.ones(len(x_range)), ls='--', label=r'$2\pi\sqrt{\frac{m}{k}}$ Classical Limit')   #plot classical limit relation 
    #labels 
plt.xlabel('$x_0$ Values [m]', fontsize=14)
plt.ylabel('Period [s]', fontsize=14)
plt.title('Relativistic/Classical Limits of Period of Spring Mass', fontsize=16)
plt.legend(loc='best', fontsize=14)
plt.show()











