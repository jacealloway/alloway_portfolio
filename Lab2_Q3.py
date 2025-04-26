import numpy as np 
import matplotlib.pyplot as plt 

    #rounding error constant
C = 10**(-16)

    #define functions f and its derivatives
def f(                  
        x: int|float
    ) -> int|float:
    """
    **Returns:** Gaussian curve exp(-x^2).
    """
    return np.exp(-x**2)

def f_prime(
        x: int|float
    ) -> int|float:
    """
    **Returns:** Derivative of Gaussian curve f(x) previously defined as exp(-x^2).
    """
    return (-2*x)*np.exp(-x**2)

def f_pprime(
        x: int|float
    ) -> int|float:
    """
    **Returns:** Second derivative of Gaussian curve f(x) previously defined as exp(-x^2).
    """
    return (4*x**2 - 2)*np.exp(-x**2)

def f_ppprime(
        x: int|float
    ) -> int|float:
    """
    **Returns:** Third derivative of Gaussian curve f(x) previously defined as exp(-x^2).
    """
    return -(8*x**3 - 12*x)*np.exp(-x**2)

    #define relative error function 
def relativeError(
        x: int|float,
        y: int|float
    ) -> int|float:
    """
    Calculate the absolute percentage (relative) error of x compared to y.
    """
    return np.abs((x-y)/y)



"""
---------------------------------- Q3 - (a)
"""
h_range = np.logspace(-16, 0, base=10, num=17) #specify range of h values of base 10

    #define a single-variable central difference function
def central_difference(
        func: callable,
        h: int|float,
        x: int|float
        ) -> int|float:
    """
    Computes the single-variable numerical derivative of a function by using the method of central differences.

    **Returns:** Approximated central derivative of func at point x using width h.
    """
    return (func(x+h) - func(x-h)) / (2*h)

    #return values
print('Range of h values:', h_range)



print() #space
"""
---------------------------------- Q3 - (b)
"""


cen_deriv_f=[]  #blank array to append derivative values for each h 
for h in h_range:
    cen_deriv_f.append(central_difference(f, h, 0.5))   #iterate through each h


f_actual = f_prime(0.5)     #determine the 'actual' numerical value of the first derivative of f
f_centralErr = []       #blank arrays to iterate over
f_cenCompareActual = []
for k in range(len(h_range)):       
    f_centralErr.append((h_range[k]**2 / 24)*f_ppprime(0.5))     #calculate the approximation error from the method 
    f_cenCompareActual.append(relativeError(cen_deriv_f[k], f_actual))  #calculate the relative error compared with actual, append both to array


# FOR DERIVATIVE ERROR
smallest_error = np.min(f_centralErr)       #find the value of the smallest error
hMin_index = (f_centralErr.index(smallest_error))   #find its index

# FOR RELATIVE ERROR
smallest_error2 = np.min(f_cenCompareActual)    #find the value of the smallest error
hMin_index2 = (f_cenCompareActual.index(smallest_error2))   #find its index

print('Respective derivative values:', cen_deriv_f)
# print('Most accurate h value central approximation (approximation) is {}'.format(h_range[hMin_index]))  #print the respective h values for each index
print('Most accurate h value central approximation (relative) is {}'.format(h_range[hMin_index2]))

exp_min_h = np.cbrt(48*C*np.abs(f(0.5)/f_ppprime(0.5)))
print('The expected h to minimize error (central) is', exp_min_h)


print() #space
"""
---------------------------------- Q3 - (c)
"""
    #define a forward difference function 
def fwd_difference(
        func: callable,
        h: int|float,
        x: int|float
    ) -> int|float:
    """
    Computes the single-variable numerical derivative of a function by using the method of forward differences.

    **Returns:** Approximated forward derivative of func at point x using width h.
    """
    return (func(x+h) - func(x)) / (h)



fwd_deriv_f=[]      
for h in h_range:       #iterate through h and calculate each of the fwd difference derivative values 
    fwd_deriv_f.append(fwd_difference(f, h, 0.5))     

f_fwdErr = []       #blank arrays for the error propagation calculations
f_fwdCompareActual = []
for k in range(len(h_range)):
    f_fwdErr.append((h_range[k] / 2)*f_pprime(0.5))     #calculate error due to approximating derivative (2nd order)
    f_fwdCompareActual.append(relativeError(fwd_deriv_f[k], f_actual))  #calculate relative error with actual (as before)


# FOR DERIVATIVE ERROR
smallest_error3 = np.min(f_fwdErr)                      #find value of smallest error
hMin_index3 = (f_fwdErr.index(smallest_error3))         #find it's index

# FOR RELATIVE ERROR ----repeat for relative error 
smallest_error4 = np.min(f_fwdCompareActual)
hMin_index4 = (f_fwdCompareActual.index(smallest_error4))

print('Respective derivative values:', fwd_deriv_f)
# print('Most accurate h value forward approximation (approximation) is {}'.format(h_range[hMin_index3]))     #print h for smallest index
print('Most accurate h value forward approximation (relative) is {}'.format(h_range[hMin_index4]))

exp_min_h = np.sqrt(4*C*np.abs(f(0.5)/f_pprime(0.5)))
print('The expected h to minimize error (fwd) is', exp_min_h)


"""
---------------------------------- Q3 - (d)
"""


fig = plt.figure()  #new figure 
plt.plot(h_range, f_cenCompareActual, label="Central Relative Error", color='red')      #plot the comparison between the central and forward difference relative errors previously calculated
plt.plot(h_range, f_fwdCompareActual, label="Forward Relative Error", color='blue', ls='--')
plt.xscale('log')   #log scale for both 
plt.yscale('log')
    #labels 
plt.legend(loc='best', fontsize=14)
plt.xlabel(r'$h$ Values (log scale)', fontsize=14)
plt.ylabel(r'Percentage Error Value (log scale)', fontsize=14)
plt.title('Comparison of Central and Forward Approximation Relative Error', fontsize=16)
plt.show()



"""
---------------------------------- Q3 - (e)
"""

"""
At high h, approximation error dominates. 
At low h, rounding error dominates. 

"""
    #show relationship between errors (approximation vs rounding and which dominates at which h )
fig, (ax1, ax2)= plt.subplots(1,2)  
ax1.plot(h_range, np.abs(f_centralErr), label='Central Approximation Error')        #plot abs of errors (can't have negative values for log scale)
ax1.plot(h_range, np.abs(C*np.array(cen_deriv_f)), label='Central Rounding Error')
    #labels 
ax1.legend(loc='best', fontsize=14)
ax1.set_title('Central Differences Error Comparison', fontsize=16)
ax1.set_xlabel(r'$h$ Range (log scale)', fontsize=14)
ax1.set_xscale('log')
ax1.set_ylabel('Error Value (log scale)', fontsize=14)
ax1.set_yscale('log')

ax2.plot(h_range, np.abs(f_fwdErr), label='Forward Approximation Error')            #repeat as before
ax2.plot(h_range, np.abs(C*np.array(fwd_deriv_f)), label='Forward Rounding Error')
    #labels 
ax2.legend(loc='best', fontsize=14)
ax2.set_title('Forward Differences Error Comparison (abs. val)', fontsize=16)
ax2.set_xlabel(r'$h$ Range (log scale)', fontsize=14)
ax2.set_xscale('log')
ax2.set_ylabel('Error Value (log scale)', fontsize=14)
ax2.set_yscale('log')

plt.show()




print() #space

"""
---------------------------------- Q3 - (f)
"""


def g(              #define the g function
        x: int|float
    ) -> int|float:
    """
    **Returns:** g(x) = np.exp(2x).
    """
    return np.exp(2*x)




  
def delta(                  # calculate the m-th derivative of f at x using recursion.
        func: callable,
        x: int|float, 
        m: int,
        h: int|float
        ) -> int|float:
    """
    Calculate the *m*-th derivative of a function *func* at a point *x* with width *h*.

    **Returns:** [integer or float] value of derivative. 
    """
    if m > 1:
        return (delta(func, x + h, m - 1, h) - delta(func, x - h, m-1, h))/(2*h)     #compute value of m-th derivatve according to central differences
    elif m==1:
        return (func(x + h) - func(x - h))/(2*h)      #if m==1, just apply central differences once
    elif m==0:
        return func(x)      #if m==0, do not do anything (return original value of func at x)



iterations = 5+1      #set the number of iterations (num of iterations is iterations - 1)
counter = 0         #set counter
g_derivs = []      #set the blank g derivative array 
while counter < iterations:     #repeat for number of iterations until counter reaches maximum
    g_derivs.append(delta(g, 0, counter, 10.**(-6)))    #append derivative array with the m-th derivative using delta  
    counter = counter + 1   #up the counter

print('The numerical values of the derivatives of g at x=0 with h=10e-6 are: {}'.format(g_derivs)) #print the values 


# """
# Note we could do this symoblically with sympy, but we are to use finite differences method.
# Analytically, every time we take a derivative, a 2 just comes down. For n derivatives then, we have 2**n
# Can factor out this constant and multiply each central differences operation by it, 
# since python will not let us call a recusive function isolated along a specific axis to take a derivative with 
# (we get a 'maximum recursion depth' error for calling central_difference(g) within itself, cannot defined vars
# like x or h anymore with function nesting...)
# """
