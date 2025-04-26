import numpy as np
from time import time


C = 10**(-16)
    #define integration functions to be quickly imported

"""
    PSEUDOCODE

    For trapezoidal:
INITIALIZE trapezoidal functional, taking in func, bounds, num_slices, and the first derivative of func
IF len(bounds) != 2, RAISE ValueError, since we should only work with an upper and lower bound
DEFINE constants: lower and upper bounds, h (width)
COMPUTE slice width h based on num_slices and bounds
INITIALIZE array of 0's of length num_slices - 1, the number of area elements
FOR each entry in empty array, COMPUTE the trapezoidal value of the k-th slice
COMPUTE sum of aray
COMPUTE approximation error by CALLING derivative of func
RETURN integral value, error value

    For Simpsons:
INITIALIZE Simpson's functional, taking in func, bounds, num_slices, and the third derivative of func
IF len(bounds) =! 2, RAISE ValueError, since we should only work with an upper and lower bound
IF num_slices %2 != 0, RAISE ValueError, since the number of slices must be even to execute 
DEFINE constants: lower and upper bounds, h (width)
COMPUTE slice width h based on num_slices and bounds
INITIALIZE array of 0's of length int(num_slices/2), the number of area elements (we iterate through every other element)
FOR each entry in empty array, COMPUTE the area element based on the left, center, and rightmost elements
COMPUTE sum of array 
COMPUTE approximation error (3rd order) by CALLING third derivative of func
RETURN integral value, error value

    For integral computation:
INITIALIZE rat_func, rat_funcprime (1st deriv), rat_funcppprime (3rd deriv)
CALL trapezoidal / simpsons functionals, CALL rat_func and derivatives
COMPUTE integrals
RETURN values and relative error
"""



def trapezoidal(
        func: callable, 
        bounds: list[int|float], 
        num_slices: int|float, 
        func_dx: callable
        ) -> list[int|float]:
    """
    Compute the single-variable integral of a function using the trapezoidal rule
    """
    if len(bounds) != 2:
        raise ValueError('Input bounds must be array of length 2!')

    lower = bounds[0]
    upper = bounds[1]
    h = (upper-lower) / num_slices #width of each slice 
    A = np.zeros(num_slices-1)
    for k in range(num_slices - 1):
        A[k] = 0.5*h*(func(lower + (k-1)*h) + func(lower + k*h)) #area of kth rectangle 

    value = np.sum(A)  #sum over the whole array to add to the value of the integral 
    app_error = (1/12) * (h**2) * (func_dx(lower) - func_dx(upper)) #compute error using input deriv
    int_error = np.sqrt(app_error**2 + (C*value)**2) #add errors
    return (value, int_error) 



def simpsons(
        func: callable, 
        bounds: list[int|float], 
        num_slices: int|float, 
        func_dx3: callable
        ) -> list[int|float]:
    """
    Compute the single-variabel integral of a function using quadratic curves
    """
    if len(bounds) != 2:
        raise ValueError('Input bounds must be array of length 2!')
    
    if num_slices%2 != 0:
        raise ValueError("Input variable 'num_slices' must be an even value!")

    lower = bounds[0]
    upper = bounds[1]
    h = (upper-lower) / num_slices #width of each slice 
    A = np.zeros(int(num_slices/2)) #blank array for output length
    for k in range(int(num_slices/2)):
        A[k] = (1/3)*h*(func(lower + 2*k*h) + 4*func(lower + h*(1+2*k) ) + func(lower + 2*h*(1+k))) #sum of lefthand, center, and righthand
    
    value =  np.sum(A)
    app_error = (1/180) * (h**4) * (func_dx3(lower) - func_dx3(upper)) #compute error using input deriv
    int_error = np.sqrt(app_error**2 + (C*value)**2) #add errors
    return (value, int_error) 



    #find error estimate for trapezoidal rule between slice differences

def trapezoidal_ErrEst(
        I1: int|float, 
        I2: int|float
        ) -> int|float:
    return (1/3)*(I2 - I1)


def simpsons_ErrEst(
        I1: int|float, 
        I2: int|float
        ) -> int|float:
    return (1/15)*(I2 - I1)


"""
---------------------- (a) - pi;    
---------------------- (b)
"""

    #define our rational functions and its derivatives for the integration purposes
def rat_func(
        x: int|float
        )-> int|float:
    return 4/(1+x**2)  #actual integral should be pi

def rat_funcPrimed(
        x: int|float
        ) -> int|float:
    return (-8*x) / ((x**2 + 1)**2)

def rat_funcPPPrimed(
        x: int|float
        ) -> int|float:
    return (-96*x*(x**2 - 1)) / ((x**2 + 1)**4)

    #get the integral values, compare with pi, and find errors
print('Trapezoidal Percentage Error:', np.abs(trapezoidal(rat_func, (0,1), 4, rat_funcPrimed)[0] - np.pi)/np.pi)
print('Simpsons Percentage Error:', np.abs(simpsons(rat_func, (0,1), 4, rat_funcPPPrimed)[0] - np.pi)/np.pi)
print('Trapezoidal Error:', trapezoidal(rat_func, (0,1), 4, rat_funcPrimed)[1])
print('Simpsons Error:', simpsons(rat_func, (0,1), 4, rat_funcPPPrimed)[1])





"""
---------------------- (c)
    PSEUDOCODE

INITIALIZE array of powers for 2^n
INITIALIZE t-range and s-range toggles to toggle if statements on or off once a condition is satisfied
FOR n in powers:
    INITIALIZE order (10^-9)
    COMPUTE N = 2**n
    COMPUTE error from trapezoidal integration using correction formula
    COMPUTE error from simpsons integration using correction formula
    COMPUTE time differences for integrations

    IF the t-range counter is FALSE and the trapezoidal error is less than the order:
        RETURN the 2**n-value, error, and time 
        SET t-range = TRUE to not return to this if statement
    
    IF the s-range counter is FALSE and the simpsons error is less than the order:
        RETURN the 2**n-value, error, and time 
        SET s-range = TRUE to not return to this if statement

    IF s-range AND t-range are both TRUE:
        BREAK the for-loop 

"""

    #generate array of n values; 2**n
powers = np.arange(2, 20)

    #create a parameter to exit an if-statement mid for-loop
trange = False
srange = False

    #calculate integral error values for powers of n in sequence
for n in powers:
    t1  = time() #timestamps; define vars
    order = 10**(-8) #set the minimum order; anything O(10e-9) will be integrated
    N = 2**n
    trap_err = trapezoidal(rat_func, (0,1), N, rat_funcPrimed)[1] #get error from trap method
    t2 = time()
    simp_err = simpsons(rat_func, (0,1), N, rat_funcPPPrimed)[1] #get error from simp method
    t3 = time()
    trap_comp_time = t2-t1  #time differences
    simp_comp_time = t3-t2


    if trange == False and trap_err < order: #if the trap error exceeds the order, return to top of for loop and skip this section furtheron
        print('Trapezoidal: num_slices', 2**n,'( n = ', n , ') with error', trap_err, 'integrated at a time of', trap_comp_time) 
        trange = True #do not return back to this loop

    if srange == False and simp_err < order: #if the simp error exceeds the order, return to top of for loop and skip this section furtheron
        print('Simpsons: num_slices', 2**n,'( n = ', n , ') with error', simp_err, 'integrated at a time of', simp_comp_time) 
        srange = True #do not return back to this loop

    if srange and trange: #break the loop once both n values are found for trap and simp
        break






"""
---------------------- (d)
"""


#compute error estimate for N1 = 16; N2 = 32 steps. O(h^2). e_2 = (1/3) * (I2 - I1)
N1 = 16
N2 = 32

    #find integral values
I1 = trapezoidal(rat_func, (0,1), N1, rat_funcPrimed)[0]
I2 = trapezoidal(rat_func, (0,1), N2, rat_funcPrimed)[0]

trapezoidal_error_est = trapezoidal_ErrEst(I1, I2) #use error estimate formula

print('Trapezoidal Practical Error Estimation Value of %i slices:'%N2, trapezoidal_error_est)




"""
---------------------- (e)

wont work for simpsons rule because our approximation error, by derivation, is zero when evaluated between 0 and 1. 
to fix, just implement formula e2 = (1/15) * (I2 - I1) for N1, N2 difference number of slices
"""


    #find integral values
I1 = simpsons(rat_func, (0,1), N1, rat_funcPPPrimed)[0]
I2 = simpsons(rat_func, (0,1), N2, rat_funcPPPrimed)[0]
print(simpsons_ErrEst(I1, I2))


    