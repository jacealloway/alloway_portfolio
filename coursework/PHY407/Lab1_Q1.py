import numpy as np



"""
---------------------- (a)
PSEUDOCODE


INITIALIZE st_Dev1, st_Dev2 function calculations from eq(1) and eq(2)
INITIALIZE rel_Err function calculation as percentage error formula
OBTAIN cdata.txt with np.loadtxt() as the INIT input array
COMPUTE numpy standard deviation by calling np.std on input array
COMPUTE eq(1) and eq(2) standard deviations by calling st_Dev function
COMPUTE relative error by calling st_Dev function
COMPUTE difference of rel_Err output with np.std() output
PRINT relative error outputs and differences



in initializing the st_Dev1, st_Dev2, rel_Err functions:

INITIALIZE st_Dev1 with 1 input array: x 
CALCULATE length of x, the number of elements
IF len(x)<=1, raise a value error, since len(x) must be greater than 1
CALCULATE mean of x 
INITIALIZE an array of 0's of len(x)
CALCULATE the (x[i] - mean(x))**2  elements, overwrite the 0's array 
CALCULATE square root, multiply by (1/(n-1))
RETURN value 

INITIALIZE st_Dev2 with 1 input array: x
CALCULATE length of x, the number of elements 
IF len(x)<=1, raise a value error, since len(x) must be greater than 1
CALCULATE the square root element in one pass, sum(x[i]**2) - ((1/n)*(sum(x)))**2
IF the square root value is negative, raise a value error, cannot cast complex values into np.float64 type
CALCULATE square root, multiply by (1/(n-1))
RETURN value

INITIALIZE rel_Err with two input arrays, measured and actual 
CALCULATE the absolute difference of measured - actual, divided by actual 
RETURN value 

"""


    #define functions
def st_Dev1(
        x: list[int|float]
        ) -> list[int|float]:
    """
    Calculate the mean and standard deviations of an array of numbers given two forms of the same variance calculation.

    **Returns:** dev1(x)
    """
    n = len(x)
    if n <= 1:  #exit the function if len(x) = 1, else 1/(n-1) is undefined in the std calculation
        raise ValueError('Input array length must be greater than 1!')
    
    mean =  (1/n) * np.sum(x) #calculate the mean value
    sums = np.zeros(n)    #calculate internal square root for eq (1) deviation
    for i in range(n):
        sums[i] = (x[i] - mean)**2
    dev1 = np.sqrt( (1/(n-1)) * np.sum(sums)) #take root

    return dev1 #return values


def st_Dev2(x: list[int|float]
        ) -> list[int|float]:
    """
    Calculate the mean and standard deviations of an array of numbers given two forms of the same variance calculation.

    **Returns:** dev2(x)
    """
    n = len(x) #just find the length, no need to pass thru data
    if n <= 1:  #exit the function if len(x) = 1, else 1/(n-1) is undefined in the std calculation
        raise ValueError('Input array length must be greater than 1!')
    
    sqrt_in = (1/(n-1)) * (np.sum(np.square(x)) - (n * (((1/n)*np.sum(x))**2) ))  #calculate the internal square root term for eq(2) deviation
    if sqrt_in < 0: #check if we're going to take root of negative number...  since there could be error between the x^2 term and the n*xbar term
        raise ValueError('Square root input negative! Cannot cast complex values into float64 variable.')

    dev2 = np.sqrt( sqrt_in ) #take root
    return dev2



    #relative error calculation
def rel_Err(
        measured: int|float, 
        actual: int|float
        ) -> int|float:
    """
    Calculate the percentage (relative) error between two values.
    """
    return np.abs(measured-actual) / actual



"""
---------------------- (b)
"""

def compare_devMethods(
        some_array: list[int|float]
        ) -> None:
    """
    Compare standard deviation methods using two variance calculations with numpy.std().

    **Returns:** Relative error differences with np.std from dev1 and dev2 equations.
    """
    #     #load data, initialize input
    # cdata = np.loadtxt('cdata.txt')    #-going to comment this out for now because I am using this as a function
        #calculate the numpy std 
    std_actual = np.std(some_array, ddof=1) 
        #calculate the std off of equations (1) and (2) with recursion
    std_measured1 = st_Dev1(some_array) 
    std_measured2 = st_Dev2(some_array) 
        #compute relative error on std equations, respectively, using rel_Err function
    rel_error1 = rel_Err(std_measured1, std_actual) 
    rel_error2 = rel_Err(std_measured2, std_actual)
        #find approximate difference between the error values
    rel_difference = rel_error1 - rel_error2
        #print the outputs
    print(f'Relative Error (1):', rel_error1, 'Relative Error (2):', rel_error2)
    if rel_difference > 0:
        print(f'Error from eq(1) is larger.')
    elif rel_difference < 0:
        print(f'Error from eq(2) is larger.')
    elif rel_difference == 0:
        print(f'Errors from (1) and (2) are identical.')


print(f'Outputs from cdata.txt:')
compare_devMethods(np.loadtxt('cdata.txt'))
print(    )






"""
---------------------- (c) 
"""


dist1 = np.random.normal(0., 1., 2000)
dist2 = np.random.normal(1.e7, 1., 2000)


print(f'Outputs from distribution 1 - normal(0., 1., 2000):')
compare_devMethods(dist1)
print(    )  #spacing
print(f'Outputs from distribution 2 - normal(1.e7, 1., 2000):')
compare_devMethods(dist2)
print(    )  #spacing

#Error will accumulate more with eq(2) since there are 2 rounding differences







"""
---------------------- (d) 
PSEUDOCODE 

INITIALIZE reduceErrEq2 function, which takes in an array
COMPUTE iterable variance difference with regards to each term according to Welford's algorithm (single-pass)
RETURN value 



in initializing reduceErrEq2 function:

DEFINE AND INITIALIZE new variables for accumulated mean M, accumulated variance (squared) S
COMPUTE length of x, INITIALIZE as n
COMPUTE iterable terms 'sample' and 'previous mean' via FOR loop iterating through x
COMPUTE iterated (squared) variance, iterated mean
COMPUTE square root of (squared) variance over n-1, for non-biased variance
RETURN value 



"""
def reduceErrEq2(
        x: list[int|float]
        ) -> int|float:
    """
    Computes the standard deviation from equation (2) using a different approach.

    x: _ArrayLike_Float

    **Returns:** Deviation computed from dev 1 using Welford's algorithm.
    """
    M = 0 #initialize new mean variables
    S = 0 #initialize new variance variables
    n = len(x) #specify number of samples
    for k in range(1, n+1):
        sample = x[k-1]   
        prev_M = M
        M = M + ((sample - M) / k)    #new propagated mean 
        S = S + ((sample - M) * (sample - prev_M)) #new propagated variance squared
    
    return np.sqrt(S/(n-1)) #return square root


workaround_d1 = reduceErrEq2(dist1)
workaround_d2 = reduceErrEq2(dist2)
print("The relative error with np.std from the Welford algorithm to dist1, dist2 is: (RE.dist1, RE.dist2) = {}".format((rel_Err(workaround_d1, np.std(dist1, ddof=1)), rel_Err(workaround_d2, np.std(dist2, ddof=1)))))





