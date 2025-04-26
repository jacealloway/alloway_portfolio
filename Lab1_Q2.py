import numpy as np 
import matplotlib.pyplot as plt
'%matplotlib inline' #VSCODE does not compile images inline terminal#

    #use latex in plots
plt.rcParams['text.usetex'] = True
    #define fractional float error
C = 10**(-16)



"""
---------------------- (a)
"""
print('-------------------PART (a)')

def relativeError(
        measured: list[int|float],
        actual: list[int|float]
        ) -> list[int|float]:
    """
    Compute the percentage error between a measured and actual value.
    """
    if type(actual) or type(measured) == float:   #if the input is a single number, do this
        return np.abs(measured - actual) / actual
    else:       #but if the input is an array of values, do this for every value
        err = []
        for i in range(len(actual)):
            err.append(np.abs(measured[i] - actual[i]) / actual[i])
        return err


def p(
        u: int|float
        ) -> int|float:
    """
    Polynomial.

    Returns value and rounding error.
    """
    C = 10**(-16)
    P = (1-u)**8
    sig = C*u**8
    return (P, sig)


def q(
        u: int|float
        ) -> list[int|float]:
    """
    Taylor expansion of p(u) close to u = 1. 

    Returns value of polynomial and rounding error.
    """
    Q = (1) - (8*u) + (28*u**2) - (56*u**3) + (70*u**4) - (56*u**5) + (28*u**6) - (8*u**7) + (u**8) 

        #calculate the rounding error
    C = 10**(-16)
    mult = []
    num_terms = 9
    for i in range(1,num_terms):
        # mult.append(np.sqrt(i)*C*(u**i))   #this is the rounding error of each of the indivial terms
        mult.append(np.abs(C*u**i))
    sig = np.sqrt(np.sum(np.square(mult), axis=0)) #add the terms with the adding error method
    return (Q, sig)



    #define an interval
interval = np.linspace(0.98, 1.02, 500)
    #plot 
plt.plot(interval, q(interval)[0], label=r'$q(u)$', color='red', lw=2)
plt.plot(interval, p(interval)[0], label=r'$p(u)$', color='black', lw=2)
    #labels and legends
plt.xlabel(r'$u$', fontsize=14)
plt.ylabel(r'Output', fontsize=14)
plt.title(r'Error Comparison', fontsize=16)
plt.legend(loc='best', fontsize=14)
plt.show()




"""
---------------------- (b)
"""
print('-------------------PART (b)')

    #calculate the difference in the polynomials
diff = p(interval)[0] - q(interval)[0]

    #define a function to estimate the errors for us 
def dev_estimate_pq(
        values: list[int|float],
        P_err: list[int|float], 
        Q_err: list[int|float]
        ) -> list[int|float]:
    """
    Estimate the standard deviation using rounding error variances on functions p(u) and q(u). 
    
    Values: array_like

        The difference p(u) - q(u).

    Returns: (numpy standard deviation, manual polynomial deviation, relative error between numpy and manual)
    """
    num_points = len(P_err)
    poly_sig = np.zeros(num_points) #Num values to iterate arrays over each entry in the interval
    for i in range(num_points):
        poly_sig[i] = np.sqrt(np.square(P_err[i]) + np.square(Q_err[i]))  #calculate square of variance
        
    poly_sig = np.sqrt(num_points)*np.sqrt(np.mean(np.square(poly_sig)))   #take mean value to find average error for each input step
    numpy_sig = np.std(values, ddof=1) #do the same thing with numpy function
    rel_err = np.abs(poly_sig - numpy_sig) / numpy_sig #compute relative error
    return (numpy_sig, poly_sig, rel_err)




    #plot the difference
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(interval, diff, label=r'$p(u) - q(u)$ Difference', color='black', lw=2)
ax2.hist(diff, bins=50, label=r'Value')
    #labels
ax1.set_xlabel(r'$u$', fontsize=14)
ax1.set_ylabel(r'Output', fontsize=14)
ax1.set_title(r'$p(u) - q(u)$ Error Difference', fontsize=16)
ax2.set_xlabel(r'$p(u) - q(u)$ Difference Value', fontsize=14)
ax2.set_ylabel(r'Frequency', fontsize=14)
ax2.set_title(r'$p(u) - q(u)$ Difference Distribution', fontsize=16)
plt.show()

    #print the comparison between the numpy difference error np.std and the rounding errors 
print("The deviation error estimate for the difference p(u) - q(u) is {}, the numpy standard deviation, the manually determined polynomial deviation, and the percentage error between the two.".format(dev_estimate_pq(diff, p(interval)[1], q(interval)[1])))






"""
---------------------- (c)
"""
print('-------------------PART (c)')

    #do everything again but with the interval being different
interval = np.linspace(0.980, 0.984, 200)
    #calculate the difference in the polynomials
diff = p(interval)[0] - q(interval)[0]

    #define a function to compute the fractional error as in eq(4)
def varFractionalError(
        array: list[int|float]
        ) -> int|float:
    """
    Returns the fractional (relative) error sigma / (sum array) compared to C
    """
    C = 10**(-16) #constants
    N = len(array)
    frac = (np.sqrt(np.mean(np.square(array)))) / (np.mean(array)) #calculate the mean-fractional component
    return (C / (np.sqrt(N))) * np.abs(frac) #return with all products




    #call p, q arrays 
p_arr = p(interval)[0]
q_arr = q(interval)[0]
    #call functions to determine RELATIVE and FRACTIONAL error respectively
compare_abs = relativeError(q_arr, p_arr)
print("The fractional error of the difference p(u) - q(u) is {}*C, approximately 1.0*C".format(varFractionalError(p_arr - q_arr)/C))


    #plot the RELATIVE error
fig = plt.figure()
plt.plot(interval, compare_abs, lw = 2, color='k', label="Relative Error")
    #labels, legend
plt.xlabel(r"$u$", fontsize=14)
plt.ylabel(r"Fraction (decimal)", fontsize=14)
plt.title(r"Relative Error of $p(u) - q(u)$ for $u$ close to 1", fontsize=16)
plt.legend(loc = 'best', fontsize=14)
plt.show()






"""
---------------------- (d)
"""
print('-------------------PART (d)')

    #define the f function
def f(
        x: list[int|float]
        ) -> list[int|float]:
    """
    Calculate fractional polynomial and rounding error.

    Returns: value, rounding error.
    """
    val = (x**8) / ((x**4)*(x**4))  
    
    num_err = C*x**8 #numerator error
    den_err = C*np.sqrt(2)*((x**4)*(x**4)) #denominator error
    err = np.sqrt( (num_err / (x**8))**2 + (den_err / ((x**4)*(x**4)))**2 )*val #concatenate them according to error propagation formula
    return (val, err)




    #use the same inteval as in parts (a), (b)
interval = np.linspace(0.98, 1.02, 500)
    #calculate the same difference
diff = f(interval)[0] - 1
    #compare the result f(interval)[1] error, np.sqrt(np.sum(np.square( [array] ))) with eq 4.5 in text
stdval = np.std(diff)
    #use the textvalue to determine the error in multiplying terms together
textval = C*np.sqrt(2)*np.mean(f(interval)[0]) 
    #take the mean of my calculated value from eq(3)
myval = np.mean(f(interval)[1])



    #print all of the error difference values
print("The np.std value, the eq(4.5) value from text, and my manually computed deviation in error is {}, respectively.".format((stdval, textval, myval)))





    #plot
fig = plt.figure()
plt.plot(interval, diff, color='k', label = r"Error Difference Near $u=1$")
    #labels
plt.xlabel(r"$u$", fontsize=14)
plt.ylabel(r"$f(u) - 1$ Value", fontsize=14)
plt.title(r"Error magnitude of $f(u)$ for $u$ close to 1", fontsize=16)
plt.legend(loc='best', fontsize=14)
plt.show()

