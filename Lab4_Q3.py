import numpy as np 
import matplotlib.pyplot as plt 

    #latex in plots 
plt.rcParams.update({"text.usetex": True})



"""
------------------------------------------------ Q3 (a)
"""

def burger_solve(u_x0: list[list], 
                 u_0t: float,
                 u_Lt: float,
                 eps: float, 
                 Lx: float, 
                 dx: float, 
                 Tf:float, 
                 dt:float
                 ) -> list[list]:
    """
    Solves Burger's equation using a FTCS method.

    
    *u_t0:* input initial condition at t=0.

    *u_x0:* float. Boundary u(t, 0) condition.

    *u_xL:* float. Boundary u(t, Lx) condition.

    *eps:* float. Equation parameter.

    *Lx:* float. Spatial length. Function writes spatial axis as the interval [-Lx/2, Lx/2] at step dx.

    *dx:* float. Spatial stepsize.

    *Tf:* float. Total amount of time specifying the solution. 

    *dt:* float. Temporal stepsize.
    """
    J = int(Lx/dx)+1    #define array lengths by the integer values of total / step
    N = int(Tf/dt)+1
    

    beta = eps * (dt/dx)    #define beta    

    if len(u_x0) != J:  #if the initial condition u(x, t=0) is not the same size as J, raise a value error 
        raise ValueError(f"Initial condition u_init must be same size as axis specified by Lx and dx.")

    u = np.zeros((N, J))    #specify NxJ array to account for each timestep and space value 
    u[0, :] = u_x0  #set the initial condition by taking the slice at t=0

            #iterate over the n and j values 
    for n in range(1, N-1): #n first
        for j in range(0, J):   #j last, since we evaluate each j for each n 
            if j == 0:      #boundary conditions at x=0
                u[n, j] = u_0t
            if j == J-1:        #boundary conditions at x=L
                u[n, j] = u_Lt
            else:       #step forward in time for every other j  according to leapfrog method 
                u[n+1, j] = u[n-1, j] - (beta/2)*((u[n, j+1]**2) - (u[n, j-1]**2))

    return u



"""
------------------------------------------------ Q3 (b)
"""
    #initial conditions 
dx = 0.02 #also try 0.1;  there is a problem with step sizes, smaller step sizes diverge too quickly and can be unrealistic.
space = np.arange(0, 2*np.pi, dx)
unot = np.sin(space)

    #evaluate u by calling our function 
    #u is fct array of x and t 
u_out = burger_solve(unot, 0, 0, 1, 2*np.pi, dx, 2, 0.005)

    #iterate over time values to plot them 
for time in (0, 0.5, 1, 1.5):
    k = int(time/0.005) #determine index 
    plt.plot(space, u_out[k, :], label = r'$t$ = %.1f'%time)    #take slice and plot 

    #labels 
plt.legend(loc='best', fontsize=16)
plt.title(r'Evolution of $u(x,t)$ Solution to Burger', fontsize=18)
plt.xlabel('$x$ (m)', fontsize=16)
plt.ylabel('Wave Speed (m/s)', fontsize=16)
plt.show()

