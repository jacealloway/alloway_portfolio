import numpy as np 
import matplotlib.pyplot as plt 

    #latex in plots 
plt.rcParams.update({"text.usetex": True})


def shallow_solve(
        Lx: float,  #x length 
        dx: float,  #x step
        Tf: float,  #time length
        dt: float,  #timestep 
        g: float,   #gravity 
        u_0t: float,    #initial u(x=0, t) condition
        u_Lt: float,    #initial u(x=L, t) condition
        u_x0: list,     #initial u(x, t=0) condition
        eta_x0: list,   #initial eta(x, t=0) condition
        eta_b: list     #eta_b, the ground topography 
        ) -> list[list]:
    """
    Solves shallow water equations. 
    """
    J = int(Lx/dx)      #J = num of x vals
    N = int(Tf/dt)      #N = num of t vals

    u = np.zeros(J, float)  #make arrays of size J for eta and u
    eta = np.zeros(J, float)

            #if the initial shapes dont match the shapes specified by axes, return an error 
    if len(eta_x0) != J or len(eta_b) != J:
        raise ValueError("eta_x0 / eta_b shapes must match input length!")

    if len(u_x0) != J:
        raise ValueError("u_x0 shape must match input length!")
    
        #the n=0 conditions 
    u = u_x0
    eta = eta_x0
    beta = dt/(2*dx)    #defined beta 

    u_new = np.copy(u)      #copy to new arrays 
    eta_new = np.copy(eta)
    n = 0   #time iteration counter 

    while n < N+1:
        n += 1  #up the counter for each timestep 
        for j in range(J):
            if j==0:    #boundary x=0: only evaluate one step instead of 2  
                u_new[j] = u_0t - (dt/dx)*(0.5*u[j+1]**2 + g*eta[j+1]  -  0.5*u[j]**2 - g*eta[j]) 

            if j==J-1:  #boundary x=L: only evaluate one step instead of 2
                u_new[j] = u_Lt - (dt/dx)*(0.5*u[j]**2 + g*eta[j]  -  0.5*u[j-1]**2 - g*eta[j-1]) 

            else:       #step forward in time for every other point, taking 2 step-differences 
                u_new[j] = u[j] - beta*( 0.5*u[j+1]**2 + g*eta[j+1]  -  0.5*u[j-1]**2 - g*eta[j-1])
                eta_new[j] = eta[j] - beta*( (eta[j+1] - eta_b[j+1])*u[j+1]  -  (eta[j-1] - eta_b[j-1])*u[j-1])

        u = np.copy(u_new)      #copy the arrays over and repeat by taking  n -> n+1
        eta = np.copy(eta_new)


    return u, eta       #return once complete 


        #initial eta condition 
def eta_init(x, H, A, mu, sigma):
    J = len(x)  
    output = np.zeros(J, float)
    exponent = np.zeros(J, float)
    for j in range(J):      #calculate the exponent seperately so we can average it 
        exponent[j] = A*np.exp(-(x[j]-mu)**2/(sigma**2))

    for j in range(J):
        output[j] = H + exponent[j] - np.mean(exponent) #calculate output 

    return output

    #initial conditions 
dx = 0.02
L = 1
x = np.arange(0, L, dx)
eta_x0 = eta_init(x, 0.01, 0.002, 0.5, 0.05)
u_x0 = np.zeros(len(x))
    
        #loop over times that we want to plot 
for time in (4, 1, 0):      #by calling the functionw e defined 
    u_out, eta_out = shallow_solve(L, dx, time, 0.01, 9.81, 0, 0, u_x0, eta_x0, np.zeros(len(x), float))

    plt.plot(x, eta_out, label='eta, time=%i'%time, lw = 1.2)   #plot 

    #labels 
plt.legend(loc='best', fontsize=16)
plt.xlabel(r'$x$ (m)', fontsize=16)
plt.ylabel('Wave Height (m)', fontsize=16)
plt.title(r'Free Surface of Shallow Water Wave $\eta(x, t)$', fontsize=18)
plt.show()