import numpy as np 
import matplotlib.pyplot as plt 

plt.rcParams.update({"text.usetex": True})



"""
------------------------------------------------ Q1 (a)
"""
M = 100 #number of gridpoints in x and y 
width = 10 #in cm 
x = np.linspace(0, width, M+1)
X, Y = np.meshgrid(x, x)    #create the grid 
a= x[1] - x[0] #should be 0.1
phi = np.zeros((M+1, M+1), dtype = float) #empty phi array 

        #when x = 2cm and 8cm, find these indices 
v1idx = int(2/a) 
v2idx = int(8/a)

    #by symmetry, set the corresponding values to 1 / -1
phi[v1idx:v2idx, v1idx] = 1
phi[v1idx:v2idx, v2idx] = -1

    #copy to next updated array 
phinew = phi.copy()

    #set accuracy 
target = 10e-6
    #init grid of error tolerances for each cell
delta = np.ones((M, M), float)
it = 0 
    #set max number of iterations
maxit = 1500

    #while the max tolerance over all cells is greater than target, do the loop
while np.max(delta) > target:
        #if #iterations > max, stop (did not converge )
    if it > maxit:
        print(f"Solution did not converge within maximum number of iterations.")
        break 

        #copy new array to old array, repeate
    np.copyto(phi, phinew)

    #iterate over indices
    for k in range(M):
        for l in range(M):
                #boundary conditions (0)
            if k == 0 or k == 100 or l == 0 or l == 100:
                phinew[k,l] = phi[k,l]

                #capacitor conditions (+1, -1)
            if (k in range(v1idx, v2idx)) and (l == v1idx or l == v2idx):
                phinew[k,l] = phi[k,l]

                #determine new values using GS
            else:
                phinew[k, l] = (phi[k+1, l] + phinew[k-1, l] + phi[k, l+1] + phinew[k, l-1])/4

                #if the single cell tolerance is greater than target, compute new tolerance 
            if delta[k, l] > target:
                delta[k,l] = np.abs(phinew[k, l] - phi[k, l])

                #else, return to next step of loop
            elif delta[k, l] <= target:
                continue
                
    it += 1     #up the iterator 

    #print num iterations
print('Regular G-S iterations:', it - 1)


    #plot
fig, ax = plt.subplots(1,1)
cs = ax.contourf(X, Y, phinew, levels = np.linspace(-1, 1, 30))
ax.set_title(f'Electric Potential of Parallel Plate Capacitor', fontsize=18)
cbar = fig.colorbar(cs, ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
cbar.ax.set_yticklabels(['-1.00', '-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75', '1.00'])
cbar.set_label(r'Potential $V$', fontsize=16)
plt.xlabel(r'$x$ (cm)', fontsize=16)
plt.ylabel(r'$y$ (cm)', fontsize=16)
plt.show()






"""
------------------------------------------------ Q1 (b)
"""

        #same thing as last time, just introduce the relaxation paramter

omega_vals = (0.1, 0.5)

for omega in omega_vals:

    phi = np.zeros((M+1, M+1), dtype = float)

    phi[v1idx:v2idx, v1idx] = 1
    phi[v1idx:v2idx, v2idx] = -1

    phinew = phi.copy()

    delta = np.ones((M, M), float)

    it = 0

    while np.max(delta) > target:
        if it > maxit:
            print(f"Solution did not converge within maximum number of iterations.")
            break 

        np.copyto(phi, phinew)

        for k in range(M):
            for l in range(M):
                if k == 0 or k == 100 or l == 0 or l == 100:
                    phinew[k,l] = phi[k,l]

                if (k in range(v1idx, v2idx)) and (l == v1idx or l == v2idx):
                    phinew[k,l] = phi[k,l]

                else:
                    phinew[k, l] = ((1+omega)*(phi[k+1, l] + phinew[k-1, l] + phi[k, l+1] + phinew[k, l-1])/4) - (omega*phi[k, l])

                if delta[k, l] > target:
                    delta[k,l] = np.abs((phinew[k, l] - phi[k, l]))

                elif delta[k, l] <= target:
                    continue
                        
        it += 1


    print('Overrelaxed with omega = %.2f is'%omega, it-1)

        #plot
    fig, ax = plt.subplots(1,1)

    cs = ax.contourf(X, Y, phinew, levels = np.linspace(-1, 1, 30))
    ax.set_title(r'Electric Potential of Parallel Plate Capacitor, Over-Relaxation $\omega = $ %.1f'%omega, fontsize=18)
    cbar = fig.colorbar(cs, ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(['-1.00', '-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75', '1.00'])
    cbar.set_label(r'Potential $V$', fontsize=16)
    plt.xlabel(r'$x$ (cm)', fontsize=16)
    plt.ylabel(r'$y$ (cm)', fontsize=16)
    plt.show()  





"""
------------------------------------------------ Q1 (c)
"""
        #take gradient of potential 
Ey, Ex = np.gradient(phinew)

        #write meshgrid 
X, Y = np.meshgrid(x, x)

fig, ax = plt.subplots(1,1)
        #streamplot the vectors 
strm = ax.streamplot(x, x, -Ex, -Ey, color=phinew, density = 1.5, linewidth = 1.4, arrowsize = 2)

    #set new color of lines, add cbar
ax.set_title(f'Electric Field of Parallel Plate Capacitor', fontsize=18)
cbar = fig.colorbar(strm.lines, ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
cbar.ax.set_yticklabels(['-1.00', '-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75', '1.00'])
cbar.set_label(r'Potential $V$', fontsize=16)
plt.xlabel(r'$x$ (cm)', fontsize=16)
plt.ylabel(r'$y$ (cm)', fontsize=16)
plt.show()





