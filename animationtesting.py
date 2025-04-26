import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


x = np.arange(0, 10, 0.1)
A = np.arange(-5, 5, 1)


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
def set_axis(ax):
    ax.cla()
    ax.set_ylim([-2, 2])
    ax.set_xlim([0, 30*np.pi])
    ax.set_title("Waves")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Length")

# def animate(frame):
#     set_axis(ax1)
#     amplitude = np.cos(0.3*frame)
#     P = amplitude*np.sin(x)- amplitude*np.cos(x)
#     ax1.plot(P)




# ani = FuncAnimation(fig, animate, frames = 1000, interval = 0.5)
# plt.show()


    #a more appropriate plotting method to plot evolutions over periods of time 
amplitude_sequence = np.linspace(0, 1, 100)
output = np.zeros(len(amplitude_sequence), dtype = 'O')
for i in range(len(amplitude_sequence)):
    k = amplitude_sequence[i]
    output[i] = k*np.sin(x)

def animate(frame):
    set_axis(ax1)
    k = frame%len(output)   #the modulo operator '%' will loop the animation for every n frames 
    ax1.plot(output[k])

ani = FuncAnimation(fig, animate, frames = len(output), interval = 0.5)
# wr=FFMpegWriter(fps=60)
# ani.save(r'/users/jacealloway/Desktop/joemama.mp4', writer=wr)
plt.show()






