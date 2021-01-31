import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

delta_x = 1     #Each step
alpha = 0.25    #Coefficient

inital_value = 100
left_boundry = 0
rigth_boundry = 0

iterations = 1000

x_axis = 100    #Aka lenght

nodes = np.linspace(0, x_axis)

u_array = np.empty((iterations, x_axis))

u_array.fill(0)

#Sets heatsource for all time iterations
for i in range(iterations - 1):
    u_array[i][int(x_axis/2)] = inital_value


def calculate(u):
    for k in range(iterations - 1):        
        for i in range(x_axis - 1):
            #Skips the midlepoint (heat source)
            if(i != 50):
                u[k+1, i] = u[k][i] + (alpha * ((u[k][i+1] - 2*u[k][i] + u[k][i-1]) / (delta_x**2)))

    return u

def plotGraph(u_val, k):
    plt.clf()
    plt.title("Temperature at " + str(k))
    plt.xlabel("m")
    plt.ylabel("Temperature")

    plt.plot(u_val)
   
    return plt

def animate(k):
    plotGraph(u_array[k], k)

print("Starting calculations...")
u_array = calculate(u_array)
print("Done calculating")

print("Creating animations...")
anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=iterations, repeat=False)
anim.save("graph.gif")

print("Program done!")