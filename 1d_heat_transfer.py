import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

#Create the 3D figure object
fig = plt.figure()
fig_3d = fig.add_subplot(111, projection="3d")

#Config values
delta_x = 1         #Each step
alpha = 0.25        #Coefficient

inital_value = 100  #Initial temp value
left_boundry = 0    #Never used?
rigth_boundry = 0   #Never used?

iterations = 2000    #Is the same as time

x_axis = 100        #Is the same as lenght

nodes = np.linspace(0, x_axis)

#Create an empty 2D array
u_array = np.empty((iterations, x_axis))

u_array.fill(0)

#Needed to create 2D arrays for 3D model
x = np.arange(x_axis)
z = np.arange(iterations)


#Sets heatsource for all time iterations
for i in range(iterations - 1):
    u_array[i][int(x_axis/2)] = inital_value

#Calculates the diffusion
def calculate_forward_euler(u):
    for k in range(iterations - 1):        
        for i in range(x_axis - 1):
            #Skips the midlepoint (heat source)
            if(i != 50):
                u[k+1, i] = u[k][i] + (alpha * ((u[k][i+1] - 2*u[k][i] + u[k][i-1]) / (delta_x**2)))

    return u

def calculate_backward_euler(u):

    return

#Plot a simple 2D graph at time k
def plot_2d_graph(u_val, k):
    plt.clf()
    plt.title("Temperature at " + str(k))
    plt.xlabel("m")
    plt.ylabel("Temperature")
    plt.plot(u_val)   
    return plt

#Creates a new image for each frame
def animate(k):
    plot_2d_graph(u_array[k], k)

#Plots the model in a wireframe style
def plot_3d_mesh():
    #Creates 2D arrays for the X and Z axis 
    X, Z = np.meshgrid(x, z)
    fig_3d.set_xlabel("m")    
    fig_3d.set_ylabel("time")
    fig_3d.set_zlabel("temp")
    fig_3d.plot_wireframe(X, Z, u_array)
    plt.show()

def plot_3d_solid():
    X, Z = np.meshgrid(x, z)
    fig_3d.set_xlabel("length")    
    fig_3d.set_ylabel("time")
    fig_3d.set_zlabel("temp")
    fig_3d.plot_surface(X, Z, u_array, cmap=cm.coolwarm)
    plt.show()

print("Starting calculations...")
u_array = calculate_forward_euler(u_array)
print("Done calculating")

#Different options to run
if(False):
    print("Creating animations...")
    #Create an animation of the heat set 
    anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=iterations, repeat=False)
    anim.save("graph.gif")

else:
    plot_3d_mesh()
    #plot_3d_solid()

print("Program done!")