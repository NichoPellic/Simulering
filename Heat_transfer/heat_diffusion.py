import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

#Create the 3D figure object
fig = plt.figure()
fig_3d = fig.add_subplot(111, projection="3d")

#Config values
delta_x = 1                 #Each step
alpha = 0.25                #Coefficient

max_temp = 100              #Maximum temp
inital_value = 100          #Initial temp value

iterations = 300           #Is the same as time

x_axis = 50                 #Is the same as lenght

#test
delta_t = (delta_x**2) / (4 * alpha)
gamma = (alpha * delta_t) / (delta_x**2)

#Values for a 2D plate
x_axis_plate = 50   #Length of plate
y_axis_plate = 30   #Height of plate

nodes = np.linspace(0, x_axis)

#Create an empty 2D array
u_array = np.empty((iterations, x_axis))

#Create an empty array for a plate object
u_array_2d = np.empty((iterations, y_axis_plate, x_axis_plate))

u_array.fill(0)
u_array_2d.fill(0)

#Needed to create 2D arrays for 3D model
x = np.arange(x_axis)
y = np.arange(iterations)

#Bool vars for different modes
heated_midle = True            #If false then the endpoints are heated
constant_heat_source = True     #If false then the heat is only applied on the first iteration
heat_source_limit = 100         #Number of iterations heat is applied

#If not heated middle, set the middle of the 2D array to be uniform random values
if(not heated_midle):
    u_initial = np.random.uniform(low=40, high=100, size=(y_axis_plate,x_axis_plate))
    u_array_2d[0,:,:] = u_initial

#Calculates the diffusion
def calculate_forward_euler_1d(u):

    set_heat_source = True    

    for k in range(iterations - 1):
        
        #Applies heat for the choosen duration
        if(constant_heat_source or k <= heat_source_limit):
            set_heat_source = True        
        
        #Sets heat at designated points
        if(set_heat_source):
            if(heated_midle):
                u_array[k][int(x_axis/2)] = inital_value

            #Sets the endpoints as heated
            else:            
                u_array[k][0] = inital_value
                u_array[k][x_axis - 1] = inital_value

            set_heat_source = False

        for i in range(1, x_axis - 1):
            #Skips the midlepoint (heat source)
            if(heated_midle):
                if(i != 50):
                    u[k+1, i] = u[k][i] + (alpha * ((u[k][i+1] - 2*u[k][i] + u[k][i-1]) / (delta_x**2)))

            #Skips the endpoints (heat sources)
            else:
                if(i != 0 and i != x_axis - 1):
                    u[k+1, i] = u[k][i] + (alpha * ((u[k][i+1] - 2*u[k][i] + u[k][i-1]) / (delta_x**2)))

    return u

def calculate_forward_euler_2d(u):

    set_heat_source = True       
    
    for k in range(iterations - 1):     

        if(constant_heat_source or k <= heat_source_limit):
            set_heat_source = True

        if(set_heat_source):
            if(heated_midle):
                u_array_2d[k][int(y_axis_plate/2)][int(x_axis_plate/2)] = inital_value

        for i in range(1, y_axis_plate - 1):
            for j in range(1, x_axis_plate - 1):
                u[k + 1, i, j] = alpha * ((u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j]) /  (delta_x**2))+ u[k][i][j]

    return u 

def calculate_backward_euler_1d(u):

    return

#Plot a simple 2D graph at time k
def plot_2d_graph(u_val, k):
    plt.clf()
    plt.title("Temperature at iteration " + str(k))
    plt.xlabel("m")
    plt.ylabel("Temperature")
    #plt.pcolormesh(u_val, cmap=cm.jet, vmin=0, vmax=max_temp)  
    plt.plot(u_val)   
    return plt

def plot_2d_plate(u_val, k):
    plt.clf()
    plt.title("Temperature at iteration" + str(k))
    plt.ylabel("Height")
    plt.xlabel("Width")
    plt.pcolormesh(u_val, cmap=cm.jet, vmin=0, vmax=max_temp)    
    plt.colorbar()
    return plt

#Plot the model in wireframe style
def plot_3d_mesh():
    #plt.clf()
    #Creates 2D arrays for the X and Z axis 
    X, Y = np.meshgrid(x, y)
    fig_3d.set_xlabel("length")    
    fig_3d.set_ylabel("time")
    fig_3d.set_zlabel("temp")
    fig_3d.plot_wireframe(X, Y, u_array)
    plt.savefig("mesh_model.png")
    #plt.show()

#Plot the model in solid style with color mapping
def plot_3d_solid():
    #plt.clf()
    X, Y = np.meshgrid(x, y)
    fig_3d.set_xlabel("length")    
    fig_3d.set_ylabel("time")
    fig_3d.set_zlabel("temp")
    fig_3d.set_zlim3d(0, max_temp)
    fig_3d.plot_surface(X, Y, u_array, cmap=cm.jet)
    plt.savefig("solid_model.png")
    #plt.show()

#Creates a new image for each frame
def animate(k):
    #plot_2d_graph(u_array[k], k)
    plot_2d_plate(u_array_2d[k], k)

print("Heat Diffusion Group 2")
print("Starting calculations...")

u_array = calculate_forward_euler_1d(u_array)
u_array_2d = calculate_forward_euler_2d(u_array_2d)

print("Done calculating")

if(True):
    print("Creating animations...")
    #Create an animation of the heat set and save it as a .gif
    anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=iterations, repeat=False)
    anim.save("heat_map.gif")

else:
    plot_3d_mesh()
    plot_3d_solid()

print("Program done!")