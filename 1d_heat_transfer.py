import numpy as np
import matplotlib as plt

delta_x = 1     #Each step
alpha = 1       #Coefficient

inital_value = 100
left_boundry = 0
rigth_boundry = 0

iterations = 100

x_axis = 100    #Aka lenght

nodes = np.linespace(0, x_axis)

u_array = np.empty(iterations, nodes)

def calculate(u):
    for k in range(iterations + 1):        
        for i in range(nodes + 1): #Maybe not neccessary ( + 1)
            u[k + 1, i] = u[k, i] + alpha((u[k, i+1] - 2(u[k, i]) + u[k, i-1])/delta_x**2) 
    return u

def plot():
 return