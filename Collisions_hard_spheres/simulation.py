import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import heapq
import multiprocessing


class Particle:
    def __init__(self, rx, ry, vx, vy, radius, mass):
        self.rx = rx
        self.ry = ry
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.mass = mass

    #Collision with other particle
    def collides(self, p2):
        delta_rx = self.rx - p2.rx
        delta_ry = self.ry - p2.ry
        delta_vx = self.vx - p2.vx
        delta_vy = self.vy - p2.vy

        delta_r_delta_r = (delta_rx ** 2) + (delta_ry ** 2)
        delta_v_delta_v = (delta_vx ** 2) + (delta_vy ** 2)
        delta_v_delta_r = (delta_vx * delta_rx) + (delta_vy * delta_ry)

        d = (delta_v_delta_r ** 2) - (delta_v_delta_v * (delta_r_delta_r - ((self.radius + p2.radius) ** 2)))

        if (delta_v_delta_r >= 0):
            delta_t_collision = 9999
        elif (d < 0):
            delta_t_collision = 9999
        else:
            delta_t_collision = -(delta_v_delta_r + np.sqrt(d)) / (delta_v_delta_v)

        print("Particle collision:", delta_t_collision)
        return delta_t_collision
    
    #Collision with vertical wall
    def collidesX(self):
        if (self.vx > 0):
            delta_t_collision = ((1 - self.radius - self.rx) / self.vx)
        elif (self.vx < 0):
            delta_t_collision = ((self.radius - self.rx) / self.vx)
        else:
            delta_t_collision = 9999

        return delta_t_collision

    #Collision with horizontal wall
    def collidesY(self):
        if (self.vy > 0):
            delta_t_collision = ((1 - self.radius - self.ry) / self.vy)
        elif (self.vy < 0):
            delta_t_collision = ((self.radius - self.ry) / self.vy)
        else:
            delta_t_collision = 9999

        return delta_t_collision

    #Collision with other particle
    def bounce(self, p2):
        #Calculate impulse J
        delta_rx = self.rx - p2.rx
        delta_ry = self.ry - p2.ry
        delta_vx = self.vx - p2.vx
        delta_vy = self.vy - p2.vy
        delta_v_delta_r = (delta_vx * delta_rx) + (delta_vy * delta_ry)

        j = (2 * self.mass * p2.mass * delta_v_delta_r) / ((self.radius + p2.radius) * (self.mass + p2.mass)) 
        j_x = (j * delta_rx) / (self.radius + p2.radius)
        j_y = (j * delta_ry) / (self.radius + p2.radius)

        self.vx = self.vx - (j_x / self.mass)
        self.vy = self.vy - (j_y / self.mass)
        p2.vx = p2.vx + (j_x / p2.mass)
        p2.vy = p2.vy + (j_y / p2.mass)

    #Reverse velocity in x direction
    def bounceX(self):
        self.vx = -(self.vx)
    
    #Reverse velocity in y direction
    def bounceY(self):
        self.vy = -(self.vy)

priority_queue = []
heapq.heapify(priority_queue)
# heapq.heappush(priority_queue, 'x')
# print(heapq.heappop(priority_queue))

#               rx, ry, vx, vy, radius, mass
p1 = Particle(0.2, 0.3, 0.01, 0.04, 0.01, 1)
p2 = Particle(0.8, 0.7, 0.01, 0.03, 0.01, 1)
iterations = 100
print("Creating animations...")

def test_plot(self):
    if (p1.collides(p2) < 0.65):
        p1.bounce(p2)
    if (p1.collidesX() < 0):
        p1.bounceX()
    if (p1.collidesY() < 0):
        p1.bounceY()
    if (p2.collidesX() < 0):
        p2.bounceX()
    if (p2.collidesY() < 0):
        p2.bounceY()

    p1.rx += p1.vx
    p1.ry += p1.vy
    p2.rx += p2.vx
    p2.ry += p2.vy

    plt.clf()
    plt.title("Iteration")
    plt.xlabel("x label")
    plt.ylabel("y label")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(p1.rx, p1.ry)
    plt.scatter(p2.rx, p2.ry)
    return plt

anim = animation.FuncAnimation(plt.figure(), test_plot, interval=1, frames=iterations, repeat=False)
anim.save("test" + ".gif")
# plt.show()
print("Done")

