import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import heapq
import multiprocessing
import random


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

        delta_r_delta_r = np.square(delta_rx) + np.square(delta_ry)
        delta_v_delta_v = np.square(delta_vx) + np.square(delta_vy)
        delta_v_delta_r = (delta_vx * delta_rx) + (delta_vy * delta_ry)

        d = np.square(delta_v_delta_r) - (delta_v_delta_v * (delta_r_delta_r - (np.square(self.radius + p2.radius))))

        if (delta_v_delta_r >= 0):
            delta_t_collision = 9999
        elif (d < 0):
            delta_t_collision = 9999
        else:
            delta_t_collision = -(delta_v_delta_r + np.sqrt(d)) / (delta_v_delta_v)

        if(delta_t_collision < 5):
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

        #totalspeed = np.sqrt(np.square(self.vx)+np.square(self.vy)) + np.sqrt(np.square(p2.vx)+np.square(p2.vy))

        self.vx = self.vx - (j_x / self.mass)
        self.vy = self.vy - (j_y / self.mass)
        p2.vx = p2.vx + (j_x / p2.mass)
        p2.vy = p2.vy + (j_y / p2.mass)

        #totalspeed = np.sqrt(np.square(self.vx)+np.square(self.vy)) + np.sqrt(np.square(p2.vx)+np.square(p2.vy))
        

    #Reverse velocity in x direction
    def bounceX(self):
        self.vx = -(self.vx)
    
    #Reverse velocity in y direction
    def bounceY(self):
        self.vy = -(self.vy)

class Event:
    def __init__(self, p1, p2, time_collision):
        self.p1 = p1
        self.p2 = p2
        self.time_collision = time_collision
        
    def __lt__(self, other):
        return self.time_collision < other.time_collision
    
    def __eq__(self, other):
        return (self.p1 == other.p1) and (self.p2 == other.p2)

    def compareTo(self, other):
        if((self.p1 == other.p1) and (self.p2 == other.p2)):
            return true


    def getTime(self):
        return self.time_collision
    
    def getParticleOne(self):
        return self.p1

    def getParticleTwo(self):
        return self.p2
        

"""
#               rx, ry, vx, vy, radius, mass
p1 = Particle(0.2, 0.3, 0.01, 0.04, 0.01, 1)
p2 = Particle(0.8, 0.7, 0.01, 0.03, 0.01, 1)
"""


n_particles = 100
particles = []
for i in range(n_particles):
    particles.append(Particle(random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(-0.005, 0.005), random.uniform(-0.005, 0.005), 0.1, 1))



priority_queue = []
heapq.heapify(priority_queue)
priority_queue = sorted(priority_queue)

#p1 = Particle(0.2, 0.5, 0.01, 0.0, 0.01, 1)
#p2 = Particle(0.8, 0.5, -0.01, 0., 0.01, 1)

iterations = 200
iterator = 0
print("Creating animations...")

def test_plot(self):

    """
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
    """
    plt.clf()
    plt.title("Iteration")
    plt.xlabel("x label")
    plt.ylabel("y label")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    #plt.scatter(p1.rx, p1.ry)
    #plt.scatter(p2.rx, p2.ry)
    

    """
    1. Sjekke alle kollisjoner for partikler, mot vegg og hverandre. 
    2. Hvis kollisjon < 9999, lag event
    3. Sorter events etter kollisjons tid
    4. Vent på kollisjon <=> avanser alle partikler
    5. Start på nytt

    """
    
    global iterator
    iterator += 1
    print(iterator)
    global priority_queue
    global particles
    for i in range(n_particles - 1):
        if(particles[i].collides(particles[i + 1]) < 10):
            priority_queue.append(Event(particles[i], particles[i + 1], particles[i].collides(particles[i + 1])))

    for i in range(n_particles):
        if(particles[i].collidesX() < 5):
            event = Event(None, particles[i], particles[i].collidesX())
            for j in range(len(priority_queue)):
                if(event == priority_queue[j]):
                    priority_queue[j] = event
            priority_queue.append(event)
        
        if(particles[i].collidesY() < 5):
            event = Event(particles[i], None, particles[i].collidesY())
            for j in range(len(priority_queue)):
                if(event == priority_queue[j]):
                    priority_queue[j] = event
            
            priority_queue.append(event)
            
            #priority_queue.append(Event(particles[i], None, particles[i].collidesY()))
            
            
            
    
    priority_queue = sorted(priority_queue)
    
    for i in range(n_particles):
        particles[i].rx += particles[i].vx
        particles[i].ry += particles[i].vy

    for i in range(len(priority_queue)):
        if((priority_queue[i].getParticleOne() is not None) and (priority_queue[i].getParticleTwo() is not None) and (priority_queue[i].getTime() < 1)):
            priority_queue[i].getParticleOne().bounce(priority_queue[i].getParticleTwo())

        if((priority_queue[i].getParticleOne() is not None) and (priority_queue[i].getParticleTwo() is None) and (priority_queue[i].getParticleOne().collidesY() < 1)):
            priority_queue[i].getParticleOne().bounceY()
        if((priority_queue[i].getParticleOne() is None) and (priority_queue[i].getParticleTwo() is not None) and (priority_queue[i].getParticleTwo().collidesX() < 1)):
            priority_queue[i].getParticleTwo().bounceX()
        
    for i in range(n_particles):
        plt.scatter(particles[i].rx, particles[i].ry, s = (np.pi * np.square(particles[i].radius)))
    return plt

anim = animation.FuncAnimation(plt.figure(), test_plot, interval=1, frames=iterations, repeat=False)
#anim.save("test" + ".gif")
plt.show()
print("Done")

