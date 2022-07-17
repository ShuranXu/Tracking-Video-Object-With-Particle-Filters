#!/usr/bin/env python
# coding: utf-8

# Tracking Objects in Video with Particle Filters
# ===============================================

# Import libraries

# In[1]:


import numpy as np
import cv2

# Repeatability
np.random.seed(0)

VFILENAME = "walking.mp4"
HEIGHT = 406
WIDTH = 722


# Load video frames from file

# In[2]:


def get_frames(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            yield frame
        else:
            break
    video.release()
    yield None


# Creating a particle cloud

# In[3]:


NUM_PARTICLES = 5000 # 50
VEL_RANGE = 0.5
def initialize_particles():
    particles = np.random.rand(NUM_PARTICLES, 4)
    particles = particles * np.array( (WIDTH,HEIGHT,VEL_RANGE,VEL_RANGE) )
    particles[ :, 2:4 ] -= VEL_RANGE/2.0 # Center velocities around 0
    return particles


# Moving particles according to their velocity state

# In[4]:


def apply_velocity(particles):
    particles[ :, 0 ] += particles[ :, 2 ]  # x = x + u
    particles[ :, 1 ] += particles[ :, 3 ]
    return particles


# Prevent particles from falling off the edge of the video frame

# In[5]:


def enforce_edges(particles):
    for i in range(NUM_PARTICLES):
        particles[i,0] = max(0, min(WIDTH-1, particles[i,0]))
        particles[i,1] = max(0, min(HEIGHT-1, particles[i,1]))
    return particles


# Measure each particle's quality

# In[6]:


def compute_errors(particles, frame):
    errors = np.zeros(NUM_PARTICLES)
    TARGET_COLOUR = np.array( (189,105,82) ) # Blue top sleeve pixel colour
#    TARGET_COLOUR = np.array( (148, 73, 49) ) # Blue top sleeve pixel colour
    for i in range(NUM_PARTICLES):
        x = int(particles[i,0])
        y = int(particles[i,1])
        pixel_colour = frame[ y, x, : ]
        errors[i] = np.sum( ( TARGET_COLOUR - pixel_colour )**2 ) # MSE in colour space
    return errors


# Assign weights to the particles based on their quality of match

# In[7]:


def compute_weights(errors):
    weights = np.max(errors) - errors
    weights[ 
        (particles[ :,0 ] == 0) |
        (particles[ :,0 ] == WIDTH-1) |
        (particles[ :,1 ] == 0) |
        (particles[ :,1 ] == HEIGHT-1)
    ] = 0.0
    
    # Make weights more sensitive to colour difference.
    # Cubing a set of numbers in the interval [0,1], the farther a number is from 1, the more it gets squashed toward zero
    weights = weights**4
    
    return weights


# Resample particles according to their weights

# In[8]:


def resample(particles, weights):
    # Normalize to get valid PDF
    probabilities = weights / np.sum(weights)

    # Resample
    indices = np.random.choice(
        NUM_PARTICLES,
        size=NUM_PARTICLES,
        p=probabilities)
    particles = particles[ indices, : ]

    # Take average over all particles, best-guess for location
    x = np.mean(particles[:,0])
    y = np.mean(particles[:,1])
    return particles, (int(x),int(y))


# Fuzz the particles

# In[9]:


def apply_noise(particles):
    # Noise is good!  Noise expresses our uncertainty in the target's position and velocity
    # We add small variations to each hypothesis that were samples from the best ones in last iteration.
    # The target's position and velocity may have changed since the last frame, some of the fuzzed hypotheses will match these changes.
    POS_SIGMA = 1.0
    VEL_SIGMA = 0.5
    noise = np.concatenate(
        (
            np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
            np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
            np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1)),
            np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1))
        ),
        axis=1
    )
    particles += noise
    return particles


# Display the video frames

# In[10]:


def display(frame, particles, location):
    if len(particles) > 0:
        for i in range(NUM_PARTICLES):
            x = int(particles[i,0])
            y = int(particles[i,1])
#            cv2.circle(frame, (x,y), 1, (0,255,0), 1)
    if len(location) > 0:
        cv2.circle(frame, location, 15, (0,0,255), 5)
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) == 27: # wait n msec for user to his Esc key
        if cv2.waitKey(0) == 27: # second Esc key exits program
            return True
    return False


# Main routine

# In[11]:


particles = initialize_particles()

for frame in get_frames(VFILENAME):
    if frame is None: break

    particles = apply_velocity(particles)
    particles = enforce_edges(particles)
    errors = compute_errors(particles, frame)
    weights = compute_weights(errors)
    particles, location = resample(particles, weights)
    particles = apply_noise(particles)
    terminate = display(frame, particles, location)
    if terminate:
        break
cv2.destroyAllWindows()


# In[ ]:




