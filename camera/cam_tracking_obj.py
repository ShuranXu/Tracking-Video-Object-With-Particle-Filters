import numpy as np
import cv2

# Repeatability
np.random.seed(0)

HEIGHT = 480
WIDTH =640
NUM_PARTICLES = 4000
VEL_RANGE = 5 # the target is not moving faster then half of the pixel per frame

POS_SIGMA =  1.0 # standard deviation for the particle position: 1 pixel
VEL_SIGMA = 0.5 # standard deviation for the particle velocity: 0.5 pixel

# We choose a single pixel on the target to define the TARGET_COLOUR. 
# Most of the other pixels on the target will be blue but have slightly
# different pixel values.  

# The RGB values of the target in the camera frame
TARGET_COLOUR = np.array((255,255,255)) 

# -----------------------------------------------------------
# Capture a frame from the camera 
# -----------------------------------------------------------
def get_cam_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            yield frame
        else:
            break
    cap.release()
    yield None



# -----------------------------------------------------------
# Creating a particle cloud
# -----------------------------------------------------------
def initialize_particles():
    particles = np.random.rand(NUM_PARTICLES, 4) # 4 columns
    # the first column is the x position,  we want to scale it up to the  width of the frame
    # the second column is the y position, we want to scale it up to the height of the frame.
    # The two other components are x,y's velocity components, and we scale them
    #down to our initial velocity range
    particles = particles * np.array( (WIDTH, HEIGHT, VEL_RANGE, VEL_RANGE))
    # We want to centre the velocity around zero,  because particles have the 
    # probabiliy to go any direction, so they must be able to go positive or negative
    particles[:,2:4] -= VEL_RANGE/2.0
    # print top 20 rows 
    print(particles[:20,:])
    return particles


# -----------------------------------------------------------
# Moving particles according to their velocity state
# -----------------------------------------------------------
def apply_velocity(particles):
    particles[:,0] += particles[:,2]
    particles[:,1] += particles[:,3]
    return particles

# -----------------------------------------------------------------
# Prevent particles from falling off the edge of the camera frame 
# -----------------------------------------------------------------
def enforce_edges(particles):
    for i in range(NUM_PARTICLES):
        # we do WIDTH-1 is because the frame coordinate is zero-based, 
        # so for frame of 100-pixel width, you want to pixel to go from
        # 0 to 99.
        particles[i,0] = max(0, min(WIDTH-1, particles[i,0]))
        particles[i,1] = max(0, min(HEIGHT-1, particles[i,1]))
    return particles


# -----------------------------------------------------------
# Measure each particle's quality 
# -----------------------------------------------------------
def compute_errors(particles, frame):
    errors = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        x = int(particles[i,0])
        y  = int(particles[i,1])
        #  obtain the pixel values of the frame at (x,y)
        pixel_color = frame[ y, x, : ]
        # calculate the mean-squared difference
        errors[i] = np.sum((TARGET_COLOUR  - pixel_color)**2)
    return errors


# -------------------------------------------------------------------
# Assign weights to the particles according to their quality of match.
# -------------------------------------------------------------------
def compute_weights(particles, errors):
    # we want to give more weights to paricles with smaller errors
    weights = np.max(errors) - errors
    # prevent the partiles from piling up along the edge
    weights[
        (particles[:,0] == 0) |
        (particles[:,0] == WIDTH-1) |
        (particles[:,1] == 0) |
        (particles[:,1] == HEIGHT-1) 
    ] = 0.0
    # square weights so that large weights get exaggerated. 
    weights = weights**4
    return weights


# -----------------------------------------------------------
# Resample particles according to their weights
# -----------------------------------------------------------
def resample(particles, weights):
    # normalize all the weights and use the normalied weights as probabilities
    probabilities = weights / np.sum(weights)
    # resample particles according to these probabilities
    # i.e. we are going to build  new particles array by sampling from the
    # current particles. The ones with high weights get chosen many times
    # and those with low weights might not be chosen at all.
    index_numbers = np.random.choice(
        NUM_PARTICLES, # where to sample from: 0 to NUM_PARTICLES-1
        size=NUM_PARTICLES, # how many samples to take
        p=probabilities) # probability distribution
    
    particles = particles[index_numbers, :]
    
    # we can determine the single best guess by calculating the mean
    x = np.mean(particles[:,0])
    y = np.mean(particles[:,1])
    
    return particles, (int(x),int(y))


# -----------------------------------------------------------
# Fuzz the particles
# -----------------------------------------------------------

# We need to locate the target and keep tracking the target, even if the lighting
# conditions change. The solution is to add noise. Noise can be used to express
# the uncertainty of the target state. We will use Gaussian noise and to each
# particle.

# If the target changes in the next frame, some of the particles will have to 
# change in the same way. Thanks to the variations from the Gaussian errors added, 
# they will move along with the target. The other particles that do not move
# with the target will have more color errors and get re-sampled. 

# Increasing the standard deviations of the noise will increase the variations in 
# our particles, and these can match the increased variations we are expecting
# in the target state. 

# Note that raising the weights to a higher power just makes them more sensitive
# to colour, and using more particles can express more uncertainly indeed, but
# in a limited level.


def apply_noise(particles):
    noise = np.concatenate(
    (
        # for x-position. The size of the output, which is the noise array
        #  is size of (NUM_PARTICLES,1).
        np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
        # for y-position
        np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
        # for x-velocity
        np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1)),
        # for y-velocity
        np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1)),
    ),
    axis=1)  # concatenate them column-wise
    
    particles += noise
    return particles


# -----------------------------------------------------------
# Display the camera frames 
# -----------------------------------------------------------
def display(frame, particles, location):
    if len(particles) > 0:
        for i in range(NUM_PARTICLES):
            x = int(particles[i,0])
            y = int(particles[i,1])
            cv2.circle(frame,(x,y), 1, (0,255,0), 1)
    if len(location) > 0:
        cv2.circle(frame, location, 15, (0,0,255), 5)
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) == 27:
        if cv2.waitKey(0) == 27:
            return True
    return False


# -----------------------------------------------------------
# Main Routine
# -----------------------------------------------------------
def main():
    particles = initialize_particles()

    for frame in get_cam_frames():
        if frame is None: break

        particles = apply_velocity(particles)
        particles = enforce_edges(particles)
        errors = compute_errors(particles, frame)
        weights = compute_weights(particles, errors)
        particles, location = resample(particles, weights)
        particles = apply_noise(particles)
        terminate = display(frame, particles, location)
        if terminate:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()