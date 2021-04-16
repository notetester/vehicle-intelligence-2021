import numpy as np
from helpers import distance
import math
import copy as cp
class ParticleFilter:
    def __init__(self, num_particles):
        self.initialized = False
        self.num_particles = num_particles

    # Set the number of particles.
    # Initialize all the particles to the initial position
    #   (based on esimates of x, y, theta and their uncertainties from GPS)
    #   and all weights to 1.0.
    # Add Gaussian noise to each particle.
    def initialize(self, x, y, theta, std_x, std_y, std_theta):
        self.particles = []
        for i in range(self.num_particles):
            self.particles.append({
                'x': np.random.normal(x, std_x),
                'y': np.random.normal(y, std_y),
                't': np.random.normal(theta, std_theta),
                'w': 1.0,
                'assoc': [],
            })
        self.initialized = True

    # Add measurements to each particle and add random Gaussian noise.
    def predict(self, dt, velocity, yawrate, std_x, std_y, std_theta):
        # Be careful not to divide by zero.
        v_yr = velocity / yawrate if yawrate else 0
        yr_dt = yawrate * dt
        for p in self.particles:
            # We have to take care of very small yaw rates;
            #   apply formula for constant yaw.
            if np.fabs(yawrate) < 0.0001:
                xf = p['x'] + velocity * dt * np.cos(p['t'])
                yf = p['y'] + velocity * dt * np.sin(p['t'])
                tf = p['t']
            # Nonzero yaw rate - apply integrated formula.
            else:
                xf = p['x'] + v_yr * (np.sin(p['t'] + yr_dt) - np.sin(p['t']))
                yf = p['y'] + v_yr * (np.cos(p['t']) - np.cos(p['t'] + yr_dt))
                tf = p['t'] + yr_dt
            p['x'] = np.random.normal(xf, std_x)
            p['y'] = np.random.normal(yf, std_y)
            p['t'] = np.random.normal(tf, std_theta)

    # Find the predicted measurement that is closest to each observed
    #   measurement and assign the observed measurement to this
    #   particular landmark.
    def associate(self, predicted, observations):
        associations = []
        # For each observation, find the nearest landmark and associate it.
        #   You might want to devise and implement a more efficient algorithm.
        for o in observations:
            min_dist = -1.0
            for p in predicted:
                dist = distance(o, p)
                if min_dist < 0.0 or dist < min_dist:
                    min_dist = dist
                    min_id = p['id']
                    min_x = p['x']
                    min_y = p['y']
            association = {
                'id': min_id,
                'x': min_x,
                'y': min_y,
            }
            associations.append(association)
        # Return a list of associated landmarks that corresponds to
        #   the list of (coordinates transformed) predictions.
        return associations

    # Update the weights of each particle using a multi-variate
    #   Gaussian distribution.
    def update_weights(self, sensor_range, std_landmark_x, std_landmark_y,
                       observations, map_landmarks):
        # TODO: For each particle, do the following:
        # 1. Select the set of landmarks that are visible
        #    (within the sensor range).
        associations=[]
        tran_obs=[]
        for p in self.particles:
            px=p['x']
            py=p['y']
            pt=p['t']
            vis_landmark=[]
            for i, l in map_landmarks.items():
                if distance(p,l)<sensor_range:
                    vis_id=i
                    vis_x=l['x']
                    vis_y=l['y']
                    vis_landmark.append({'id':vis_id,'x':vis_x,'y':vis_y})
        # 2. Transform each observed landmark's coordinates from the
        #    particle's coordinate system to the map's coordinates.
            for obs in observations:
                tran_obs.append({'x':px+obs['x']*np.cos(pt)-obs['y']*np.sin(pt),'y':py+obs['x']*np.sin(pt)+obs['y']*np.cos(pt)})
            if not vis_landmark:
                continue
        # 3. Associate each transformed observation to one of the
        #    predicted (selected in Step 1) landmark positions.
        #    Use self.associate() for this purpose - it receives
        #    the predicted landmarks and observations; and returns
        #    the list of landmarks by implementing the nearest-neighbour
        #    association algorithm.
            near_landmark=self.associate(vis_landmark,tran_obs)
        # 4. Calculate probability of this set of observations based on
        #    a multi-variate Gaussian distribution (two variables being
        #    the x and y positions with means from associated positions
        #    and variances from std_landmark_x and std_landmark_y).
        #    The resulting probability is the product of probabilities
        #    for all the observations.
            p['w']=1.0
            for n in near_landmark:
                norm=1/(2*math.pi*std_landmark_x*std_landmark_y)
                power=((p['x']-n['x'])**2)/(std_landmark_x**2)+((p['y']-n['y'])**2)/(std_landmark_y**2)-2*((p['x']-n['x'])*(p['y']-n['y'])/(std_landmark_x*std_landmark_y))
        # 5. Update the particle's weight by the calculated probability.
                weight=norm*np.exp(-0.5*power)
                p['w']*=weight
                associations.append(n['id'])
            p['assoc']=associations
    # Resample particles with replacement with probability proportional to
    #   their weights.
    def resample(self):
        # TODO: Select (possibly with duplicates) the set of particles
        #       that captures the posteior belief distribution, by
        # 1. Drawing particle samples according to their weights.
        weights=[]
        for i in self.particles:
            weights+=i['w']
        # 2. Make a copy of the particle; otherwise the duplicate particles
        #    will not behave independently from each other - they are
        #    references to mutable objects in Python.
        # Finally, self.particles shall contain the newly drawn set of
        #   particles.
        resample_particles=[]
        for i in range(self.num_particles):
            r=np.random.uniform(0,weights)
            for p in self.particles:
                if r>p['w']:
                    r-=p['w']
                else:
                    resample_particles.append(cp.deepcopy(p))
                    break
        self.particles=resample_particles
    # Choose the particle with the highest weight (probability)
    def get_best_particle(self):
        highest_weight = -1.0
        for p in self.particles:
            if p['w'] > highest_weight:
                highest_weight = p['w']
                best_particle = p
        return best_particle
