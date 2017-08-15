# Graph SLAM
# Punnu LK Phairatt

'''
   X,Y model dx,dy measurement model
   
   Work in progress:
        1- offline processing
        2- change measurement model to lidar (range, beaing) not dx,dy
           because we dont know exactly the robot position so we can't
           reverse r,bearing to dx,dy from the current position?
           r,bearing -> actual meaurement
           dx,dy -> depend on the robot pose    
   
'''

from math import sin, cos, pi, atan2, sqrt
from numpy import *
from lib.logfile_reader import *
from lib.slam_ekf_library import get_observations, write_cylinders, write_error_ellipses
from lib.UdcMatrix import *
import time

class OnlineGraphSLAM:
    def __init__(self, init_state, robot_width, scanner_displacement, motion_noise_xy, motion_noise_theta, measurement_noise):
        # robot and measurement configuration
        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.motion_noise_xy = motion_noise_xy
        self.motion_noise_theta = motion_noise_theta
        self.measurement_noise = measurement_noise
        
        # x,y,theta
        self.state   = init_state
        self.Omega   = matrix()
        self.Omega.zero(2, 2)
        # set x0,y0,theta0 to the initial position (mark by both 1.0)
        self.Omega.value[0][0] = 1.0
        self.Omega.value[1][1] = 1.0
         # set vector to the initial position
        self.Xi = matrix()
        self.Xi.zero(2, 1)
        self.Xi.value[0][0] = init_state[0]
        self.Xi.value[1][0] = init_state[1]
        # initial matrix size (x0,y0,theta0)
        self.dim = 2
        # create an empty list of landmarks
        self.number_of_landmarks = 0
        self.landmark_list = []
        
        
    def dg(self, control, w):
        """the motion command dx,dy,dtheta from the current state"""
        x, y, theta = self.state[0:3]
        l, r = control
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            dx = (rad + w/2.)*(sin(theta+alpha) - sin(theta))
            dy = (rad + w/2.)*(-cos(theta+alpha) + cos(theta))
            dtheta = (alpha + pi) % (2*pi) - pi
        else:
            dx = l * cos(theta)
            dy = l * sin(theta)
            dtheta = 0
        
        motion = array([dx, dy, dtheta])  
        self.state[0:3] += motion
        return motion
            
        
    def add_landmark_to_state(self, initial_coords):
        """Enlarge the informtion matrix (Omega) and vector(Xi) to include one more
           landmark, which is given by its initial_coords (an (x, y) tuple).
           Returns the index of the newly added landmark."""
        # update state
        lmk_index = self.number_of_landmarks
        self.number_of_landmarks += 1
        self.landmark_list.append(lmk_index)
        self.state = append(self.state, array(initial_coords))
        #print("lmk xy: ", initial_coords)
        
        # extend the matrix and vector at the end of the matrices
        self.dim += 2
        dim = self.dim
        self.Omega = self.Omega.expand(dim,dim,list(range(dim-2)),list(range(dim-2)))
        self.Xi = self.Xi.expand(dim,1,list(range(dim-2)),[0])
        return lmk_index
    
    def get_landmarks(self):
        """Returns a list of (x, y) tuples of all landmark positions."""
        return ([(self.state[3+2*j], self.state[3+2*j+1])
                 for j in range( int(self.number_of_landmarks)  )])
    
    
    def motion_update(self, motion):
        """Update the motion information."""
        # extend matrix dimension for new state x,y,theta and compute range to be extended and added
        self.dim += 2
        expand_i = list(range(2)) + list(range(4, self.dim))
        # Expand matrix and vector to accomodate one step move e.g. (xi,yi) to (xi+1,yi+1) 
        self.Omega = self.Omega.expand(self.dim, self.dim,  expand_i, expand_i)
        self.Xi = self.Xi.expand(self.dim, 1, expand_i, [0])
        # diag update e.g. (x0,x0),(y0,y0),(theta0,theta0),(x1,x1),(y1,y1),(theta1,theta1)
        for b in range(4):
            # set motion noise
            motion_noise = motion_noise_xy    
            self.Omega.value[b][b] +=  1.0 / motion_noise
            
        # off-diag update e.g. (x0,x1),(x1,x0),(y0,y1),(y1,y0),(theta0,theta1),(theta1,theta0)
        for b in range(2):
            # set motion noise
            motion_noise = motion_noise_xy
                
            self.Omega.value[b  ][b+2] += -1.0 / motion_noise
            self.Omega.value[b+2][b  ] += -1.0 / motion_noise
            # update vector e.g. (x0,x1),(y0,y1)
            self.Xi.value[b  ][0]  += -motion[b] / motion_noise
            self.Xi.value[b+2][0]  +=  motion[b] / motion_noise
                  
  
    def measurement_update(self, observations):
        """Update the measurement information."""
       
        for obs in observations:
            # extract data
            (r,bearing), cylinder_world, cylinder_scanner, cylinder_index = obs
            
            lmk_index = cylinder_index
            dx_lmk = cylinder_world[0]-self.state[0]#r*cos(bearing)#
            dy_lmk = cylinder_world[1]-self.state[1]#r*sin(bearing)#
            
            if lmk_index == -1:
                # new one - need to expand matrix/vector at the end for a landmark
                lmk_index = self.add_landmark_to_state(cylinder_world)
                
                
            # measurement -> [lmk index, dx, dy from the robot position]    
            measurement = [lmk_index, dx_lmk, dy_lmk]
            
            # m is the index of the landmark coordinate in the matrix/vector
            # start after motion expansion so start from 3
            m = 2 + 2*lmk_index
            
            # update the information maxtrix/vector based on the measurement
            # measurement information update both robot x,y and lmmk x,y
            # b = 0 update x data, b = 1 update y data
            # Note: update the measurement from the new position after the motion update
            #       shift everything by 3
            for b in range(2):
                self.Omega.value[b][b]     +=  1.0 / self.measurement_noise
                self.Omega.value[m+b][m+b] +=  1.0 / self.measurement_noise
                self.Omega.value[b][m+b]   += -1.0 / self.measurement_noise
                self.Omega.value[m+b][b]   += -1.0 / self.measurement_noise
                self.Xi.value[b][0]        += -measurement[1+b] / self.measurement_noise
                self.Xi.value[m+b][0]      +=  measurement[1+b] / self.measurement_noise
        
        #print(self.Omega)        
        #mu = self.Omega.inverse() * self.Xi        
                
    def update_onlineslam(self):
        """Compute the robot and landmark estimated position"""
        # Sebastian Thrun megic fomula from udacity AI class
        # Online matrix/vector transformation - reduce size to a current position after move      
        # Omega <- Omega_p - AT*B_inv*A
        # Xi <- Xi_p - AT*B_inv*C
        Omega_p = self.Omega.take(list(range(2,self.dim)),list(range(2,self.dim)))
        Xi_p = self.Xi.take(list(range(2,self.dim)),[0])
        A = self.Omega.take(list(range(2)),list(range(2,self.dim)))
        B = self.Omega.take(list(range(2)),list(range(2)))
        C = self.Xi.take(list(range(2)),[0])
        self.Omega = Omega_p - (A.transpose() * B.inverse() * A)
        self.Xi = Xi_p - (A.transpose() * B.inverse() * C)
        # now Omega and Xi has a reduced dimension 
        self.dim -= 2
        
    def solve_onlineslam(self):
        # compute and update estimated robot and landmark state
        mu = self.Omega.inverse() * self.Xi
        '''
        # copy custom matrix() to numpy 1d array
        row = mu.dimx
        self.state = array( [mu.value[i][0] for i in range(row)] )
        self.state[2] = (self.state[2] + pi) % (2*pi) - pi
        '''
        return mu
        
        
       
if __name__ == '__main__':
    start_t = time.time()
    # Robot constants.
    scanner_displacement = 30.0             # scanner offset from x vehicle, mm
    ticks_to_mm = 0.349
    robot_width = 155.0                     # mm 

    # Cylinder extraction and matching constants.
    minimum_valid_distance = 20.0           # min scanner range, mm
    depth_jump = 100.0                      # cylinder slope thresholding, mm
    cylinder_offset = 90.0                  # cylinder centroid offset for a landmark measurement, mm
    max_cylinder_distance = 500.0           # max scanner range, mm

    # Filter constants.
    motion_noise_xy = 600.0                   # Weight error in x,y motor control
    motion_noise_theta = 1.0                # Weight error in steering control 
    measurement_noise = 300.0                 # Weight error in x,y measurement

    # Arbitrary start position.
    initial_state = array([500.0, 0.0, 45.0 / 180.0 * pi])

    # Setup filter.
    gph = OnlineGraphSLAM(initial_state, robot_width, scanner_displacement, motion_noise_xy, motion_noise_theta, measurement_noise)
    

    # Read data.
    logfile = LegoLogfile()
    logfile.read("../in_data/robot4_motors.txt")
    logfile.read("../in_data/robot4_scan.txt")

    # Loop over all motor tick records and all measurements and generate
    # filtered positions and covariances.
    # This is the EKF SLAM loop.
    f = open("../out_data/graph_slam_correction.txt", "w")
    for i in range(len(logfile.motor_ticks)-1):
    #for i in range(1):            
        # Update menasurement information
        # LK- note: In slam First observation will get no lmk match since start with empty map
        #           This is lmk association with the scans data at this tick location
        #           if matched -> return with lmk index in the system state, if no matched -> -1
        # observation = [(range, bearing),(lmk x world,lmk y world),(lmk x scanner, lmk y scanner),(lmk index)]
        observations = get_observations(logfile.scan_data[i],depth_jump, minimum_valid_distance, cylinder_offset,gph, max_cylinder_distance)
        gph.measurement_update(observations)
        
        # get motion command dx,dy,dtheta from odometry
        control = array(logfile.motor_ticks[i+1]) * ticks_to_mm
        motion = gph.dg(control,robot_width)
        # Update motion information
        gph.motion_update(motion)
        
        # Update Graph
        gph.update_onlineslam()
        
        
        # compute graph    
        mu = gph.solve_onlineslam()
        
        #print("n lmk: ", gph.number_of_landmarks)
        #print("Omega: ", gph.Omega)
        #print("Xi: ", gph.Xi)
        #print("mu: ", mu)
        
        # Save result
           
        # End of Graph SLAM - from here on, data is written.
        # Output the center of the scanner, not the center of the robot.
        xs = mu.value[0][0] + scanner_displacement * cos(gph.state[2])
        ys = mu.value[1][0] + scanner_displacement * sin(gph.state[2])
        
        print("F %f %f %f" %tuple([xs, ys, 0.0]), end = " ", file=f)
        print(file=f)
        # Write estimates of landmarks.
        write_cylinders(f, "W C", gph.get_landmarks())
        # Write cylinders detected by the scanner.
        write_cylinders(f, "D C", [(obs[2][0], obs[2][1])
                                   for obs in observations])
            
    print(mu)  
    f.close()
    
    
    