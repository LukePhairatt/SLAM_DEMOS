# Graph SLAM
# Punnu LK Phairatt

'''
   Work in progress:
        1- offline processing
        2- change measurement model to lidar (range, beaing) not dx,dy
           because we dont know exactly the robot position so we can't
           reverse r,bearing to dx,dy from the current position?
           r,bearing -> actual meaurement
           dx,dy -> depend on the robot pose
        3- velocity medel
        4- dx dy motion model
        5- udc r, bearing model
        6- using batch update + landmark counter
   
'''

from math import sin, cos, pi, atan2, sqrt
from numpy import *
from lib.logfile_reader import *
from lib.slam_ekf_library import get_observations, write_cylinders, write_error_ellipses
from lib.UdcMatrix import *
import time

class OnlineGraphSLAM:
    def __init__(self, init_state, robot_width, scanner_displacement, \
                 control_motion_factor, control_turn_factor, measurement_distance_stddev,measurement_angle_stddev):
        # robot and measurement configuration
        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = measurement_distance_stddev
        self.measurement_angle_stddev = measurement_angle_stddev
        
        # x,y,theta
        self.state   = init_state
        self.Omega   = matrix()
        self.Omega.zero(3, 3)
        # anchor x0,y0,theta0 to the initial position 
        self.Omega.value[0][0] = 1.0
        self.Omega.value[1][1] = 1.0
        self.Omega.value[2][2] = 1.0
         # set vector to the initial position
        self.Xi = matrix()
        self.Xi.zero(3, 1)
        self.Xi.value[0][0] = init_state[0]
        self.Xi.value[1][0] = init_state[1]
        self.Xi.value[2][0] = init_state[2]
        # initial matrix size (x0,y0,theta0)
        self.dim = 3
        # create an empty list of landmarks
        self.number_of_landmarks = 0
        self.landmark_list = []
        
    @staticmethod
    def g(state, control, w):
        x, y, theta = state
        l, r = control
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            g1 = x + (rad + w/2.)*(sin(theta+alpha) - sin(theta))
            g2 = y + (rad + w/2.)*(-cos(theta+alpha) + cos(theta))
            g3 = (theta + alpha + pi) % (2*pi) - pi
        else:
            g1 = x + l * cos(theta)
            g2 = y + l * sin(theta)
            g3 = theta

        return array([g1, g2, g3])
    
    @staticmethod
    def dg_dstate(state, control, w):
        theta = state[2]
        l, r = control
        if r != l:
            alpha = (r-l)/w
            theta_ = theta + alpha
            rpw2 = l/alpha + w/2.0
            m = array([[1.0, 0.0, rpw2*(cos(theta_) - cos(theta))],
                       [0.0, 1.0, rpw2*(sin(theta_) - sin(theta))],
                       [0.0, 0.0, 1.0]])
        else:
            m = array([[1.0, 0.0, -l*sin(theta)],
                       [0.0, 1.0,  l*cos(theta)],
                       [0.0, 0.0,  1.0]])
        return m
    
    
    @staticmethod
    def dg_dcontrol(state, control, w):
        theta = state[2]
        l, r = tuple(control)
        if r != l:
            rml = r - l
            rml2 = rml * rml
            theta_ = theta + rml/w
            dg1dl = w*r/rml2*(sin(theta_)-sin(theta))  - (r+l)/(2*rml)*cos(theta_)
            dg2dl = w*r/rml2*(-cos(theta_)+cos(theta)) - (r+l)/(2*rml)*sin(theta_)
            dg1dr = (-w*l)/rml2*(sin(theta_)-sin(theta)) + (r+l)/(2*rml)*cos(theta_)
            dg2dr = (-w*l)/rml2*(-cos(theta_)+cos(theta)) + (r+l)/(2*rml)*sin(theta_)
            
        else:
            dg1dl = 0.5*(cos(theta) + l/w*sin(theta))
            dg2dl = 0.5*(sin(theta) - l/w*cos(theta))
            dg1dr = 0.5*(-l/w*sin(theta) + cos(theta))
            dg2dr = 0.5*(l/w*cos(theta) + sin(theta))

        dg3dl = -1.0/w
        dg3dr = 1.0/w
        m = array([[dg1dl, dg1dr], [dg2dl, dg2dr], [dg3dl, dg3dr]])
            
        return m
    
    
    
    @staticmethod
    def h(state, landmark, scanner_displacement):
        """Takes a (x, y, theta) state and a (x, y) landmark, and returns the
           measurement (range, bearing)."""
        dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
        dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
        r = sqrt(dx * dx + dy * dy)
        alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi

        return array([r, alpha])
    
    @staticmethod
    def dh_dstate(state, landmark, scanner_displacement):
        theta = state[2]
        cost, sint = cos(theta), sin(theta)
        dx = landmark[0] - (state[0] + scanner_displacement * cost)
        dy = landmark[1] - (state[1] + scanner_displacement * sint)
        q = dx * dx + dy * dy
        sqrtq = sqrt(q)
        dr_dx = -dx / sqrtq
        dr_dy = -dy / sqrtq
        dr_dtheta = (dx * sint - dy * cost) * scanner_displacement / sqrtq
        dalpha_dx =  dy / q
        dalpha_dy = -dx / q
        dalpha_dtheta = -1 - scanner_displacement / q * (dx * cost + dy * sint)

        return array([[dr_dx, dr_dy, dr_dtheta],
                      [dalpha_dx, dalpha_dy, dalpha_dtheta]])
            
        
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
    
    
    
    #----------------------- Information Filter ----------------------------- #
    
    def motion_update(self, control):
        """Update the motion information."""
        G3 = self.dg_dstate(self.state, control, self.robot_width)
        
        X_old = self.state[0:3]
        # predicting new state
        self.state[0:3] = self.g(self.state[0:3], control, self.robot_width)
        X_new = self.state[0:3]
        
        
        '''
        left, right = control
        
        left_var = (self.control_motion_factor * left)**2 +\
                   (self.control_turn_factor * (left-right))**2
        right_var = (self.control_motion_factor * right)**2 +\
                    (self.control_turn_factor * (left-right))**2
        
        left_var = 250
        right_var = 250
        control_covariance = diag([left_var, right_var])
        V = self.dg_dcontrol(self.state, control, self.robot_width)
        R3 = dot(V, dot(control_covariance, V.T))
        '''
        R3 = array([[200.0, 0.0, 0.0],[0.0, 200.0, 0.0],[0.0,0.0,0.5]])
        I = identity(3)
        Gr = concatenate((-G3.T,I), axis=0)
        Gv = concatenate((-G3,I), axis=1)
        # information filter update
        dot_GR = dot(Gr,linalg.inv(R3))
        Omega_motion = dot(dot_GR,Gv)
        
        motion = X_new - dot(G3,X_old)
        Si_motion = dot(dot_GR,motion)
        Si_motion = Si_motion.reshape(len(Si_motion),1)
    
        #print(Omega_motion)
        # extend matrix dimension for new state x,y,theta and compute range to be extended and added
        self.dim += 3
        expand_i = list(range(3)) + list(range(6, self.dim))
        # Expand matrix and vector to accomodate one step move e.g. (xi,yi) to (xi+1,yi+1) 
        self.Omega = self.Omega.expand(self.dim, self.dim,  expand_i, expand_i)
        self.Xi = self.Xi.expand(self.dim, 1, expand_i, [0])
        
        # convert numpy to matrix form
        Omega_motion = matrix(value=Omega_motion.tolist())
        Si_motion = matrix(value=Si_motion.tolist())
        
        # expand value
        Omega_motion = Omega_motion.expand(self.dim, self.dim, list(range(6)), list(range(6)) )
        Si_motion    = Si_motion.expand(self.dim, 1, list(range(6)), [0] )
        
        # update
        self.Omega += Omega_motion
        self.Xi += Si_motion
        
        
  
    def measurement_update(self, observations):
        """Update the measurement information."""
        for obs in observations:
            # extract data
            measurement, cylinder_world, cylinder_scanner, cylinder_index = obs            
            lmk_index = cylinder_index
            
            if lmk_index == -1:
                # new one - need to expand matrix/vector at the end for a landmark
                lmk_index = self.add_landmark_to_state(cylinder_world)
                print(cylinder_world)
                
            landmark = self.state[3+2*lmk_index : 3+2*lmk_index+2]
            H3 = self.dh_dstate(self.state, landmark, self.scanner_displacement)
            H = zeros((2, 5))
            H[0:2, 0:3] = H3
            H[0:2, 3:5] = -H3[0:2, 0:2]
            Q = diag([self.measurement_distance_stddev**2, self.measurement_angle_stddev**2])
            
            dot_HQ = dot(H.T,linalg.inv(Q))
            Omega_m = dot(dot_HQ,H)
            
            innovation = array(measurement) - self.h(self.state, landmark, self.scanner_displacement)
            innovation[1] = (innovation[1] + pi)%(2*pi) - pi
            
            S = append(self.state[0:3], landmark)
            HSZ = dot(H,S) + innovation
            Si_m = dot(dot_HQ,HSZ)
            Si_m = Si_m.reshape(len(Si_m),1)
            
            # convert numpy to matrix form
            Omega_m = matrix(value=Omega_m.tolist())
            Si_m    = matrix(value=Si_m.tolist())
            
            # sparse update
            dim_row = self.Omega.dimx
            dim_col = self.Omega.dimy
            # sparse expansion
            expand_i = [0,1,2] + list(range(3+2*lmk_index,3+2*lmk_index+2))
            Omega_m = Omega_m.expand(dim_row,dim_col,expand_i,expand_i)
            Si_m = Si_m.expand(dim_row,1,expand_i,[0])
            
            # update
            self.Omega += Omega_m
            self.Xi += Si_m
                
                
                
    def update_onlineslam(self):
        """Compute the robot and landmark estimated position"""
        # Sebastian Thrun megic fomula from udacity AI class
        # Online matrix/vector transformation - reduce size to a current position after move      
        # Omega <- Omega_p - AT*B_inv*A
        # Xi <- Xi_p - AT*B_inv*C
        Omega_p = self.Omega.take(list(range(3,self.dim)),list(range(3,self.dim)))
        Xi_p = self.Xi.take(list(range(3,self.dim)),[0])
        A = self.Omega.take(list(range(3)),list(range(3,self.dim)))
        B = self.Omega.take(list(range(3)),list(range(3)))
        C = self.Xi.take(list(range(3)),[0])
        self.Omega = Omega_p - (A.transpose() * B.inverse() * A)
        self.Xi = Xi_p - (A.transpose() * B.inverse() * C)
        # now Omega and Xi has a reduced dimension 
        self.dim -= 3
        
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
    control_motion_factor = 0.35                 # Error in motor control.
    control_turn_factor = 0.6                    # Additional error due to slip when turning.
    measurement_distance_stddev = 100.0           # Distance measurement error of cylinders (mm) 40
    measurement_angle_stddev = 10. / 180.0 * pi  # Angle measurement error- radian 10

    # Arbitrary start position.
    initial_state = array([500.0, 0.0, 45.0 / 180.0 * pi])
    #initial_state = array([0.0, 0.0, 0.0 / 180.0 * pi])

    # Setup filter.
    gph = OnlineGraphSLAM(initial_state, robot_width, scanner_displacement, control_motion_factor, \
                          control_turn_factor, measurement_distance_stddev,measurement_angle_stddev)
    

    # Read data.
    logfile = LegoLogfile()
    logfile.read("../in_data/robot4_motors.txt")
    logfile.read("../in_data/robot4_scan.txt")

    # Loop over all motor tick records and all measurements and generate
    # filtered positions and covariances.
    # This is the EKF SLAM loop.
    f = open("../out_data/graph_slam_correction.txt", "w")
    #for i in range(len(logfile.motor_ticks)-1):
    for i in range(100,100+100):
        # LK- note: In slam First observation will get no lmk match since start with empty map
        #           This is lmk association with the scans data at this tick location
        #           if matched -> return with lmk index in the system state, if no matched -> -1
        # observation = [(range, bearing),(lmk x world,lmk y world),(lmk x scanner, lmk y scanner),(lmk index)]
        observations = get_observations(logfile.scan_data[i],depth_jump, minimum_valid_distance, cylinder_offset,gph, max_cylinder_distance)
        gph.measurement_update(observations)
        #print("Omega: ", gph.Omega)
        #print("Xi: ", gph.Xi)
        #print("mu ", gph.Omega.inverse() * gph.Xi)
    
        
        # get motion command dx,dy,dtheta from odometry
        control = array(logfile.motor_ticks[i+1]) * ticks_to_mm
        # Update motion information
        gph.motion_update(control)
        
        
        # Update Graph
        gph.update_onlineslam()

        # compute graph    
        #mu = gph.solve_onlineslam()
        
        #print("n lmk: ", gph.number_of_landmarks)
        #print("Omega: ", gph.Omega)
        #print("Xi: ", gph.Xi)
        #print("mu: ", mu)
        
        '''
        # Save result
           
        # End of Graph SLAM - from here on, data is written.
        # Output the center of the scanner, not the center of the robot.
        xs = mu.value[0][0] + scanner_displacement * cos(gph.state[2])
        ys = mu.value[1][0] + scanner_displacement * sin(gph.state[2])
        ts = mu.value[2][0]
        
        print("F %f %f %f" %tuple([xs, ys, ts]), end = " ", file=f)
        print(file=f)
        # Write estimates of landmarks.
        write_cylinders(f, "W C", gph.get_landmarks())
        # Write cylinders detected by the scanner.
        write_cylinders(f, "D C", [(obs[2][0], obs[2][1])
                                   for obs in observations])
            
        '''
    f.close()
    mu = gph.solve_onlineslam()
    print("mu: ", mu)
    
    
    