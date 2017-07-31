# EKF SLAM - prediction, landmark assignment and correction.
# This file is based on Claus Brenner, 20 JAN 13 tutorial
# Punnu LK Phairatt

from math import sin, cos, pi, atan2, sqrt
from numpy import *
from lib.logfile_reader import *
from lib.slam_ekf_library import get_observations, write_cylinders,\
     write_error_ellipses

import time

class ExtendedKalmanFilterSLAM:
    def __init__(self, state, covariance,
                 robot_width, scanner_displacement,
                 control_motion_factor, control_turn_factor,
                 measurement_distance_stddev, measurement_angle_stddev):
        # The state. This is the core data of the Kalman filter.
        self.state = state
        self.covariance = covariance

        # Some constants.
        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = measurement_distance_stddev
        self.measurement_angle_stddev = measurement_angle_stddev

        # Currently, the number of landmarks is zero.
        self.number_of_landmarks = 0
        # Tracing landmark- start with empty array
        self.landmark_counter = array([])

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

    def predict(self, control):
        """The prediction step of the Kalman filter."""
        # covariance' = G * covariance * GT + R
        # where R = V * (covariance in control space) * VT.
        # Covariance in control space depends on move distance.

        # ---------------------------------------------------------
        # LK- This function only enlarge G and R 
        #     However, Covariance/System state will be enlarged 
        #     in add landmark function 
        # ---------------------------------------------------------
        G3 = self.dg_dstate(self.state, control, self.robot_width)
        left, right = control
        left_var = (self.control_motion_factor * left)**2 +\
                   (self.control_turn_factor * (left-right))**2
        right_var = (self.control_motion_factor * right)**2 +\
                    (self.control_turn_factor * (left-right))**2
        control_covariance = diag([left_var, right_var])
        V = self.dg_dcontrol(self.state, control, self.robot_width)
        R3 = dot(V, dot(control_covariance, V.T))
        # LK- Compute predicted state and covariance of FULL STATES
        # Hints:
        # - The number of landmarks is self.number_of_landmarks.
        # - eye(n) is the numpy function which returns a n x n identity matrix.
        # - zeros((n,n)) returns a n x n matrix which is all zero.
        # - If M is a matrix, M[0:2,1:5] returns the submatrix which consists
        #   of the rows 0 and 1 (but not 2) and the columns 1, 2, 3, 4.
        #   This submatrix operator can be used on either side of an assignment.
        # - Similarly for vectors: v[1:3] returns the vector consisting of the
        #   elements 1 and 2, but not 3.
        # - All matrix and vector indices start at 0.

        # Now enlarge G3 and R3 to accomodate all landmarks. Then, compute the
        # state = [x,y,heading,xl1,yl1,xl2,yl2....]
        # G = [G3  0  0            R = [R3 0 0       matrix 3+2n by 3+2n
        #       0  1  0                  0 0 0       n = number of landmarks 
        #       0  0  1]                 0 0 0]      2n = xl and yl states
        n = self.number_of_landmarks
        full_state = 3+2*n
        G = eye(full_state)
        R = zeros((full_state,full_state))
        G[0:3,0:3] = G3
        R[0:3,0:3] = R3
        # new covariance matrix self.covariance
        # self.covariance is 3*3 for a time being
        # will be extended to lankmarks in the next section
        self.covariance = dot(G, dot(self.covariance, G.T)) + R 
        # state' = g(state, control)
        # only compute a predicted robot state x,y,heading -> g only take a robot state 1x3
        self.state[0:3] = self.g(self.state[0:3], control, self.robot_width)
        

    def add_landmark_to_state(self, initial_coords):
        """Enlarge the current state and covariance matrix to include one more
           landmark, which is given by its initial_coords (an (x, y) tuple).
           Returns the index of the newly added landmark."""

        # Augment the robot's state and covariance matrix.
        # Initialize the state with the given initial_coords and the
        # covariance with 1e10 (as an approximation for "infinity".
        
        # STATE- add lmk info to state 
        self.state = append(self.state, array(initial_coords))

        # COVARIANCE- add lmk info to covariance
        new_lmk = int(len(initial_coords)/2)                                            #(x,y)
        cov_lmk = 1e10*eye(2*new_lmk)                                                   #double size 1 lmk = 2 states x,y
        m = int(self.number_of_landmarks)                                               #lmk num before add
        n = int(self.number_of_landmarks+new_lmk)						                #lmk num after add
        index_old  = 3+2*m                     	 					                    #current cov data index
        index_new  = 3+2*n                                  				            #end of index
        cov_holder = zeros((3+2*n, 3+2*n))                   				            #create new size holder
        cov_holder[0:index_old, 0:index_old] = self.covariance[0:index_old,0:index_old]	#copy old covariance data
        cov_holder[index_old:index_new,index_old:index_new] = cov_lmk              	    #add new covariance lmk
        self.covariance = cov_holder
        # Notes:
        # - If M is a matrix, use M[i:j,k:l] to obtain the submatrix of
        #   rows i to j-1 and colums k to l-1. This can be used on the left and
        #   right side of the assignment operator.
        # - zeros(n) gives a zero vector of length n, eye(n) an n x n identity
        #   matrix.
        # - Do not forget to increment self.number_of_landmarks.
        # - Do not forget to return the index of the newly added landmark. I.e.,
        #   the first call should return 0, the second should return 1.
        # - Landmark index 3+2*index 
        #   [x, y, h, lmk_x0,lmk_y0,lmk_x1,lmk_y1........]
        #   [0, 1, 2, 3,     4,     5,     6    ........]        
        #             index 0       index 1  
        # INIT- lmk  index
        self.number_of_landmarks = n
        lmk_index = n-1
        # Add new landmark to the end of a counter start with 1
        self.landmark_counter = append(self.landmark_counter,1)
        return lmk_index
        
    def remove_landmark_from_state(self, landmark_index):
        """
            Remove landmark from state and covariance
        """
        print("Remove landmark...")
        #check bound
        if landmark_index >=0 and landmark_index < self.number_of_landmarks:
            index = arange(3+2*landmark_index, 3+2*landmark_index+2)
            #remove landmark from state    
            self.state = delete(self.state, index)
            #remove landmark from covariance row and col
            self.covariance = delete(self.covariance, index, 0) #row
            self.covariance = delete(self.covariance, index, 1) #col
            #reduce landmark number
            self.number_of_landmarks = self.number_of_landmarks-1
            #remove landmark counter
            self.landmark_counter = delete(self.landmark_counter,landmark_index)
            return True
        
        return False

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
        drdx = -dx / sqrtq
        drdy = -dy / sqrtq
        drdtheta = (dx * sint - dy * cost) * scanner_displacement / sqrtq
        dalphadx =  dy / q
        dalphady = -dx / q
        dalphadtheta = -1 - scanner_displacement / q * (dx * cost + dy * sint)

        return array([[drdx, drdy, drdtheta],
                      [dalphadx, dalphady, dalphadtheta]])

    def correct(self, measurement, landmark_index):
        """The correction step of the Kalman filter."""
        # Get (x_m, y_m) of the landmark from the state vector.
        landmark = self.state[3+2*landmark_index : 3+2*landmark_index+2]
        H3 = self.dh_dstate(self.state, landmark, self.scanner_displacement)

        # --->>> Add your code here to set up the full H matrix.
        lmk_n = self.number_of_landmarks
        H = zeros((2, 3+2*lmk_n))
        H[0:2, 0:3] = H3
        H[0:2, 3+2*landmark_index:3+2*landmark_index+2] = -H3[0:2, 0:2]

        # This is the old code from the EKF - no modification necessary!
        Q = diag([self.measurement_distance_stddev**2, self.measurement_angle_stddev**2])
        K = dot(self.covariance, dot(H.T, linalg.inv(dot(H, dot(self.covariance, H.T)) + Q)))
        innovation = array(measurement) - self.h(self.state, landmark, self.scanner_displacement)
        innovation[1] = (innovation[1] + pi) % (2*pi) - pi
        # LK the robot x,y, heading and lmk x,y states as well as Covariance are updated using std the Kalman filter
 
        self.state = self.state + dot(K, innovation)
        self.covariance = dot(eye(size(self.state)) - dot(K, H), self.covariance)

    def get_landmarks(self):
        """Returns a list of (x, y) tuples of all landmark positions."""
        return ([(self.state[3+2*j], self.state[3+2*j+1])
                 for j in range( int(self.number_of_landmarks)  )])

    def get_landmark_error_ellipses(self):
        """Returns a list of all error ellipses, one for each landmark."""
        ellipses = []
        for i in range( int(self.number_of_landmarks) ):
            j = 3 + 2 * i
            ellipses.append(self.get_error_ellipse(
                self.covariance[j:j+2, j:j+2]))
        return ellipses

    @staticmethod
    def get_error_ellipse(covariance):
        """Return the position covariance (which is the upper 2x2 submatrix)
           as a triple: (main_axis_angle, stddev_1, stddev_2), where
           main_axis_angle is the angle (pointing direction) of the main axis,
           along which the standard deviation is stddev_1, and stddev_2 is the
           standard deviation along the other (orthogonal) axis."""
        eigenvals, eigenvects = linalg.eig(covariance[0:2,0:2])
        angle = atan2(eigenvects[1,0], eigenvects[0,0])
        return (angle, sqrt(eigenvals[0]), sqrt(eigenvals[1]))        


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
    measurement_distance_stddev = 600.0          # Distance measurement error of cylinders (mm)
    measurement_angle_stddev = 45. / 180.0 * pi  # Angle measurement error- radian

    # Arbitrary start position.
    initial_state = array([500.0, 0.0, 45.0 / 180.0 * pi])

    # Covariance at start position.
    initial_covariance = zeros((3,3))

    # Setup filter.
    kf = ExtendedKalmanFilterSLAM(initial_state, initial_covariance,
                                  robot_width, scanner_displacement,
                                  control_motion_factor, control_turn_factor,
                                  measurement_distance_stddev,
                                  measurement_angle_stddev)

    # Read data.
    logfile = LegoLogfile()
    logfile.read("../in_data/robot4_motors.txt")
    logfile.read("../in_data/robot4_scan.txt")

    # Loop over all motor tick records and all measurements and generate
    # filtered positions and covariances.
    # This is the EKF SLAM loop.
    f = open("../out_data/ekf_slam_correction_full.txt", "w")
    for i in range(len(logfile.motor_ticks)):

        # Cleanup not good landmark after sometime (landmark counter < threshold) e.g. after 50 steps
        # If we have not seen this landmark any more (it doesn't increase for a while then we remove it from the state)
        # This is useful when the map is growing larger and need to clear some memory space
        # Check landmark last active
        '''
        if i != 0 and not(i%100):
            for lmk_indx in xrange(len(active_counter)):
                if active_counter[lmk_indx] < 20:
                    #remove in active landmarks
                    kf.remove_landmark_from_state(lmk_indx)
                    
        ''' 
        # Prediction.
        control = array(logfile.motor_ticks[i]) * ticks_to_mm
        kf.predict(control)

        # Correction.
        # LK- note: In slam First observation will get no lmk match since start with empty map
        #           get observation return (range, bearing) (lmk x world,lmk y world) (lmk x scanner, lmk y scanner) (lmk index)
        #           This is lmk association with the scans data at this tick location
        #           if matched -> return with lmk index in the system state, if no matched -> -1
        
        observations = get_observations(
            		logfile.scan_data[i],
            		depth_jump, minimum_valid_distance, cylinder_offset,
            		kf, max_cylinder_distance)
        
        for obs in observations:
            measurement, cylinder_world, cylinder_scanner, cylinder_index = obs
            if cylinder_index == -1:
                cylinder_index = kf.add_landmark_to_state(cylinder_world)
            else:
                if cylinder_index < len(kf.landmark_counter):
                    kf.landmark_counter[cylinder_index] += 1
                    
            #Only do the correction after a number of times this landmark has been observed
            if kf.landmark_counter[cylinder_index] > 2:
                kf.correct(measurement, cylinder_index)
            
        # End of EKF SLAM - from here on, data is written.
        # Output the center of the scanner, not the center of the robot.
        print("F %f %f %f" % \
            tuple(kf.state[0:3] + [scanner_displacement * cos(kf.state[2]),
                                   scanner_displacement * sin(kf.state[2]),
                                   0.0]), end = " ", file=f)
        
        print(file=f)
        # Write covariance matrix in angle stddev1 stddev2 stddev-heading form.
        # e = (main axis angle, main axis size, minor axis size)
        # get_error_ellipse only extract vehicle state covariance FROM a full state covariance (position, heading, landmarks)
        e = ExtendedKalmanFilterSLAM.get_error_ellipse(kf.covariance)
        # LK tuple concat short cut (angle, major minor)+(heading,) -> (angle, major minor, heading cov)
        print( "E %f %f %f %f" % (e + ( sqrt(kf.covariance[2,2]), )),end=" ", file=f   )
        print(file=f)
        # Write estimates of landmarks.
        write_cylinders(f, "W C", kf.get_landmarks())
        # Write error ellipses of landmarks.
        write_error_ellipses(f, "W E", kf.get_landmark_error_ellipses())
        # Write cylinders detected by the scanner.
        write_cylinders(f, "D C", [(obs[2][0], obs[2][1])
                                   for obs in observations])

    f.close()
    
    
    