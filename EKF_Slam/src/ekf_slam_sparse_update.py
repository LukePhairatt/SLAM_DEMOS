# EKF SLAM - prediction, landmark assignment and correction, landmark initialisation
# Work based on  Claus Brenner, and Joan Sola tutorials
# Punnu Phairatt

import time
from math import sin, cos, pi, atan2, sqrt
from numpy import *
from lib.logfile_reader import *
from lib.slam_ekf_library import get_observations_ML, write_cylinders,\
     write_error_ellipses


'''
This class performs EKF slam with sparse matrix update and maximum likelihood
measurement-landmark association, and a new landmark uncertainty init.
'''
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
        # EQUATION #
        # covariance' = G * covariance * GT + R
        #              where R = V * (control covariance or noise N) * VT.
        #              Covariance in control space depends on move distance.
        # NOTE: 
        # state = [x,y,heading,xl1,yl1,xl2,yl2....]
        # G = [G3  0  0            R = [R3 0 0       matrix 3+2n by 3+2n
        #       0  1  0                  0 0 0       n = number of landmarks 
        #       0  0  1]                 0 0 0]      2n = xl and yl states
        
        G3 = self.dg_dstate(self.state, control, self.robot_width)
        left, right = control
        left_var = (self.control_motion_factor * left)**2 +\
                   (self.control_turn_factor * (left-right))**2
        right_var = (self.control_motion_factor * right)**2 +\
                    (self.control_turn_factor * (left-right))**2
        control_covariance = diag([left_var, right_var])
        V = self.dg_dcontrol(self.state, control, self.robot_width)
        R3 = dot(V, dot(control_covariance, V.T))

        # Now enlarge G3 and R3 to accomodate all landmarks. Then, compute the  
        n = self.number_of_landmarks
        
        # Modification to the original codes
        # new PREDICTED covariance matrix self.covariance
        # ---------------------------------------------------------
        # Calculating Sparse state covariance matrix 
        # ---------------------------------------------------------
        # Covariance P = [Prr Prm]
        #                [Pmr Pmm]
        # Prr_predict = dg_dstate * Prr * dg_dstate(T) + dg_dcontrol * N * dg_dcontrol(T)
        # Prm_predict = dg_dstate * Prm
        # Pmr_predict = Prm_predict(T)
        # Pmm_predict = Pmm
        r = arange(0,3)          #[0,1,2]
        m = arange(3,3+2*n)      #[3,4,5,6...]
        # Using numpy ix_ built-in for slicing 2d array
        Prr = self.covariance[ix_(r,r)]
        Prm = self.covariance[ix_(r,m)]     
        Prr_predict = dot(G3, dot(Prr, G3.T)) + R3
        Prm_predict = dot(G3, Prm)
        Pmr_predict = Prm_predict.T
        # Update sparse covariance
        self.covariance[ix_(r,r)] = Prr_predict
        self.covariance[ix_(r,m)] = Prm_predict 
        self.covariance[ix_(m,r)] = Pmr_predict
        # No change to Pmm here - this will be change at the correction step
        # state' = g(state, control)
        # only compute a predicted robot state x,y,heading -> g only take a robot state 1x3
        self.state[0:3] = self.g(self.state[0:3], control, self.robot_width) 

    
    def add_landmark_to_state_jacobian(self, initial_coords, measurement):
        """Enlarge the current state and covariance matrix to include one more
           landmark, which is given by its initial_coords (an (x, y) tuple).
           Returns the index of the newly added landmark.
           
           Punnu Phairatt improvement
           Before, we initialise a new landmark uncertainty with a big number e.g. 1e10
           This version will compute landmark covariance from the robot,
           measurement, and other landmark uncertainty
        """
        print("Add lmk")
        # Augment the robot's state and covariance matrix.
        # Initialize the state with the given initial_coords and the
        # covariance with 1e10 (as an approximation for "infinity".
        
        # STATE- add lmk info to state 
        self.state = append(self.state, array(initial_coords))

        # COVARIANCE- add lmk info to covariance
        # This is the inverse observation model
        # Pll = GR * Prr * GR.T + Gy_new * Q * Gy_new.T
        # Plx = GR * [Prr Prm] = GR * Prx
        r = measurement[0]                   # measurement range
        theta = measurement[1]               # measurement bearing
        heading = self.state[2]              # vehicle heading
        dx = self.scanner_displacement       # scaner offset x
        dy = 0.0                             # scaner offset y
        beta = 0.0                           # scnner offset angle
        
        # lmk jacobian wrt to robot x,y,heading state
        GR = array([[1.0, 0.0, -dx*sin(heading)-dy*sin(heading)-r*sin(theta+heading+beta)],
                    [0.0, 1.0, -dy*sin(heading)+dx*cos(heading)+r*cos(theta+heading+beta)]])
        
        # lmk jacobian wrt to measurement r,bearing state 
        Gy_new =  array([[cos(theta+heading+beta), -r*sin(theta+heading+beta)],
                         [sin(theta+heading+beta),  r*cos(theta+heading+beta)]])
        
        
        # measurement noise in range, bearing
        Q = diag([self.measurement_distance_stddev**2, self.measurement_angle_stddev**2])
        
        lmk_n = int(self.number_of_landmarks)                    # Current number of landmarks in the state
        r = arange(0,3)                                          #e.g. [0,1,2]
        m = arange(3,3+2*lmk_n)                                  #e.g. [3,4,5,6...end-1,end]
        rm = concatenate((r,m))                                  #e.g. [0,1,2,3,4,5,6,7,8,., end-1, end]
        
        Prr = self.covariance[ix_(r,r)]
        Prx = self.covariance[ix_(r,rm)]
        # Compute the inverse observation
        Pll = dot(GR, dot(Prr,GR.T)) + dot(Gy_new, dot(Q,Gy_new.T)) #covariance lmk (2x2)
        Plx = dot(GR, Prx)                                          #covariance lmk (2x3+2*lmk)
        Pxl = Plx.T
        
        # Augment state and covariance
        new_lmk = int(len(initial_coords)/2)                                            #(x,y)
        m = int(self.number_of_landmarks)                                               #lmk num before add
        n = int(self.number_of_landmarks+new_lmk)						                #lmk num after add
        index_old  = 3+2*m                     	 					                    #current cov data index
        index_new  = 3+2*n                                  				            #end of index
        cov_holder = zeros((3+2*n, 3+2*n))                   				            #create new size holder
        cov_holder[0:index_old, 0:index_old] = self.covariance[0:index_old,0:index_old]	#copy old covariance data
        cov_holder[index_old:index_new,index_old:index_new] = Pll              	        #add new covariance lmk
        cov_holder[index_old:index_new,0:index_old] = Plx              	                #add new covariance lmk
        cov_holder[0:index_old,index_old:index_new] = Pxl            	                #add new covariance lmk
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
        # LK - Landmark index 3+2*index 
        #      [x,y,h,lmk_x0,lmk_y0,lmk_x1,lmk_y1........]
        #      [0,1,2,3     ,4	   ,5     ,6     ........]        
        #             index 0       index 1  
        
        # INIT- lmk  and index
        self.number_of_landmarks = n
        lmk_index = n-1
        return lmk_index

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
    
        
    def correct_ML(self, observations):
        """The correction step of the Kalman filter using
           Maximum likelihood
        """
        # EQUATION - Sparse matrix correction
        # -----------------------------------------------------
        # z(innovation) = y - h(x)
        # Z = [Hr Hl] [Prr Prl] [Hr.T]   + Q
        #             [Plr Pll] [Hl.T]
        # -----------------------------------------------------
        # K = [Prr Prl] [Hr.T]  * Z'
        #     [Pmr Pml] [Hl.T]
        # -----------------------------------------------------
        # x_update = x_predict + K*z
        # P_update = P_predict - K*Z*K.T
        # -----------------------------------------------------
        # Maximum likelihood of the landmark measurements
        # ML = z.T * Z' * z
        # -----------------------------------------------------
        # Get range and bearing measurement
        lmk_n = int(self.number_of_landmarks)                      # Current number of landmarks in the state
        r = arange(0,3)                                            #e.g. [0,1,2]
        m = arange(3,3+2*lmk_n)                                    #e.g. [3,4,5,6...end-1,end]
        new_lmks = len(observations) * [True]                      #tracking new landmarks for initialisation            
        for obs_indx, obs in enumerate(observations):
            measurement, cylinder_world, cylinder_scanner, cylinder_index = obs
            # check likelihood and correction against all current landmarks in the state
            ML_best = 9.0
            best_index = -1
            Hx_best = []
            Z_best  = []
            rm_best = []
            rl_best = []
            z_best  = []
            
            for landmark_index in range(lmk_n):
                # sparse matrix index
                l = arange(3+2*landmark_index, 3+2*landmark_index+2)     #e.g. [7,8]
                rl = concatenate((r,l))                                  #e.g. [0,1,2,7,8]
                rm = concatenate((r,m))                                  #e.g. [0,1,2,3,4,5,6,7,8,., end-1, end]
                # Get (x_m, y_m) of the landmark from the state vector.
                landmark = self.state[3+2*landmark_index : 3+2*landmark_index+2]
                z = array(measurement) - self.h(self.state, landmark, self.scanner_displacement)
                z[1] = (z[1] + pi) % (2*pi) - pi
                Hr = self.dh_dstate(self.state, landmark, self.scanner_displacement)
                Hl = -Hr[0:2, 0:2]
                Hx = concatenate((Hr, Hl), axis=1)
                Q = diag([self.measurement_distance_stddev**2, self.measurement_angle_stddev**2])
                Z = dot(Hx, dot(self.covariance[ix_(rl,rl)], Hx.T)) + Q
                # Likelihoood- find the best match
                ML = dot(z.T, dot(linalg.inv(Z), z) )
                #print ML, z
                if ML < ML_best:
                    ML_best = ML
                    Hx_best = Hx  
                    Z_best  = Z   
                    rm_best = rm
                    rl_best = rl
                    z_best  = z
                    best_index = landmark_index
                    
            # correct with the best likelihood        
            if best_index > -1:
                # this observation matchs this landmark i, do the correction
                K = dot( self.covariance[ix_(rm_best,rl_best)], dot(Hx_best.T, linalg.inv(Z_best)  ))
                # LK the robot x,y, heading and lmk x,y states as well as Covariance are updated using std the Kalman filter
                self.state = self.state + dot(K, z_best)
                self.covariance = self.covariance - dot(K, dot(Z_best, K.T))
                # not a new landmark on this observation
                new_lmks[obs_indx] = False
                
        # init new land mark from the observation list
        for indx, new_lmk in enumerate(new_lmks):  
            if new_lmk == True:
                measurement, cylinder_world, cylinder_scanner, cylinder_index = observations[indx]
                self.add_landmark_to_state_jacobian(cylinder_world, measurement)

    def get_landmarks(self):
        """Returns a list of (x, y) tuples of all landmark positions."""
        return ([(self.state[3+2*j], self.state[3+2*j+1])
                 for j in range( int(self.number_of_landmarks) )])

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
    f = open("../out_data/ekf_slam_correction_sp.txt", "w")
    for i in range(len(logfile.motor_ticks)):
    #for i in xrange(5):
        # Prediction.
        control = array(logfile.motor_ticks[i]) * ticks_to_mm
        kf.predict(control)

        # Correction- tradition only range diff
        # LK- note: In slam First observation will get no lmk match since start with empty map
        #           get observation return (range, bearing) (lmk x world,lmk y world) (lmk x scanner, lmk y scanner) (lmk index)
        #           This is lmk association with the scans data at this tick location
        #           if matched -> return with lmk index in the system state, if no matched -> -1  
        
        observations = get_observations_ML(logfile.scan_data[i],depth_jump, minimum_valid_distance, cylinder_offset,kf, max_cylinder_distance)
           
        # Correction- using likelihood association
        kf.correct_ML(observations)
 
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
