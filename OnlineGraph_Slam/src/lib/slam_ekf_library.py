# This file contains helper functions for Unit D of the SLAM lecture,
# most of which were developed in earlier units.
# Claus Brenner, 11 DEC 2012
from math import sin, cos, pi, atan2
from lib.logfile_reader import LegoLogfile
from numpy import *

# Utility to write a list of cylinders to (one line of) a given file.
# Line header defines the start of each line, e.g. "D C" for a detected
# cylinder or "W C" for a world cylinder.
def write_cylinders(file_desc, line_header, cylinder_list):
    print(line_header, end = " ", file=file_desc)
    for c in cylinder_list:
        print("%.1f %.1f" % tuple(c), end = " ", file=file_desc)
    print(file=file_desc)

    
# Utility to write a list of error ellipses to (one line of) a given file.
# Line header defines the start of each line.
def write_error_ellipses(file_desc, line_header, covariance_matrix_list):
    print(line_header, end=" ", file=file_desc)
    for m in covariance_matrix_list:
        eigenvals, eigenvects = linalg.eig(m)
        angle = atan2(eigenvects[1,0], eigenvects[0,0])
        print("%.3f %.1f %.1f" % (angle, sqrt(eigenvals[0]), sqrt(eigenvals[1])), end=" ", file=file_desc)
    print(file=file_desc)

def write_error_ellipses(file_desc, line_header, error_ellipse_list):
    print(line_header, end=" ", file=file_desc)
    for e in error_ellipse_list:
        print("%.3f %.1f %.1f" % e, end=" ", file=file_desc)
    print(file=file_desc)

# Find the derivative in scan data, ignoring invalid measurements.
def compute_derivative(scan, min_dist):
    jumps = [ 0 ]
    for i in range(1, len(scan) - 1):
        l = scan[i-1]
        r = scan[i+1]
        if l > min_dist and r > min_dist:
            derivative = (r - l) / 2.0
            jumps.append(derivative)
        else:
            jumps.append(0)
    jumps.append(0)
    return jumps

# Cylinders as landmark features
# For each area between a left falling edge and a right rising edge,
# determine the average ray number and the average depth.
def find_cylinders(scan, scan_derivative, jump, min_dist):
    cylinder_list = []
    on_cylinder = False
    sum_ray, sum_depth, rays = 0.0, 0.0, 0

    for i in range(len(scan_derivative)):
        if scan_derivative[i] < -jump:
            # Start a new cylinder, independent of on_cylinder.
            on_cylinder = True
            sum_ray, sum_depth, rays = 0.0, 0.0, 0
        elif scan_derivative[i] > jump:
            # Save cylinder if there was one.
            if on_cylinder and rays:
                cylinder_list.append((sum_ray/rays, sum_depth/rays))
            on_cylinder = False
        # Always add point, if it is a valid measurement.
        elif scan[i] > min_dist:
            sum_ray += i
            sum_depth += scan[i]
            rays += 1
    return cylinder_list


# Wall landmark as RANSAC 
# This function finds wall lmks by projecting a point to lines extracted using RANSAC
def find_lmkwalls(i_Ransac, model, measurement, scanner_pose, landmark_point, line_search):
    # --------------- Formating measurement data for a RANSAC function ------------------- #
    shape = (len(measurement),1)
    angle = zeros( shape ) 
    distance = zeros( shape )
    xm = zeros( shape ) 
    ym = zeros( shape )
    ix = arange( shape[0] ).reshape(shape)
    for i,c in enumerate(measurement):
        angle[i] = LegoLogfile.beam_index_to_angle(i)
        distance[i] = c
        # Compute x, y data in world coordinates.
        xs, ys = distance[i]*cos(angle[i]), distance[i]*sin(angle[i])
        xm[i], ym[i] = LegoLogfile.scanner_to_world(scanner_pose, (xs, ys))
        
    # Scope the line within this min-max box (only for visualisation) 
    xmin = min(xm)
    xmax = max(xm)
    ymin = min(ym)
    ymax = max(ym)
    xy_box = array([xmin,xmax,ymin,ymax])
    
    # --------------- RANSAC- finding walls (fitting lines) ------------------------------- #
    arena_data = hstack( (ix,xm,ym,distance,angle) )
    uncheck_data = copy(arena_data)
    iteration = 0
    minimum_size = i_Ransac.d
    line_index = []
    line_fit   = []
    data_size = len(uncheck_data)
    while(iteration < line_search and data_size > minimum_size):
        #print '---------------------',iteration,'---------------------------'
        data_size = len(uncheck_data)
        #2#3#4#5 run RANSAC loop
        ransac_fit, ransac_index = i_Ransac.run(uncheck_data, model, arena_data)    # run ransac line fit
        #6#7
        if ransac_fit != None:
            line_index.append(ransac_index)                                         # save found line index
            line_fit.append(ransac_fit)
            uncheck_data = delete(uncheck_data, ransac_index, axis=0)               # delete checked data
        iteration+=1
    
    # --------------------------- Filter lines and get intersection point -------------------- #
    # Check Ransac found any lines before running this
    if(len(line_fit)>0):
        # Combine similar lines to a single one with similarity threshold and within bounding box min/max
        # TODO- passing line threshold configuration
        model_lines, x_points = model.get_lines(line_fit, xmin, xmax, ymin, ymax)
        #print 'Found %d lines',len(model_lines) 
    else:
        return 0,0,0,0
    
    # ---------------------------- Project a landmark point ----------------------------------- #
    projected_p = model.get_projected_point(model_lines, landmark_point)
        
    return model_lines, projected_p, xy_box, arena_data

# This function does all processing needed to obtain the cylinder observations.
# It matches the cylinders and returns distance and angle observations together
# with the cylinder coordinates in the world system, the scanner
# system, and the corresponding cylinder index (in the list of estimated parameters).
# In detail:
# - It takes scan data and detects cylinders.
# - For every detected cylinder, it computes its world coordinate using
#   the polar coordinates from the cylinder detection and the robot's pose,
#   taking into account the scanner's displacement.
# - Using the world coordinate, it finds the closest cylinder in the
#   list of current (estimated) landmarks, which are part of the current state.
#   
# - If there is such a closest cylinder, the (distance, angle) pair from the
#   scan measurement (these are the two observations), the (x, y) world
#   coordinates of the cylinder as determined by the measurement, the (x, y)
#   coordinates of the same cylinder in the scanner's coordinate system,
#   and the index of the matched cylinder are added to the output list.
#   The index is the cylinder number in the robot's current state.
# - If there is no matching cylinder, the returned index will be -1.
def get_observations(scan, jump, min_dist, cylinder_offset,
                     robot, max_cylinder_distance):
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    # Compute scanner pose from robot pose.
    scanner_pose = (
        robot.state[0] + cos(robot.state[2]) * robot.scanner_displacement,
        robot.state[1] + sin(robot.state[2]) * robot.scanner_displacement,
        robot.state[2])

    # For every detected cylinder which has a closest matching pole in the
    # cylinders that are part of the current state, put the measurement
    # (distance, angle) and the corresponding cylinder index into the result list.
    result = []
    for c in cylinders:
        # Compute the angle and distance measurements.
        angle = LegoLogfile.beam_index_to_angle(c[0])                      # bearing angle
        distance = c[1] + cylinder_offset                                  # range 
        # Compute x, y of cylinder in world coordinates.
        xs, ys = distance*cos(angle), distance*sin(angle)                  # lmks in a scanner frame
        x, y = LegoLogfile.scanner_to_world(scanner_pose, (xs, ys))        # lmks in a world frame
        # Find closest cylinder in the state.
        best_dist_2 = max_cylinder_distance * max_cylinder_distance
        best_index = -1
        for index in range(int(robot.number_of_landmarks)):
            pole_x, pole_y = robot.state[3+2*index : 3+2*index+2]
            dx, dy = pole_x - x, pole_y - y
            dist_2 = dx * dx + dy * dy
            if dist_2 < best_dist_2:
                best_dist_2 = dist_2
                best_index = index
        # Always add result to list. Note best_index may be -1.
        result.append(((distance, angle), (x, y), (xs, ys), best_index))

    return result


# - This is my implemenntation of sparse matrix and maximum likelihood association
def get_observations_ML(scan, jump, min_dist, cylinder_offset, robot, max_cylinder_distance):
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    # Compute scanner pose from robot pose.
    scanner_pose = (
        robot.state[0] + cos(robot.state[2]) * robot.scanner_displacement,
        robot.state[1] + sin(robot.state[2]) * robot.scanner_displacement,
        robot.state[2])

    # For every detected cylinder which has a closest matching pole in the
    # cylinders that are part of the current state, put the measurement
    # (distance, angle) and the corresponding cylinder index into the result list.
    result = []
    for c in cylinders:
        # Compute the angle and distance measurements.
        angle = LegoLogfile.beam_index_to_angle(c[0])
        distance = c[1] + cylinder_offset
        # Compute x, y of cylinder in world coordinates.
        xs, ys = distance*cos(angle), distance*sin(angle)
        x, y = LegoLogfile.scanner_to_world(scanner_pose, (xs, ys))
        # Find closest cylinder in the state.
        best_index = -1
        best_ML    =  9.0                                            #Maximum likelihood threshold 
        measurement = array([distance, angle])
        for index in range(int(robot.number_of_landmarks)): 
            # using likelihood
            lmk_n = robot.number_of_landmarks                        #Current number of landmarks in the state
            r = arange(0,3)                                          #robot state e.g. [0,1,2]
            l = arange(3+2*index, 3+2*index+2)                       #landmark state in pairs e.g. [7,8] #index 0,1,2 ....n
            m = arange(3,3+2*lmk_n)                                  #landmark map state e.g. [3,4,5,6...end-1,end]
                                                                     
            rl = concatenate((r,l))                                  #robot and landmark l co-matrix e.g. [0,1,2,7,8]
            rm = concatenate((r,m))                                  #robot and map co-matrix e.g. [0,1,2,3,4,5,6,7,8,., end-1, end]
            
            # Get (x_m, y_m) of the landmark from the state vector.
            landmark = robot.state[3+2*index : 3+2*index+2]
            innovation = array(measurement) - robot.h(robot.state, landmark, robot.scanner_displacement)
            innovation[1] = (innovation[1] + pi) % (2*pi) - pi
            
            Hr = robot.dh_dstate(robot.state, landmark, robot.scanner_displacement)
            Hl = -Hr[0:2, 0:2]
            Hx = concatenate((Hr, Hl), axis=1)
            Q = diag([robot.measurement_distance_stddev**2, robot.measurement_angle_stddev**2])
            Z = dot(Hx, dot(robot.covariance[ix_(rl,rl)], Hx.T)) + Q
            # Do the correction only if Maximum likelihood < threshold
            # ML = z.T * Z' * z
            ML = dot(innovation.T, dot(linalg.inv(Z), innovation) )
            if ML < best_ML:
                best_ML = ML
                best_index = index
                
        # Always add result to list. Note best_index may be -1.
        result.append( ((distance, angle), (x, y), (xs, ys), best_index) )

    return result