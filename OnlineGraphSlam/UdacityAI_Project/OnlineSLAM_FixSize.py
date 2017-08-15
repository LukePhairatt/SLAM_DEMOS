# NO WORKING!
# Correct Algorithm run but the solution check original dimension size


# ------------
# User Instructions
#
# In this problem you will implement a more manageable
# version of graph SLAM in 2 dimensions. 
#
# Define a function, online_slam, that takes 5 inputs:
# data, N, num_landmarks, motion_noise, and
# measurement_noise--just as was done in the last 
# programming assignment of unit 6. This function
# must return TWO matrices, mu and the final Omega.
#
# Just as with the quiz, your matrices should have x
# and y interlaced, so if there were two poses and 2
# landmarks, mu would look like:
#
# mu = matrix([[Px0],
#              [Py0],
#              [Px1],
#              [Py1],
#              [Lx0],
#              [Ly0],
#              [Lx1],
#              [Ly1]])
#
# Enter your code at line 566.

# -----------
# Testing
#
# You have two methods for testing your code.
#
# 1) You can make your own data with the make_data
#    function. Then you can run it through the
#    provided slam routine and check to see that your
#    online_slam function gives the same estimated
#    final robot pose and landmark positions.
# 2) You can use the solution_check function at the
#    bottom of this document to check your code
#    for the two provided test cases. The grading
#    will be almost identical to this function, so
#    if you pass both test cases, you should be
#    marked correct on the homework.

from UdcMatrix import *
from Robot import *
from math import *
import random


# print the result of SLAM, the robot pose(s) and the landmarks
def print_result(N, num_landmarks, result):
    print
    print 'Estimated Pose(s):'
    for i in range(N):
        print '    ['+ ', '.join('%.3f'%x for x in result.value[2*i]) + ', ' \
            + ', '.join('%.3f'%x for x in result.value[2*i+1]) +']'
    print 
    print 'Estimated Landmarks:'
    for i in range(num_landmarks):
        print '    ['+ ', '.join('%.3f'%x for x in result.value[2*(N+i)]) + ', ' \
            + ', '.join('%.3f'%x for x in result.value[2*(N+i)+1]) +']'
        

# compare with given the result
def solution_check(result, answer_mu, answer_omega):

    if len(result) != 2:
        print "Your function must return TWO matrices, mu and Omega"
        return False
    
    user_mu = result[0]
    user_omega = result[1]
    
    if user_mu.dimx == answer_omega.dimx and user_mu.dimy == answer_omega.dimy:
        print "It looks like you returned your results in the wrong order. Make sure to return mu then Omega."
        return False
    
    if user_mu.dimx != answer_mu.dimx or user_mu.dimy != answer_mu.dimy:
        print "Your mu matrix doesn't have the correct dimensions. Mu should be a", answer_mu.dimx, " x ", answer_mu.dimy, "matrix."
        return False
    else:
        print "Mu has correct dimensions."
        
    if user_omega.dimx != answer_omega.dimx or user_omega.dimy != answer_omega.dimy:
        print "Your Omega matrix doesn't have the correct dimensions. Omega should be a", answer_omega.dimx, " x ", answer_omega.dimy, "matrix."
        return False
    else:
        print "Omega has correct dimensions."
        
    if user_mu != answer_mu:
        print "Mu has incorrect entries."
        return False
    else:
        print "Mu correct."
        
    if user_omega != answer_omega:
        print "Omega has incorrect entries."
        return False
    else:
        print "Omega correct."
        
    print "Test case passed!"
    return True


# this routine makes the robot data
def make_data(N, num_landmarks, world_size, measurement_range, motion_noise, 
              measurement_noise, distance):

    complete = False
    while not complete:
        data = []
        # make robot and landmarks
        r = robot(world_size, measurement_range, motion_noise, measurement_noise)
        r.make_landmarks(num_landmarks)
        seen = [False for row in range(num_landmarks)]
    
        # guess an initial motion
        orientation = random.random() * 2.0 * pi
        dx = cos(orientation) * distance
        dy = sin(orientation) * distance
    
        for k in range(N-1):
    
            # sense
            Z = r.sense()

            # check off all landmarks that were observed 
            for i in range(len(Z)):
                seen[Z[i][0]] = True
    
            # move
            while not r.move(dx, dy):
                # if we'd be leaving the robot world, pick instead a new direction
                orientation = random.random() * 2.0 * pi
                dx = cos(orientation) * distance
                dy = sin(orientation) * distance

            # memorize data
            data.append([Z, [dx, dy]])

        # we are done when all landmarks were observed; otherwise re-run
        complete = (sum(seen) == num_landmarks)

    print ' '
    print 'Landmarks: ', r.landmarks
    print r
    return data
    

# --------------------------------
#
# Online_slam - retains all landmarks but only most recent robot pose
# See the document for Online Graph Slam equations
# It separates X,Y positions and landmarks 

def online_slam(data, N, num_landmarks, motion_noise, measurement_noise):    
    #    - My Matrix separation approach
    #    - Online dimension = M landmark + 1 position only
    dim = 1 + num_landmarks
    
    # init 1 element matrix
    Omega = matrix([[1.0]])             # X0 constraint
    Xi = matrix([[world_size / 2.0]])   # X0 position
    Yi = matrix([[world_size / 2.0]])   # X0 position
    # matrix pre-allocation by expanding
    Omega = Omega.expand(dim,dim,[0],[0])
    Xi = Xi.expand(dim,1,[0],[0])
    Yi = Yi.expand(dim,1,[0],[0])
    
    
    # process the data and update measurement and then motion one by one
    for k in range(len(data)):
        # measurement data
        measurement = data[k][0]   #(lmk id, rx,ry)....
        
        # measurement update at this current location
        for i in range(len(measurement)):
            # m is the index of the landmark coordinate in the matrix/vector X or Y
            # start from 1 (online slam) after the current location
            m = 1 + measurement[i][0] 
            # update the information maxtrix between X/Y and a Landmark
            Omega.value[0][0] +=  1.0 / measurement_noise
            Omega.value[m][m] +=  1.0 / measurement_noise
            Omega.value[0][m] += -1.0 / measurement_noise
            Omega.value[m][0] += -1.0 / measurement_noise
            # update the information vector X
            Xi.value[0][0]    += -measurement[i][1] / measurement_noise
            Xi.value[m][0]    +=  measurement[i][1] / measurement_noise
            # update the information vector Y
            Yi.value[0][0]    += -measurement[i][2] / measurement_noise
            Yi.value[m][0]    +=  measurement[i][2] / measurement_noise
            
                         
        # motion update - Add a new row and column 
        motion      = data[k][1]   #(dx,dy)
        dx = motion[0]
        dy = motion[1]
        wm = 1./motion_noise
        # Expand matrix and vector to accomodate one step move e.g. (xi,yi) to (xi+1,yi+1) 
        # compute range to be extended and added
        expand_i = [0] + range(2,dim+1)
        #([0, 1+1...dim+1], [0, 1+1...dim+1])
        Omega = Omega.expand(dim+1,dim+1,expand_i,expand_i)
        Xi = Xi.expand(dim+1,1,expand_i,[0])
        Yi = Yi.expand(dim+1,1,expand_i,[0])
        
        # MATRIX
        # diag update e.g. (0,0) and (1,1)
        Omega.value[0][0]   +=  1.0 / motion_noise
        Omega.value[1][1]   +=  1.0 / motion_noise
        # diag update e.g. (0,1) and (1,0)
        Omega.value[0][1]   += -1.0 / motion_noise
        Omega.value[1][0]   += -1.0 / motion_noise
        
        # VECTOR - (0,0) and (1,0)
        Xi.value[0][0]      += -motion[0] / motion_noise
        Xi.value[1][0]      +=  motion[0] / motion_noise
        Yi.value[0][0]      += -motion[1] / motion_noise
        Yi.value[1][0]      +=  motion[1] / motion_noise
                    
        # Online matrix/vector transformation - reduce size to a current position after move      
        # Omega <- Omega_p - AT*B_inv*A
        # Xi <- Xi_p - AT*B_inv*C
        Omega_p = Omega.take(range(1,dim+1),range(1,dim+1))
        Xi_p = Xi.take(range(1,dim+1),[0])
        Yi_p = Yi.take(range(1,dim+1),[0])
        A  = Omega.take([0],range(1,dim+1))
        B  = Omega.take([0],[0])
        CX = Xi.take([0],[0])
        CY = Yi.take([0],[0])
        Omega = Omega_p - (A.transpose() * B.inverse() * A)
        Xi = Xi_p - (A.transpose() * B.inverse() * CX)
        Yi = Yi_p - (A.transpose() * B.inverse() * CY)
           
    
    # Compute best estimate of the robot and landmark positions
    mu_x = Omega.inverse() * Xi
    mu_y = Omega.inverse() * Yi
    # building mu
    # mu =  matrix([ [Px0],
    #                [Py0],
    #                [Px1],
    #                [Py1],
    #                [Lx0],
    #                [Ly0],
    #                [Lx1],
    #                [Ly1]])
    mu = matrix()
    mu.zero(2*dim,1)
    for i in range(dim):
        mu.value[2*i][0]   = mu_x.value[i][0]
        mu.value[2*i+1][0] = mu_y.value[i][0]
    
    return mu, Omega # make sure you return both of these matrices to be marked correct.



# ------------------------------------------------------------------------
# Main routines- Online Graph Slam Testing
# ------------------------------------------------------------------------

num_landmarks      = 5        # number of landmarks
N                  = 10       # time steps
world_size         = 100.0    # size of world
measurement_range  = 30.0     # range at which we can sense landmarks
motion_noise       = 2.0      # noise in robot motion
measurement_noise  = 2.0      # noise in the measurements
distance           = 5.0     # distance by which robot (intends to) move each iteration

# Uncomment the following three lines to run the online_slam routine.
data = make_data(N, num_landmarks, world_size, measurement_range, motion_noise, measurement_noise, distance)
result = online_slam(data, N, num_landmarks, motion_noise, measurement_noise)
print_result(1, num_landmarks, result[0])

# -----------
# Test Case 1

testdata1          = [[[[1, 21.796713239511305, 25.32184135169971], [2, 15.067410969755826, -27.599928007267906]], [16.4522379034509, -11.372065246394495]],
                      [[[1, 6.1286996178786755, 35.70844618389858], [2, -0.7470113490937167, -17.709326161950294]], [16.4522379034509, -11.372065246394495]],
                      [[[0, 16.305692184072235, -11.72765549112342], [2, -17.49244296888888, -5.371360408288514]], [16.4522379034509, -11.372065246394495]],
                      [[[0, -0.6443452578030207, -2.542378369361001], [2, -32.17857547483552, 6.778675958806988]], [-16.66697847355152, 11.054945886894709]]]

answer_mu1         = matrix([[81.63549976607898],
                             [27.175270706192254],
                             [98.09737507003692],
                             [14.556272940621195],
                             [71.97926631050574],
                             [75.07644206765099],
                             [65.30397603859097],
                             [22.150809430682695]])

answer_omega1      = matrix([[0.36603773584905663, -0.169811320754717, -0.011320754716981133, -0.1811320754716981],
                             [-0.169811320754717, 0.6509433962264151, -0.05660377358490567, -0.40566037735849064],
                             [-0.011320754716981133, -0.05660377358490567, 0.6962264150943396, -0.360377358490566],
                             [-0.1811320754716981, -0.4056603773584906, -0.360377358490566, 1.2339622641509433]])

result = online_slam(testdata1, 5, 3, 2.0, 2.0)
#print 'mu', result[0]
#print 'omega', result[1]
solution_check(result, answer_mu1, answer_omega1)

# -----------
# Test Case 2

testdata2          = [[[[0, 12.637647070797396, 17.45189715769647], [1, 10.432982633935133, -25.49437383412288]], [17.232472057089492, 10.150955955063045]],
                      [[[0, -4.104607680013634, 11.41471295488775], [1, -2.6421937245699176, -30.500310738397154]], [17.232472057089492, 10.150955955063045]],
                      [[[0, -27.157759429499166, -1.9907376178358271], [1, -23.19841267128686, -43.2248146183254]], [-17.10510363812527, 10.364141523975523]],
                      [[[0, -2.7880265859173763, -16.41914969572965], [1, -3.6771540967943794, -54.29943770172535]], [-17.10510363812527, 10.364141523975523]],
                      [[[0, 10.844236516370763, -27.19190207903398], [1, 14.728670653019343, -63.53743222490458]], [14.192077112147086, -14.09201714598981]]]

answer_mu2         = matrix([[63.37479912250136],
                             [78.17644539069596],
                             [61.33207502170053],
                             [67.10699675357239],
                             [62.57455560221361],
                             [27.042758786080363]])

answer_omega2      = matrix([[0.22871751620895048, -0.11351536555795691, -0.11351536555795691],
                             [-0.11351536555795691, 0.7867205207948973, -0.46327947920510265],
                             [-0.11351536555795691, -0.46327947920510265, 0.7867205207948973]])

result = online_slam(testdata2, 6, 2, 3.0, 4.0)
solution_check(result, answer_mu2, answer_omega2)



