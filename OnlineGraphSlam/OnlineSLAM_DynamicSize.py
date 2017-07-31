# ------------
# Quick notes

# Experiment with Dynamic array size extension for each motion and landmark update
# Simulate data has been generated with landmark is observed in order to run with this programme.

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
# -----------
# Testing
#    You can use the solution_check function at the
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

# --------------------------------------------------------------------------
# online_slam - retains all landmarks but only most recent robot pose
# See the document for Online Graph Slam equations
# It separates X,Y positions and landmarks
# Dynamic array size extension for each landmark and motion update 
# --------------------------------------------------------------------------

def online_slam(data, N, num_landmarks, motion_noise, measurement_noise):    
    # init 1 element matrix
    Omega = matrix([[1.0]])             # X0 constraint
    Xi = matrix([[world_size / 2.0]])   # X0 position
    Yi = matrix([[world_size / 2.0]])   # X0 position
    # initial matrix size
    dim = 1
    # create an empty list of landmarks
    landmark_list = []
    # process the data and update measurement and then motion one by one
    for k in range(len(data)):
        # measurement data
        measurement = data[k][0]   #(lmk id, rx,ry)....
        
        # measurement update at this current location
        for i in range(len(measurement)):
            # find landmark association
            lmk_index = measurement[i][0]
            # check if the new landmark
            if not lmk_index in landmark_list:
                # new one - need to expand matrix/vector at the end for a landmark before updating this motion and landmark
                dim += 1
                # record this landmark in the list
                landmark_list.append(lmk_index)
                # extend the matrix and vector
                Omega = Omega.expand(dim,dim,range(dim-1),range(dim-1))
                Xi = Xi.expand(dim,1,range(dim-1),[0])
                Yi = Yi.expand(dim,1,range(dim-1),[0])
                
            
            # let update
            # m is the landmark index in the matrix/vector X or Y
            # start from 1 (online slam) after the current location
            m = 1 + lmk_index 
            # update the information maxtrix between the current location and this Landmark
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

world_size         = 100.0    # size of world (don't chnage- it has been used to generate these simulate data)

# My test case edit- reorder landmark index to suit dynamic expansion of online slam
testdata0          = [[[[0, 21.796713239511305, 25.32184135169971], [1, 15.067410969755826, -27.599928007267906]], [16.4522379034509, -11.372065246394495]],
                      [[[0, 6.1286996178786755, 35.70844618389858], [1, -0.7470113490937167, -17.709326161950294]], [16.4522379034509, -11.372065246394495]],
                      [[[1, -17.49244296888888, -5.371360408288514],[2, 16.305692184072235, -11.72765549112342]], [16.4522379034509, -11.372065246394495]],
                      [[[1, -32.17857547483552, 6.778675958806988], [2, -0.6443452578030207, -2.542378369361001]], [-16.66697847355152, 11.054945886894709]]]


answer_mu0 = matrix([[81.63549976607895], 
                    [27.17527070619227], 
                    [71.97926631050572], 
                    [75.07644206765099], 
                    [65.30397603859095], 
                    [22.150809430682695], 
                    [98.09737507003689], 
                    [14.556272940621184]])

answer_omega0 =  matrix([[0.36603773584905663, -0.011320754716981133, -0.1811320754716981, -0.169811320754717],
                        [-0.011320754716981133, 0.6962264150943396, -0.360377358490566, -0.05660377358490567],
                        [-0.1811320754716981, -0.360377358490566, 1.2339622641509433, -0.4056603773584906],
                        [-0.169811320754717, -0.05660377358490567, -0.40566037735849064, 0.6509433962264151]])

result = online_slam(testdata0, 5, 3, 2.0, 2.0)
solution_check(result, answer_mu0, answer_omega0)
#print 'mu', result[0]
#print 'omega', result[1]

testdata1 = [
            [[[0, -12.850125925252064, -3.933785533519887], [1, -14.324015614935629, 8.270248886213079]], [2.2010206987985454, -4.489488599324021]], 
            [[[0, -15.47003158897079, 0.928280438201802]], [2.2010206987985454, -4.489488599324021]], 
            [[[0, -16.515307649920178, 4.107336898194161]], [2.2010206987985454, -4.489488599324021]], 
            [[[0, -19.535442789045753, 8.989800734534178]], [2.2010206987985454, -4.489488599324021]], 
            [[[2, 20.10158277769682, -7.0324267786819235]], [2.2010206987985454, -4.489488599324021]], 
            [[[2, 17.021879342684294, -2.1431275702209], [3, -10.587702680230684, -17.59474128923241]], [2.2010206987985454, -4.489488599324021]], 
            [[[2, 13.662516211758625, 0.9192978671348002], [3, -13.876421239708055, -10.778382331109766],[4, 6.968683661274824, -21.955659708283196]], [2.2010206987985454, -4.489488599324021]],
            [[[2, 14.176976969179602, 4.6439917757447935], [3, -14.454290933801005, -10.163817305992154], [4, 7.577655713891584, -16.737405183566676]], [2.2010206987985454, -4.489488599324021]], 
            [[[2, 8.597657662181138, 8.786371826813125], [3, -20.875202583756874, -4.040766546254422],[4, 0.9701263264120459, -15.943513382737184]], [2.2010206987985454, -4.489488599324021]]
            ]

answer_mu1 = matrix([[71.91500027698375],
                    [11.973548279423852],
                    [37.11639935255842],
                    [46.03965657797581],
                    [35.67598385113673],
                    [58.270248173017904],
                    [78.4798416029697],
                    [25.31046043855889],
                    [50.089749720920125],
                    [11.564969573660846],
                    [71.49145422962918],
                    [1.9840548998883207]])

answer_omega1 = matrix([[0.39564108485876953, -0.00028617618411671983, -5.020634809065259e-06, -0.13252969705489565, -0.1320426954784163, -0.13076745423691374],
                        [-0.00028617618411671983, 0.7308036028075388, -0.17138941047706066, -0.1662683629718142, -0.04149554669692436, -0.008585285523501593],
                        [-5.020634809065259e-06, -0.17138941047706066, 0.39173001034250776, -0.0029169888240669147, -0.0007279920473144624, -0.0001506190442719577],
                        [-0.13252969705489565, -0.1662683629718142, -0.0029169888240669147, 1.5002460111056442, -0.7168060729598652, -0.4758909116468687],
                        [-0.1320426954784163, -0.04149554669692436, -0.0007279920473144624, -0.7168060729598651, 1.3538091556296377, -0.4612808643524888],
                        [-0.13076745423691374, -0.008585285523501593, -0.0001506190442719577, -0.4758909116468687, -0.4612808643524888, 1.0769763728925885]])

result = online_slam(testdata1, 10, 5, 2.0, 2.0)
solution_check(result, answer_mu1, answer_omega1)
#print 'mu', result[0]
#print 'omega', result[1]
