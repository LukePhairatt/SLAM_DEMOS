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
#
# Robot position: x,y (no heading)
# Robot motion: dx,dy from the current position
# Measurement: lx,ly from the current position
# Graph sequence: Observation then motion update

# -----------
# Note on Testing
#    You can use the solution_check function at the
#    bottom of this document to check your code
#    for the two provided test cases. 


from UdcMatrix import *
from Robot import *
from math import *
import random

# ######################################################################
#
# online_slam - retains all landmarks but only most recent robot pose
#
# ######################################################################

def online_slam(data, N, num_landmarks, motion_noise, measurement_noise):
    # LK- Dynamic expansion
    # init 2 elements matrix (x0,y0)
    n_state = 2
    Omega = matrix()
    Omega.zero(2, 2)
    # set x0,y0 to the initial position (mark by both 1.0)
    Omega.value[0][0] = 1.0
    Omega.value[1][1] = 1.0
    
    # init 2 elements matrix (x0,y0)
    Xi = matrix()
    Xi.zero(2, 1)
    # set vector to the initial position
    Xi.value[0][0] = world_size / 2.0
    Xi.value[1][0] = world_size / 2.0
    
    # initial matrix size (x0,y0)
    dim = n_state
    # create an empty list of landmarks
    landmark_list = []
    
    # process the data and update measurement and then motion one by one
    print(len(data))
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
                dim += 2
                # record this landmark in the list
                landmark_list.append(lmk_index)
                # extend the matrix and vector
                Omega = Omega.expand(dim,dim,list(range(dim-2)),list(range(dim-2)))
                Xi = Xi.expand(dim,1,list(range(dim-2)),[0])
            
            # let update
            # m is the landmark index in the matrix/vector X or Y
            # start from 1 (online slam) after the current location
            
            # m is the index of the landmark coordinate in the matrix/vector
            # start after n_state
            m = n_state + 2*lmk_index
            # update the information maxtrix/vector based on the measurement
            # measurement information update both robot x,y and lmmk x,y
            # b = 0 update x data, b = 1 update y data
            for b in range(2):
                Omega.value[b][b]     +=  1.0 / measurement_noise
                Omega.value[m+b][m+b] +=  1.0 / measurement_noise
                Omega.value[b][m+b]   += -1.0 / measurement_noise
                Omega.value[m+b][b]   += -1.0 / measurement_noise
                Xi.value[b][0]        += -measurement[i][1+b] / measurement_noise
                Xi.value[m+b][0]      +=  measurement[i][1+b] / measurement_noise
            
            
                         
        # motion update - Add a new row and column 
        motion      = data[k][1]   #(dx,dy)
        
        # Expand matrix and vector to accomodate one step move e.g. (xi,yi) to (xi+1,yi+1) 
        # compute range to be extended and added
        expand_i = list(range(n_state)) + list(range(2*n_state,dim+n_state))
        
        Omega = Omega.expand(dim+n_state,dim+n_state,expand_i,expand_i)
        Xi = Xi.expand(dim+n_state,1,expand_i,[0])
        # diag update e.g. (x0,x0),(y0,y0),(x1,x1),(y1,y1)
        for b in range(2*n_state):
            Omega.value[b][b] +=  1.0 / motion_noise
        # off-diag update e.g. (x0,x1),(x1,x0),(y0,y1),(y1,y0)
        for b in range(n_state):
            Omega.value[b  ][b+n_state] += -1.0 / motion_noise
            Omega.value[b+n_state][b  ] += -1.0 / motion_noise
            # update vector e.g. (x0,x1),(y0,y1)
            Xi.value[b  ][0]        += -motion[b] / motion_noise
            Xi.value[b+n_state][0]  +=  motion[b] / motion_noise
        
        
        # Sebastian Thrun megic fomula
        # Online matrix/vector transformation - reduce size to a current position after move      
        # Omega <- Omega_p - AT*B_inv*A
        # Xi <- Xi_p - AT*B_inv*C
        Omega_p = Omega.take(list(range(n_state,dim+n_state)),list(range(n_state,dim+n_state)))
        Xi_p = Xi.take(list(range(n_state,dim+n_state)),[0])
        A = Omega.take(list(range(n_state)),list(range(n_state,dim+n_state)))
        B = Omega.take(list(range(n_state)),list(range(n_state)))
        C = Xi.take(list(range(n_state)),[0])
        Omega = Omega_p - (A.transpose() * B.inverse() * A)
        Xi = Xi_p - (A.transpose() * B.inverse() * C)
        
        step_mu = Omega.inverse() * Xi
        print("robot position x,y: ",step_mu.value[0][0]," ",step_mu.value[1][0])
           
    # compute best estimate
    mu = Omega.inverse() * Xi 
    
    return mu, Omega # make sure you return both of these matrices to be marked correct.

# --------------------------------------------------------------------------
#
# print the result of SLAM, the robot pose(s) and the landmarks
#
# --------------------------------------------------------------------------

def print_result(N, num_landmarks, result):
    print()
    print('Estimated Pose(s):')
    for i in range(N):
        print( '    ['+ ', '.join('%.3f'%x for x in result.value[2*i]) + ', ' \
            + ', '.join('%.3f'%x for x in result.value[2*i+1]) +']')
    print() 
    print('Estimated Landmarks:')
    for i in range(num_landmarks):
        print('    ['+ ', '.join('%.3f'%x for x in result.value[2*(N+i)]) + ', ' \
            + ', '.join('%.3f'%x for x in result.value[2*(N+i)+1]) +']')
        

##########################################################
# TESTING
#
# Uncomment one of the test cases below to check that your
# online_slam function works as expected.

def solution_check(result, answer_mu, answer_omega):

    if len(result) != 2:
        print("Your function must return TWO matrices, mu and Omega")
        return False
    
    user_mu = result[0]
    user_omega = result[1]
    
    if user_mu.dimx == answer_omega.dimx and user_mu.dimy == answer_omega.dimy:
        print("It looks like you returned your results in the wrong order. Make sure to return mu then Omega.")
        return False
    
    if user_mu.dimx != answer_mu.dimx or user_mu.dimy != answer_mu.dimy:
        print("Your mu matrix doesn't have the correct dimensions. Mu should be a", answer_mu.dimx, " x ", answer_mu.dimy, "matrix.")
        return False
    else:
        print("Mu has correct dimensions.")
        
    if user_omega.dimx != answer_omega.dimx or user_omega.dimy != answer_omega.dimy:
        print("Your Omega matrix doesn't have the correct dimensions. Omega should be a", answer_omega.dimx, " x ", answer_omega.dimy, "matrix.")
        return False
    else:
        print("Omega has correct dimensions.")
        
    if user_mu != answer_mu:
        print("Mu has incorrect entries.")
        return False
    else:
        print("Mu correct.")
        
    if user_omega != answer_omega:
        print("Omega has incorrect entries.")
        return False
    else:
        print("Omega correct.")
        
    print("Test case passed!")
    return True

# -----------
# My test case edit- reorder landmark index to suit dynamic expansion of online slam
world_size = 100
testdata0          = [[[[0, 21.796713239511305, 25.32184135169971], [1, 15.067410969755826, -27.599928007267906]], [16.4522379034509, -11.372065246394495]],
                      [[[0, 6.1286996178786755, 35.70844618389858], [1, -0.7470113490937167, -17.709326161950294]], [16.4522379034509, -11.372065246394495]],
                      [[[1, -17.49244296888888, -5.371360408288514],[2, 16.305692184072235, -11.72765549112342]], [16.4522379034509, -11.372065246394495]],
                      [[[1, -32.17857547483552, 6.778675958806988], [2, -0.6443452578030207, -2.542378369361001]], [-16.66697847355152, 11.054945886894709]]]


#result = online_slam(testdata0, 5, 3, 2.0, 2.0)
#print( 'mu', result[0])
#print( 'omega', result[1])

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

#result = online_slam(testdata1, 10, 5, 2.0, 2.0)
#solution_check(result, answer_mu1, answer_omega1)
#print('mu', result[0])
#print('omega', result[1])


testdata4 = [
            [[[0, -18.585768597715386, 8.981826179939848], [1, 32.467536736319694, -13.83376106828661]], [-19.820683428347554, 2.672172979660681]],
            [[[0, 2.6058328002327964, 7.050693788541891], [2, -17.186412218190988, 24.551860132408414]], [-19.820683428347554, 2.672172979660681]],
            [[[0, 23.20700167215044, 1.5947337629642337], [2, 2.467011314873435, 20.28416261109431]], [16.390467532396432, -11.460915062483377]],
            [[[0, 6.165706931772941, 11.454221307788625], [2, -14.012290136826305, 26.93115188638665]], [16.390467532396432, -11.460915062483377]],
            [[[3, 17.47030463011561, -24.578630215698446], [0, -14.903919584119414, 23.292995147953356], [1, 36.65748599063523, 3.6737719163551366]],[16.390467532396432, -11.460915062483377]],
            [[[3, 1.5051898256900231, -14.358954156301174], [1, 23.258126552850428, 12.419499267062552]], [16.390467532396432, -11.460915062483377]],
            [[[3, -13.576302129133044, -6.482190637818383], [1, 4.951219994499219, 24.613547563939957]], [16.390467532396432, -11.460915062483377]],
            [[[3, -29.162773307633273, 8.263054422736078], [1, -11.82363095005573, 37.06312911163883]], [-8.17521480915073, 18.252831638522352]],
            [[[3, -21.7186717498676, -10.57571533368422], [1, -0.5448037511866954, 13.996107390513664]], [-8.17521480915073, 18.252831638522352]],
            [[[3, -11.232906308574094, -29.09393168021315], [1, 7.364005487047098, -1.856177144624739]], [-8.17521480915073, 18.252831638522352]],
            [[[0, -38.29910914762921, -0.994854196841763], [1, 13.246499953206893, -23.57863285919469]], [-8.17521480915073, 18.252831638522352]],
            [[[4, 35.52494238699079, 11.569322541007324], [0, -28.54767823065727, -19.48415986208803]], [-8.17521480915073, 18.252831638522352]],
            [[], [4.310795695299155, -19.529901189544976]],
            [[[0, -27.730611054288797, -17.99109127274661], [2, -48.145265020033456, -1.0349724273533352]],[4.310795695299155, -19.529901189544976]]]

'''
testdata4 robot position
ground truth:  [[50.0, 50.0],
                [30.179316571652446, 52.67217297966068],
                [10.358633143304893, 55.34434595932136],
                [26.749100675701325, 43.88343089683798],
                [43.13956820809776, 32.4225158343546],
                [59.530035740494185, 20.961600771871222],
                [75.92050327289061, 9.500685709387845],
                [92.31097080528704, -1.9602293530955315],
                [84.13575599613631, 16.29260228542682],
                [75.96054118698558, 34.545433923949176],
                [67.78532637783485, 52.79826556247153],
                [59.61011156868412, 71.05109720099388],
                [51.43489675953339, 89.30392883951623],
                [55.74569245483254, 69.77402764997126],
                [60.0564881501317, 50.24412646042629]]

'''
result = online_slam(testdata4, 15, 5, 2.0, 2.0)
print("online slam: final position + lmk",result[0])






