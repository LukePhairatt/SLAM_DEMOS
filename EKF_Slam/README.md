## **EKF-SLAM**
![project][image0]
---

[//]: # (Image References)
[image0]: ./images/viewer.png "result"


# **Overview**
The work of this project shows the step of implementing Extended Kalman Filter for SLAM. The work in this project uses the offline data and visualisation package taken from the tutorial given by Prof. Claus Brenner.

# **EKF-SLAM Summary**

**STEP 1-Prediction:** Predicting the next state,

_EKF Formula (Prediction)_

```sh

	X[t]_         = g(X[t-1],u[t])  
	Sig_State[t]_ = G[t]*Sig_State[t-1]*GT[t] + V[t]*Sig_Control*VT[t]


	where
		g  = Motion model  
		X_ = (Predict) System state = {x,y,theta,landmarks}    
		u = Control inputs = {left wheel distance, right wheel distance}  
		G = Jacobian matrix w.r.t State  
		V = Jacobian matrix w.r.t Controls    
		Sig_State_   = (Predict) State covariances

		Sig_Control = Control covariances = [left_var     0    ]  
                            	    		    [ 0       right_var]

		left_var  = (control_motion_factor * left)^2  + (control_turn_factor * (left-right))^2
		right_var = (control_motion_factor * right)^2 + (control_turn_factor * (left-right))^2

```

_Differential Drive Motion Model (g)_


	
		l, r := left, right control input (encoder ticks * meter_per_tick)
		w = wheel base distance (width)
		r != l:
		    alpha = (r - l) / w
		    rad = l/alpha
		    x' = x + (rad + w/2.)*(sin(theta+alpha) - sin(theta))
		    y' = y + (rad + w/2.)*(-cos(theta+alpha) + cos(theta))
		    theta' = (theta + alpha + pi) % (2*pi) - pi
		r == l
		    x' = x + l * cos(theta)
		    y' = y + l * sin(theta)
		    theta' = theta

_State Jacobian Matrix(without landmark)_


		l, r = control
		r != l:
		    alpha = (r-l)/w
		    theta_ = theta + alpha
		    rpw2 = l/alpha + w/2.0
		    m = array([[1.0, 0.0, rpw2*(cos(theta_) - cos(theta))],
		               [0.0, 1.0, rpw2*(sin(theta_) - sin(theta))],
		               [0.0, 0.0, 1.0]])
		r == l
		    m = array([[1.0, 0.0, -l*sin(theta)],
		               [0.0, 1.0,  l*cos(theta)],
		               [0.0, 0.0,  1.0]])


_Control Jacobian Matrix(without landmarks)_


		l, r = control
		r != l:
		    rml = r - l
		    rml2 = rml * rml
		    theta_ = theta + rml/w
		    dg1dl = w*r/rml2*(sin(theta_)-sin(theta))  - (r+l)/(2*rml)*cos(theta_)
		    dg2dl = w*r/rml2*(-cos(theta_)+cos(theta)) - (r+l)/(2*rml)*sin(theta_)
		    dg1dr = (-w*l)/rml2*(sin(theta_)-sin(theta)) + (r+l)/(2*rml)*cos(theta_)
		    dg2dr = (-w*l)/rml2*(-cos(theta_)+cos(theta)) + (r+l)/(2*rml)*sin(theta_)
		    
		r == l
		    dg1dl = 0.5*(cos(theta) + l/w*sin(theta))
		    dg2dl = 0.5*(sin(theta) - l/w*cos(theta))
		    dg1dr = 0.5*(-l/w*sin(theta) + cos(theta))
		    dg2dr = 0.5*(l/w*cos(theta) + sin(theta))

		dg3dl = -1.0/w
		dg3dr = 1.0/w
		m = array([[dg1dl, dg1dr], [dg2dl, dg2dr], [dg3dl, dg3dr]])



**STEP 2-Correction:**
_EKF Formula (Correction)_

```sh

	K = 	 Sig_State_*HT  
              ----------------------         
               (H*Sig_State_*HT + Q)

        X  =     X_ + K*(Z - z)

        Sig_state = (I - Kt*H)*X_   

```

_Measurement Covariance (Q)_

	Q =     [var_range,        0      ]
         	[     0   ,   var_bearing ]

	where
		var_range   = range std. deviation^2
		var_bearing = bearing std. deviation^2
	
_Measurement function (h)_  

	z = {r, alpha} = h(X) given by

	dx = lmk_x - (x + scanner_displacement * cos(theta))
        dy = lmk_y - (y + scanner_displacement * sin(theta))
        r = sqrt(dx * dx + dy * dy)
        alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi

_Measurement Jacobian Matrix (H)_
  	w.r.t the robot pose (x,y,theta)
	 	[dr_dx,     dr_dy,     dr_dtheta    ]
         	[dalpha_dx, dalpha_dy, dalpha_dtheta]
	
	where
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

        w.r.t the landmark i position (lmk_x, lmk_y) 
		[dr_dlmk_x,     dr_dlmk_y     ]
         	[dalpha_dlmk_x, dalpha_dlmk_y ]

 	where
		dr_dlmk_x 	= -(dr_dx) 
		dr_dlmk_y	= -(dr_dy)
		dalpha_dlmk_x	= -(dalpha_dx)
		dalpha_dlmk_y	= -(dalpha_dy)

_Measurement Observation and Correction Porcess_
	for each observation		
		* Do measurement association: 
			get_observations(...)

		* Do landmark initialisation for a new observation
			add_landmark_to_state(...)

		* Do a state correction
			correct(...)



# **Runing project.
Need python 3.x to run

To run the simulation, in src folder
'$ python ekf_slam_full_update.py'


To view result, in src/lib folder
'$ logfile_viewer.py (and select load ekf_slam_correction_full.txt)'



