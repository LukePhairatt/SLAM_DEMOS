## **Online Graph SLAM- Motion/Lidar Data Post-Processing**

---

# **Overview**
The work of this project shows the step of implementing Online Graph for SLAM. The work in this project uses the offline data and visualisation package taken from the tutorial given by Prof. Claus Brenner.


The state vector(mu) has x and y interlaced, so if there were two poses and 2 landmarks,  
mu would look like:

mu = matrix([[Px0],
             [Py0],
	     [h0 ],
             [Px1],
             [Py1],
 	     [h1 ],
             [Lx0],
             [Ly0],
             [Lx1],
             [Ly1]])
 

Robot position: Px,Py,h(heading)
Landmark position: Lx,Ly
Robot wheel motion: dl,dr from the current robot pose
Measurement: range, bearing 
Graph sequence: Observation then motion update

The code has been exyended with a dynamic expansion of the matrices to simulate SLAM idea when observe the new landmarks.
It allocates/expands the matrix for a new motion/landmark measurement.

### **Online Graph SLAM steps**
**Initialisation:**  
The system starts at the initial position x0,y0,h0 with the matrix dimension of 3x3
	(Omega matrix)

		x0  	y0 	h0
	x0	1	
	y0		1
	h0			1


	(Xi vector)
	
	x0	Px0	
	y0	Py0
	h0      h0




**Measurement Update:**  
Update corresponding the current robot position (x0,y0) and landmark observation (Lxi,Lyi)
Note: 
      1: if the new landmark, we need to expand the matrix/vector before the update	
      2: update/add the information from the previous state

_Measurement Update_:
```sh
	Omega update
		Add Hi_T*Q_inv*Hi at corresponding x, mi

	Xi update
		Add Hi_T*Q_inv*(zi-zi'+ Hi*Si)

	where
		x   = [x,y,heading,landmarks]
		zi' = h(x) = [range, bearing]
		q   = dx * dx + dy * dy
		Hi  = dhi/dx =	[-dx/sqrtq,-dy/sqrtq, (dx*sint-dy*cost)*scanner_displacement/sqrtq, dx/sqrtq,dy/sqrtq]
				[ dy/q,    -dx/q,     -1 - scanner_displacement/q*(dx*cost+dy*sint),-dy/q,  ,dx/q    ]
		Q   = [2x2] measurement noise
		Si  = [x, lmki]

	
```
		
**Motion Update:**  
Update corresponding the current robot position (x0,y0) to the next position (x1,y1)
Note: 
      1: we need to expand the matrix/vector before the motion update
      2: update/add the information from the previous state


_Motion Model(g)_:
```sh
		l, r = dl,dr
		w = wheel base width
		if r != l:
		    alpha = (r - l) / w
		    rad = l/alpha
		    x' = x + (rad + w/2.)*(sin(theta+alpha) - sin(theta))
		    y' = y + (rad + w/2.)*(-cos(theta+alpha) + cos(theta))
		    theta' = (theta + alpha + pi) % (2*pi) - pi
		else:
		    x' = x + l * cos(theta)
		    y' = y + l * sin(theta)
		    theta' = theta
```


_Motion Update_:
```sh
	Omega update (X_prvious to X)
		-GT  * R_inv * (-GT I)
		 I


	Xi update (X_previous to X)
		-GT  * R_inv * (X-GT*X_previous)
		 I

	Where
		 X = robot state = [x,y,heading]
		 G = State Jacobian = dg/dstate
		 R = [3x3] control noise
	

	dg/dstate := 

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
		


```

**Reduce Position:**  

[see this](/src) for a definition
```sh
	Omega = Omega' - (AT * B.inverse() * A)
        Xi = Xi' - (AT * B.inverse() * C)
			
```

**Retrieve the robot and landmark positions:**  

To retrieve the robot and landmark positions, we use the following equation

```sh
	mu = Omega.inverse() * Xi 
```



**Running the code**  
```sh
$ python online_graph_slam.py
```
