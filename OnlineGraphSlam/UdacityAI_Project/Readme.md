## **Online Graph SLAM- Landmark/Motion Simulation**

The following python codes has been implemented from Udacity AI course on Online Graph SLAM.  

The state vector(mu) has x and y interlaced, so if there were two poses and 2 landmarks,  
mu would look like:

mu = matrix([[Px0],
             [Py0],
             [Px1],
             [Py1],
             [Lx0],
             [Ly0],
             [Lx1],
             [Ly1]])
 

Robot position: Px,Py (no heading)
Landmark position: Lx,Ly
Robot motion: dx,dy from the current position
Measurement: lx,ly from the current position
Graph sequence: Observation then motion update


The code has been exyended with a dynamic expansion of the matrices to simulate SLAM idea when observe the new landmarks.
It allocates/expands the matrix for a new motion/landmark measurement.

### **Online Graph SLAM steps**
**Initialisation:**  
The system starts at the initial position x0,y0 with the matrix dimension of 2x2
	(Omega matrix)

		x0  	y0 
	x0	1	
	y0		1


	(Xi vector)
	
	x0	Px0	
	y0	Py0




**Measurement Update:**  
Update corresponding the current robot position (x0,y0) and landmark observation (Lxi,Lyi)
Note: 1: not included measurement noises
      2: if the new landmark, we need to expand the matrix/vector before the update	
      3: update/add the information from the previous state
```sh

			(Omega matrix)

		x0  	y0  ............Lxi    Lyi
	x0     +1			-1
	y0	 	+1		       -1
	:
	:
	Lxi    -1			+1
	Lyi	       -1		       +1


			(Xi vector)
	
	x0	-lxi	
	y0	-lyi
	:
	:
	Lxi      lxi
	Lyi      lyi
	
```
		
**Motion Update:**  
Update corresponding the current robot position (x0,y0) to the next position (x1,y1)
Note: 1: not included motion noises
      2: we need to expand the matrix/vector before the motion update
      3: update/add the information from the previous state

```sh
			(Omega matrix)

		x0  	y0  	x1	y1	
	x0	+1	        -1
	y0		+1		-1
	x1	-1		+1
	y1		-1		+1



			(Xi vector)
	
	x0	-dx	
	y0	-dy
	x1	+dx
	y1	+dy

```

**Reduce Position:**  

```sh
				(Omega)

		x0  	y0   |	x1	y1	Lx0    Ly0	Lx1	Ly1
	x0		     |
	y0         [B]       |			[A]
	--------------------------------------------------------------------
	x1                   |
	y1                   |
	Lx0                  |
	Ly0       [AT]       |                  [Omega']
	Lx1                  |
	Ly1                  |
	
	



				(Xi Vector)

	x0
	y0        [C]
        -----------------
	x1
	y1
	Lx0
	Ly0       [Xi']
	Lx1
	Ly1



	Omega = Omega' - (AT * B.inverse() * A)
        Xi = Xi' - (AT * B.inverse() * C)

```

**Retrieve the robot and landmarks position:**  

To retrieve the robot and landmark positions, we use the following equation

```sh
	mu = Omega.inverse() * Xi 
```



**Running the code**  
```sh
$ python OnlineSLAM_DynamicLmkUpdate.py
```
