# ---------------------------------------------------------------------------------
# Modified slam project from Claus Brenner to build probability grid map approach
# see Probabilistic Robotics by Thrun etc. for details
# Punnu LK Phairatt, 17.03.2016
# ---------------------------------------------------------------------------------

from lib.logfile_reader import *
from lib.slam_library import get_cylinders_from_scan, write_cylinders,\
	write_error_ellipses, get_mean, get_error_ellipse_and_heading_variance,\
	print_particles
from time import *
from gridmap import *
from math import *
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


'''
  Helper class for FastSLAM approach: state transition, jacobian matrix and measurement likelihood
'''
class Particle:
	def __init__(self, pose):
		self.pose = pose
		self.landmark_positions = []
		self.landmark_covariances = []
	
	def number_of_landmarks(self):
		"""Utility: return current number of landmarks in this particle."""
		return len(self.landmark_positions)

	@staticmethod
	def g(state, control, w):
		"""State transition. This is exactly the same method as in the Kalman filter."""
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
		return np.array([g1, g2, g3])
		
	def move(self, left, right, w):
		"""Given left, right control and robot width, move the robot."""
		self.pose = self.g(self.pose, (left, right), w)
		
	@staticmethod
	def h(state, landmark, scanner_displacement):
		"""Measurement function. Takes a (x, y, theta) state and a (x, y) 
		landmark, and returns the corresponding (range, bearing)."""
		dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
		dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
		r = sqrt(dx * dx + dy * dy)
		alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi
		return np.array([r, alpha])
		
	@staticmethod
	def dh_dlandmark(state, landmark, scanner_displacement):
		"""Derivative with respect to the landmark coordinates. This is related
		to the dh_dstate function we used earlier (it is: -dh_dstate[0:2,0:2])."""
		theta = state[2]
		cost, sint = cos(theta), sin(theta)
		dx = landmark[0] - (state[0] + scanner_displacement * cost)
		dy = landmark[1] - (state[1] + scanner_displacement * sint)
		q = dx * dx + dy * dy
		sqrtq = sqrt(q)
		dr_dmx = dx / sqrtq
		dr_dmy = dy / sqrtq
		dalpha_dmx = -dy / q
		dalpha_dmy =  dx / q

		return np.array([[dr_dmx, dr_dmy], [dalpha_dmx, dalpha_dmy]])
		
	def expected_measurement(self, landmark_number, scanner_displacement):
		"""Returns the expected distance and bearing measurement for a given
		landmark number and the pose of this particle (Robot position)."""
		return self.h(self.pose, self.landmark_positions[landmark_number], scanner_displacement)
		
	def measurement_jacobian_covariance(self, landmark_number, Qt_measurement_covariance, scanner_displacement):
		"""Computes Jacobian H of measurement function at the particle's position (Robot position)
		and the landmark given by landmark_number. Also computes the measurement covariance matrix."""
		# Notes:
		# LK- H is dh_dlandmark (2x2) NOT full EKF H state
		#   - Qt is measurement variance r and bearing 
		#     Qt = Qt_measurement_covariance (2x2)
		#     Cov_lmk = self.landmark_covariances
		#   - H is computed using dh_dlandmark.
		#   - To compute Ql, you will need the product of two matrices, which is np.dot(A, B).
		#     H = 2x2, Ql = 2x2
		H = self.dh_dlandmark(self.pose, self.landmark_positions[landmark_number], scanner_displacement)
		Cov_lmk = self.landmark_covariances[landmark_number]
		# LK- Ql = H * Cov_lmk * HT + Qt 
		Ql = np.dot(np.dot(H,Cov_lmk), H.T) + Qt_measurement_covariance
		return (H, Ql)
		
	def measurement_likelihood(self, measurement,
										landmark_number,
										Qt_measurement_covariance,
										scanner_displacement):
		"""For a given measurement and landmark_number, returns the likelihood
		that the measurement corresponds to the landmark."""
		# Notes:
		# - You will need delta_z, which is the measurement minus the
		#   expected_measurement_for_landmark()
		# - Ql is obtained using a call to
		#   H_Ql_jacobian_and_measurement_covariance_for_landmark(). You
		#   will only need Ql, not H
		# - np.linalg.det(A) computes the determinant of A
		# - np.dot() does not distinguish between row and column vectors.
		z_hat = self.expected_measurement(landmark_number, scanner_displacement)      
		z = measurement
		dz = z-z_hat
		Ql = self.measurement_jacobian_covariance(
				landmark_number, Qt_measurement_covariance, scanner_displacement)[1]

		# LK- likelihood = e(-0.5*dzT*Ql_inv*dz)/(2pi*sqrt(det Ql))
		Ql_inv = np.linalg.inv(Ql)
		zQ = np.dot(np.dot(dz.T, Ql_inv), dz)   	
		l = exp(-0.5*zQ)/(2*pi*sqrt(np.linalg.det(Ql)))
		return l
		
	def compute_correspondence_likelihoods(self, measurement,
											number_of_landmarks,
											Qt_measurement_covariance,
											scanner_displacement):
		"""For a given measurement, returns a list of all correspondence
		likelihoods (from index 0 to number_of_landmarks-1)."""
		likelihoods = []
		for i in range(number_of_landmarks):
			likelihoods.append(self.measurement_likelihood(measurement, i, Qt_measurement_covariance,scanner_displacement))
		return likelihoods

	def initialize_new_landmark(self, measurement_in_scanner_system,
								Qt_measurement_covariance,
								scanner_displacement):
		"""Given a (x, y) measurement in the scanner's system, initializes a
		new landmark and its covariance."""
		#scanner_pose = (self.pose[0] + cos(self.pose[2]) * scanner_displacement,
		#				self.pose[1] + sin(self.pose[2]) * scanner_displacement,
		#				self.pose[2])
		scanner_pose = LegoLogfile.get_scanner_pose(self.pose,(scanner_displacement,0,0))
		# This code initialize a new landmark after no match found in the lmk association step (10_b: Corresponding likelihood)
		# Need to added - Landmark Position x,y in the world frame 
		#               - Landmark Covariance =  H_inv*Qt*(H_inv).T
		# Notes:
		# - LegoLogfile.scanner_to_world() (from lego_robot.py) will return
		#   the world coordinate, given the scanner pose and the coordinate in
		#   the scanner's system.
		lmk_xs = measurement_in_scanner_system[0]
		lmk_ys = measurement_in_scanner_system[1]
		lmk_point  = tuple([lmk_xs, lmk_ys])
		lmk_x, lmk_y = LegoLogfile.scanner_to_world(scanner_pose, lmk_point)
		self.landmark_positions.append(np.array([lmk_x, lmk_y]))
		H_inv = np.linalg.inv( self.dh_dlandmark(self.pose, tuple([lmk_x, lmk_y]), scanner_displacement) )
		Q_lmk = np.dot(np.dot(H_inv, Qt_measurement_covariance), H_inv.T)       
		self.landmark_covariances.append(Q_lmk)
		
	def update_landmark(self, landmark_number, measurement,
						Qt_measurement_covariance, scanner_displacement):
		"""Update a landmark's estimated position and covariance."""
		# This code update a matched lmk found in the lmk association step (10_b: Corresponding likelihood)
		# We need to compute - an updated lmk position = old lmk position + K(z-z_hat)
		#                    - an update covariance = (I-KH) * old_covariance
		# where
		#                      K = Cov_old*HT(H*Cov_old*HT) = Cov_old*HT(Ql_inv) 
		#                          Cov_old -> matched landmark covariance before update
		#			   		   H = dh_dlandmark at this moment
		#                          z_hat = expected mesurement self.h(..)
		#                          z = actual measurement	
		# Notess:
		# - H and Ql are computed using
		#   H_Ql_jacobian_and_measurement_covariance_for_landmark()
		# - Delta z is measurement minus expected measurement
		# - Expected measurement can be computed using
		#   h_expected_measurement_for_landmark()
		# - Remember to update landmark_positions[landmark_number] as well
		#   as landmark_covariances[landmark_number].
		H, Ql = self.measurement_jacobian_covariance(landmark_number, Qt_measurement_covariance, scanner_displacement)
		z_hat = self.expected_measurement(landmark_number, scanner_displacement)
		z = measurement        
		Cov_old = self.landmark_covariances[landmark_number]
		Ql_inv = np.linalg.inv(Ql) 
		K = np.dot(np.dot(Cov_old, H.T), Ql_inv)
		I = np.identity(2)
		# Updated lmk position = old lmk position + K(z-z_hat)
		self.landmark_positions[landmark_number] =  self.landmark_positions[landmark_number] + np.dot(K, (z-z_hat)) 
		#an update covariance = (I-K*H) * old_covariance
		self.landmark_covariances[landmark_number] = np.dot((I- np.dot(K,H)), Cov_old)
		return 1
		
	def update_particle(self, measurement, measurement_in_scanner_system,
						number_of_landmarks,
						minimum_correspondence_likelihood,
						Qt_measurement_covariance, scanner_displacement):
		"""Given a measurement (ONLY ONE MEASUREMENT), computes the likelihood that it belongs to any
		of the landmarks in the particle. If there are none, or if all likelihoods are below the minimum_correspondence_likelihood
		threshold, add a landmark to the particle. Otherwise, update the (existing) landmark with the largest likelihood."""
		# LK- This function only computes a maximum likelihood of one measurement as an output weight
		#   - In this function, it also ADDS new landmark OR UPDATE landmark depend on the likelihood!
		#   - Compute likelihood of correspondence of measurement to all landmarks
		#     (from 0 to number_of_landmarks-1) 

		likelihoods = self.compute_correspondence_likelihoods(measurement,
										number_of_landmarks,
										Qt_measurement_covariance,
										scanner_displacement)
		
		# If the likelihood list is empty(no match found! New lmk!), or the max correspondence likelihood
		# is still smaller than minimum_correspondence_likelihood, THEN setup a new landmark
		if not likelihoods or\
			max(likelihoods) < minimum_correspondence_likelihood:
			# Insert a new landmark.
			self.initialize_new_landmark(measurement_in_scanner_system,
										Qt_measurement_covariance,
										scanner_displacement)
			return minimum_correspondence_likelihood
			
		# Else update the particle's EKF for the corresponding particle.
		# Match found with acceptable likelihod THEN Update position, Covariance
		else:
			# This computes (max, argmax) of measurement_likelihoods.
			# Find weight, the maximum likelihood, and the corresponding landmark index.
			w, landmark_number = max((v,i) for i,v in enumerate(likelihoods))
			# Update_landmark
			self.update_landmark(landmark_number, measurement, Qt_measurement_covariance, scanner_displacement)
			return w


'''
   Main Particle Filter prediction/correction/building grid map
'''
class FastSLAM:
	def __init__(self, initial_particles,
				robot_width, scanner_displacement,
				control_motion_factor, control_turn_factor,
				measurement_distance_stddev, measurement_angle_stddev,
				minimum_correspondence_likelihood):
		# The particles.
		self.particles = initial_particles
		# Some constants.
		self.robot_width = robot_width
		self.scanner_displacement = scanner_displacement
		self.control_motion_factor = control_motion_factor
		self.control_turn_factor = control_turn_factor
		self.measurement_distance_stddev = measurement_distance_stddev
		self.measurement_angle_stddev = measurement_angle_stddev
		self.minimum_correspondence_likelihood = minimum_correspondence_likelihood
		
	def predict(self, control):
		"""The prediction step of the particle filter."""
		left, right = control
		left_std  = sqrt((self.control_motion_factor * left)**2 +\
						(self.control_turn_factor * (left-right))**2)
		right_std = sqrt((self.control_motion_factor * right)**2 +\
						(self.control_turn_factor * (left-right))**2)
		# Modify list of particles: for each particle, predict its new position.
		for p in self.particles:
			l = random.gauss(left, left_std)
			r = random.gauss(right, right_std)
			p.move(l, r, self.robot_width)
				
	def update_and_compute_weights_gridmap(self, GlobalMap, LocalMap, scanner_offset, Range_Lidar):
		"""Updates all particles and returns a list of their weights."""
		weights = []
		# compute weight for all particles (vehicle positions)
		for p in self.particles:
			# For each particle do.... vehicle_pose is p.pose
			vehicle_pose = p.pose
			# Compute local grid map
			# TODO - LocalMap.ComputeGridProbability((0.0,0.0,vehicle_pose[2]), scanner_displacement, Range_Lidar)		
			# compute grid map correlation with global_map
			# TODO- replace this with LocalMap.grid_map_correlation(vehicle_pose, GlobalMap)
			LocalMap.ComputeGridProbability((0.0,0.0,vehicle_pose[2]), scanner_offset, Range_Lidar)
			weight = LocalMap.grid_map_correlation(vehicle_pose, GlobalMap)
			# Append overall weight of this particle to weight list.
			weights.append(weight)
		return weights

		
	def resample(self, weights):
		"""Return a list of particles which have been resampled, proportional
		to the given weights."""
		new_particles = []
		max_weight = max(weights)
		index = random.randint(0, len(self.particles) - 1)
		offset = 0.0
		for i in range(len(self.particles)):
			offset += random.uniform(0, 2.0 * max_weight)
			while offset > weights[index]:
				offset -= weights[index]
				index = (index + 1) % len(weights)
				
			new_particles.append(copy.deepcopy(self.particles[index]))
		return new_particles
				
	def correct_gridmap(self, GlobalMap, LocalMap, scanner_offset, Range_Lidar):
		# Compute weight of each particle
		weights = self.update_and_compute_weights_gridmap(GlobalMap, LocalMap, scanner_offset, Range_Lidar)
		# resampling the good one
		self.particles = self.resample(weights)


if __name__ == '__main__':
	# Robot constants.
	scanner_displacement = 30.0
	scanner_offset = (scanner_displacement, 0.0, 0.0)
	ticks_to_mm = 0.349
	robot_width = 155.0
	
	# Cylinder extraction and matching constants.
	minimum_valid_distance = 20.0
	depth_jump = 100.0
	cylinder_offset = 90.0
	
	# Filter constants.
	control_motion_factor = 0.35       		# Error in motor control.
	control_turn_factor = 0.6  				# Additional error due to slip when turning.
	measurement_distance_stddev = 200.0  	# Distance measurement error of cylinders.
	measurement_angle_stddev = 15.0 / 180.0 * pi  	# Angle measurement error.
	minimum_correspondence_likelihood = 0.001  		# Min likelihood of correspondence.
	
	# Generate initial particles. Each particle is (x, y, theta).
	number_of_particles = 50
	start_state = np.array([500.0, 0.0, 45.0 / 180.0 * pi])
	initial_particles = [copy.copy(Particle(start_state)) for _ in range(number_of_particles)]
	
	# Setup filter.
	fs = FastSLAM(initial_particles,
				  robot_width, scanner_displacement,
				  control_motion_factor, control_turn_factor,
				  measurement_distance_stddev,
				  measurement_angle_stddev,
				  minimum_correspondence_likelihood)
	
	# Read data.
	logfile = LegoLogfile()
	logfile.read("../in_data/robot4_motors.txt")
	logfile.read("../in_data/robot4_scan.txt")

	# Set measurement configuration for grid map update
	zt = logfile.scan_data[0]
	AngleIncrement = abs(LegoLogfile.beam_index_to_angle(1)-LegoLogfile.beam_index_to_angle(0))
	AngleStart = LegoLogfile.beam_index_to_angle(0)
	ResamplingFactor = 10.0
	# Init range measurement object data resampling configuration (do it once!)
	# See gridmap.py
	Range_Lidar = RangeMeasurement(AngleIncrement, AngleStart, ResamplingFactor)
	# Set lidar data based on given configuration
	Range_Lidar.set_resample(zt)

	# Construct grid map global 
	glb_GridDimension = (61, 61)
	glb_CentreGrid    = (31, 31)
	glb_Resolution    = 100
	GlobalMap = Gridmap(glb_GridDimension, glb_CentreGrid, glb_Resolution)
	# Construct grid map local 
	loc_GridDimension = (31, 31)
	loc_CentreGrid    = (16, 16)
	loc_Resolution    = glb_Resolution
	LocalMap = Gridmap(loc_GridDimension, loc_CentreGrid, loc_Resolution)	
	# Read or init global map
	print("Init GlobalMap............")
	GlobalMap.ComputeGridProbability(start_state, scanner_offset, Range_Lidar)
		
	# Loop over all motor tick records.
	# This is the FastSLAM filter loop, with prediction and correction.
	f = open("../out_data/fast_slam_correction.txt", "w")	
	for i in range(0,int(len(logfile.motor_ticks)/5)):
		# ------------------------------------------------------------ #
		#                         Prediction                           #
		# ------------------------------------------------------------ #
		# Prediction- move each particle to a new position with gaussian distribution
		# TODO- if control = 0,0 ignore the prediction, correction but update the global map
		control = map(lambda x: x * ticks_to_mm, logfile.motor_ticks[i])
		fs.predict(control)
		
		# ------------------------------------------------------------ #
		#                         Correction                           #
		# ------------------------------------------------------------ #
		# LK-TODO correction using gridmap correlation
		#   -Remove get_cylinder...
		#   -Compute local grid map and pass to fs.correction to use for weight calculation....
		
		# Approach 1- Correction using landmark weight 
		# cylinders = get_cylinders_from_scan(logfile.scan_data[i], depth_jump, minimum_valid_distance, cylinder_offset)
		# fs.correct(cylinders)
		
		# Approach 2- Correction using gridmap correlation (Global VS this measurement map aka Local map)
		zt = logfile.scan_data[i]
		Range_Lidar.set_resample(zt)
		fs.correct_gridmap(GlobalMap, LocalMap, scanner_offset, Range_Lidar)
		
		# ------------------------------------------------------------ #
		#                  Output best or mean particles.              #
		# ------------------------------------------------------------ #
		print_particles(fs.particles, f)
		
		# Output state estimated from all particles.
		mean = get_mean(fs.particles)
		print("F %.0f %.0f %.3f" %\
			  (mean[0] + scanner_displacement * cos(mean[2]),
			   mean[1] + scanner_displacement * sin(mean[2]),
			   mean[2]), end=" ", file=f)
		print(file=f)
		
		# Output error ellipse and standard deviation of heading.
		errors = get_error_ellipse_and_heading_variance(fs.particles, mean)
		print("E %.3f %.0f %.0f %.3f" % errors, end=" ", file=f)
		print(file=f)
		
		# Output landmarks of particle which is closest to the mean position.
		output_particle = min([(np.linalg.norm(mean[0:2] - fs.particles[k].pose[0:2]),k) for k in range(len(fs.particles)) ])[1]
		# Write estimates of landmarks.
		write_cylinders(f, "W C", fs.particles[output_particle].landmark_positions)
		# Write covariance matrices.
		write_error_ellipses(f, "W E", fs.particles[output_particle].landmark_covariances)
		
		# ----------------------------------------------------------------------------------------- #
		#                    Update Global gridmap with corrected pose                              #
		# ----------------------------------------------------------------------------------------- #
		print(i)
		vehicle_corrected_pose = mean#fs.particles[output_particle].pose
		# Update a global map every n step
		if(i%2==0): 
			print("Run probability grid update....",i)
			GlobalMap.ComputeGridProbability(vehicle_corrected_pose, scanner_offset, Range_Lidar)
			#UpdatePlotGridMap(im, gridmap)

	f.close()
	#visualising map
	#plt.ioff()
	#FinalPlotGridMap(fig, gridmap)
	if(1):
		T = np.ones(glb_GridDimension)
		fig, ax_lst = plt.subplots(1,1)
		im = ax_lst.imshow(T, interpolation='nearest',
						   origin='bottom',
						   aspect='auto',
						   vmin=0.0,
						   vmax=1.0,
						   cmap='hot',extent=[0,1,0,1])
		fig.colorbar(im)
		ax = fig.add_subplot(111)
		ax.set_title('colorMap')	
		ax.set_aspect('equal')
		cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
		cax.get_xaxis().set_visible(False)
		cax.get_yaxis().set_visible(False)
		cax.patch.set_alpha(0)
		cax.set_frame_on(False)	
		plt.imshow(GlobalMap.gridmap)
		plt.colorbar(orientation='vertical')
		plt.show()
