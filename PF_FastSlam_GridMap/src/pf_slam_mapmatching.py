# ---------------------------------------------------------------------------------
# Modified slam project from Claus Brenner to build probability grid map approach
# see Probabilistic Robotics by Thrun etc. for details
# Punnu LK Phairatt, 17.03.2016
# ---------------------------------------------------------------------------------

## Improve with a new resampling say delete p weight < 0.9, sum w to 1?,
## use binarymap to thing up speed-up (occ map take some time to run)
## having a map beforehand and use it as global map for each particle

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


class Particle:
	def __init__(self, pose):
		self.pose = pose				# x,y,theta pose
		self.map_particle = None     	# grid map of this particle

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
		

'''
   Main Particle Filter prediction/correction/building grid map probability
'''
class FastSLAM:
	def __init__(self, initial_particles,
				robot_width, scanner_displacement,
				control_motion_factor, control_turn_factor,
				minimum_correspondence_likelihood, GridDimension, CentreGrid,Resolution):
		# The particles.
		self.particles = initial_particles
		# Some constants.
		self.robot_width = robot_width
		self.scanner_displacement = scanner_displacement
		self.control_motion_factor = control_motion_factor
		self.control_turn_factor = control_turn_factor
		self.minimum_correspondence_likelihood = minimum_correspondence_likelihood
		self.GridDimension = GridDimension					
		self.CentreGrid    = CentreGrid 					
		self.Resolution    = Resolution							
			
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
				
	def update_and_compute_weights_gridmap(self, scanner_offset, measurement, show=False):
		"""Updates all particles and returns a list of their weights.
		   measurement = (range, bearing)
		
		"""
		weights = []
		# compute weight for all particles (vehicle positions)
		for p in self.particles:
			# For each particle do.... vehicle_pose is p.pose
			vehicle_pose = p.pose
			# init particle map with the measurement
			if p.map_particle is None:
				p.map_particle = Gridmap(self.GridDimension, self.CentreGrid, self.Resolution)
				p.map_particle.gridmap = p.map_particle.MeasurementToMap(vehicle_pose, scanner_offset, measurement)
				weight = self.minimum_correspondence_likelihood
			else:
				# Convert the measurement to gridmap
				map_m = Gridmap(self.GridDimension, self.CentreGrid, self.Resolution)
				map_z = map_m.MeasurementToMap(vehicle_pose, scanner_offset, measurement)
				weight = grid_map_correlation(p.map_particle.gridmap, map_z)
				# Debug
				if show:
					print("weight:", weight, "particle:", vehicle_pose)
					plt.figure(1)
					plt.imshow(p.map_particle.gridmap)
					plt.figure(2)
					plt.imshow(map_z)
					plt.show()
					
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
				
	def correct_gridmap(self,scanner_offset,measurement, showPlot):
		# Compute weight of each particle
		weights = self.update_and_compute_weights_gridmap(scanner_offset, measurement,showPlot)
		weights = np.array(weights,dtype=np.float16)
		weights = weight_normalise(weights)
		print("particle normailsed weight: ", weights)
		# resampling the good one
		self.particles = self.resample(weights)


if __name__ == '__main__':
	# Robot constants.
	scanner_displacement = 30.0
	scanner_offset = (scanner_displacement, 0.0, 0.0)
	ticks_to_mm = 0.349
	robot_width = 155.0
	
	# Filter constants.
	control_motion_factor = 0.35       			# Error in motor control. 0.35
	control_turn_factor = 0.6  					# Additional error due to slip when turning 0.6
	minimum_correspondence_likelihood = 0.001  	# Min likelihood of correspondence.
	
	# Particle grid map size
	GridDimension = (101,101)					# map size row x col
	CentreGrid    = (51,51) 					# map centre
	Resolution    = 50							# mm/cell
	
	
	# Generate initial particles. Each particle is (x, y, theta).
	number_of_particles = 70
	start_state = np.array([500.0, 0.0, 45.0 / 180.0 * pi])
	initial_particles = [copy.copy(Particle(start_state)) for _ in range(number_of_particles)]
	
	# Setup filter.
	fs = FastSLAM(initial_particles,
				  robot_width, scanner_displacement,
				  control_motion_factor, control_turn_factor,
				  minimum_correspondence_likelihood, GridDimension, CentreGrid, Resolution)
	
	# Read data.
	logfile = LegoLogfile()
	logfile.read("../in_data/robot4_motors.txt")
	logfile.read("../in_data/robot4_scan.txt")

	# Set measurement configuration for grid map update
	# Init range measurement object data resampling configuration (do it once!)
	# See gridmap.py
	AngleIncrement = abs(LegoLogfile.beam_index_to_angle(1)-LegoLogfile.beam_index_to_angle(0))
	AngleStart = LegoLogfile.beam_index_to_angle(0)
	ResamplingFactor = 10.0
	Range_Lidar = RangeMeasurement(AngleIncrement, AngleStart, ResamplingFactor)

	# Loop over all motor tick records.
	# This is the FastSLAM filter loop, with prediction and correction.
	f = open("../out_data/fast_slam_correction_gridmap.txt", "w")
	for i in range(0,int(len(logfile.motor_ticks))):
		print("loop: ",i)
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
		# down sampling measurement (range,bearing)
		measurement = Range_Lidar.data_sampling(logfile.scan_data[i])
		
		fs.correct_gridmap(scanner_offset, measurement, False)
		
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

		
		# ----------------------------------------------------------------------------------------- #
		#                    (Option) Build Global gridmap with the orrected pose                   #
		# ----------------------------------------------------------------------------------------- #

	f.close()
	
	'''
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
	'''
