# ---------------------------------------------------------------------------------
# Occpancy grid map 
# see Probabilistic Robotics by Thrun etc. for details
# Punnu LK Phairatt, 17.03.2016
# ---------------------------------------------------------------------------------
from lib.logfile_reader import *
from lib.slam_library import get_cylinders_from_scan, write_cylinders,\
	write_error_ellipses, get_mean, get_error_ellipse_and_heading_variance,\
	print_particles
from math import *
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

'''
  Compute grid map occcupancy and matching                         
  1.0 = occupied, 0.0 = free, 0.5 = unknown/not explored
  ALL UNITS ARE IN 'mm'

'''
class Gridmap:
	def __init__(self, GridDimension, CentreGrid, Resolution):
		self.g_row = GridDimension[0]                               # N rows
		self.g_col = GridDimension[1]	                            # N cols
		self.CentreGrid  = CentreGrid                               # x=0,y=0 coordinate
		self.Resolution  = Resolution                               # mm/cell
		self.lo = 0.0
		self.locc = 10.0
		self.lfree = -10.0
		self.lmax  = 15.0
		self.l  = np.zeros(shape=(self.g_row, self.g_col)) 			# map log scale matrix
		self.gridmap = 0.5*np.ones(shape=(self.g_row, self.g_col)) 	# create grid map rowxcol with unknow space
		self.beta = 15.0 * pi/180.0    								# +- beam width radian bound
		self.wall = 100                								# +- wall thickness 100 mm bound 	
		self.zmax = 3000               								# maximum range measurement in mm

	def GridToXY(self, Row, Col):
		'''
		   Convert a given row(-Y),col(+X) to X,Y world
		'''
		Xi =  self.Resolution * (Col - self.CentreGrid[1])
		Yi = -self.Resolution * (Row - self.CentreGrid[0])
		return (Xi,Yi)

	def XYToGrid(self, X, Y):
		'''
		   Convert a given X,Y world to row(-Y),col(+X)
		'''
		Col = floor(self.CentreGrid[0] + X/self.Resolution)
		Row = floor(self.CentreGrid[1] - Y/self.Resolution)
		return (Row, Col)
	
	def MeasurementToMap(self, vehicle_pose, scanner_offset, measurement):
		'''
		   Update range and bearing measurement to the grid map
		'''
		dist = measurement[0]
		bearing = measurement[1]
		#update ray cast measurement to the grid world space from the scanner pose
		scanner_pose = LegoLogfile.get_scanner_pose(vehicle_pose, scanner_offset)
		for i in range(len(dist)):
			bearing_s = bearing[i]          								# bearing 
			point_s = dist[i]*cos(bearing[i]), dist[i]*sin(bearing[i])      # X,Y in the scanner frame
			point_w = LegoLogfile.scanner_to_world(scanner_pose, point_s)   # X,Y in the world frame
			row, col = self.XYToGrid(point_w[0],point_w[1])       			# Mapping X,Y to Grid Row Col
			# Update map-check if lmk is within the map range
			# TODO: Dynamic map range extension/ uisng fixed map for now
			if(row >= 0 and row <= self.gridmap.shape[0] and col >=0 and col < self.gridmap.shape[1]):
				self.gridmap.itemset((row,col),1.0)                                     	# set this grid occupied

	def ComputeGridProbability(self, vehicle_pose, scanner_displacement, Lidar):
		'''
		   This function compute and update ccupancy grid given the particle pose and measurements
		'''
		#get current scanner pose
		scanner_pose = LegoLogfile.get_scanner_pose(vehicle_pose, scanner_displacement)     #scanner XY in the map
		scanner_grid = self.XYToGrid(scanner_pose[0],scanner_pose[1])             			#scanner grid in the map
		range_bound  = floor(self.zmax/self.Resolution)									    #n-grid bound
		#loop through all cells within max measurement bound (won't check all elements in the map) 
		for i in range(int(scanner_grid[0]-range_bound), int(scanner_grid[0]+range_bound)):
			if (i<0 or i>self.g_row-1):#outside boundary ignore
				continue
			for j in range(int(scanner_grid[1]-range_bound), int(scanner_grid[1]+range_bound)):
				if (j<0 or j>self.g_col-1):#outside boundary ignore
					continue
				mapXY = self.GridToXY(i, j)
				#range_ = sqrt((scanner_pose[0]-mapXY[0])**2 + (scanner_pose[1]-mapXY[1])**2)
				#if mi cell is in the perceptual field of zt
				l_xy  = self.l.item((i,j)) + self.Inverse_sensor_model(mapXY, scanner_pose, Lidar) - self.lo	
				# avoid exp math overflow (accumulate l) by limit to exp 10
				if abs(l_xy) < self.lmax:
					self.l.itemset((i,j),l_xy)
				else:
					self.l.itemset((i,j), np.sign(l_xy)*self.lmax)
				#compute p(m|zt,xt)	
				self.gridmap.itemset((i,j), 1.0 - 1.0/(1.0+exp(self.l.item(i,j) )))


	def Inverse_sensor_model(self, mapXY, xs, Lidar):
		zt = Lidar.zt
		xi = mapXY[0]
		yi = mapXY[1]
		x = xs[0]
		y = xs[1]
		heading_s = xs[2]
		r = sqrt((xi-x)**2 + (yi-y)**2)
		br = (atan2(yi-y, xi-x) - heading_s + pi) % (2*pi) - pi               # angle need to be in -pi to pi!
		#find zk matching from bearing angle by k = argmin_j(br-bj)
		#finding match zt ray with this cell from the bearing angle
		db = [abs(br- Lidar.index_to_rad(j)) for j in range(len(zt))]
		(min_k,k) = min((min_k,k) for k,min_k in enumerate(db))
		zk = zt[k]
		bk = Lidar.index_to_rad(k)
		#validating zt_k match for assigning probability
		if(r > min(self.zmax, zk + self.wall/2.0) or (abs(br-bk)>self.beta/2.0) ):
			return self.lo
		if(zk < self.zmax and abs(r-zk)< self.wall/2.0 ):
			return self.locc
		if r <= zk:
			return self.lfree
		# if nothing match, return unknown 
		#print("r:",r," zk:",zk, " bearing: ",br)
		return self.lo
		
	def grid_map_correlation(self, vehicle_pose, gridmap_global):
		#extract world map data that corresponding to the local map 						(note: move local map to world)
		row_o, col_o = gridmap_global.XYToGrid(vehicle_pose[0],vehicle_pose[1])                       			#vehicle location in the world
		#find corresponding row col data of the world map 									(note: ignore non-overlap data)
		#work out world map bound index from local map  overlay
		row_top_w   = max(row_o - floor(self.gridmap.shape[0]/2), 0)
		col_left_w  = max(col_o - floor(self.gridmap.shape[1]/2), 0)
		row_low_w   = min(row_o + floor(self.gridmap.shape[0]/2), gridmap_global.gridmap.shape[0])
		col_right_w = min(col_o + floor(self.gridmap.shape[1]/2), gridmap_global.gridmap.shape[1])
		#local r,c centre to the world rc by a relation of adding row_o, col_o 
		#local r,c to world r,c ->  world r,c = local r,c + row_o, col_o - local r,c centre
		#world r,c to local r,c ->  local r,c = world r,c - row_o, col_o + local r,c centre
		
		#now work out local map overlap corresponding to just selected world map
		row_top_l   = row_top_w   - row_o + floor(self.gridmap.shape[0]/2)
		col_left_l  = col_left_w  - col_o + floor(self.gridmap.shape[1]/2)
		row_low_l   = row_low_w   - row_o + floor(self.gridmap.shape[0]/2)
		col_right_l = col_right_w - col_o + floor(self.gridmap.shape[1]/2)
		
		#make sure local and world map have the same size
		m_world = gridmap_global.gridmap[row_top_w:row_low_w+1, col_left_w:col_right_w+1]  #numpy need to add 1 in the end
		m_local = self.gridmap[row_top_l:row_low_l+1, col_left_l:col_right_l+1]            #numpy need to add 1 in the end
		if m_world.shape != m_local.shape:
			print("What the heck, grid map extraction is not equal, map weight can not be done")
		
		#image blur for smoothness
		m_world = cv2.GaussianBlur(m_world,(5,5),0)
		m_local = cv2.GaussianBlur(m_local,(5,5),0)
		#now computing weight
		m_ = (sum(sum(m_local+m_world)))/float(2* m_local.size)
		mn = m_*np.ones(m_local.shape)
		A = sum(sum((m_local-mn) * (m_world-mn)))
		B = sum(sum((m_local-mn)**2)) * sum(sum((m_world-mn)**2))
		matched_score = A/(sqrt(B))  #similarity weight -1 to +1
		return matched_score
		
	@staticmethod	
	def FinalPlotGridMap(fig, gridmap):
		ax = fig.add_subplot(111)
		ax.set_title('colorMap')	
		ax.set_aspect('equal')
		cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
		cax.get_xaxis().set_visible(False)
		cax.get_yaxis().set_visible(False)
		cax.patch.set_alpha(0)
		cax.set_frame_on(False)	
		plt.imshow(gridmap)
		plt.colorbar(orientation='vertical')
		plt.show()
	
		
class RangeMeasurement:
	def __init__(self, DegreeResolution, StartAngle, ResamplingFactor): 
		self.start_angle    = StartAngle
		#self.zt, self.resolution	= self.data_resample(RangeMeasurement, ResamplingFactor, DegreeResolution)
		self.angle_resolution  = ResamplingFactor * DegreeResolution
		self.zt                = None
		self.resampling_factor = ResamplingFactor
		
	def set_resample(self, data_in):
		self.zt = tuple([data_in[i] for i in range(0,len(data_in),int(self.resampling_factor))])
	
	def index_to_rad(self, index):
		return (self.start_angle + (index * self.angle_resolution) + pi)%(2*pi) - pi
		
	def data_sampling(self, data_in):
		distance = []
		bearing  = []
		for i in range(0, len(data_in), int(self.resampling_factor)):
			distance.append(data_in[i])
			bearing.append(self.index_to_rad(i))
		return distance, bearing
	
	
if __name__ == '__main__':
	print('test probability grid map and matching...')
	#Read data.
	logfile = LegoLogfile()
	logfile.read("../in_data/robot4_scan.txt")
	vehicle_corrected_pose = (0.0, 0.0, 0.0)
	scanner_displacement = (0.0, 0.0, 0.0)
	scanner_pose = LegoLogfile.get_scanner_pose(vehicle_corrected_pose, scanner_displacement)
	#Set scanner data for resampling
	data = logfile.scan_data[35]
	AngleIncrement = abs(LegoLogfile.beam_index_to_angle(1)-LegoLogfile.beam_index_to_angle(0))
	AngleStart = LegoLogfile.beam_index_to_angle(0)
	ResamplingFactor = 30.0
	#Reduce scanner data
	Lidar = RangeMeasurement(AngleIncrement, AngleStart, ResamplingFactor)
	Lidar.set_resample(data)
	#measurement = Range_Lidar.data_sampling(data)

	
	#Construct grid map global object e.g. particle global map
	glb_GridDimension = (201, 201)
	glb_CentreGrid    = (101, 101)
	glb_Resolution    = 50             # mm/cell
	GlobalMap = Gridmap(glb_GridDimension, glb_CentreGrid, glb_Resolution)
	GlobalMap.ComputeGridProbability(vehicle_corrected_pose, scanner_displacement, Lidar)
	#Construct grid map local object e.g. particle current observation map (within max range)
	loc_GridDimension = (101, 101)
	loc_CentreGrid    = (51, 51)
	loc_Resolution    = glb_Resolution
	LocalMap = Gridmap(loc_GridDimension, loc_CentreGrid, loc_Resolution)
	LocalMap.ComputeGridProbability((0.0,0.0,45*pi/180), scanner_displacement, Lidar)
	#weight = LocalMap.grid_map_correlation(vehicle_corrected_pose, GlobalMap)
	#print("Correlation score = ", weight)
	
	#Plot computed grid map
	if(1):
		T = np.ones(loc_GridDimension)
		fig, ax_lst = plt.subplots(1,1)
		im = ax_lst.imshow(T, interpolation='nearest',
						   origin='bottom',
						   aspect='auto',
						   vmin=0.0,
						   vmax=1.0,
						   cmap='hot',extent=[0,1,0,1])
		#fig.colorbar(im)
		#ax = fig.add_subplot(111)
		#ax.set_title('colorMap')	
		#ax.set_aspect('equal')
		#cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
		#cax.get_xaxis().set_visible(False)
		#cax.get_yaxis().set_visible(False)
		#cax.patch.set_alpha(0)
		#cax.set_frame_on(False)	
		plt.imshow(LocalMap.gridmap)
		plt.colorbar(orientation='vertical')
		plt.show()
		
		
		T = np.ones(glb_GridDimension)
		fig, ax_lst = plt.subplots(1,1)
		im = ax_lst.imshow(T, interpolation='nearest',
						   origin='bottom',
						   aspect='auto',
						   vmin=0.0,
						   vmax=1.0,
						   cmap='hot',extent=[0,1,0,1])
		#fig.colorbar(im)
		#ax = fig.add_subplot(111)
		#ax.set_title('colorMap')	
		#ax.set_aspect('equal')
		#cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
		#cax.get_xaxis().set_visible(False)
		#cax.get_yaxis().set_visible(False)
		#cax.patch.set_alpha(0)
		#cax.set_frame_on(False)	
		plt.imshow(GlobalMap.gridmap)
		plt.colorbar(orientation='vertical')
		plt.show()
	
	
	