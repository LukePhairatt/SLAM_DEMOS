## **Particle Filter- FastSLAM**
![project][image0]
---

[//]: # (Image References)
[image0]: ./images/viewer.png "result"


# **Overview**
The work of this project shows the step of implementing Particle Filter for SLAM. The work is based on the offline data and visualisation package is taken from the tutorial given by Prof. Claus Brenner. In addition to the original experiment with the feature base approach of landmark association, the project is including an ongoing work on a map base data association without particular landmark matchin ('pf_slam_mapmatching.py'). 


# *FASTSLAM Summary**

**STEP 1-Prediction:**


**STEP 2-Correction:**



# **Runing project.
Need python 3.x to run

To run the simulation, in src folder
'''sh
$ python pf_slam.py
'''

To view result, in src/lib folder
'''sh
$ logfile_viewer.py (and select load load fast_slam_correction.txt)
'''
