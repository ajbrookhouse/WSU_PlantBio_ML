import os
from os.path import sep
import h5py
import numpy as np
from scipy.ndimage import grey_closing
from connectomics.utils.process import bc_watershed
import time
from matplotlib import pyplot as plt




greyClosing=10
thres1=.85
thres2=.15
thres3=.8
thres_small=1000
outputFile = "C:\\Users\\Aaron\\Documents\\chloroOutLabelTest.h5"





class TimeCounter:
	"""A class that is used to calculate time remaining for long tasks with known number of incremental steps

	This class is used to calculate time left in long tasks and print them out to the user.
	Time can be displayed in hours, minutes, or seconds.
	To use the TimeCounter class, make an instance of the class. To do this, you need to know how many steps the process you are tracking will be
	Each step of the calculation, call the tick function of the instance.
	Anytime you want to display the estimated time remaining, call the print function of the

	Example, if you are doing 10 long calculations and it takes a while:

	counter = TimeCounter(10) #Ten because there are ten steps or calculations
	for i in range(10):
		long_calculation() #Placeholder, any function or block of code would work
		counter.tick() #Updates the TimeCounter, so it knows how far along the total task you are
		counter.print() #Prints out the remaining time left, as calculated by the TimeCounter

	This will print out the estimated time left after calculating each number
	"""

	def __init__(self, numSteps, timeUnits = 'hours', prefix=''):
		"""Initiallizes a TimeCounter instance

		Parameters
		----------
		numSteps : int
			The number of steps that your long calculation / process has, after each step you must call tick()

		timeUnits : {'hours', 'minutes', 'seconds'}
			Sets the time unit that you want to use when displaying progress through the print function

		prefix : str, optional
			If you use this parameter, when printing this prefix will be included before the rest of the output.
			Could use this to provide more detail to the output, or change prefix midway to show in more detail what step the process is on.
		"""

		self.numSteps = numSteps
		self.startTime = time.time()
		self.index = 0
		self.timeUnits = timeUnits
		if timeUnits == 'hours':
			self.scaleFactor = 3600
		if timeUnits == 'minutes':
			self.scaleFactor = 60
		if timeUnits == 'seconds':
			self.scaleFactor = 1
		self.prefix=prefix
		self.remainingTime = None

	def tick(self):
		"""Must be called every time a step is made in the long calculation

		For example, if your long calculation has 10 long steps, and you want to use them to calculate time left:
		Initiallize a TimeCounter with 10 steps, then after each step make sure to call tick()
		You can then use print to print out progress whenever you want.
		"""

		self.index += 1
		currentTime = time.time()
		fractionComplete = self.index / self.numSteps
		timeSofar = currentTime - self.startTime
		totalTime = 1/fractionComplete * timeSofar
		self.remainingTime = totalTime - timeSofar

	def print(self):
		"""Prints the current progress of the TimeCounter
		"""
		print(self.__str__())

	def __str__ (self):
		"""Returns the string value that would be printed by print()
		"""
		if self.remainingTime:
			return self.prefix + ' ' + "{:.2f}".format(self.remainingTime / self.scaleFactor) + ' ' + self.timeUnits + ' left'
		else:
			return "Cannot calculate time left, either just started or close to end"

def instanceProcess(inputH5Filename, cubeSize=1000, stride = 250, combineThreshold=.7):
	
	print("Starting!")
	f = h5py.File(inputH5Filename, 'r+')
	dataset = f['vol0']
	
	if 'processed' in f.keys():
		del(f['processed'])
	if 'processingMap' in f.keys():
		del(f['processingMap'])
		
	f.create_dataset('processed', (dataset.shape[1], dataset.shape[2], dataset.shape[3]), dtype=np.uint16, chunks=True)
	f.create_dataset('processingMap', (dataset.shape[1], dataset.shape[2], dataset.shape[3]), dtype=np.uint8, chunks=True)
	h5out = f['processed']
	map_ = f['processingMap']
	firstChunk = True
	instanceCounter = 1
	
	testCount = 0
	for xiteration in range(0,dataset.shape[1], stride):
		for yiteration in range(0, dataset.shape[2], stride):
			for ziteration in range(0, dataset.shape[3], stride):
				testCount += 1
				
	print('Iterations of Processing Needed:', testCount)
	deltaTracker = TimeCounter(testCount)
	
	for xiteration in range(0,dataset.shape[1], stride):
		for yiteration in range(0, dataset.shape[2], stride):
			for ziteration in range(0, dataset.shape[3], stride):
				print('Loading', xiteration, yiteration, ziteration)
				print("Iteration:",deltaTracker.index,"Out of:",deltaTracker.numSteps)
				print()
				
				xmin = xiteration
				xmax = min(xiteration + cubeSize, dataset.shape[1])
				ymin = yiteration
				ymax = min(yiteration + cubeSize, dataset.shape[2])
				zmin = ziteration
				zmax = min(ziteration + cubeSize, dataset.shape[3])
				
				startSlice = dataset[:,xmin:xmax, ymin:ymax, zmin:zmax]
				plt.imshow(startSlice[0,0,:,:])
				plt.show()
				startSlice[1] = grey_closing(startSlice[1], size=(greyClosing,greyClosing,greyClosing))
				seg = bc_watershed(startSlice, thres1=thres1, thres2=thres2, thres3=thres3, thres_small=thres_small)
				del(startSlice)
				h5Temp = h5out[xmin:xmax, ymin:ymax, zmin:zmax]
				overlapMask = map_[xmin:xmax, ymin:ymax, zmin:zmax] == 1
				notOverlapMask = np.logical_not(overlapMask)

				print("Overlap Count Percent %", np.count_nonzero(overlapMask) / np.count_nonzero(notOverlapMask))
				
				if firstChunk:
					h5Temp = seg
					instanceCounter = seg.max() + 1
					firstChunk = False
					print("FirstChunk", np.unique(seg, return_counts=True))
				else:
					for subid in np.unique(seg[notOverlapMask]):
						print("Subid:", subid)
						if subid == 0:
							continue
						subidMask = seg == subid
						subidInOverlapMask = np.logical_and(subidMask,overlapMask)
						subidInOverlapSize = np.count_nonzero(subidInOverlapMask)
						
						largestOldSubid=0
						largestOldSubidJaccard = 0
						for oldSubid in np.unique(h5Temp[subidInOverlapMask]):
							if subid == 0:
								continue
							oldSubidMask = h5Temp == oldSubid
							unionValue = np.count_nonzero(np.logical_or(oldSubidMask, subidInOverlapMask))
							intersectionValue = np.count_nonzero(np.logical_and(oldSubidMask, subidInOverlapMask))
							jaccard = intersectionValue / unionValue
							if jaccard > largestOldSubidJaccard:
								largestOldSubid = oldSubid
								largestOldSubidJaccard = jaccard
						print("Largest Jaccard:", largestOldSubidJaccard)
						if largestOldSubidJaccard > combineThreshold:
							print("Old Instance")
							h5Temp[np.logical_and(subidMask,notOverlapMask)] = largestOldSubid
						else:
							print("New Instance!")
							h5Temp[subidMask] = instanceCounter
							instanceCounter += 1
						
				#Wrap Up
				print("Writing")
				h5out[xmin:xmax, ymin:ymax, zmin:zmax] = h5Temp
				map_[xmin:xmax, ymin:ymax, zmin:zmax] = 1
				del(h5Temp)
				deltaTracker.tick()
				deltaTracker.print()
				print('==============================')
				print()


instanceProcess(outputFile)