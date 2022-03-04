"""

https://wiki.tcl-lang.org/page/List+of+ttk+Themes
adapta is liked
aqua is ok

"""

#######################################
# Imports                             #
#######################################
import matplotlib as mpl
mpl.use('Agg')
import plyer
from tkinter import colorchooser
from scipy.ndimage import grey_closing
from ttkthemes import ThemedTk
import tkinter as tk
import tkinter.ttk as ttk
# from ttkbootstrap import Style as bootStyle
from PIL import ImageColor
from pygubu.widgets.pathchooserinput import PathChooserInput
from connectomics.config import *
import yaml
import yacs
from tkinter import StringVar
from os.path import isdir
import time
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from os.path import sep
from os import listdir
from os import mkdir
import h5py
import open3d as o3d
from tkinter import filedialog as fd
from skimage import measure
from os.path import isfile
from PIL import Image, ImageSequence
from io import StringIO
import threading
from contextlib import redirect_stdout
from contextlib import redirect_stderr
import contextlib
from connectomics.engine import Trainer
from connectomics.config import *
import paramiko
from scipy.ndimage.measurements import label as label2
import torch
import connectomics
import traceback
import sys
import ast
from connectomics.utils.process import bc_watershed
import argparse
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

def rgb2hex(colorTuple):
	r, g, b = colorTuple
	return "#{:02x}{:02x}{:02x}".format(r,g,b)

def getWeightsFromLabels(labelStack): #TODO, make it look at more than just the first image
	im = Image.open(labelStack)
	data = np.array(im)
	unique, nSamples = np.unique(data, return_counts=True)
	m = max(nSamples)
	normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
	return normedWeights

class TimeCounter:
	def __init__(self, numSteps, timeUnits = 'hours', prefix=''):
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

	def tick(self):
		self.index += 1
		currentTime = time.time()

		fractionComplete = self.index / self.numSteps
		timeSofar = currentTime - self.startTime
		totalTime = 1/fractionComplete * timeSofar
		self.remainingTime = totalTime - timeSofar

	def print(self):
		print(self.prefix, self.remainingTime / self.scaleFactor, self.timeUnits, 'left')

def instanceArrayToMesh(d, uniqueList=None):
	if not uniqueList:
		uniqueList, countList = np.unique(d, return_counts=True)

	maxIndex = list(countList).index(max(countList))
	maxValueProb0 = uniqueList[maxIndex]
	fullMesh = o3d.geometry.TriangleMesh()
	deltaTracker = TimeCounter(len(uniqueList), timeUnits='minutes', prefix='Creating Point Cloud: ')
	for subUnique in uniqueList:
		if subUnique == maxValueProb0:
			continue

		trueMask = d == subUnique

		verts, faces, normals, values = measure.marching_cubes(trueMask, 0)
		verts = o3d.utility.Vector3dVector(verts)
		faces = o3d.utility.Vector3iVector(faces)
		subMesh = o3d.geometry.TriangleMesh(verts, faces)
		subMesh.compute_vertex_normals()
		subMesh.paint_uniform_color(np.random.rand(3))
		fullMesh = fullMesh + subMesh
		deltaTracker.tick()
		deltaTracker.print()
	fullMesh = fullMesh.simplify_vertex_clustering(3)
	fullMesh = fullMesh.filter_smooth_taubin(number_of_iterations=100)#filter_smooth_simple(number_of_iterations=10)
	fullMesh.compute_vertex_normals()
	return fullMesh


def instanceArrayToPointCloud(d, uniqueList=None):
	if not uniqueList:
		uniqueList, countList = np.unique(d, return_counts=True)

	maxIndex = list(countList).index(max(countList))
	maxValueProb0 = uniqueList[maxIndex]
	fullCloud = o3d.geometry.PointCloud()
	deltaTracker = TimeCounter(len(uniqueList), timeUnits='minutes', prefix='Creating Point Cloud: ')
	for subUnique in uniqueList:
		#print(subUnique)
		if subUnique == maxValueProb0:
			continue
		pointListWhere = np.where(d == subUnique)
		pointListWhere = np.array(pointListWhere)
		pointListWhere = pointListWhere.transpose()
		subCloud = o3d.geometry.PointCloud()
		subCloud.points = o3d.utility.Vector3dVector(pointListWhere)
		subCloud.paint_uniform_color(np.random.rand(3))
		fullCloud = fullCloud + subCloud
		deltaTracker.tick()
		deltaTracker.print()
	return fullCloud

def InstanceSegmentProcessing(inputH5Filename, greyClosing=10, thres1=.85, thres2=.15, thres3=.8, thres_small=25000, cubeSize=1000):
	f = h5py.File(inputH5Filename, 'r+')
	dataset = f['vol0']

	if 'processed' in f.keys():
		del(f['processed'])
	if 'processingMap' in f.keys():
		del(f['processingMap'])

	print('Creating Datasets in .h5 file')
	f.create_dataset('processed', (dataset.shape[1], dataset.shape[2], dataset.shape[3]), dtype=np.uint16, chunks=True)
	f.create_dataset('processingMap', (dataset.shape[1], dataset.shape[2], dataset.shape[3]), dtype=np.uint8, chunks=True)
	h5out = f['processed']
	map_ = f['processingMap']

	halfCube = int(cubeSize/2)
	quarterCube = int(cubeSize/4)
	offsetList = [0, quarterCube, halfCube]
	countDic = {}
	completeCount = 0

	print(dataset.shape)
	testCount = 0
	for xiteration in range(0,dataset.shape[1], int(cubeSize)):
		for yiteration in range(0, dataset.shape[2], int(cubeSize)):
			for ziteration in range(0, dataset.shape[3], int(cubeSize)):
				testCount += 1
	print('Iterations of Processing Needed', testCount * len(offsetList))
	deltaTracker = TimeCounter(testCount * len(offsetList))

	for offsetStart in offsetList:
		for xiteration in range(offsetStart,dataset.shape[1], int(cubeSize)):
			for yiteration in range(offsetStart, dataset.shape[2], int(cubeSize)):
				for ziteration in range(offsetStart, dataset.shape[3], int(cubeSize)):

					xmin = xiteration
					xmax = min(xiteration + cubeSize, dataset.shape[1]) #TODO should the -1 be here?
					ymin = yiteration
					ymax = min(yiteration + cubeSize, dataset.shape[2])
					zmin = ziteration
					zmax = min(ziteration + cubeSize, dataset.shape[3])

					h5Temp = h5out[xmin:xmax, ymin:ymax, zmin:zmax] 
					mapTemp = map_[xmin:xmax, ymin:ymax, zmin:zmax]

					startSlice = dataset[:,xmin:xmax, ymin:ymax, zmin:zmax]
					startSlice[1] = grey_closing(startSlice[1], size=(greyClosing,greyClosing,greyClosing))
					seg = bc_watershed(startSlice, thres1=thres1, thres2=thres2, thres3=thres3, thres_small=thres_small)
					del(startSlice)

					mapTemp[seg == 0] = 2
					seg[mapTemp == 1] = 0

					for subid in np.unique(seg):
						if subid == 0:
							continue
						idlist = np.where(seg == subid)

						'''
						Following if and elif check to see if the subid blob is touching the edge of
						the scanning box. This is only acceptable if the edge it is touching is
						the edge of the entire dataset. The final else only adds the chloroplasts to
						the processed file if these conditions are met. Otherwise they are ignored and
						hopefully will be picked up when the process reiterates through with a different
						initial offset
						'''
						if 0 in idlist[0] and not xiteration == 0:
							pass
						elif 0 in idlist[1] and not yiteration == 0:
							pass
						elif 0 in idlist[2] and not ziteration == 0:
							pass
						elif xmax - xmin - 1 in idlist[0] and not xmax == dataset.shape[1]:
							pass
						elif ymax - ymin - 1 in idlist[1] and not ymax == dataset.shape[2]:
							pass
						elif zmax - zmin - 1 in idlist[2] and not zmax == dataset.shape[2]:
							pass
						else:
							completeCount += 1
							tempMask = seg == subid
							h5Temp[tempMask] = completeCount
							mapTemp[tempMask] = 1
							countDic[completeCount] = np.count_nonzero(seg == subid)
							del(tempMask)

					h5out[xmin:xmax, ymin:ymax, zmin:zmax] = h5Temp
					map_[xmin:xmax, ymin:ymax, zmin:zmax] = mapTemp
					del(h5Temp)
					del(mapTemp)

					deltaTracker.tick()
					deltaTracker.print()
					print('==============================')

	h5out.attrs['countDictionary'] = str(countDic)
	f.close()

def getMultiClassImage(imageFilepath, uniquePixels=[]):
	if type(imageFilepath) == type('Test'):
		im = Image.open(imageFilepath)
	else:
		im = imageFilepath
	im = im.convert("RGBA")
	data = np.array(im)
	info = np.iinfo(data.dtype) # Get the information of the incoming image type
	data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
	data = 255 * data # Now scale by 255
	a = data.astype(np.uint8)
	b = np.zeros((a.shape[0],a.shape[1]))
	c = list(b)
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			r,g,b,alpha = a[i,j]
			value = (r,g,b)
			if value in uniquePixels:
				c[i][j] = uniquePixels.index(value)
			else:
				uniquePixels.append(value)
				c[i][j] = uniquePixels.index(value)
	d = np.array(c)
	# toReturn = Image.fromarray(d)
	# return toReturn, uniquePixels
	return d, uniquePixels

def getMultiClassImageStack(imageFilepath,uniquePixels=[]):
	labelStack = []
	unique = []
	im = Image.open(imageFilepath)
	for i, imageSlice in enumerate(ImageSequence.Iterator(im)):
		labels, unique = getMultiClassImage(imageSlice, uniquePixels=unique)
		labelStack.append(labels)
	return np.array(labelStack)

def createH5FromNumpy(npArray, filename):
	h5f = h5py.File(filename, 'w')
	h5f.create_dataset('dataset_1', data=npArray)
	h5f.close()



#######################################
# Array Manipulations Function        #
####################################### 

def getImagesForLabels(d, index):
	indexesToCheck = []
	for i in range(d.shape[0]):
		if i == index:
			continue
		indexesToCheck.append(i)
	print('first index')
	mask = d[indexesToCheck[0]] < d[index]
	for i in indexesToCheck[1:]:
		print(i,indexesToCheck)
		mask = mask & d[i] < d[index]
	#TODO output as images

def getPointCloudForIndex(d, index, filename=''):
	indexesToCheck = []
	for i in range(d.shape[0]):
		if i == index:
			continue
		indexesToCheck.append(i)
	print('first index')
	mask = d[indexesToCheck[0]] < d[index]
	for i in indexesToCheck[1:]:
		print(i,indexesToCheck)
		mask = mask & d[i] < d[index]

	print('Creating Point List')
	pointListWhere = np.where(mask == True)
	pointListWhere = np.array(pointListWhere)
	pointListWhere = pointListWhere.transpose()
	print('Creating Cloud')
	cloud = o3d.geometry.PointCloud()
	cloud.points = o3d.utility.Vector3dVector(pointListWhere)
	del(pointListWhere)
	return cloud

def arrayToMesh(d, index):
	indexesToCheck = []
	for i in range(d.shape[0]):
		if i == index:
			continue
		indexesToCheck.append(i)
	print('first index')
	mask = d[indexesToCheck[0]] < d[index]
	for i in indexesToCheck[1:]:
		print(i,indexesToCheck)
		mask = mask & d[i] < d[index]

	array = mask
	print('marching_cubes')
	verts, faces, normals, values = measure.marching_cubes(array, 0)
	print('verts and faces')
	verts = o3d.utility.Vector3dVector(verts)
	faces = o3d.utility.Vector3iVector(faces)
	print('creating triangles')
	mesh = o3d.geometry.TriangleMesh(verts, faces)
	print('calculating normals')
	mesh.compute_vertex_normals()
	print('Simplification Calculations')
	mesh = mesh.simplify_vertex_clustering(3)
	mesh = mesh.filter_smooth_taubin(number_of_iterations=100)#filter_smooth_simple(number_of_iterations=10)
	print('Calculating final Normals')
	mesh.compute_vertex_normals()
	return mesh

#######################################
# Thread Workers                      #
####################################### 

def OutputToolsGetStatsThreadWorker(h5path, streamToUse):
	with redirect_stdout(streamToUse):
		try:
			print('Loading H5 File')
			h5f = h5py.File(h5path, 'r')

			metadata = ast.literal_eval(h5f['vol0'].attrs['metadata'])
			configType = metadata['configType'].lower()

			if 'instance' in configType:
				print('H5 Loaded, reading Stats (Output will be in nanometers^3)')
				countDic = h5f['processed'].attrs['countDictionary']
				metadata = h5f['vol0'].attrs['metadata']
				print(metadata)
				countDic = ast.literal_eval(countDic)
				metadata = ast.literal_eval(metadata)
				xScale, yScale, zScale = metadata['x_scale'], metadata['y_scale'], metadata['z_scale']
				h5f.close()
				countList = []
				for key in countDic.keys():
					countList.append(countDic[key])
				countList = np.array(countList) * xScale * yScale * zScale
				print()
				print('==============================')
				print()
				print('H5File Raw Counts')
				print()
				print(sorted(countList))
				print()
				print('==============================')
				print()
				print('H5File Stats')
				print('Min:', min(countList))
				print('Max:', max(countList))
				print('Mean:', np.mean(countList))
				print('Median:', np.median(countList))
				print('Standard Deviation:', np.std(countList))
				print('Sum:', sum(countList))
				print('Total Number:', len(countList))

			elif 'semantic' in configType:
				d = h5f['vol0'][:]

				for index in range(1, d.shape[0]):
					print()
					print('==============================')
					print('Outputting Stats for layer:', index)
					indexesToCheck = []
					for i in range(d.shape[0]):
						if i == index:
							continue
						indexesToCheck.append(i)
					mask = d[indexesToCheck[0]] < d[index]
					for i in indexesToCheck[1:]:
						mask = mask & d[i] < d[index] ##TODO finish
					labels_out = cc3d.connected_components(mask, connectivity=26)
					del(mask)
			else:
				pass #Unknown File Type
		except:
			print('Critical Error:')
			traceback.print_exc()

def OutputToolsMakeGeometriesThreadWorker(h5path, makeMeshs, makePoints, streamToUse):
	with redirect_stdout(streamToUse):
		try:
			print('Loading H5 File')
			h5f = h5py.File(h5path, 'r')

			metadata = ast.literal_eval(h5f['vol0'].attrs['metadata'])
			configType = metadata['configType'].lower()
			outputFilenameShort = h5path[:-3]

			if 'instance' in configType:
				'''
				INSTANCE SECTION
				'''
				if makeMeshs:
					d = np.array(h5f['processed'])
					#d = d[::3,::3,::3] # Remove
					print('Loaded')
					mesh = instanceArrayToMesh(d)
					print('Finished Calculating Mesh, saving ' + outputFilenameShort + '_instance.ply')
					o3d.io.write_triangle_mesh(outputFilenameShort + '_instance.ply', mesh)
					print('Finished')
				if makePoints:
					d = np.array(h5f['processed'])
					#d = d[::3,::3,::3] # Remove
					print('Loaded')
					cloud = instanceArrayToPointCloud(d)
					print('Finished Calculating Point Cloud, saving')
					o3d.io.write_point_cloud(outputFilenameShort + '_instance.pcd', cloud)
					print('Finished')

			elif 'semantic' in configType:
				'''
				SEMANTIC SECTION
				'''
				d = np.array(h5f['vol0'])
				numIndexes = d.shape[0]
				rootFolder, h5Filename = head_tail = os.path.split(h5path)
				h5Filename = h5Filename[:-3]

				if makeMeshs:
					print('Starting Mesh Creation')
					for index in range(1, numIndexes):
						print('Creating Mesh for Index:', index)
						mesh = arrayToMesh(d, index)
						o3d.io.write_triangle_mesh(outputFilenameShort + '_semantic_' + str(index) + '.ply', mesh)
					print('Finished with making Meshes')
					print()
				if makePoints:
					print('Starting Point Cloud Creation')
					for index in range(1, numIndexes):
						print('Creating Point Cloud for Index:', index)
						cloud = getPointCloudForIndex(d, index)
						o3d.io.write_point_cloud(outputFilenameShort + '_semantic_' + str(index) + '.pcd', cloud)
					print('Finished with making Point Clouds')
					print()
			elif '2D' in configType:
				'''
				2D SECTION
				'''
				pass
			else:
				print('Unrecognized File')

		except:
			print('Critical Error:')
			traceback.print_exc()

def runRemoteServer(url, uname, passw, trainStack, trainLabels, configToUse, submissionScriptString, folderToUse, pytorchFolder, submissionCommand):
	client = paramiko.SSHClient()
	client.load_system_host_keys()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	client.connect(url, username=uname, password=passw)
	scp = SCPClient(client.get_transport())
	scp.put(trainStack, folderToUse + sep + 'trainImages.tif')
	scp.put(trainLabels, folderToUse + sep + 'trainLabels.tif')
	scp.put(configToUse, folderToUse + sep + 'config.yaml')
	with open('submissionScript.sb','w') as outFile:
		outFile.write(submissionScriptString)
	scp.put(submissionScript, folderToUse + sep + 'submissionScript.sb')
	scp.close()
	stdin, stdout, stderr = client.exec_command(submissionCommand)
	output = stdout.read().decode().strip()
	stdin.close()
	stdout.close()
	stderr.close()
	client.close()
	return output

def getMultiClassImage(imageFilepath, uniquePixels=[]):
	if type(imageFilepath) == type('Test'):
		im = Image.open(imageFilepath).convert('RGB')
	else:
		im = imageFilepath.convert('RGB')
	#print('imtype',type(im))
	data = np.array(im)
	#print(data)
	info = np.iinfo(data.dtype) # Get the information of the incoming image type
	data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
	data = 255 * data # Now scale by 255
	a = data.astype(np.uint8)
	b = np.zeros((a.shape[0],a.shape[1]))
	c = list(b)
	#print('atype',type(a),a.shape)
	#print(np.unique(a))
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			#print(a[i,j])
			value = tuple(a[i,j])
			if value in uniquePixels:
				c[i][j] = uniquePixels.index(value)
			else:
				uniquePixels.append(value)
				c[i][j] = uniquePixels.index(value)
	d = np.array(c)
	#toReturn = Image.fromarray(d)
	return d, uniquePixels

def getMultiClassImageStack(imageFilepath,uniquePixels=[]):
	labelStack = []
	unique = []
	im = Image.open(imageFilepath)
	for i, imageSlice in enumerate(ImageSequence.Iterator(im)):
		labels, unique = getMultiClassImage(imageSlice, uniquePixels=unique)
		labelStack.append(labels)
	return np.array(labelStack), unique

def makeLabels(filename):
	labels, unique = getMultiClassImageStack(filename)
	return labels

def getImageForLabelNaming(images, labelArray, index, filename):
	if type(images) == type('str'):
		images = Image.open(images)
	if type(labelArray) == type('str'):
		h5f = h5py.File(labelArray, 'r')
		labelArray = np.array(h5f['dataset_1'])
		h5f.close()
		if labelArray.ndim == 3:
			labelArray = labelArray[0]
	images = images.convert('L').convert('RGB')
	images = np.array(images)
	mask = labelArray == index
	unlabelledImage = np.copy(images)
	images[mask] = (255,0,0)
	plt.subplot(1,2,1)
	plt.imshow(unlabelledImage)
	plt.title('Image')
	plt.subplot(1,2,2)
	plt.title('Label')
	plt.imshow(images)
	plt.savefig(filename)

#######################################
# Remote Server Functions             #
####################################### 

def getRemoteFile(url, uname, passw, filenameToGet, filenameToStore):
	client = paramiko.SSHClient()
	client.load_system_host_keys()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	client.connect(url, username=uname, password=passw)
	scp = SCPClient(client.get_transport())
	scp.get(filenameToGet, filenameToStore)
	scp.close()
	client.close()

def checkStatusRemoteServer(url, name, passw, jobOutputFile):
	client = paramiko.SSHClient()
	client.load_system_host_keys()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	client.connect(url, username=uname, password=passw)
	command = 'cat ' + jobOutputFile
	stdin, stdout, stderr = client.exec_command(command)
	outputLines = stdout.readlines()
	stdin.close()
	stdout.close()
	stderr.close()
	fullOutput = ''
	for line in outputLines:
		fullOutput += line.strip() + '\n'
	client.close()
	return fullOutput

def getSubmissionScriptAsString(template, memory, time, config, outputDirectory):
	file = open(template,'r')
	stringTemplate = file.read()
	file.close()
	stringTemplate.replace('{memory}',memory)
	stringTemplate.replace('{time}',time)
	stringTemplate.replace('{config}',config)
	stringTemplate.replace('{outputFileDir}',outputDirectory)
	return stringTemplate

#######################################
# Machine Learning Helper Functions   #
####################################### 

def get_args():
	parser = argparse.ArgumentParser(description="Model Training & Inference")
	parser.add_argument('--config-file', type=str,
						help='configuration file (yaml)')
	parser.add_argument('--config-base', type=str,
						help='base configuration file (yaml)', default=None)
	parser.add_argument('--inference', action='store_true',
						help='inference mode')
	parser.add_argument('--distributed', action='store_true',
						help='distributed training')
	parser.add_argument('--local_rank', type=int,
						help='node rank for distributed training', default=None)
	parser.add_argument('--checkpoint', type=str, default=None,
						help='path to load the checkpoint')
	# Merge configs from command line (e.g., add 'SYSTEM.NUM_GPUS 8').
	parser.add_argument(
		"opts",
		help="Modify config options using the command-line",
		default=None,
		nargs=argparse.REMAINDER,
	)
	args = parser.parse_args()
	return args

def get_args_modified(modifiedArgs):
	parser = argparse.ArgumentParser(description="Model Training & Inference")
	parser.add_argument('--config-file', type=str,
						help='configuration file (yaml)')
	parser.add_argument('--config-base', type=str,
						help='base configuration file (yaml)', default=None)
	parser.add_argument('--inference', action='store_true',
						help='inference mode')
	parser.add_argument('--distributed', action='store_true',
						help='distributed training')
	parser.add_argument('--local_rank', type=int,
						help='node rank for distributed training', default=None)
	parser.add_argument('--checkpoint', type=str, default=None,
						help='path to load the checkpoint')
	# Merge configs from command line (e.g., add 'SYSTEM.NUM_GPUS 8').
	parser.add_argument(
		"opts",
		help="Modify config options using the command-line",
		default=None,
		nargs=argparse.REMAINDER,
	)
	args = parser.parse_args(modifiedArgs)
	return args

def trainFromMain(config):
	args = get_args_modified(['--config-file', config])

	# if args.local_rank == 0 or args.local_rank is None:
	args.local_rank = None
	print("Command line arguments: ", args)

	manual_seed = 0 if args.local_rank is None else args.local_rank
	np.random.seed(manual_seed)
	torch.manual_seed(manual_seed)

	cfg = load_cfg(args)
	if args.local_rank == 0 or args.local_rank is None:
		# In distributed training, only print and save the
		# configurations using the node with local_rank=0.
		print("PyTorch: ", torch.__version__)
		print(cfg)

		if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
			print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
			os.makedirs(cfg.DATASET.OUTPUT_PATH)
			save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

	if args.distributed:
		assert torch.cuda.is_available(), \
			"Distributed training without GPUs is not supported!"
		dist.init_process_group("nccl", init_method='env://')
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("Rank: {}. Device: {}".format(args.local_rank, device))
	cudnn.enabled = True
	cudnn.benchmark = True

	mode = 'test' if args.inference else 'train'
	trainer = Trainer(cfg, device, mode,
					  rank=args.local_rank,
					  checkpoint=args.checkpoint)

	# Start training or inference:
	if cfg.DATASET.DO_CHUNK_TITLE == 0:
		test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
		test_func() if args.inference else trainer.train()
	else:
		trainer.run_chunk(mode)

	print("Rank: {}. Device: {}. Process is finished!".format(
		  args.local_rank, device))

def predFromMain(config, checkpoint, metaData=''):
	args = get_args_modified(['--inference', '--checkpoint', checkpoint, '--config-file', config])

	# if args.local_rank == 0 or args.local_rank is None:
	args.local_rank = None
	print("Command line arguments: ", args)

	manual_seed = 0 if args.local_rank is None else args.local_rank
	np.random.seed(manual_seed)
	torch.manual_seed(manual_seed)

	cfg = load_cfg(args)
	print('loaded config')
	if args.local_rank == 0 or args.local_rank is None:
		# In distributed training, only print and save the
		# configurations using the node with local_rank=0.
		print("PyTorch: ", torch.__version__)
		print(cfg)

		if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
			print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
			os.makedirs(cfg.DATASET.OUTPUT_PATH)
			save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

	if args.distributed:
		assert torch.cuda.is_available(), \
			"Distributed training without GPUs is not supported!"
		dist.init_process_group("nccl", init_method='env://')
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("Rank: {}. Device: {}".format(args.local_rank, device))
	cudnn.enabled = True
	cudnn.benchmark = True

	mode = 'test' if args.inference else 'train'
	trainer = Trainer(cfg, device, mode,
					  rank=args.local_rank,
					  checkpoint=args.checkpoint)

	print('About to start training')
	# Start training or inference:
	if cfg.DATASET.DO_CHUNK_TITLE == 0:
		test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
		test_func() if args.inference else trainer.train()
	else:
		trainer.run_chunk(mode)

	print("Rank: {}. Device: {}. Process is finished!".format(
		  args.local_rank, device))

	h = h5py.File(os.path.join(cfg["INFERENCE"]["OUTPUT_PATH"] + sep + cfg['INFERENCE']['OUTPUT_NAME']), 'r+')
	h['vol0'].attrs['metadata'] = metaData
	h.close()

@contextlib.contextmanager
def redirect_argv(*args):
	sys._argv = sys.argv[:]
	sys.argv=args
	yield
	sys.argv = sys._argv

def trainThreadWorker(cfg, stream):
	with redirect_stdout(stream):
		try:
			trainFromMain(cfg)
		except:
			traceback.print_exc()

def useThreadWorker(cfg, stream, checkpoint, metaData=''):
	with redirect_stdout(stream):
		try:
			print('About to pred from main')
			predFromMain(cfg, checkpoint, metaData=metaData)
			print('Done with pred from main')

			with open(cfg,'r') as file:
				config = yaml.load(file, Loader=yaml.FullLoader)

			outputFile = os.path.join(config["INFERENCE"]["OUTPUT_PATH"], config['INFERENCE']['OUTPUT_NAME'])
			print('OutputFile:',outputFile)
			InstanceSegmentProcessing(outputFile, greyClosing=10, thres1=.85, thres2=.15, thres3=.8, thres_small=100, cubeSize=1000)
			print('Completely done, output is saved in', outputFile)
		except:
			print('Critical Error')
			traceback.print_exc()

def trainThreadWorkerCluster(cfg, stream, button, url, username, password, trainStack, trainLabels, submissionScriptString, folderToUse, pytorchFolder, submissionCommand):
	with redirect_stdout(stream):
		with redirect_stderr(stream):
			runRemoteServer(url, username, password, trainStack, trainLabels, configToUse, submissionScriptString, folderToUse, pytorchFolder, submissionCommand)
	button['state'] = 'normal'

def ImageToolsCombineImageThreadWorker(pathToCombine, outputFile, streamToUse):
	# with redirect_stdout(streamToUse):
	# 	try:
	# 		if not outputFile[-3:] == '.h5':
	# 			outputFile = outputFile + '.h5'

	# 		im = Image.open(filesToCombine[0])
	# 		height, width = im.size

	# 		h = h5py.File(outputFile, 'w')
	# 		h.create_dataset('dataset_1', (len(filesToCombine), width, height), dtype=np.uint8)
	# 		outData = h['dataset_1']

	# 		index = 0
	# 		for image in list(sorted(filesToCombine)):
	# 			print("Reading image:", image)
	# 			im = Image.open(image)
	# 			im = np.array(im)
	# 			# im = np.transpose(im)

	# 			outData[index,:,:] = im
	# 			index += 1

	# 		h.close()
	# 		print('Finished, files combined to: ', outputFile)

	# 	except:
	# 		print('Critical Error:')
	# 		traceback.print_exc()
	with redirect_stdout(streamToUse):
		try:
			images = []
			for image in list(sorted(listdir(pathToCombine))):
				print("Reading image:", image)
				if not image == '_combined.tif':
					im = Image.open(pathToCombine + sep + image)
					images.append(im)
			print("Writing Combined image:", pathToCombine + sep + '_combined.tif')
			images[0].save(outputFile, save_all=True, append_images=images[1:])
			print("Finished Combining Images")
		except:
			print('Critical Error:')
			traceback.print_exc()


# files = listdir('/media/aaron/Spectroscopy_Images/_InstanceSegmentation/toInfer')
# files2 = []
# files = list(sorted(files))
# for file in files:
# 	if not file[-4:] == '.tif':
# 		continue
# 	files2.append('/media/aaron/Spectroscopy_Images/_InstanceSegmentation/toInfer/' + file)
# ImageToolsCombineImageThreadWorker(files2, '/media/aaron/fullToPredict.h5', sys.stdout)
# exit()



def VisualizeThreadWorker(filesToVisualize, streamToUse, voxel_size=1):
	with redirect_stdout(streamToUse):
		try:
			geometries_to_draw = []
			for file in filesToVisualize:
				filename = file[0]
				filecolor = file[1]
				print('Loading File ' + filename + '(, may take a while)')
				print(filecolor)

				if filename[-4:] == '.ply': #Mesh
					toAdd = o3d.io.read_triangle_mesh(filename)
					if not 'instance' in filename:
						toAdd.paint_uniform_color(np.array(filecolor)/255)
					toAdd.compute_vertex_normals()
					geometries_to_draw.append(toAdd)
				elif filename[-4:] == '.pcd': #Point Cloud
					toAdd = o3d.io.read_point_cloud(filename)
					if not 'instance' in filename:
						toAdd.paint_uniform_color(np.array(filecolor)/255)
					toAdd = o3d.geometry.VoxelGrid.create_from_point_cloud(toAdd, voxel_size=voxel_size)
					geometries_to_draw.append(toAdd)
				else: #Unknown Filetype
					pass
			o3d.visualization.draw_geometries(geometries_to_draw)

		except:
			print('Critical Error:')
			traceback.print_exc()



#######################################
# TK Helper Functions                 #
####################################### 

def complimentColor(hexValue=None, rgbTuple=None): #adopted from stackoverflow, lost link
	if hexValue and rgbTuple:
		raise Exception("Must provide either hexValue or rgbTuple, not both")
	if not hexValue and not rgbTuple:
		raise Exception("Must provide either hexValue or rgbTuple")

	if rgbTuple:
		r, g, b = rgbTuple
	if hexValue:
		r, g, b = ImageColor.getcolor(hexValue, "RGB")

	# https://stackoverflow.com/a/3943023/112731
	if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
		return "#000000"
	else:
		return "#FFFFFF"

class TextboxStream(StringIO): # Replaced by MemoryStream, works a lot nicer with the threads. This was causing issues.
	def __init__(self, widget, maxLen = None):
		super().__init__()
		self.widget = widget

	def write(self, string):
		self.widget.insert("end", string)
		self.widget.see('end')


def MessageBox(message, title=None):
	print(message)
	tk.messagebox.showinfo(title=title, message=message)


class MemoryStream(StringIO):
	def __init__(self):
		super().__init__()
		self.text = ''

	def write(self, string):
		self.text = self.text + string


class FileChooser(ttk.Frame):
	def __init__(self, master=None, labelText='File: ', changeCallback=False, mode='open', title='', buttonText='Choose File', **kw):

		self.changeCallback = changeCallback
		ttk.Frame.__init__(self, master, **kw)
		self.label = ttk.Label(self)
		self.label.configure(text=labelText)
		self.label.grid(column='0', row='0')

		self.sv = StringVar()
		self.sv.trace_add("write", self.entryChangeCallback)
		self.entry = ttk.Entry(self, textvariable=self.sv)
		self.entry.grid(column='1', row='0')
		self.sv.set('')

		self.button = ttk.Button(self)
		self.button.configure(cursor='arrow', text=buttonText)
		self.button.grid(column='3', row='0')
		self.button.configure(command=self.ChooseFileButtonPress)

		self.filepath = self.entry.get()
		self.filepaths = None
		self.mode = mode
		self.title=title
		if self.title == '':
			self.title = 'Select File(s)'

	def entryChangeCallback(self, sv, three, four):
		self.filepath = self.getFilepath()
		if self.changeCallback != False:
			self.changeCallback()


	def ChooseFileButtonPress(self):
		filename = ''
		if self.mode == 'open':
			filename = fd.askopenfilename(title=self.title)
		elif self.mode == 'create':
			filename = fd.asksaveasfilename(title=self.title)
		elif self.mode == 'openMultiple':
			filename = fd.askopenfilenames(title=self.title)
			self.filepaths = filename
		elif self.mode == 'folder':
			filename = fd.askdirectory(title=self.title)
		self.filepath = str(filename)
		self.sv.set(str(self.filepath))
		self.entry.xview("end")

	def getFilepath(self):
		return self.entry.get()

	def getMultiFilepahts(self):
		return self.filepaths


class LayerVisualizerRow(ttk.Frame):
	def __init__(self, master, color, index, changeCallback=False, **kw):
		ttk.Frame.__init__(self, master, **kw)

		self.fileChooser = FileChooser(self, changeCallback = changeCallback)
		self.fileChooser.grid(column='0', row='0')

		self.colorButton = tk.Button(self)
		self.colorButton.configure(cursor='arrow', text='Choose Color')
		self.colorButton.grid(column='1', row='0')
		self.colorButton.configure(command=self.ChooseColor, bg=color, fg=complimentColor(hexValue=color))
		self.color = color

		self.master = master
		self.index = index

	def ChooseColor(self):
		colorTuple = colorchooser.askcolor(title ="Choose Color For Layer " + str(self.index))[0]
		self.color = rgb2hex(colorTuple)
		self.colorButton.configure(bg=self.color, fg = complimentColor(hexValue=self.color))

	def GetColor(self):
		return self.color

	def GetFile(self):
		return self.fileChooser.getFilepath().strip()


class LayerVisualizerContainer(ttk.Frame):
	def __init__(self, master=None, **kw):
		ttk.Frame.__init__(self, master, **kw)

		self.frameToExpand = ttk.Frame(self)
		self.frameToExpand.configure(height='200', width='200')
		self.frameToExpand.pack(side='top')

		self.LayerVisualizerRows = []
		firstVisualizerRow = LayerVisualizerRow(master = self.frameToExpand, color = self.getSuggestedColor(0), index=0, changeCallback = self.changeCallback)
		firstVisualizerRow.grid(column='0', row='0')
		self.LayerVisualizerRows.append(firstVisualizerRow)

	def changeCallback(self): #Carefull if modifying, look for recursion due to passing self.changeCallback to constructor of LayerVisualizerRow
		if len(self.LayerVisualizerRows) == 0: #Function may get called way to early by initializers
			return

		if self.LayerVisualizerRows[-1] == None: #Needed to stop Recursion, this none step is important
			return

		lastFilename = self.LayerVisualizerRows[-1].GetFile()
		twoBackFilename = None

		if len(self.LayerVisualizerRows) > 1: #If the list is long enough, get the second back filename in list
			twoBackFilename = self.LayerVisualizerRows[-2].GetFile()

		if (not twoBackFilename == None) and (twoBackFilename.strip() == lastFilename.strip()) and (lastFilename.strip() == ''): #If the last two are empty, get rid of the last row
			self.LayerVisualizerRows[-1].grid_forget()
			del(self.LayerVisualizerRows[-1])

		elif not lastFilename.strip() == '': #If the last row gets filled, create another row.
			newIndex = len(self.LayerVisualizerRows)
			self.LayerVisualizerRows.append(None)
			nextVisualizerRow = LayerVisualizerRow(master = self.frameToExpand, color = self.getSuggestedColor(newIndex), index=newIndex, changeCallback = self.changeCallback)
			nextVisualizerRow.grid(column='0', row=str(newIndex))
			self.LayerVisualizerRows[-1] = nextVisualizerRow

	def getSuggestedColor(self, index):
		# https://sashamaps.net/docs/resources/20-colors/
		# Using colors from above website at 99.99% accessability, removed white
		colors = ['#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9', '#ffffff', '#000000']
		if index < len(colors):
			colorToReturn = colors[index]
		else:
			colorToReturn = "#000000"
		return colorToReturn

	def getFiles(self):
		filesToReturn = []
		for layer in self.LayerVisualizerRows:
			fileToAdd = layer.GetFile()
			colorToAdd = layer.GetColor()
			if type(colorToAdd) == type('test'):
				colorToAdd = np.array(ImageColor.getcolor(colorToAdd, "RGB"))
			if not fileToAdd == '':
				filesToReturn.append((fileToAdd, colorToAdd))
		return filesToReturn


# print(getWeightsFromLabels('/home/aaron/Documents/WSU_PlantBio_ML/ExampleData/plasmSemanticLabels.tif'))
# exit()

#######################################
# Main Application Class              #
####################################### 

class TabguiApp():
	def __init__(self, master=None):
		self.root = master
		self.root.title("Anatomics MLT")
		# style = bootStyle(theme='sandstone')
		self.root.option_add("*font", "Times_New_Roman 12")

		self.RefreshVariables(firstTime=True)

		# build ui
		self.tabHolder = ttk.Notebook(master)
		self.frameTrain = ttk.Frame(self.tabHolder)
		self.numBoxTrainGPU = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainGPU.configure(from_='0', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxTrainGPU.delete('0', 'end')
		self.numBoxTrainGPU.insert('0', _text_)
		self.numBoxTrainGPU.grid(column='1', row='6')
		self.numBoxTrainCPU = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainCPU.configure(from_='1', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxTrainCPU.delete('0', 'end')
		self.numBoxTrainCPU.insert('0', _text_)
		self.numBoxTrainCPU.grid(column='1', row='7')
		self.pathChooserTrainImageStack = PathChooserInput(self.frameTrain)
		self.pathChooserTrainImageStack.configure(type='file')
		self.pathChooserTrainImageStack.grid(column='1', row='0')
		self.pathChooserTrainLabels = PathChooserInput(self.frameTrain)
		self.pathChooserTrainLabels.configure(type='file')
		self.pathChooserTrainLabels.grid(column='1', row='1')
		self.label1 = ttk.Label(self.frameTrain)
		self.label1.configure(text='Image Stack (.tif or .h5): ')
		self.label1.grid(column='0', row='0')
		self.label2 = ttk.Label(self.frameTrain)
		self.label2.configure(text='Labels (.tif or .h5): ')
		self.label2.grid(column='0', row='1')
		self.label4 = ttk.Label(self.frameTrain)
		self.label4.configure(text='# GPU: ')
		self.label4.grid(column='0', row='6')
		self.label5 = ttk.Label(self.frameTrain)
		self.label5.configure(text='# CPU: ')
		self.label5.grid(column='0', row='7')
		self.label17 = ttk.Label(self.frameTrain)
		self.label17.configure(text='Base LR: ')
		self.label17.grid(column='0', row='16')
		self.label18 = ttk.Label(self.frameTrain)
		self.label18.configure(text='Iteration Step: ')
		self.label18.grid(column='0', row='17')
		self.label19 = ttk.Label(self.frameTrain)
		self.label19.configure(text='Iteration Save: ')
		self.label19.grid(column='0', row='18')
		self.label20 = ttk.Label(self.frameTrain)
		self.label20.configure(text='Iteration Total: ')
		self.label20.grid(column='0', row='19')
		self.label21 = ttk.Label(self.frameTrain)
		self.label21.configure(text='Samples Per Batch: ')
		self.label21.grid(column='0', row='20')
		self.numBoxTrainBaseLR = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainBaseLR.configure(increment='.001', to='1000')
		_text_ = '''.01'''
		self.numBoxTrainBaseLR.delete('0', 'end')
		self.numBoxTrainBaseLR.insert('0', _text_)
		self.numBoxTrainBaseLR.grid(column='1', row='16')
		self.numBoxTrainIterationStep = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainIterationStep.configure(from_='1', increment='1', to='10000000000')
		_text_ = '''1'''
		self.numBoxTrainIterationStep.delete('0', 'end')
		self.numBoxTrainIterationStep.insert('0', _text_)
		self.numBoxTrainIterationStep.grid(column='1', row='17')
		self.numBoxTrainIterationSave = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainIterationSave.configure(from_='1', increment='1000', to='10000000000')
		_text_ = '''5000'''
		self.numBoxTrainIterationSave.delete('0', 'end')
		self.numBoxTrainIterationSave.insert('0', _text_)
		self.numBoxTrainIterationSave.grid(column='1', row='18')
		self.numBoxTrainIterationTotal = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainIterationTotal.configure(from_='1', increment='5000', to='10000000000')
		_text_ = '''100000'''
		self.numBoxTrainIterationTotal.delete('0', 'end')
		self.numBoxTrainIterationTotal.insert('0', _text_)
		self.numBoxTrainIterationTotal.grid(column='1', row='19')
		self.numBoxTrainSamplesPerBatch = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainSamplesPerBatch.configure(from_='1', increment='1', to='100000000000')
		_text_ = '''1'''
		self.numBoxTrainSamplesPerBatch.delete('0', 'end')
		self.numBoxTrainSamplesPerBatch.insert('0', _text_)
		self.numBoxTrainSamplesPerBatch.grid(column='1', row='20')

		# self.separator2 = ttk.Separator(self.frameTrain)
		# self.separator2.configure(orient='horizontal')
		# self.separator2.grid(column='0', columnspan='2', row='2')
		# self.separator2.rowconfigure('2', minsize='30')

		self.configChooserVariable = tk.StringVar(master)
		self.configChooserVariable.set(self.configs[0])
		self.configChooserSelect = ttk.OptionMenu(self.frameTrain, self.configChooserVariable, self.configs[0], *self.configs)
		self.configChooserSelect.grid(column='1', row='2')
		self.labelConfig = ttk.Label(self.frameTrain)
		self.labelConfig.configure(text='Training Config: ')
		self.labelConfig.grid(column='0', row='2')

		self.xyzTrainSubFrame = ttk.Frame(self.frameTrain)
		self.xyzTrainSubFrame.grid(column='0', row='3', columnspan='2')
		self.labelTrainX = ttk.Label(self.xyzTrainSubFrame)
		self.labelTrainX.configure(text='X nm/pixel: ')
		self.labelTrainX.grid(column='0', row='0')
		self.labelTrainY = ttk.Label(self.xyzTrainSubFrame)
		self.labelTrainY.configure(text='Y nm/pixel: ')
		self.labelTrainY.grid(column='0', row='1')
		self.labelTrainZ = ttk.Label(self.xyzTrainSubFrame)
		self.labelTrainZ.configure(text='Z nm/pixel: ')
		self.labelTrainZ.grid(column='0', row='2')

		self.entryTrainX = ttk.Entry(self.xyzTrainSubFrame)
		self.entryTrainX.grid(column='1', row='0')
		self.entryTrainY = ttk.Entry(self.xyzTrainSubFrame)
		self.entryTrainY.grid(column='1', row='1')
		self.entryTrainZ = ttk.Entry(self.xyzTrainSubFrame)
		self.entryTrainZ.grid(column='1', row='2')


		self.separator3 = ttk.Separator(self.frameTrain)
		self.separator3.configure(orient='horizontal')
		self.separator3.grid(column='0', columnspan='2', row='21')
		self.separator3.rowconfigure('21', minsize='30')
		self.label25 = ttk.Label(self.frameTrain)
		self.label25.configure(text='Name: ')
		self.label25.grid(column='0', row='22')
		self.entryTrainModelName = ttk.Entry(self.frameTrain)
		self.entryTrainModelName.grid(column='1', row='22')
		self.checkbuttonTrainClusterRun = ttk.Checkbutton(self.frameTrain)
		self.checkbuttonTrainClusterRun.configure(text='Run On Compute Cluster')
		self.checkbuttonTrainClusterRun.grid(column='0', columnspan='2', row='23')
		self.checkbuttonTrainClusterRun.configure(command=self.trainUseClusterCheckboxPress)

		self.label26 = ttk.Label(self.frameTrain)
		self.label26.configure(text='Cluster URL: ')
		self.label26.grid(column='0', row='24')
		self.label27 = ttk.Label(self.frameTrain)
		self.label27.configure(text='Username: ')
		self.label27.grid(column='0', row='25')
		self.buttonTrainTrain = ttk.Button(self.frameTrain)
		self.buttonTrainTrain.configure(text='Train')
		self.buttonTrainTrain.grid(column='0', columnspan='2', row='27')
		self.buttonTrainTrain.configure(command=self.trainTrainButtonPress)


		self.textTrainOutput = tk.Text(self.frameTrain)
		self.textTrainOutput.configure(height='10', width='50')
		_text_ = '''Training Output Will Be Here'''
		self.textTrainOutput.insert('0.0', _text_)
		self.textTrainOutput.grid(column='0', columnspan='2', row='29')


		self.label28 = ttk.Label(self.frameTrain)
		self.label28.configure(text='Password: ')
		self.label28.grid(column='0', row='26')
		self.entryTrainClusterURL = ttk.Entry(self.frameTrain)
		self.entryTrainClusterURL.configure(state='disabled')
		self.entryTrainClusterURL.grid(column='1', row='24')
		self.entryTrainClusterUsername = ttk.Entry(self.frameTrain)
		self.entryTrainClusterUsername.configure(state='disabled')
		self.entryTrainClusterUsername.grid(column='1', row='25')
		self.entryTrainClusterPassword = ttk.Entry(self.frameTrain, show='*')
		self.entryTrainClusterPassword.configure(state='disabled')
		self.entryTrainClusterPassword.grid(column='1', row='26')
		self.buttonTrainCheckCluster = ttk.Button(self.frameTrain)
		self.buttonTrainCheckCluster.configure(state='disabled', text='Check Cluster Status')
		self.buttonTrainCheckCluster.grid(column='0', columnspan='2', row='28')
		self.buttonTrainCheckCluster.configure(command=self.trainCheckClusterButtonPress)
		self.frameTrain.configure(height='200', width='200')
		self.frameTrain.pack(side='top')
		self.tabHolder.add(self.frameTrain, text='Train')


		#######################################################################################


		self.framePredict = ttk.Frame(self.tabHolder)
		self.pathChooserUseImageStack = FileChooser(self.framePredict, labelText='Image Stack (.tif or .h5): ', mode='open')
		# self.pathChooserUseImageStack.configure(type='file')
		self.pathChooserUseImageStack.grid(column='0', row='0', columnspan='2')
		self.pathChooserUseOutputFile = FileChooser(self.framePredict, labelText='Output File: ', mode='create')
		# self.pathChooserUseOutputFile.configure(type='file')
		self.pathChooserUseOutputFile.grid(column='0', row='1', columnspan='2')
		self.entryUsePadSize = ttk.Entry(self.framePredict)
		_text_ = '''[56, 56, 56]'''
		self.entryUsePadSize.delete('0', 'end')
		self.entryUsePadSize.insert('0', _text_)
		self.entryUsePadSize.grid(column='1', row='6')
		self.entryUseAugMode = ttk.Entry(self.framePredict)
		_text_ = "'mean'"
		self.entryUseAugMode.delete('0', 'end')
		self.entryUseAugMode.insert('0', _text_)
		self.entryUseAugMode.grid(column='1', row='7')
		self.entryUseAugNum = ttk.Entry(self.framePredict)
		_text_ = '''16'''
		self.entryUseAugNum.delete('0', 'end')
		self.entryUseAugNum.insert('0', _text_)
		self.entryUseAugNum.grid(column='01', row='8')
		self.numBoxUseSamplesPerBatch = ttk.Spinbox(self.framePredict)
		self.numBoxUseSamplesPerBatch.configure(from_='1', increment='1', to='100000')
		_text_ = '''1'''
		self.numBoxUseSamplesPerBatch.delete('0', 'end')
		self.numBoxUseSamplesPerBatch.insert('0', _text_)
		self.numBoxUseSamplesPerBatch.grid(column='01', row='10')
		# self.label23 = ttk.Label(self.framePredict)
		# self.label23.configure(text='Image Stack (.tif): ')
		# self.label23.grid(column='0', row='0')
		# self.label24 = ttk.Label(self.framePredict)
		# self.label24.configure(text='Output File: ')
		# self.label24.grid(column='0', row='1')
		self.label29 = ttk.Label(self.framePredict)
		self.label29.configure(text='Pad Size')
		self.label29.grid(column='0', row='6')
		self.label30 = ttk.Label(self.framePredict)
		self.label30.configure(text='Aug Mode: ')
		self.label30.grid(column='0', row='7')
		self.label31 = ttk.Label(self.framePredict)
		self.label31.configure(text='Aug Num: ')
		self.label31.grid(column='0', row='8')
		self.label32 = ttk.Label(self.framePredict)
		self.label32.configure(text='Samples Per Batch: ')
		self.label32.grid(column='0', row='10')
		self.label33 = ttk.Label(self.framePredict)
		self.label33.configure(text='Stride: ')
		self.label33.grid(column='0', row='9')
		self.entryUseStride = ttk.Entry(self.framePredict)
		_text_ = '''[56, 56, 56]'''
		self.entryUseStride.delete('0', 'end')
		self.entryUseStride.insert('0', _text_)
		self.entryUseStride.grid(column='1', row='9')
		self.checkbuttonUseCluster = ttk.Checkbutton(self.framePredict)
		self.checkbuttonUseCluster.configure(text='Run On Compute Cluster')
		self.checkbuttonUseCluster.grid(column='0', columnspan='2', row='23')
		self.checkbuttonUseCluster.configure(command=self.UseModelUseClusterCheckboxPress)
		self.label3 = ttk.Label(self.framePredict)
		self.label3.configure(text='Cluster URL: ')
		self.label3.grid(column='0', row='24')
		self.label8 = ttk.Label(self.framePredict)
		self.label8.configure(text='Username: ')
		self.label8.grid(column='0', row='25')
		self.buttonUseLabel = ttk.Button(self.framePredict)
		self.buttonUseLabel.configure(text='Label')
		self.buttonUseLabel.grid(column='0', columnspan='2', row='27')
		self.buttonUseLabel.configure(command=self.UseModelLabelButtonPress)

		self.textUseOutput = tk.Text(self.framePredict)
		self.textUseOutput.configure(height='10', width='50')
		_text_ = '''Labelling Output Will Be Here'''
		self.textUseOutput.insert('0.0', _text_)
		self.textUseOutput.grid(column='0', columnspan='2', row='28')

		self.label9 = ttk.Label(self.framePredict)
		self.label9.configure(text='Password: ')
		self.label9.grid(column='0', row='26')
		self.entryUseClusterURL = ttk.Entry(self.framePredict)
		self.entryUseClusterURL.configure(state='disabled')
		self.entryUseClusterURL.grid(column='1', row='24')
		self.entryUseClusterUsername = ttk.Entry(self.framePredict)
		self.entryUseClusterUsername.configure(state='disabled')
		self.entryUseClusterUsername.grid(column='1', row='25')
		self.entryUseClusterPassword = ttk.Entry(self.framePredict)
		self.entryUseClusterPassword.configure(state='disabled')
		self.entryUseClusterPassword.grid(column='1', row='26')
		self.separator4 = ttk.Separator(self.framePredict)
		self.separator4.configure(orient='horizontal')
		self.separator4.grid(column='0', columnspan='2', row='11')
		self.separator4.rowconfigure('11', minsize='45')
		self.separator5 = ttk.Separator(self.framePredict)
		self.separator5.configure(orient='horizontal')
		self.separator5.grid(column='0', columnspan='2', row='2')
		self.separator5.rowconfigure('2', minsize='45')
		self.framePredict.configure(height='200', width='200')
		self.framePredict.pack(side='top')
		self.tabHolder.add(self.framePredict, text='Auto-Label')


		self.modelChooserVariable = tk.StringVar(master)
		self.modelChooserVariable.set(self.models[0])
		self.modelChooserSelect = ttk.OptionMenu(self.framePredict, self.modelChooserVariable, self.models[0], *self.models)
		self.modelChooserSelect.grid(column='1', row='2')

		self.labelPredictModelChooser = ttk.Label(self.framePredict)
		self.labelPredictModelChooser.configure(text='Model to Use: ')
		self.labelPredictModelChooser.grid(column='0', row='2')

		######################################################################

		self.frameEvaluate = ttk.Frame(self.tabHolder)
		self.label40 = ttk.Label(self.frameEvaluate)
		self.label40.configure(text='Model Output (.h5): ')
		self.label40.grid(column='0', row='0')
		self.label41 = ttk.Label(self.frameEvaluate)
		self.label41.configure(text='Ground Truth Label(.h5): ')
		self.label41.grid(column='0', row='1')
		self.buttonEvaluateEvaluate = ttk.Button(self.frameEvaluate)
		self.buttonEvaluateEvaluate.configure(text='Evaluate')
		self.buttonEvaluateEvaluate.grid(column='0', columnspan='2', row='2')
		self.buttonEvaluateEvaluate.configure(command=self.EvaluateModelEvaluateButtonPress)
		self.pathchooserinputEvaluateLabel = PathChooserInput(self.frameEvaluate)
		self.pathchooserinputEvaluateLabel.configure(type='file')
		self.pathchooserinputEvaluateLabel.grid(column='1', row='1')
		self.pathchooserinputEvaluateModelOutput = PathChooserInput(self.frameEvaluate)
		self.pathchooserinputEvaluateModelOutput.configure(type='file')
		self.pathchooserinputEvaluateModelOutput.grid(column='1', row='0')
		self.frameEvaluate.configure(height='200', width='200')
		self.frameEvaluate.pack(side='top')
		self.tabHolder.add(self.frameEvaluate, text='Evaluate Model')

		##################################################################################################################
		# Section Image Tools
		##################################################################################################################

		self.frameImage = ttk.Frame(self.tabHolder)

		self.fileChooserImageToolsInput = FileChooser(self.frameImage, labelText='Folder to Combine (input .tif files): ', mode='folder', title='Files To Combine', buttonText='Choose Folder of Images to Combine')
		self.fileChooserImageToolsInput.grid(column='0', row='0', columnspan='2')

		self.fileChooserImageToolsOutput = FileChooser(self.frameImage, labelText='Output Filename ', mode='create', title='Output Filename', buttonText='Choose Output File')
		self.fileChooserImageToolsOutput.grid(column='0', row='1', columnspan='2')

		self.buttonImageCombine = ttk.Button(self.frameImage)
		self.buttonImageCombine.configure(text='Combine Images Into H5 File Stack')
		self.buttonImageCombine.grid(column='0', row='2')
		self.buttonImageCombine.configure(command=self.ImageToolsCombineImageButtonPress)

		self.textImageTools = tk.Text(self.frameImage)
		self.textImageTools.configure(height='10', width='75')
		_text_ = '''Image Tools Output Will Be Here'''
		self.textImageTools.insert('0.0', _text_)
		self.textImageTools.grid(column='0', columnspan='2', row='3')

		self.frameImage.configure(height='200', width='200')
		self.frameImage.pack(side='top')
		self.tabHolder.add(self.frameImage, text='Image Tools')

		##################################################################################################################
		# 
		##################################################################################################################

		self.frameOutputTools = ttk.Frame(self.tabHolder)
		self.fileChooserOutputStats = FileChooser(master=self.frameOutputTools, labelText='Model Output (.h5): ', changeCallback=False, mode='open', title='Choose File', buttonText='Choose File')
		self.fileChooserOutputStats.grid(column='1', row='0')

		self.buttonOutputGetStats = ttk.Button(self.frameOutputTools)
		self.buttonOutputGetStats.configure(text='Get Model Output Stats')
		self.buttonOutputGetStats.grid(column='0', columnspan='2', row='4')
		self.buttonOutputGetStats.configure(command=self.OutputToolsModelOutputStatsButtonPress)

		self.textOutputOutput = tk.Text(self.frameOutputTools)
		self.textOutputOutput.configure(height='25', width='75')
		_text_ = '''Output Goes Here'''
		self.textOutputOutput.insert('0.0', _text_)
		self.textOutputOutput.grid(column='0', columnspan='2', row='5')

		self.checkbuttonOutputMeshs = ttk.Checkbutton(self.frameOutputTools)
		self.checkbuttonOutputMeshs.configure(text='Meshs')
		self.checkbuttonOutputMeshs.grid(column='0', row='2')

		self.checkbuttonOutputPointClouds = ttk.Checkbutton(self.frameOutputTools)
		self.checkbuttonOutputPointClouds.configure(text='Point Clouds')
		self.checkbuttonOutputPointClouds.grid(column='1', row='2')

		self.buttonOutputMakeGeometries = ttk.Button(self.frameOutputTools)
		self.buttonOutputMakeGeometries.configure(text='Make Geometries')
		self.buttonOutputMakeGeometries.grid(column='0', columnspan='2', row='3')
		self.buttonOutputMakeGeometries.configure(command=self.OutputToolsMakeGeometriesButtonPress)

		self.frameOutputTools.configure(height='200', width='200')
		self.frameOutputTools.pack(side='top')
		self.tabHolder.add(self.frameOutputTools, text='Output Tools')

		############################################################################################

		self.frameVisualize = ttk.Frame(self.tabHolder)
		self.frameVisualize.configure(height='200', width='200')
		self.frameVisualize.pack(side='top')

		self.visualizeRowsHolder = LayerVisualizerContainer(self.frameVisualize)
		self.visualizeRowsHolder.grid(column='0', row='1')

		self.buttonVisualize = ttk.Button(self.frameVisualize)
		self.buttonVisualize.configure(text='Visualize')
		self.buttonVisualize.grid(column='0', row='2')
		self.buttonVisualize.configure(command=self.VisualizeButtonPress)

		self.textVisualizeOutput = tk.Text(self.frameVisualize)
		self.textVisualizeOutput.configure(height='10', width='50')
		_text_ = '''Output Goes Here'''
		self.textVisualizeOutput.insert('0.0', _text_)
		self.textVisualizeOutput.grid(column='0', row='3')

		self.tabHolder.add(self.frameVisualize, text='Visualize')

		self.checkbuttonUseCluster.invoke()
		self.checkbuttonUseCluster.invoke()
		self.checkbuttonTrainClusterRun.invoke()
		self.checkbuttonTrainClusterRun.invoke()
		self.checkbuttonOutputPointClouds.invoke()
		self.checkbuttonOutputMeshs.invoke()

		############################################################################################
		"""
		self.frameConfig = ttk.Frame(self.tabHolder)
		self.label10 = ttk.Label(self.frameConfig)
		self.label10.configure(text='Architecture: ')
		self.label10.grid(column='0', row='5')
		self.label10.columnconfigure('0', minsize='0')
		self.label11 = ttk.Label(self.frameConfig)
		self.label11.configure(text='Input Size: ')
		self.label11.grid(column='0', row='6')
		self.label11.columnconfigure('0', minsize='0')
		self.label12 = ttk.Label(self.frameConfig)
		self.label12.configure(text='Output Size: ')
		self.label12.grid(column='0', row='7')
		self.label12.columnconfigure('0', minsize='0')
		self.label13 = ttk.Label(self.frameConfig)
		self.label13.configure(text='In Planes: ')
		self.label13.grid(column='0', row='8')
		self.label13.columnconfigure('0', minsize='0')
		self.label14 = ttk.Label(self.frameConfig)
		self.label14.configure(text='Out Planes: ')
		self.label14.grid(column='0', row='9')
		self.label14.columnconfigure('0', minsize='0')
		self.label15 = ttk.Label(self.frameConfig)
		self.label15.configure(text='Loss Option: ')
		self.label15.grid(column='0', row='10')
		self.label15.columnconfigure('0', minsize='0')
		self.label16 = ttk.Label(self.frameConfig)
		self.label16.configure(text='Loss Weight: ')
		self.label16.grid(column='0', row='11')
		self.label16.columnconfigure('0', minsize='0')
		self.label22 = ttk.Label(self.frameConfig)
		self.label22.configure(text='Target Opt: ')
		self.label22.grid(column='0', row='12')
		self.label22.columnconfigure('0', minsize='0')
		self.label34 = ttk.Label(self.frameConfig)
		self.label34.configure(text='Weight Opt')
		self.label34.grid(column='0', row='13')
		self.label34.columnconfigure('0', minsize='0')
		self.label35 = ttk.Label(self.frameConfig)
		self.label35.configure(text='Pad Size: ')
		self.label35.grid(column='0', row='14')
		self.label35.columnconfigure('0', minsize='0')
		self.label36 = ttk.Label(self.frameConfig)
		self.label36.configure(text='LR_Scheduler: ')
		self.label36.grid(column='0', row='15')
		self.label36.columnconfigure('0', minsize='0')
		self.label37 = ttk.Label(self.frameConfig)
		self.label37.configure(text='Base LR: ')
		self.label37.grid(column='0', row='16')
		self.label37.columnconfigure('0', minsize='0')
		self.label38 = ttk.Label(self.frameConfig)
		self.label38.configure(text='Steps: ')
		self.label38.grid(column='0', row='21')
		self.label38.columnconfigure('0', minsize='0')
		self.entryConfigArchitecture = ttk.Entry(self.frameConfig)
		_text_ = '''unet_residual_3d'''
		self.entryConfigArchitecture.delete('0', 'end')
		self.entryConfigArchitecture.insert('0', _text_)
		self.entryConfigArchitecture.grid(column='1', row='5')
		self.entryConfigInputSize = ttk.Entry(self.frameConfig)
		_text_ = '''[112, 112, 112]'''
		self.entryConfigInputSize.delete('0', 'end')
		self.entryConfigInputSize.insert('0', _text_)
		self.entryConfigInputSize.grid(column='1', row='6')
		self.entryConfigOutputSize = ttk.Entry(self.frameConfig)
		_text_ = '''[112, 112, 112]'''
		self.entryConfigOutputSize.delete('0', 'end')
		self.entryConfigOutputSize.insert('0', _text_)
		self.entryConfigOutputSize.grid(column='1', row='7')
		self.numBoxConfigInPlanes = ttk.Spinbox(self.frameConfig)
		self.numBoxConfigInPlanes.configure(from_='1', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxConfigInPlanes.delete('0', 'end')
		self.numBoxConfigInPlanes.insert('0', _text_)
		self.numBoxConfigInPlanes.grid(column='1', row='8')
		self.numBoxConfigOutPlanes = ttk.Spinbox(self.frameConfig)
		self.numBoxConfigOutPlanes.configure(from_='1', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxConfigOutPlanes.delete('0', 'end')
		self.numBoxConfigOutPlanes.insert('0', _text_)
		self.numBoxConfigOutPlanes.grid(column='1', row='9')
		self.entryConfigLossOption = ttk.Entry(self.frameConfig)
		_text_ = '''[['WeightedBCE', 'DiceLoss']]'''
		self.entryConfigLossOption.delete('0', 'end')
		self.entryConfigLossOption.insert('0', _text_)
		self.entryConfigLossOption.grid(column='1', row='10')
		self.entryConfigLossWeight = ttk.Entry(self.frameConfig)
		_text_ = '''[[1.0, 1.0]]'''
		self.entryConfigLossWeight.delete('0', 'end')
		self.entryConfigLossWeight.insert('0', _text_)
		self.entryConfigLossWeight.grid(column='1', row='11')
		self.entryConfigTargetOpt = ttk.Entry(self.frameConfig)
		_text_ = '''['0']'''
		self.entryConfigTargetOpt.delete('0', 'end')
		self.entryConfigTargetOpt.insert('0', _text_)
		self.entryConfigTargetOpt.grid(column='1', row='12')
		self.entryConfigWeightOpt = ttk.Entry(self.frameConfig)
		_text_ = '''[['1', '0']]'''
		self.entryConfigWeightOpt.delete('0', 'end')
		self.entryConfigWeightOpt.insert('0', _text_)
		self.entryConfigWeightOpt.grid(column='1', row='13')
		self.entryConfigPadSize = ttk.Entry(self.frameConfig)
		_text_ = '''[56, 56, 56]'''
		self.entryConfigPadSize.delete('0', 'end')
		self.entryConfigPadSize.insert('0', _text_)
		self.entryConfigPadSize.grid(column='1', row='14')
		self.entryConfigLRScheduler = ttk.Entry(self.frameConfig)
		_text_ = '''"WarmupMultiStepLR"'''
		self.entryConfigLRScheduler.delete('0', 'end')
		self.entryConfigLRScheduler.insert('0', _text_)
		self.entryConfigLRScheduler.grid(column='1', row='15')
		self.spinboxConfigBaseLR = ttk.Spinbox(self.frameConfig)
		self.spinboxConfigBaseLR.configure(increment='.001', to='1000')
		_text_ = '''.01'''
		self.spinboxConfigBaseLR.delete('0', 'end')
		self.spinboxConfigBaseLR.insert('0', _text_)
		self.spinboxConfigBaseLR.grid(column='1', row='16')
		self.entryConfigSteps = ttk.Entry(self.frameConfig)
		_text_ = '''(80000, 90000)'''
		self.entryConfigSteps.delete('0', 'end')
		self.entryConfigSteps.insert('0', _text_)
		self.entryConfigSteps.grid(column='1', row='21')
		self.label39 = ttk.Label(self.frameConfig)
		self.label39.configure(text='Name: ')
		self.label39.grid(column='0', row='23')
		self.label39.columnconfigure('0', minsize='0')
		self.entryConfigName = ttk.Entry(self.frameConfig)
		self.entryConfigName.grid(column='1', row='23')
		self.buttonConfigSave = ttk.Button(self.frameConfig)
		self.buttonConfigSave.configure(text='Save Config')
		self.buttonConfigSave.grid(column='0', columnspan='2', row='24')
		self.buttonConfigSave.columnconfigure('0', minsize='0')
		self.buttonConfigSave.configure(command=self.SaveConfigButtonPress)
		self.separator1 = ttk.Separator(self.frameConfig)
		self.separator1.configure(orient='horizontal')
		self.separator1.grid(column='0', columnspan='2', row='22')
		self.separator1.rowconfigure('22', minsize='50')
		self.separator1.columnconfigure('0', minsize='0')
		self.frameConfig.configure(height='750', width='200')
		self.frameConfig.pack(side='top')
		self.tabHolder.add(self.frameConfig, text='Make Configs')
		"""
		self.tabHolder.pack(side='top')

		# Main widget
		self.mainwindow = self.tabHolder

	def trainUseClusterCheckboxPress(self):
		status = self.checkbuttonTrainClusterRun.instate(['selected'])
		if status == True:
			status = 'normal'
		else:
			status = 'disabled'
		self.entryTrainClusterURL['state'] = status
		self.entryTrainClusterUsername['state'] = status
		self.entryTrainClusterPassword['state'] = status
		self.buttonTrainCheckCluster['state'] = status

	def VisualizeButtonPress(self):
		self.buttonVisualize['state'] = 'disabled'
		filesToVisualize = self.visualizeRowsHolder.getFiles()
		memStream = MemoryStream()
		t = threading.Thread(target=VisualizeThreadWorker, args=(filesToVisualize, memStream, 5))
		t.setDaemon(True)
		t.start()
		self.longButtonPressHandler(t, memStream, self.textVisualizeOutput, [self.buttonVisualize])


	def longButtonPressHandler(self, thread, memStream, textBox, listToReEnable, refreshTime=1000):
		textBox.delete(1.0,"end")
		textBox.insert("end", memStream.text)
		textBox.see('end')

		if thread.is_alive():
			self.root.after(refreshTime, lambda: self.longButtonPressHandler(thread, memStream, textBox, listToReEnable, refreshTime))
		else:
			for element in listToReEnable:
				element['state'] = 'normal'
			self.RefreshVariables()

	def trainTrainButtonPress(self):
		self.buttonTrainTrain['state'] = 'disabled'
		try:

			cluster = self.checkbuttonTrainClusterRun.instate(['selected'])
			configToUse = self.configChooserVariable.get()
			gpuNum = self.numBoxTrainGPU.get()
			cpuNum = self.numBoxTrainCPU.get()
			lr = self.numBoxTrainBaseLR.get()
			itStep = self.numBoxTrainIterationStep.get()
			itSave = self.numBoxTrainIterationSave.get()
			itTotal = self.numBoxTrainIterationTotal.get()
			samples = self.numBoxTrainSamplesPerBatch.get()

			name = self.entryTrainModelName.get()
			image = self.pathChooserTrainImageStack.entry.get()
			labels = self.pathChooserTrainLabels.entry.get()

			sizex = int(self.entryTrainX.get())
			sizey = int(self.entryTrainY.get())
			sizez = int(self.entryTrainZ.get())

			if isdir('Data' + sep + 'models' + sep + name):
				pass #TODO Check if want to continue, if so get latest checkpoint

			with open('Data' + sep + 'configs' + sep + configToUse,'r') as file:
				config = yaml.load(file, Loader=yaml.FullLoader)

			config['SYSTEM']['NUM_GPUS'] = gpuNum
			config['SYSTEM']['NUM_CPUS'] = cpuNum
			config['DATASET']['IMAGE_NAME'] = image
			config['DATASET']['LABEL_NAME'] = labels
			config['DATASET']['OUTPUT_PATH'] = 'Data' + sep + 'models' + sep + name + sep
			config['SOLVER']['BASE_LR'] = lr
			config['SOLVER']['ITERATION_STEP'] = itStep
			config['SOLVER']['ITERATION_SAVE'] = itSave
			config['SOLVER']['ITERATION_TOTAL'] = itTotal
			config['SOLVER']['SAMPLES_PER_BATCH'] = samples

			if 'semantic' in configToUse.lower():
				weightsToUse = []
				weights = list(getWeightsFromLabels(labels))
				for weight in weights:
					weightsToUse.append(float(weight))

				config['MODEL']['TARGET_OPT'] = ['9-' + str(len(weights))] #Target Opt
				config['MODEL']['OUT_PLANES'] = len(weights) #Output Planes
				config['MODEL']['LOSS_KWARGS_VAL'] = list([[[weightsToUse]]]) #Class Weights

			if not isdir('Data' + sep + 'models' + sep + name):
				mkdir('Data' + sep + 'models' + sep + name)

			with open("Data" + sep + "models" + sep + name + sep + "config.yaml", 'w') as file:
				yaml.dump(config, file)

			metaDictionary = {} #TODO add more info to metadaa like nm/pixel
			metaDictionary['configType'] = configToUse
			metaDictionary['x_scale'] = sizex
			metaDictionary['y_scale'] = sizey
			metaDictionary['z_scale'] = sizez
			with open("Data" + sep + "models" + sep + name + sep + "metadata.yaml", 'w') as file:
				yaml.dump(metaDictionary, file)

			if cluster:
				url = self.entryTrainClusterURL.get()
				username = self.entryTrainClusterUsername.get()
				password = self.entryTrainClusterPassword.get()
				submissionScriptString = getSubmissionScriptAsString('configServer2.yaml','20gb','00:10:00','configServer2.yaml','output')
				t = threading.Thread(target=trainThreadWorkerCluster, args=(cfg, self.textTrainOutputStream, self.buttonTrainTrain, url, username, password, image, labels, submissionScriptString, 'projects/pytorch_connectomics', 'projects/pytorch_connectomics', 'sbatch submissionScript.sb'))
				# cfg, stream, button, url, username, password, trainStack, trainLabels, submissionScriptString, folderToUse, pytorchFolder, submissionCommand
				t.setDaemon(True)
				t.start()
			else:
				memStream = MemoryStream()
				t = threading.Thread(target=trainThreadWorker, args=("Data" + sep + "models" + sep + name + sep + "config.yaml", memStream))
				t.setDaemon(True)
				t.start()
				self.longButtonPressHandler(t, memStream, self.textTrainOutput, [self.buttonTrainTrain])
		except:
			self.buttonTrainTrain['state'] = 'normal'
			traceback.print_exc()

	def trainCheckClusterButtonPress(self):
		pass

	def UseModelUseClusterCheckboxPress(self):
		status = self.checkbuttonUseCluster.instate(['selected'])
		if status == True:
			status = 'normal'
		else:
			status = 'disabled'
		self.entryUseClusterURL['state'] = status
		self.entryUseClusterUsername['state'] = status
		self.entryUseClusterPassword['state'] = status

	def getConfigForModel(self, model):
		return "Data" + sep + "models" + sep + model + sep + "config.yaml"

	def getLastCheckpointForModel(self, model):
		checkpointFiles = os.listdir('Data' + sep + 'models' + sep + model)
		biggestCheckpoint = 0

		for subFile in checkpointFiles:
			try:
				checkpointNumber = int(subFile.split('_')[1][:-8])
			except:
				pass
			if checkpointNumber > biggestCheckpoint:
				biggestCheckpoint = checkpointNumber
			#print('biggest checkpoint',biggestCheckpoint)
		biggestCheckpoint = 'Data' + sep + 'models' + sep + model + sep + 'checkpoint_' + str(biggestCheckpoint).zfill(5) + '.pth.tar'
		return biggestCheckpoint

	def getMetadataForModel(self, model):
		with open('Data' + sep + 'models' + sep + model + sep + 'metadata.yaml','r') as file:
			metaData = yaml.load(file, Loader=yaml.FullLoader)
		metaDataStr = str(metaData)
		print('Metadata For Model:', model, '|', metaDataStr)
		return metaDataStr

	def UseModelLabelButtonPress(self):
		self.buttonUseLabel['state'] = 'disabled'
		try:
			cluster = self.checkbuttonUseCluster.instate(['selected'])
			model = self.modelChooserVariable.get()
			gpuNum = 1 #self.numBoxTrainGPU.get()
			cpuNum = 1 #self.numBoxTrainCPU.get()
			samples = self.numBoxUseSamplesPerBatch.get()
			outputFile = self.pathChooserUseOutputFile.entry.get()
			image = self.pathChooserUseImageStack.entry.get()

			padSize = self.entryUsePadSize.get()
			augMode = self.entryUseAugMode.get()
			augNum = self.entryUseAugNum.get()
			stride = self.entryUseStride.get()

			configToUse = self.getConfigForModel(model)
			print('Config to use:', configToUse)
			with open(configToUse,'r') as file:
				config = yaml.load(file, Loader=yaml.FullLoader)

			config['SYSTEM']['NUM_GPUS'] = gpuNum
			config['SYSTEM']['NUM_CPUS'] = cpuNum

			config['INFERENCE']['OUTPUT_PATH'] = os.path.split(outputFile)[0]
			outName = os.path.split(outputFile)[1]
			if not outName.split('.')[-1] == 'h5':
				outName += '.h5'
			config['INFERENCE']['OUTPUT_NAME'] = outName
			config['INFERENCE']['IMAGE_NAME'] = image
			config['INFERENCE']['SAMPLES_PER_BATCH'] = samples

			with open('temp.yaml','w') as file:
				yaml.dump(config, file)

			print(config)

			if cluster: #TODO Fix Cluster
				url = self.entryTrainClusterURL.get()
				username = self.entryTrainClusterUsername.get()
				password = self.entryTrainClusterPassword.get()
				submissionScriptString = getSubmissionScriptAsString('configServer2.yaml','20gb','00:10:00','configServer2.yaml','output')
				t = threading.Thread(target=useThreadWorkerCluster, args=(cfg, self.textUseOutputStream, self.buttonUseLabel, url, username, password, image, labels, submissionScriptString, 'projects/pytorch_connectomics', 'projects/pytorch_connectomics', 'sbatch submissionScript.sb'))
				# cfg, stream, button, url, username, password, trainStack, trainLabels, submissionScriptString, folderToUse, pytorchFolder, submissionCommand
				t.setDaemon(True)
				t.start()
			else:
				print('Starting Non Cluster')				
				biggestCheckpoint = self.getLastCheckpointForModel(model)
				metaData = self.getMetadataForModel(model)
				memStream = MemoryStream()
				t = threading.Thread(target=useThreadWorker, args=('temp.yaml', memStream, biggestCheckpoint, metaData))
				t.setDaemon(True)
				t.start()
				self.longButtonPressHandler(t, memStream, self.textUseOutput, [self.buttonUseLabel])
		except:
			traceback.print_exc()
			self.buttonUseLabel['state'] = 'normal'

	def EvaluateModelEvaluateButtonPress(self):
		labelImage = self.pathChooserEvaluateLabels.entry.get()
		modelOutput = self.pathChooserEvaluateModelOutput.entry.get()
		labels = []
		im = Image.open(labelImage)
		for i, frame in enumerate(ImageSequence.Iterator(im)):
			framearr = np.asarray(frame)
			labels.append(framearr)
		labels = np.array(labels)

		h = h5py.File(modelOutput,'r')
		pred = np.array(h['vol0'][0])
		h.close()

		cutoffs = []
		ls = []
		ps = []
		percentDiffs = []
		precisions = []
		accuracies = []
		recalls = []
		for cutoff in range(0, 30, 1):
			cutoffs.append(cutoff)
			workingPred = np.copy(pred)
			workingPred[workingPred >= cutoff] = 255
			workingPred[workingPred != 255] = 0

			tp = np.sum((workingPred == labels) & (labels==255))
			tn = np.sum((workingPred == labels) & (labels==0))
			fp = np.sum((workingPred != labels) & (labels==0))
			fn = np.sum((workingPred != labels) & (labels==255))

			percentDiff = 1 - (np.count_nonzero(labels==255) - np.count_nonzero(workingPred==255))/np.count_nonzero(workingPred==255)
			percentDiffs.append(percentDiff)

			precisions.append(tp/(tp+fp))
			recalls.append(tp/(tp+fn))
			accuracies.append((tp + tn)/(tp + fp + tn + fn))

			ls.append(np.count_nonzero(labels))
			ps.append(np.count_nonzero(workingPred))
			del(workingPred)

		precisions = np.array(precisions)
		recalls = np.array(recalls)
		f1 = 2 * (precisions * recalls)/(precisions + recalls)

		plt.plot(cutoffs, precisions, label='precision')
		plt.plot(cutoffs, recalls, label='recall')
		plt.plot(cutoffs, f1, label='f1')
		#plt.plot(cutoffs, accuracies, label='accuracy')
		plt.plot(cutoffs, percentDiffs, label='percent differences')
		plt.legend()
		plt.grid()
		plt.show()

	def ImageToolsCombineImageButtonPress(self):
		try:
			memStream = MemoryStream()
			self.buttonImageCombine['state'] = 'disabled'
			filesToCombine = self.fileChooserImageToolsInput.getFilepath()
			outputFile = self.fileChooserImageToolsOutput.getFilepath() + '.tif'
			if not outputFile[-4:] == '.tif':
				outputFile = outputFile + '.tif'
			t = threading.Thread(target=ImageToolsCombineImageThreadWorker, args=(filesToCombine, outputFile, memStream))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textImageTools, [self.buttonImageCombine])
		except:
			traceback.print_exc()
			self.buttonTrainTrain['state'] = 'normal'

	def OutputToolsModelOutputStatsButtonPress(self):
		try:
			memStream = MemoryStream()
			self.buttonOutputGetStats['state'] = 'disabled'
			filename = self.fileChooserOutputStats.getFilepath() #TODO get file name
			t = threading.Thread(target=OutputToolsGetStatsThreadWorker, args=(filename, memStream))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textOutputOutput, [self.buttonOutputGetStats])
		except:
			traceback.print_exc()
			self.buttonOutputGetStats['state'] = 'normal'

	def OutputToolsMakeGeometriesButtonPress(self):
		try:
			memStream = MemoryStream()
			self.buttonOutputMakeGeometries['state'] = 'disabled'
			h5Path = self.fileChooserOutputStats.getFilepath()
			makeMeshs = self.checkbuttonOutputMeshs.instate(['selected'])
			makePoints = self.checkbuttonOutputPointClouds.instate(['selected'])
			t = threading.Thread(target=OutputToolsMakeGeometriesThreadWorker, args=(h5Path, makeMeshs, makePoints, memStream))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textOutputOutput, [self.buttonOutputMakeGeometries])
		except:
			traceback.print_exc()
			self.buttonOutputMakeGeometries['state'] = 'normal'

	def RefreshVariables(self, firstTime=False):

		configs = sorted(listdir('Data' + sep + 'configs'))
		for file in configs:
			if not file[-5:] == '.yaml':
				configs.remove(file)
		configs = list(sorted(configs))
		self.configs = configs

		modelList = []
		if not os.path.isdir('Data' + sep + 'models'):
			os.makedirs('Data' + sep + 'models')
		models = listdir('Data' + sep + 'models')
		for model in models:
			if os.path.isdir('Data' + sep + 'models' + sep + model):
				modelList.append(model)
		if len(modelList) == 0:
			modelList.append('No Models Yet')
		modelList = list(sorted(modelList))
		self.models = modelList

		if not firstTime:
			self.modelChooserSelect.set_menu(self.models[0], *self.models)
			self.configChooserSelect.set_menu(self.configs[0], *self.configs)

	# def SaveConfigButtonPress(self):
	# 	name = self.entryConfigName.get()

	# 	architecture = self.entryConfigArchitecture.get()
	# 	inputSize = self.entryConfigInputSize.get()
	# 	outputSize = self.entryConfigOutputSize.get()
	# 	inplanes = self.numBoxConfigInPlanes.get()
	# 	outplanes = inplanes
	# 	lossOption = self.entryConfigLossOption.get()
	# 	lossWeight = self.entryConfigLossWeight.get()
	# 	targetOpt = self.entryConfigTargetOpt.get()
	# 	weightOpt = self.entryConfigWeightOpt.get()
	# 	padSize = self.entryConfigPadSize.get()
	# 	LR_Scheduler = self.entryConfigLRScheduler.get()
	# 	baseLR = self.numBoxTrainBaseLR.get()
	# 	steps = self.entryConfigSteps.get()

	# 	with open('Data' + sep + 'configs' + sep + 'default.yaml','r') as file:
	# 		defaultConfig = yaml.load(file, Loader=yaml.FullLoader)
	# 	defaultConfig['MODEL']['ARCHITECTURE'] = architecture
	# 	defaultConfig['MODEL']['INPUT_SIZE'] = inputSize
	# 	defaultConfig['MODEL']['OUTPUT_SIZE'] = outputSize
	# 	defaultConfig['MODEL']['IN_PLANES'] = inplanes
	# 	defaultConfig['MODEL']['OUT_PLANES'] = outplanes
	# 	defaultConfig['MODEL']['LOSS_OPTION'] = lossOption
	# 	defaultConfig['MODEL']['LOSS_WEIGHT'] = lossWeight
	# 	defaultConfig['MODEL']['TARGET_OPT'] = targetOpt
	# 	defaultConfig['MODEL']['WEIGHT_OPT'] = weightOpt
	# 	defaultConfig['DATASET']['PAD_SIZE'] = padSize
	# 	defaultConfig['SOLVER']['LR_SCHEDULER_NAME'] = LR_Scheduler
	# 	defaultConfig['SOLVER']['BASE_LR'] = baseLR
	# 	defaultConfig['SOLVER']['STEPS'] = steps

	# 	with open('Data' + sep + 'configs' + sep + name + '.yaml','w') as file:
	# 		yaml.dump(defaultConfig, file)

	# 	MessageBox('Config Saved')
	# 	self.RefreshVariables()


	def run(self):
		self.mainwindow.mainloop()


#######################################
# Boilerplate TK Create & Run Window  #
####################################### 

if __name__ == '__main__':
	# root = tk.Tk()
	root = ThemedTk(theme='adapta')
	app = TabguiApp(root)
	# s = ttk.Style(root)
	# print(s.theme_names())
	# s.theme_use('clam')
	app.run()