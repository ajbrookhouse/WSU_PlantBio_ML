from utils import *
from Remote import *
from dataManipulation import *

from scipy.ndimage import grey_closing
import shutil
import glob
from connectomics.config import * #TODO probably shouldn't import all
from connectomics.utils.process import binary_watershed,bc_watershed
import yaml
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import os
from os.path import sep
from os import listdir
import cc3d
import h5py
import open3d as o3d
from contextlib import redirect_stdout
from contextlib import redirect_stderr
import contextlib
from connectomics.engine import Trainer
from connectomics.config import *
import torch
import traceback
import sys
import imageio
import neuroglancer
import ast
import argparse
import numpy as np
import re
import csv
import pandas as pd
import webbrowser

kill_neuroglancer=False

def openURLcallback(url):
    webbrowser.open_new(url)

def openNeuroGlancerThread(images, labels, labelToChange, scale=(20,20,20), mode='pre'):
	def ngLayer(data,res,oo=[0,0,0],tt='segmentation'):
		return neuroglancer.LocalVolume(data,dimensions=res,volume_type=tt,voxel_offset=oo)

	# global kill_neuroglancer #tried to use this to turn neuroglancer off, I don't think it's currently working
	# kill_neuroglancer=False
	# try:
	# 	segThreshold=int(segThreshold)
	# except:
	# 	print("Error with SegThreshold, setting to 255/2")

	ip = 'localhost' #or public IP of the machine for sharable display
	port = 9999 #change to an unused port number
	neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
	viewer=neuroglancer.Viewer()

	# SNEMI (# 3d vol dim: z,y,x)
	D0='./'
	res = neuroglancer.CoordinateSpace(
			names=['z', 'y', 'x'],
			units=['nm', 'nm', 'nm'],
			scales=scale)
	if mode=='post':
		im = imageio.volread(images)
		with h5py.File(labels, 'r') as fl:
			keys = list(fl.keys()) # get the keys and put them in a list
			with viewer.txn() as s:
				s.layers.append(name='images',layer=ngLayer(im,res,tt='image'))
				gt = np.array(fl[keys[0]][0])
				s.layers.append(name='masks',layer=ngLayer(gt,res,tt='segmentation'))
				# s.layers.append(name='images',layer=ngLayer(im,res,tt='segmentation'))
				# if len(keys)==1: # extract the datasetname automatically and apply to following visualization code
					
					# gt2 = np.array(fl[keys[0]][1])
					# print('gt',gt.shape)
					# print('gt2',gt2.shape)
					# gt = gt/255.0
					# s.layers.append(name='masks',layer=ngLayer(gt,res,tt='segmentation'))
					# s.layers.append(name='masks2',layer=ngLayer(gt2,res,tt='image'))
				# else:
				# 	for planeIndex in range(1,fl['vol0'].shape[0]):
				# 		planeTemp = np.array(fl['vol0'][planeIndex])
				# 		planeTemp[planeTemp < segThreshold] = 0
				# 		planeTemp[planeTemp != 0] = planeIndex
				# 		s.layers.append(name='plane_' + str(planeIndex),layer=ngLayer(planeTemp,res,tt='segmentation'))
	elif mode=='pre':
		im = imageio.volread(images)
		seg = imageio.volread(labels)

		with viewer.txn() as s:
			s.layers.append(name='images',layer=ngLayer(im,res,tt='image'))
			s.layers.append(name='labels',layer=ngLayer(seg,res,tt='segmentation'),selected_alpha=0.3)

	labelToChange.configure(text=str(viewer))
	labelToChange.bind("<Button-1>", lambda e: openURLcallback(str(viewer)))

	# while(not kill_neuroglancer):
	# 	time.sleep(2)

def closeNeuroglancerThread():
	global kill_neuroglancer
	kill_neuroglancer=True

# Machine Learning

def combineChunks(chunkFolder, predictionName, outputFile, metaData=''):
	#listOfFiles = [chunkFolder + sep + f for f in os.listdir(chunkFolder) if re.search(predictionName[:-3] + '_\\[[^\\]]*\\].h5', f)]
	listOfFiles = [chunkFolder + sep + f for f in os.listdir(chunkFolder) if f[-3:] == '.h5']

	globalXmax = 0
	globalYmax = 0
	globalZmax = 0

	for f in listOfFiles:
		head, tail = os.path.split(f)
		coords = tail[tail.rindex("_") + 1:-3].split('-')
		coords = list(filter(lambda val: val !=  '', coords) )
		coords = np.array(coords, dtype=int)
		zmin, zmax, ymin, ymax, xmin, xmax = coords

		globalZmax = max(globalZmax, zmax)
		globalYmax = max(globalYmax, ymax)
		globalXmax = max(globalXmax, xmax)

	h5Initial = h5py.File(listOfFiles[0], 'r')
	d = h5Initial['vol0'][:]
	h5Initial.close()
	del(h5Initial)
	dTypeToUse = d.dtype
	numPlanes = d.shape[0]

	newH5 = h5py.File(outputFile, 'w')
	newH5.create_dataset('vol0', (numPlanes, globalZmax+1, globalYmax+1, globalXmax+1), dtype=dTypeToUse, chunks=True)
	dataset = newH5['vol0']

	deltaTracker = TimeCounter(len(listOfFiles), timeUnits='hours', prefix='Combining Chunks: ')
	deltaTracker.print()

	for f in listOfFiles:
		head, tail = os.path.split(f)
		coords = tail[tail.rindex("_") + 1:-3  ].split('-')
		coords = list(filter(lambda val: val !=  '', coords) )
		coords = np.array(coords, dtype=int)
		zmin, zmax, ymin, ymax, xmin, xmax = coords

		h5in = h5py.File(f, 'r')
		d = h5in['vol0'][:]
		dataset[:,zmin:zmax, ymin:ymax, xmin:xmax] = d

		h5in.close()
		del(d)
		deltaTracker.tick()
		deltaTracker.print()
		print('Did file ' + f)

	if type(metaData) != type(' '):
		metaData = str(metaData)

	newH5['vol0'].attrs['metadata'] = metaData
	newH5.close()

@contextlib.contextmanager
def redirect_argv(*args):
	"""
	"""
	sys._argv = sys.argv[:]
	sys.argv=args
	yield
	sys.argv = sys._argv

def get_args():
	"""Copied from pytorch_connectomics. #TODO import it instead
	"""

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
	"""Copied from pytorch_connectomics and modified to take in a string as an argument

	Parameters
	----------
	modifiedArgs : str
		A string that replicates the command line args that you would normally give to pytorch_connectomics

	Returns
	-------
	unknown, possibly dict?
		Returns args, whatever argparse.ArgumentParser is (possibly a dict?)
	"""

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
	"""Taken from pytorch_connectomics and modified to to be used here, always trains

	Parameters
	----------
	config : str
		The filepath of the config file you are using for training.
		Note: do not send the path to the configs in Data/configs, usually, parts of the gui program create a temp.yaml file from the fields in the gui and pass that path to this
	"""

	args = get_args_modified(['--config-file', config])

	# if args.local_rank == 0 or args.local_rank is None:
	args.local_rank = None
	print("Command line arguments: ", args)

	# manual_seed = 0 if args.local_rank is None else args.local_rank
	# np.random.seed(manual_seed)
	# torch.manual_seed(manual_seed)

	cfg = load_cfg(args)
	# path = 

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
		print('current device? ',device)

	print("Rank: {}. Device: {}".format(args.local_rank, device))
	cudnn.enabled = True
	cudnn.benchmark = True

	# mode = 'test' if args.inference else 'train' ###
	mode = 'train'
	trainer = Trainer(cfg, device, mode,
					  rank=args.local_rank,
					  checkpoint=args.checkpoint)

	# Start training or inference:
	if cfg.DATASET.DO_CHUNK_TITLE == 0:
		# test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
		# test_func() if args.inference else trainer.train()
		print("RUNNING TRAIN")
		trainer.train() 
	else:
		trainer.run_chunk(mode)

	print("Rank: {}. Device: {}. Process is finished!".format(
		  args.local_rank, device))

def predFromMain(config, checkpoint, metaData='', recombineChunks=False):
	"""Taken from pytorch_connectomics and modified to to be used here, always predicts

	Parameters
	----------
	config : str
		The filepath of the config file you are using for model prediction.
		Note: do not send the path to the configs in Data/configs, usually, parts of the gui program create a temp.yaml file from the fields in the gui and pass that path to this
	
	checkpoint : str
		The filepath in Data/models to the saved checkpoint of the model to use for prediction.
		#TODO add a specific example here

	metadata : str, optional
		The string metadata of the model

	recombineChunks : bool
		Whether or not many chunks (smaller subsets of the full prediction) need to be combined.
		Normally this is only true when using .json datasets and .txt datasets
	"""

	args = get_args_modified(['--inference', '--checkpoint', checkpoint, '--config-file', config])

	# if args.local_rank == 0 or args.local_rank is None:
	args.local_rank = None

	manual_seed = 0 if args.local_rank is None else args.local_rank
	# np.random.seed(manual_seed)
	# torch.manual_seed(manual_seed)

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

	# if args.distributed:
	# 	assert torch.cuda.is_available(), \
	# 		"Distributed training without GPUs is not supported!"
	# 	dist.init_process_group("nccl", init_method='env://')
	# 	torch.cuda.set_device(args.local_rank)
	# 	device = torch.device("cuda", args.local_rank)
	# else:
	# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("Rank: {}. Device: {}".format(args.local_rank, device))
	cudnn.enabled = True
	cudnn.benchmark = True

	# mode = 'test' if args.inference else 'train'
	mode = 'test'
	trainer = Trainer(cfg, device, mode,
					  rank=args.local_rank,
					  checkpoint=args.checkpoint)

	# Start training or inference:
	if cfg.DATASET.DO_CHUNK_TITLE == 0:
		test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
		test_func() if args.inference else trainer.train()
		# print("RUNNING TEST")
	else:
		trainer.run_chunk(mode)

	print("Rank: {}. Device: {}. Process is finished!".format(
		  args.local_rank, device))

	# print('Recombine Chunks:', recombineChunks)
	# if not recombineChunks:
	# 	h = h5py.File(os.path.join(cfg["INFERENCE"]["OUTPUT_PATH"] + sep + cfg['INFERENCE']['OUTPUT_NAME'] + '.h5'),'a')
	# 	h['vol0'].attrs['metadata'] = metaData
	# 	h.close()

#f is a an opened h5 file
def InstanceSegmentProcessArray(f, cropDic, greyClosing=10, thres1=.85, thres2=.15, thres3=.8, thres_small=1000):
	dataset = f['vol0']

	xmin = cropDic['xmin']
	xmax = cropDic['xmax']
	ymin = cropDic['ymin']
	ymax = cropDic['ymax']
	zmin = cropDic['zmin']
	zmax = cropDic['zmax']

	startSlice = dataset[:,xmin:xmax, ymin:ymax, zmin:zmax]
	startSlice[1] = grey_closing(startSlice[1], size=(greyClosing,greyClosing,greyClosing))
	seg = bc_watershed(startSlice, thres1=thres1, thres2=thres2, thres3=thres3, thres_small=thres_small)
	return seg

def InstanceSegmentProcessing(inputH5Filename, greyClosing=10, thres1=.85, thres2=.15, thres3=.8, thres_small=1000, cubeSize=1000):
	"""Take the raw, two plane output of the machine learning output, and turn it into a one plane, usable format

	The machine learning model has an edge, and a volume prediction
	This function uses a watershed function to turn this into one plane
	This function adds a 'processed' dataset to the h5 file that has 0 as background, and each instance gets a unique integer id
	
	Parameters
	----------
	inputH5Filename : str
		The filepath of the H5 file that you need to InstanceSegment

	greyClosing : int, optional, default = 10
		How much grey closing to do on the instance arrays before watershed. Grey closing closes up small gaps between surfaces in the edge detection
		# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_closing.html

	thresh1, thres2, thresh3 : float between 0 and 1
		Parameters for the watershed function. Look at the pytorch_connectomics code for bc_watershed in connectomics.util.process

	thresh_small : int, optional, default is 1000
		I believe that the watershed ignores any regions that are smaller than this threshold

	cubeSize : int, optional, default is 1000
		The ImageSegmentation postproccessing cannot be done on large arrays at once, so it needs to be done in smaller pieces.
		`cubeSize` is the size of the cube that can get watershed processing at once. The function would break down the larger dataset into chunks of this size

	Returns
	-------
	None
		Creates a new 'processed' dataset in the H5 file with the processed ImageSegmentation
	"""

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
	thirdCube = int(cubeSize/3)
	quarterCube = int(cubeSize/4)
	offsetList = [0, thirdCube, quarterCube, halfCube]
	countDic = {}
	completeCount = 0

	print('Dataset Shape:', dataset.shape)
	print('Chunks:', dataset.chunks)
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
					xmax = min(xiteration + cubeSize, dataset.shape[1])
					ymin = yiteration
					ymax = min(yiteration + cubeSize, dataset.shape[2])
					zmin = ziteration
					zmax = min(ziteration + cubeSize, dataset.shape[3])

					print('Loading', xiteration, yiteration, ziteration)

					h5Temp = h5out[xmin:xmax, ymin:ymax, zmin:zmax] 
					mapTemp = map_[xmin:xmax, ymin:ymax, zmin:zmax]

					startSlice = dataset[:,xmin:xmax, ymin:ymax, zmin:zmax]
					print('Loaded, processing')
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
					print('Writing')
					h5out[xmin:xmax, ymin:ymax, zmin:zmax] = h5Temp
					map_[xmin:xmax, ymin:ymax, zmin:zmax] = mapTemp
					del(h5Temp)
					del(mapTemp)

					deltaTracker.tick()
					deltaTracker.print()
					print('==============================')
					print()

	h5out.attrs['countDictionary'] = str(countDic)
	f.close()


# Thread Workers

def trainThreadWorker(cfg, stream):
	"""Call this function as a seperate thread (or not if you want it to block) to train a model

	Parameters
	----------
	cfg : str
		The filepath of the config file you are using for training.
		Note: do not send the path to the configs in Data/configs, usually, parts of the gui program create a temp.yaml file from the fields in the gui and pass that path to this

	stream : MemoryStream from utils.py
		Stream to hold the output of what gets printed from the training process. Look at gui.py for an example in the ButtonPress functions

	Returns
	-------
	None
		Trains a model to be saved in Data/models
	"""

	with redirect_stdout(stream):
		try:
			trainFromMain(cfg)
		except:
			traceback.print_exc()
def writeH5(filename, dtarray, datasetname='vol0'):
	fid=h5py.File(filename,'w')
	if isinstance(datasetname, (list,)):
		for i,dd in enumerate(datasetname):
			ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
			ds[:] = dtarray[i]
	else:
		ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
		ds[:] = dtarray
	fid.close()
	del dtarray

def useThreadWorker(cfg, stream, checkpoint, metaData='', recombineChunks=False):
	"""Call this function as a seperate thread (or not if you want it to block) to use a model for prediction

	Parameters
	----------
	cfg : str
		The filepath of the config file you are using for training.
		Note: do not send the path to the configs in Data/configs, usually, parts of the gui program create a temp.yaml file from the fields in the gui and pass that path to this

	stream : MemoryStream from utils.py
		Stream to hold the output of what gets printed from the training process. Look at gui.py for an example in the ButtonPress functions

	checkpoint : str
		The filepath in Data/models to the saved checkpoint of the model to use for prediction.
		#TODO add a specific example here

	metadata : str, optional
		The string metadata of the model

	recombineChunks : bool
		Whether or not many chunks (smaller subsets of the full prediction) need to be combined.
		Normally this is only true when using .json datasets and .txt datasets

	Returns
	-------
	None
		Creates a new H5 file with the model prediction
	"""

	with redirect_stdout(stream):
		try:
			print('About to pred from main')
			predFromMain(cfg, checkpoint, metaData=metaData, recombineChunks=recombineChunks)
			print('Done with pred from main\n')

			if type(metaData) == type('test'):
				metaData = ast.literal_eval(metaData)
			configType = metaData['configType']

			# if '2D' in configType or 'instance' in configType.lower() or recombineChunks:

			with open(cfg,'r') as file:
				config = yaml.load(file, Loader=yaml.FullLoader)

			if 'semantic2d' in configType.lower():
				print('Semantic 2D Post-Processing Required. Please click the button below to begin.')
				# modelOutputFilePath=os.path.join(config["INFERENCE"]["OUTPUT_PATH"], config['INFERENCE']['OUTPUT_NAME'])
				# f = h5py.File(modelOutputFilePath, "r")
				# post_arr=np.array(f['vol0'])
				# f.close()
				# del f
				# print('\n',post_arr.shape)
				# post_arr=np.invert(post_arr)

				# Recombine=[]
				# for layer in post_arr[0]:
				# 	new_layer=np.expand_dims(layer, axis=0)
				# 	new_layer=binary_watershed(new_layer,thres1=0.8,thres2=0.85, thres_small=1024,seed_thres=35)
				# 	# print(np.unique(new_layer))
				# 	Recombine.append(new_layer)
				
				# post_arr=np.stack(Recombine, axis=0)
				# del Recombine
				# print('after combine',post_arr.shape)
				# post_arr=np.expand_dims(post_arr, axis=0)
				# print(post_arr.shape)
				# # write and store
				# writeH5(modelOutputFilePath+'_s2D_out',np.array(post_arr))
				# del post_arr
				# print("Finished Semantic2D Process! Please find the 'Model Output' with its original name + _s2D_out")
			elif 'semantic3d' in configType.lower():
				print('Semantic 3D Post-Processing Required. Please click the button below to begin.')
				# modelOutputFilePath=os.path.join(config["INFERENCE"]["OUTPUT_PATH"], config['INFERENCE']['OUTPUT_NAME'])
				# # open file
				# f = h5py.File(modelOutputFilePath, "r")
				# post_arr=np.array(f['vol0'][:2])
				# f.close()
				# del f
				# print('\n',post_arr.shape)
				# post_arr=np.invert(post_arr)

				# post_arr=bc_watershed(post_arr,thres1=0.9,thres2=0.8,thres3=0.8,thres_small=1024,seed_thres=35)
				# post_arr=np.expand_dims(post_arr, axis=0)
				# print(post_arr.shape)

				# # write and store
				# writeH5(modelOutputFilePath+'_s3D_out',np.array(post_arr))
				# del post_arr
				# print("Finished Semantic3D Process! Please find the 'Model Output' with its original name + _s3D_out")
			elif 'instance2d' in configType.lower():
				print('Instance 2D Post-Processing Required. Please click the button below to begin.')
				# modelOutputFilePath=os.path.join(config["INFERENCE"]["OUTPUT_PATH"], config['INFERENCE']['OUTPUT_NAME'])
				
				# f = h5py.File(modelOutputFilePath, "r")
				# post_arr=np.array(f['vol0'])
				# f.close()
				# del f
				# print('\n',post_arr.shape)
				
				# Recombine=[]
				# for layer in post_arr[0]:
				# 	new_layer=np.expand_dims(layer, axis=0)
				# 	new_layer=bc_watershed(new_layer,thres1=0.9,thres2=0.8,thres3=0.8,thres_small=1024,seed_thres=35)
				# 	# print(np.unique(new_layer))
				# 	Recombine.append(new_layer)
				
				# post_arr=np.stack(Recombine, axis=0)
				# del Recombine
				# print('after combine',post_arr.shape)
				# post_arr=np.expand_dims(post_arr, axis=0)
				# print(post_arr.shape)
				# # write and store
				# writeH5(modelOutputFilePath+'_i2D_out',np.array(post_arr))
				# del post_arr
				# print("Finished Instance2D Process! Please find the 'Model Output' with its original name + _i2D_out")
			elif 'instance3d' in configType.lower():
				print('Instance 3D Post-Processing Required. Please click the button below to begin.')
				# modelOutputFilePath=os.path.join(config["INFERENCE"]["OUTPUT_PATH"], config['INFERENCE']['OUTPUT_NAME'])
				
				# f = h5py.File(modelOutputFilePath, "r")
				# post_arr=np.array(f['vol0'][:2])
				# print(post_arr.dtype)
				# f.close()
				# del f
				# print('\n',post_arr.shape)
				# # watershed
				# # from connectomics.utils.process import bcd_watershed
				# # post_arr=bcd_watershed(post_arr,thres1=0.9, thres2=0.8, thres3=0.8, thres4=0.4, thres5=0.0, thres_small=128,seed_thres=35)
				# post_arr=bc_watershed(post_arr,thres1=0.9,thres2=0.8,thres3=0.8,thres_small=1024,seed_thres=32)
				# post_arr=np.expand_dims(post_arr, axis=0)
				# print(post_arr.shape)
				# # split and store
				# n=post_arr.shape[0]//100
				# if n!=0:
				# 	res=np.array_split(post_arr,n,axis=0)
				# 	for i,a in enumerate(res):
				# 		# print(i,a.shape)
				# 		writeH5(modelOutputFilePath+'_i3D_out_'+str(i),a)
				# else:
				# 	writeH5(modelOutputFilePath+'_i3D_out_0',post_arr)
					
				# del post_arr
				# print("Finished Instance Process! Please find the 'Model Output' with its original name + _i3D_out")
				

				# if 'instance' in configType.lower() and not '2D' in configType and not recombineChunks: #3D instance, all in memory
				# 	print('3D Instance Post-Processing')
				# 	outputFile = os.path.join(config["INFERENCE"]["OUTPUT_PATH"], config['INFERENCE']['OUTPUT_NAME'])
				# 	print('OutputFile:',outputFile)
				# 	InstanceSegmentProcessing(outputFile, greyClosing=10, thres1=.85, thres2=.15, thres3=.8, thres_small=100, cubeSize=1000)
				# 	print('Completely done, output is saved in', outputFile)
				# elif not '2D' in configType and recombineChunks:
				# 	print('Path, outputName', config["INFERENCE"]["OUTPUT_PATH"], config["INFERENCE"]["OUTPUT_NAME"])
				# 	outputPath = config["INFERENCE"]["OUTPUT_PATH"]
				# 	outputName = config["INFERENCE"]["OUTPUT_NAME"]
				# 	newOutputName = outputPath[:outputPath.rindex(sep) + 1] + outputName
				# 	combineChunks(outputPath, outputName, newOutputName, metaData=metaData)
				# 	shutil.rmtree(outputPath)
					
				# 	if 'instance' in configType.lower():
				# 		print('Starting Instance Post-Processing')
				# 		InstanceSegmentProcessing(newOutputName, greyClosing=10, thres1=.85, thres2=.15, thres3=.8, thres_small=100, cubeSize=1000)						
				# 	elif 'semantic' in configType.lower():
				# 		pass

				# 	print('Completely done, output is saved in', newOutputName)
				# elif '2D' in configType and not 'instance' in configType.lower(): #Semantic 2D
				# 	outputPath = config["INFERENCE"]["OUTPUT_PATH"]
				# 	outputName = config["INFERENCE"]["OUTPUT_NAME"]
				# 	newOutputName = outputPath[:outputPath.rindex(sep) + 1] + outputName
				# 	toCombineFileList = list(sorted(glob.glob(outputPath + sep + outputName[:-3] + '_*.h5')))
				# 	numFiles = len(toCombineFileList)

				# 	h5f = h5py.File(toCombineFileList[0], 'r')
				# 	d = h5f['vol0'][:]
				# 	h5f.close()
				# 	numPlanes = d.shape[0]
				# 	width = d.shape[3]
				# 	height = d.shape[2]
				# 	del(d)

				# 	newH5 = h5py.File(newOutputName, 'w')
				# 	newH5.create_dataset('vol0', (numPlanes, numFiles, height, width), dtype=np.uint8)
				# 	dataset = newH5['vol0']

				# 	for index, file in enumerate(toCombineFileList):
				# 		h5f = h5py.File(file, 'r')
				# 		d = h5f['vol0'][:]
				# 		h5f.close()
				# 		d = d.squeeze()
				# 		dataset[:,index,:,:] = d

				# 	newH5['vol0'].attrs['metadata'] = str(metaData)
				# 	newH5.close()
				# 	shutil.rmtree(outputPath)
				# elif '2D' in configType and 'instance' in configType.lower():
				# 	outputPath = config["INFERENCE"]["OUTPUT_PATH"]
				# 	outputName = config["INFERENCE"]["OUTPUT_NAME"]
				# 	newOutputName = outputPath[:outputPath.rindex(sep) + 1] + outputName
				# 	toCombineFileList = list(sorted(glob.glob(outputPath + sep + outputName[:-3] + '_*.h5')))
				# 	numFiles = len(toCombineFileList)

				# 	h5f = h5py.File(toCombineFileList[0], 'r')
				# 	d = h5f['vol0'][:]
				# 	h5f.close()
				# 	numPlanes = d.shape[0]
				# 	width = d.shape[3]
				# 	height = d.shape[2]
				# 	del(d)

				# 	newH5 = h5py.File(newOutputName, 'w')
				# 	newH5.create_dataset('vol0', (numPlanes, numFiles, height, width), dtype=np.uint8)
				# 	dataset = newH5['vol0']

				# 	for index, file in enumerate(toCombineFileList): #Put Raw Data into H5
				# 		h5f = h5py.File(file, 'r')
				# 		d = h5f['vol0'][:]
				# 		h5f.close()
				# 		d = d.squeeze()
				# 		dataset[:,index,:,:] = d

				# 	newH5['vol0'].attrs['metadata'] = str(metaData)

				# 	#Process Instance and put under processed
				# 	newH5.create_dataset('processed', (numFiles, height, width), dtype=np.uint16)
				# 	dataset = newH5['processed']
				# 	for index, file in enumerate(toCombineFileList):
				# 		h5f = h5py.File(file, 'r')
				# 		d = h5f['vol0'][:]
				# 		h5f.close()
				# 		d = d.squeeze()
				# 		greyClosing = 10
				# 		thres1=.9  
				# 		thres2=.8	
				# 		thres3=.85	
				# 		thres_small=1
				# 		#d[0] = grey_closing(d[0], size=(greyClosing,greyClosing))
				# 		d = bc_watershed(d, thres1=thres1, thres2=thres2, thres3=thres3, thres_small=thres_small)
				# 		dataset[index] = d

				# 	newH5['processed'].attrs['metadata'] = str(metaData)
				# 	newH5.close()
				# 	shutil.rmtree(outputPath)

				# print('All Post-process are Completely Finished')
		except:
			print('Critical Error')
			traceback.print_exc()

# def trainThreadWorkerCluster(cfg, stream, button, url, username, password, trainStack, trainLabels, submissionScriptString, folderToUse, pytorchFolder, submissionCommand):
# 	"""Will train within a server instead of the local computer
# 	#TODO should be in Remote.py, check to see if it is safe to move
# 	"""

# 	with redirect_stdout(stream):
# 		with redirect_stderr(stream):
# 			runRemoteServer(url, username, password, trainStack, trainLabels, configToUse, submissionScriptString, folderToUse, pytorchFolder, submissionCommand)
# 	button['state'] = 'normal'

def ImageToolsCombineImageThreadWorker(pathToCombine, outputFile, streamToUse):
	"""Combines a folder with a stack of images into one .tif, .json, or .txt dataset

	Parameters
	----------
	pathToCombine : str
		The filepath of the folder containing all of the images that you want to combine

	outputFile : str
		the filename/path of the .tif, .json, or .txt dataset that you want to create

	streamToUse : MemoryStream from utils.py
		Memory Stream to capture printed output, look for examples of how this is used in gui.py, particularly in the ButtonPress Functions

	Returns
	-------
	None
		Creates a new dataset (.tif, .json, or .txt)
	"""

	with redirect_stdout(streamToUse):
		try:
			images = []
			listOfImages = list(sorted(listdir(pathToCombine)))
			if not pathToCombine[-1] == sep:
				pathToCombine += sep

			for image in listOfImages:
				images.append(pathToCombine + image)

			if outputFile[-4:] == '.txt':
				createTxtFileFromImageList(images, outputFile)

			elif outputFile[-4:] == '.tif':
				createTifFromImageList(images, outputFile)

			elif outputFile[-5:] == '.json':
				writeJsonForImages(images, outputFile)

		except:
			print('Critical Error:')
			traceback.print_exc()

def OutputToolsGetStatsThreadWorker(h5path, streamToUse, outputFile, cropBox = [0, 0, 0, 0, 0, 0]):
	"""Measures volumes in an H5 file, and saves them in a .csv

	Parameters
	----------
	h5path : str
		Path to the H5 file that you want to measure volumes from

	streamToUse : MemoryStream from utils.py
		Memory Stream to capture printed output, look for examples of how this is used in gui.py, particularly in the ButtonPress Functions

	outputFile : str
		The filename/path of the .csv file that you want to save volumes to.

	cropBox : list of ints (6 long), optional
		minx, miny, minz, maxx, maxy, maxz order
		Can be used if you want to only measure from a specified area

	Returns
	-------
	None
		Creates a CSV file
	"""

	with redirect_stdout(streamToUse):
		try:
			minx, miny, minz, maxx, maxy, maxz = cropBox
			xmin = minx
			ymin = miny
			zmin = minz
			xmax = maxx
			ymax = maxy
			zmax = maxz
			if cropBox == [0, 0, 0, 0, 0, 0] or cropBox == ['', '', '', '', '', '']:
				cropped = False
			else:
				cropped = True

			print('Loading H5 File')
			h5f = h5py.File(h5path, 'r') #TODO make sure file is always closed properly using with:

			imageIndexList = []
			planeIndexList = []
			areaList = []

			dataset = np.array(h5f['vol0'])
			h5f.close()
			print("Begin Processing")

			IDlist=np.unique(dataset)
			dictt={}
			dictt['Instance ID']=IDlist
			dictt['Area(in pixel)']=[]

			# imageIndexList = []
			# for i in range(dataset.shape[0]):
			# 	dFile = dataset[i]
			# 	for id in IDList:
			# 		temp_lst=[]
			# 		if unique == 0:
			# 			continue
			# 		imageIndexList.append(i)
			# 		temp_lst.append(np.count_nonzero(dFile == id))

			for id in IDlist:
				dictt['Area(in pixel)'].append(np.count_nonzero(dataset == id))
		
			df = pd.DataFrame.from_dict(dictt)
			df.to_csv(outputFile,index=True)
			print("Wrote CSV to " + outputFile)

			print()
			print('==============================')
			# print()
			# print('H5File Raw Counts')
			# print()
			# print(sorted(countList))
			# print()
			# print('==============================')
			print()
			print('Instance Stats')
			print('Min:', min(df['Area(in pixel)']))
			print('Max:', max(df['Area(in pixel)']))
			print('Mean:', np.mean(df['Area(in pixel)']))
			print('Median:', np.median(df['Area(in pixel)']))
			print('Standard Deviation:', np.std(df['Area(in pixel)']))
			print('Sum:', sum(df['Area(in pixel)']))
			print('Total Number:', len(df['Area(in pixel)']))

			print('\nStats Successfully Generated!')
		except:
			print('Critical Error:')
			traceback.print_exc()

def VisualizeThreadWorker(filesToVisualize, streamToUse, voxel_size=1): #TODO check to see if this could be removed. It is replaced by the new visualizationGUI.py
	with redirect_stdout(streamToUse):
		try:
			geometries_to_draw = []
			for file in filesToVisualize:
				filename = file[0]
				filecolor = file[1]
				print('Loading File ' + filename + '(, may take a while)')
				print(filecolor)

				if '_mesh_' in filename: #Mesh
					toAdd = o3d.io.read_triangle_mesh(filename)
					if not '_instance_' in filename:
						toAdd.paint_uniform_color(np.array(filecolor)/255)
					toAdd.compute_vertex_normals()
					geometries_to_draw.append(toAdd)
				elif '_pointCloud_' in filename: #Point Cloud
					toAdd = o3d.io.read_point_cloud(filename)
					if not '_instance_' in filename:
						toAdd.paint_uniform_color(np.array(filecolor)/255)
					toAdd = o3d.geometry.VoxelGrid.create_from_point_cloud(toAdd, voxel_size=voxel_size)
					geometries_to_draw.append(toAdd)
				else: #Unknown Filetype
					pass

			# visWindow = o3d.visualization.Visualizer()
			# visWindow.create_window()
			# for geometry in geometries_to_draw:
			# 	visWindow.add_geometry(geometry)
			# visWindow.draw()

			o3d.visualization.draw(geometries_to_draw, 'Visualization Window')

			#o3d.visualization.draw_geometries(geometries_to_draw)

		except:
			print('Critical Error:')
			traceback.print_exc()

def OutputToolsMakeGeometriesThreadWorker(h5path, makeMeshs, makePoints, streamToUse, downScaleFactor=1):
	"""Makes 3D geometries from a H5File

	Parameters
	----------
	h5path : str
		The filepath of the dataset you want to create 3D geometries from

	makeMeshs : bool
		If true, this function will create a 3D Mesh

	makePoints : bool
		If true, this function will create a PointCloud

	streamToUse : MemoryStream from utils.py
		Memory Stream to capture printed output, look for examples of how this is used in gui.py, particularly in the ButtonPress Functions

	downScaleFactor : int, optional
		If 1, no downscaling
		2 would mean each axis is sampled at 1/2 of the points, so will have 1/8 total volume

	Returns
	-------
	None
		Writes 3D files to disk	
	"""

	with redirect_stdout(streamToUse):
		try:
			print('Loading H5 File')
			h5f = h5py.File(h5path, 'r')

			# metadata = {'configType':'instance.yaml'}
			metadata = ast.literal_eval(h5f['vol0'].attrs['metadata'])
			configType = metadata['configType'].lower()
			outputFilenameShort = h5path[:-3]

			if '2D' in configType:
				print('Cannot Make 3D geometries for a 2D sample')
				h5f.close()
				return

			elif 'instance' in configType:
				'''
				INSTANCE SECTION
				'''
				dataset = h5f['processed']
				if not downScaleFactor == 1:
					d = subSampled3DH5(dataset, downScaleFactor)
				else:
					d = dataset[:]
				h5f.close()
				print('Loaded H5 File')
				if makeMeshs:
					print('Creating Mesh')
					mesh = instanceArrayToMesh(d)
					print('Finished Calculating Mesh, saving ' + outputFilenameShort + '_instance_mesh_downscaled_' + str(downScaleFactor) + '_.ply')
					o3d.io.write_triangle_mesh(outputFilenameShort + '_instance_mesh_.ply', mesh)
					print('Finished making meshes')
				if makePoints:
					print('Loaded H5 File, creating point cloud')
					cloud = instanceArrayToPointCloud(d)
					print('Finished Calculating Point Cloud, saving')
					o3d.io.write_point_cloud(outputFilenameShort + '_instance_pointCloud_downscaled_' + str(downScaleFactor) + '_.ply', cloud)
					print('Finished')

			elif 'semantic' in configType:
				'''
				SEMANTIC SECTION
				'''
				dataset = h5f['vol0']
				if not downScaleFactor == 1:
					d = subSampled3DH5(dataset, downScaleFactor)
				else:
					d = dataset[:]
				h5f.close()
				print('Loaded H5 File')
				numIndexes = d.shape[0]
				rootFolder, h5Filename = head_tail = os.path.split(h5path)
				h5Filename = h5Filename[:-3]

				if makeMeshs:
					print('Starting Mesh Creation')
					for index in range(1, numIndexes):
						print('Creating Mesh for Index:', index)
						mesh = arrayToMesh(d, index)
						o3d.io.write_triangle_mesh(outputFilenameShort + '_semantic_mesh_' + str(index) + '_downscaled_' + str(downScaleFactor) + '_.ply', mesh)
					print('Finished with making Meshes')
					print()
				if makePoints:
					print('Starting Point Cloud Creation')
					for index in range(1, numIndexes):
						print('Creating Point Cloud for Index:', index)
						cloud = getPointCloudForIndex(d, index)
						o3d.io.write_point_cloud(outputFilenameShort + '_semantic_pointCloud_' + str(index)  + '_downscaled_' + str(downScaleFactor) + '_.ply', cloud)
					print('Finished with making Point Clouds')
					print()

			else:
				print('Unrecognized File')
				h5f.close()
				return

		except:
			print('Critical Error:')
			traceback.print_exc()