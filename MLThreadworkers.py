from utils import *
from Remote import *
from dataManipulation import *

from scipy.ndimage import grey_closing
import shutil
import glob
from connectomics.config import *
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
import ast
from connectomics.utils.process import bc_watershed
import argparse
import numpy as np
import re
import csv
import pandas as pd



# Machine Learning

def combineChunks(chunkFolder, predictionName, outputFile, metaData=''):
	listOfFiles = [chunkFolder + sep + f for f in os.listdir(chunkFolder) if re.search(predictionName[:-3] + '_\\[[^\\]]*\\].h5', f)]

	globalXmax = 0
	globalYmax = 0
	globalZmax = 0

	for f in listOfFiles:
		head, tail = os.path.split(f)
		coords = tail[tail.rindex('[') + 1:tail.rindex(']')].split(' ')
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
		coords = tail[tail.rindex('[') + 1:tail.rindex(']')].split(' ')
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
	sys._argv = sys.argv[:]
	sys.argv=args
	yield
	sys.argv = sys._argv

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

def predFromMain(config, checkpoint, metaData='', recombineChunks=False):
	args = get_args_modified(['--inference', '--checkpoint', checkpoint, '--config-file', config])

	# if args.local_rank == 0 or args.local_rank is None:
	args.local_rank = None
	#print("Command line arguments: ", args)

	manual_seed = 0 if args.local_rank is None else args.local_rank
	np.random.seed(manual_seed)
	torch.manual_seed(manual_seed)

	cfg = load_cfg(args)
	#print('loaded config')
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

	#print('About to start training')
	# Start training or inference:
	if cfg.DATASET.DO_CHUNK_TITLE == 0:
		test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
		test_func() if args.inference else trainer.train()
	else:
		trainer.run_chunk(mode)

	print("Rank: {}. Device: {}. Process is finished!".format(
		  args.local_rank, device))

	print('Recombine Chunks:', recombineChunks)
	if not recombineChunks:
		h = h5py.File(os.path.join(cfg["INFERENCE"]["OUTPUT_PATH"] + sep + cfg['INFERENCE']['OUTPUT_NAME']), 'r+')
		h['vol0'].attrs['metadata'] = metaData
		h.close()

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
	with redirect_stdout(stream):
		try:
			trainFromMain(cfg)
		except:
			traceback.print_exc()

def useThreadWorker(cfg, stream, checkpoint, metaData='', recombineChunks=False):
	with redirect_stdout(stream):
		try:
			print('About to pred from main')
			predFromMain(cfg, checkpoint, metaData=metaData, recombineChunks=recombineChunks)
			print('Done with pred from main')

			if type(metaData) == type('test'):
				metaData = ast.literal_eval(metaData)
			configType = metaData['configType']

			if '2D' in configType or 'instance' in configType.lower() or recombineChunks:
				print('Starting Post-Processing')

				with open(cfg,'r') as file:
					config = yaml.load(file, Loader=yaml.FullLoader)

				if 'instance' in configType.lower() and not '2D' in configType and not recombineChunks: #3D instance, all in memory
					print('3D Instance Post-Processing')
					outputFile = os.path.join(config["INFERENCE"]["OUTPUT_PATH"], config['INFERENCE']['OUTPUT_NAME'])
					print('OutputFile:',outputFile)
					InstanceSegmentProcessing(outputFile, greyClosing=10, thres1=.85, thres2=.15, thres3=.8, thres_small=100, cubeSize=1000)
					print('Completely done, output is saved in', outputFile)

				elif not '2D' in configType and recombineChunks:
					print('Path, outputName', config["INFERENCE"]["OUTPUT_PATH"], config["INFERENCE"]["OUTPUT_NAME"])
					outputPath = config["INFERENCE"]["OUTPUT_PATH"]
					outputName = config["INFERENCE"]["OUTPUT_NAME"]
					newOutputName = outputPath[:outputPath.rindex(sep) + 1] + outputName
					combineChunks(outputPath, outputName, newOutputName, metaData=metaData)
					shutil.rmtree(outputPath)

					if 'instance' in configType.lower():
						print('Starting Instance Post-Processing')
						InstanceSegmentProcessing(newOutputName, greyClosing=10, thres1=.85, thres2=.15, thres3=.8, thres_small=100, cubeSize=1000)						
					elif 'semantic' in configType.lower():
						pass

					print('Completely done, output is saved in', newOutputName)

				elif '2D' in configType and not 'instance' in configType.lower(): #Semantic 2D
					outputPath = config["INFERENCE"]["OUTPUT_PATH"]
					outputName = config["INFERENCE"]["OUTPUT_NAME"]
					newOutputName = outputPath[:outputPath.rindex(sep) + 1] + outputName
					toCombineFileList = list(sorted(glob.glob(outputPath + sep + outputName[:-3] + '_*.h5')))
					numFiles = len(toCombineFileList)

					h5f = h5py.File(toCombineFileList[0], 'r')
					d = h5f['vol0'][:]
					h5f.close()
					numPlanes = d.shape[0]
					width = d.shape[3]
					height = d.shape[2]
					del(d)

					newH5 = h5py.File(newOutputName, 'w')
					newH5.create_dataset('vol0', (numPlanes, numFiles, height, width), dtype=np.uint8)
					dataset = newH5['vol0']

					for index, file in enumerate(toCombineFileList):
						h5f = h5py.File(file, 'r')
						d = h5f['vol0'][:]
						h5f.close()
						d = d.squeeze()
						dataset[:,index,:,:] = d

					newH5['vol0'].attrs['metadata'] = str(metaData)
					newH5.close()
					shutil.rmtree(outputPath)
				elif '2D' in configType and 'instance' in configType.lower():
					outputPath = config["INFERENCE"]["OUTPUT_PATH"]
					outputName = config["INFERENCE"]["OUTPUT_NAME"]
					newOutputName = outputPath[:outputPath.rindex(sep) + 1] + outputName
					toCombineFileList = list(sorted(glob.glob(outputPath + sep + outputName[:-3] + '_*.h5')))
					numFiles = len(toCombineFileList)

					h5f = h5py.File(toCombineFileList[0], 'r')
					d = h5f['vol0'][:]
					h5f.close()
					numPlanes = d.shape[0]
					width = d.shape[3]
					height = d.shape[2]
					del(d)

					newH5 = h5py.File(newOutputName, 'w')
					newH5.create_dataset('vol0', (numPlanes, numFiles, height, width), dtype=np.uint8)
					dataset = newH5['vol0']

					for index, file in enumerate(toCombineFileList): #Put Raw Data into H5
						h5f = h5py.File(file, 'r')
						d = h5f['vol0'][:]
						h5f.close()
						d = d.squeeze()
						dataset[:,index,:,:] = d

					newH5['vol0'].attrs['metadata'] = str(metaData)

					#Process Instance and put under processed
					newH5.create_dataset('processed', (numFiles, height, width), dtype=np.uint16)
					dataset = newH5['processed']
					for index, file in enumerate(toCombineFileList):
						h5f = h5py.File(file, 'r')
						d = h5f['vol0'][:]
						h5f.close()
						d = d.squeeze()
						greyClosing = 10
						thres1=.85
						thres2=.15
						thres3=.8
						thres_small=1
						#d[0] = grey_closing(d[0], size=(greyClosing,greyClosing))
						d = bc_watershed(d, thres1=thres1, thres2=thres2, thres3=thres3, thres_small=thres_small)
						dataset[index] = d

					newH5['processed'].attrs['metadata'] = str(metaData)
					newH5.close()
					shutil.rmtree(outputPath)

				print('Completely Finished')


		except:
			print('Critical Error')
			traceback.print_exc()

def trainThreadWorkerCluster(cfg, stream, button, url, username, password, trainStack, trainLabels, submissionScriptString, folderToUse, pytorchFolder, submissionCommand):
	with redirect_stdout(stream):
		with redirect_stderr(stream):
			runRemoteServer(url, username, password, trainStack, trainLabels, configToUse, submissionScriptString, folderToUse, pytorchFolder, submissionCommand)
	button['state'] = 'normal'

def ImageToolsCombineImageThreadWorker(pathToCombine, outputFile, streamToUse):
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
	with redirect_stdout(streamToUse):
		try:

			minx, miny, minz, maxx, maxy, maxz = cropBox
			if cropBox == [0, 0, 0, 0, 0, 0] or cropBox == ['', '', '', '', '', '']:
				cropped = True
			else:
				cropped = False

			print('Loading H5 File')
			h5f = h5py.File(h5path, 'r') #TODO make sure file is always closed properly using with:

			imageIndexList = []
			planeIndexList = []
			areaList = []

			metadata = ast.literal_eval(h5f['vol0'].attrs['metadata'])
			#metadata = {'configType':'2d', 'x_scale':1, 'y_scale':1}
			configType = metadata['configType'].lower()
			if '2d' in configType:
				xScale, yScale = metadata['x_scale'], metadata['y_scale']
				if 'instance' in configType:
					dataset = h5f['processed']
					imageIndexList = []
					areaList = []
					for imageIndex in range(dataset.shape[0]):
						dFile = dataset[imageIndex]
						for unique in np.unique(dFile):
							if unique == 0:
								continue
							imageIndexList.append(imageIndex)
							areaList.append(np.count_nonzero(dFile == unique) * xScale * yScale)
					df = pd.DataFrame({"Image Index":imageIndexList, "Area":areaList})
					df.to_csv(outputFile)
					print("Wrote 2D CSV to " + outputFile)

				else: # Semantic
					dataset = h5f['vol0']
					for imageIndex in range(dataset.shape[1]): #Iterate over 2d images
						d = dataset[:,imageIndex,:,:]

						for index in range(1, d.shape[0]): #Iterate over planes

							indexesToCheck = []
							for i in range(d.shape[0]):
								if i == index:
									continue
								indexesToCheck.append(i)
							mask = d[indexesToCheck[0]] < d[index]
							for i in indexesToCheck[1:]:
								mask = mask & d[i] < d[index]

							labels_out = cc3d.connected_components(mask, connectivity=8)
							num, count = np.unique(labels_out, return_counts=True)
							countList = count[1:]

							xScale, yScale = metadata['x_scale'], metadata['y_scale']
							countList = np.array(countList) * xScale * yScale

							for element in list(sorted(countList)):
								imageIndexList.append(imageIndex)
								planeIndexList.append(index)
								areaList.append(element)

					df = pd.DataFrame({"Image Index":imageIndexList, "Plane Index":planeIndexList, "Area":areaList})
					df.to_csv(outputFile)
					print("Wrote 2D CSV to " + outputFile)
			else: # 3d
				if 'instance' in configType: # Instance 3d
					if cropped: # Instance Cropped 3d
						d = h5f['processed'][zmin:zmax, xmin:xmax, ymin:ymax]
						metadata = ast.literal_eval(h5f['vol0'].attrs['metadata'])
						xScale, yScale, zScale = metadata['x_scale'], metadata['y_scale'], metadata['z_scale']
						h5f.close()

						num, count = np.unique(d, return_counts=True)
						countList = np.array(count)[1:]
						countList = countList * xScale * yScale * zScale

						with open(outputFile, 'w') as outFile:
							wr = csv.writer(outFile)
							wr.writerow(countList)

					else: # Instance 3d using the pre-calculated countDic from the whole dataset
						print('H5 Loaded, reading Stats from instance segmentation (Output will be in nanometers^3)')
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

						with open(outputFile, 'w') as outFile:
							wr = csv.writer(outFile)
							wr.writerow(countList)

				elif 'semantic' in configType: #Semantic 3D
					if cropped:
						d = h5f['vol0'][:]
					else:
						d = h5f['vol0'][:,zmin:zmax,xmin:xmax,ymin:ymax]
					h5f.close()
					df = {}

					for index in range(1, d.shape[0]):
						print()
						print('==============================')
						if cropped:
							print('Cropped To:', xmin, xmax, ymin, ymax, zmin, zmax)
							print('xmin, xmax, ymin, ymax, zmin, zmax')
							print()
						print('Outputting Stats for layer:', index)
						indexesToCheck = []
						for i in range(d.shape[0]):
							if i == index:
								continue
							indexesToCheck.append(i)
						mask = d[indexesToCheck[0]] < d[index]
						for i in indexesToCheck[1:]:
							mask = mask & d[i] < d[index]
						labels_out = cc3d.connected_components(mask, connectivity=26)
						num, count = np.unique(labels_out, return_counts=True)
						countList = count[1:]

						xScale, yScale, zScale = metadata['x_scale'], metadata['y_scale'], metadata['z_scale']
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

						df[index] = np.array(countList)

					indexColumn = []
					valueColumn = []
					for index in range(1, d.shape[0]):
						indexColumn += list(np.ones(len(df[index]), dtype=int) * index)
						valueColumn += list(df[index])
					df2 = pd.DataFrame({"Plane Index":indexColumn, "volume":valueColumn})
					df2.to_csv(outputFile)
		except:
			print('Critical Error:')
			traceback.print_exc()

def VisualizeThreadWorker(filesToVisualize, streamToUse, voxel_size=1):
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
					print('Finished Calculating Mesh, saving ' + outputFilenameShort + '_instance_mesh_.ply')
					o3d.io.write_triangle_mesh(outputFilenameShort + '_instance_mesh_.ply', mesh)
					print('Finished making meshes')
				if makePoints:
					print('Loaded H5 File, creating point cloud')
					cloud = instanceArrayToPointCloud(d)
					print('Finished Calculating Point Cloud, saving')
					o3d.io.write_point_cloud(outputFilenameShort + '_instance_pointCloud_.ply', cloud)
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
						o3d.io.write_triangle_mesh(outputFilenameShort + '_semantic_mesh_' + str(index) + '.ply', mesh)
					print('Finished with making Meshes')
					print()
				if makePoints:
					print('Starting Point Cloud Creation')
					for index in range(1, numIndexes):
						print('Creating Point Cloud for Index:', index)
						cloud = getPointCloudForIndex(d, index)
						o3d.io.write_point_cloud(outputFilenameShort + '_semantic_pointCloud_' + str(index) + '.ply', cloud)
					print('Finished with making Point Clouds')
					print()

			else:
				print('Unrecognized File')
				h5f.close()
				return

		except:
			print('Critical Error:')
			traceback.print_exc()