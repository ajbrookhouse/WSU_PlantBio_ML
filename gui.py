"""

https://wiki.tcl-lang.org/page/List+of+ttk+Themes
adapta is liked
aqua is ok

"""

import tkinter as tk
import tkinter.ttk as ttk
# from ttkbootstrap import Style as bootStyle
from pygubu.widgets.pathchooserinput import PathChooserInput
from connectomics.config import *
import yaml
import yacs
import time
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from os.path import sep
from os import listdir
import h5py
import open3d as o3d
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
import torch
import connectomics
import traceback
import sys
import argparse
import numpy as np

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

def OutputToolsMakeGeometriesThread(h5path, makeMeshs, makePoints, buttonToEnable, streamToUse):
	with redirect_stdout(streamToUse):
		with redirect_stderr(streamToUse):
			print('Loading H5 File')
			h5f = h5py.File(h5path, 'r')
			d = np.array(h5f['vol0'])
			print('Loaded')
			numIndexes = d.shape[0]
			rootFolder, h5Filename = head_tail = os.path.split(h5path)
			h5Filename = h5Filename[:-3]

			if makeMeshs:
				print('Starting Mesh Creation')
				for index in range(1, numIndexes):
					print('Creating Mesh for Index:', index)
					mesh = arrayToMesh(d, index)
					o3d.io.write_triangle_mesh(rootFolder + h5Filename + '_mesh_' + str(index) + '.ply', mesh)
				print('Finished with making Meshes')
				print()
			if makePoints:
				print('Starting Point Cloud Creation')
				for index in range(1, numIndexes):
					print('Creating Point Cloud for Index:', index)
					cloud = getPointCloudForIndex(d, index)
					o3d.io.write_point_cloud(rootFolder + h5Filename + '_pointCloud_' + str(index) + '.pcd', cloud)
				print('Finished with making Point Clouds')
				print()		
	buttonToEnable['state'] = 'normal'

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

def MessageBox(message):
	print(message)
	tk.messagebox.showinfo(title=None, message=message)

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

def trainFromMain():
	args = get_args()
	if args.local_rank == 0 or args.local_rank is None:
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

def predFromMain(config, checkpoint):
	args = get_args_modified(['--inference', '--checkpoint', checkpoint, '--config-file', config])

	if args.local_rank == 0 or args.local_rank is None:
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

@contextlib.contextmanager
def redirect_argv(*args):
	sys._argv = sys.argv[:]
	sys.argv=args
	yield
	sys.argv = sys._argv

def trainThreadWorker(cfg, stream, button):
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	try:
		with redirect_stdout(stream):
			with redirect_stderr(stream):
				with redirect_argv('','--config-file=' + cfg):
					# args = get_args()
					# trainer = Trainer(cfg, device, 'train',
					# 	rank=args.local_rank,
					# 	checkpoint=args.checkpoint)
					# trainer.train()
					#save_all_cfg(cfg, '')
					trainFromMain()
	except:
		traceback.print_exc()
	button['state'] = 'normal'

def useThreadWorker(cfg, stream, button, checkpoint):
	print('In Use Thread Worker')
	try:
		print('HERERERERERERE')
		with redirect_stdout(stream):
			with redirect_stderr(stream):
				print('HERERERERERERE22222')
				print('Here, ', cfg)
				predFromMain(cfg, checkpoint)
	except:
		traceback.print_exc()
		print('Except Here in useThreadWorker')
	button['state'] = 'normal'

def trainThreadWorkerCluster(cfg, stream, button, url, username, password, trainStack, trainLabels, submissionScriptString, folderToUse, pytorchFolder, submissionCommand):
	with redirect_stdout(stream):
		with redirect_stderr(stream):
			runRemoteServer(url, username, password, trainStack, trainLabels, configToUse, submissionScriptString, folderToUse, pytorchFolder, submissionCommand)
	button['state'] = 'normal'

def ImageToolsCombineImageThread(pathToCombine, buttonToEnable, streamToUse):
	with redirect_stdout(streamToUse):
		with redirect_stderr(streamToUse):
			images = []
			for image in list(sorted(listdir(pathToCombine))):
				print("Reading image:", image)
				if not image == '_combined.tif':
					im = Image.open(pathToCombine + sep + image)
					images.append(im)
			print("Writing Combined image:", pathToCombine + sep + '_combined.tif')
			images[0].save(pathToCombine + sep + '_combined.tif', save_all=True, append_images=images[1:])
			print("Finished Combining Images")
			buttonToEnable['state'] = 'normal'

class TextboxStream(StringIO):
	def __init__(self, widget, maxLen = None):
		super().__init__()
		self.widget = widget

	def write(self, string):
		self.widget.insert("end", string)
		self.widget.see('end')

class TabguiApp():
	def __init__(self, master=None):
		self.root = master
		self.root.title("The Title")
		# style = bootStyle(theme='sandstone')
		self.root.option_add("*font", "Times_New_Roman 12")

		self.RefreshVariables()

		# build ui
		self.tabHolder = ttk.Notebook(master)
		self.frameTrain = tk.Frame(self.tabHolder)
		self.numBoxTrainGPU = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainGPU.configure(from_='0', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxTrainGPU.delete('0', 'end')
		self.numBoxTrainGPU.insert('0', _text_)
		self.numBoxTrainGPU.grid(column='1', row='3')
		self.numBoxTrainCPU = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainCPU.configure(from_='1', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxTrainCPU.delete('0', 'end')
		self.numBoxTrainCPU.insert('0', _text_)
		self.numBoxTrainCPU.grid(column='1', row='4')
		self.pathChooserTrainImageStack = PathChooserInput(self.frameTrain)
		self.pathChooserTrainImageStack.configure(type='file')
		self.pathChooserTrainImageStack.grid(column='1', row='0')
		self.pathChooserTrainLabels = PathChooserInput(self.frameTrain)
		self.pathChooserTrainLabels.configure(type='file')
		self.pathChooserTrainLabels.grid(column='1', row='1')
		self.label1 = ttk.Label(self.frameTrain)
		self.label1.configure(text='Image Stack (.tif): ')
		self.label1.grid(column='0', row='0')
		self.label2 = ttk.Label(self.frameTrain)
		self.label2.configure(text='Labels (.h5)')
		self.label2.grid(column='0', row='1')
		self.label4 = ttk.Label(self.frameTrain)
		self.label4.configure(text='# GPU: ')
		self.label4.grid(column='0', row='3')
		self.label5 = ttk.Label(self.frameTrain)
		self.label5.configure(text='# CPU: ')
		self.label5.grid(column='0', row='4')
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
		self.configChooserSelect = tk.OptionMenu(self.frameTrain, self.configChooserVariable, *self.configs)
		self.configChooserSelect.grid(column='0', columnspan='2', row='2')

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
		self.textTrainOutputStream = TextboxStream(self.textTrainOutput)


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
		self.pathChooserUseImageStack = PathChooserInput(self.framePredict)
		self.pathChooserUseImageStack.configure(type='file')
		self.pathChooserUseImageStack.grid(column='1', row='0')
		self.pathChooserUseOutputFile = PathChooserInput(self.framePredict)
		self.pathChooserUseOutputFile.configure(type='file')
		self.pathChooserUseOutputFile.grid(column='1', row='1')
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
		self.label23 = ttk.Label(self.framePredict)
		self.label23.configure(text='Image Stack (.tif): ')
		self.label23.grid(column='0', row='0')
		self.label24 = ttk.Label(self.framePredict)
		self.label24.configure(text='Output File: ')
		self.label24.grid(column='0', row='1')
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
		self.textUseOutputStream = TextboxStream(self.textUseOutput)

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
		self.modelChooserSelect = tk.OptionMenu(self.framePredict, self.modelChooserVariable, *self.models)
		self.modelChooserSelect.grid(column='0', columnspan='2', row='2')

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
		self.label42 = ttk.Label(self.frameImage)
		self.label42.configure(text='Folder Of Images: ')
		self.label42.grid(column='0', row='0')
		# self.label43 = ttk.Label(self.frameImage)
		# self.label43.configure(text='Folder Of Label Images')
		# self.label43.grid(column='0', row='1')
		self.pathchooserinputImageImageFolder = PathChooserInput(self.frameImage)
		self.pathchooserinputImageImageFolder.configure(type='folder')
		self.pathchooserinputImageImageFolder.grid(column='1', row='0')
		# self.pathchooserinputImageLabelFolder = PathChooserInput(self.frameImage)
		# self.pathchooserinputImageLabelFolder.configure(type='folder')
		# self.pathchooserinputImageLabelFolder.grid(column='1', row='1')
		self.buttonImageCombine = ttk.Button(self.frameImage)
		self.buttonImageCombine.configure(text='Combine Images Into Stack')
		self.buttonImageCombine.grid(column='2', row='0')
		self.buttonImageCombine.configure(command=self.ImageToolsCombineImageButtonPress)
		# self.buttonImageMakeLabel = ttk.Button(self.frameImage)
		# self.buttonImageMakeLabel.configure(text='Make Label File')
		# self.buttonImageMakeLabel.grid(column='2', row='1')
		# self.buttonImageMakeLabel.configure(command=self.ImageToolsMakeLabelButtonPress)

		self.textImageTools = tk.Text(self.frameImage)
		self.textImageTools.configure(height='10', width='50')
		_text_ = '''Image Tools Output Will Be Here'''
		self.textImageTools.insert('0.0', _text_)
		self.textImageTools.grid(column='0', columnspan='3', row='1')
		self.textUseImageToolsStream = TextboxStream(self.textImageTools)

		self.frameImage.configure(height='200', width='200')
		self.frameImage.pack(side='top')
		self.tabHolder.add(self.frameImage, text='Image Tools')

		##################################################################################################################
		# 
		##################################################################################################################

		self.frameOutputTools = ttk.Frame(self.tabHolder)
		self.label44 = ttk.Label(self.frameOutputTools)
		self.label44.configure(text='Model Output (.h5): ')
		self.label44.grid(column='0', row='0')
		self.pathchooserinputOutputModelOutput = PathChooserInput(self.frameOutputTools)
		self.pathchooserinputOutputModelOutput.configure(type='file')
		self.pathchooserinputOutputModelOutput.grid(column='1', row='0')
		self.buttonOutputGetStats = ttk.Button(self.frameOutputTools)
		self.buttonOutputGetStats.configure(text='Get Model Output Stats')
		self.buttonOutputGetStats.grid(column='0', columnspan='2', row='4')
		self.buttonOutputGetStats.configure(command=self.OutputToolsModelOutputStatsButtonPress)
		self.textOutputOutput = tk.Text(self.frameOutputTools)
		self.textOutputOutput.configure(height='10', width='50')
		_text_ = '''Output Goes Here'''
		self.textOutputOutput.insert('0.0', _text_)
		self.textOutputOutput.grid(column='0', columnspan='2', row='5')
		self.textOutputOutputStream = TextboxStream(self.textOutputOutput)
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

		self.labelVisualizeFile = ttk.Label(self.frameVisualize)
		self.labelVisualizeFile.configure(text='File To Visulize: ')
		self.labelVisualizeFile.grid(column='0', row='0')

		self.pathchooserVisualize = PathChooserInput(self.frameVisualize)
		self.pathchooserVisualize.configure(type='file')
		self.pathchooserVisualize.grid(column='1', row='0')

		self.buttonOutputMakeGeometries = ttk.Button(self.frameVisualize)
		self.buttonOutputMakeGeometries.configure(text='Visualize')
		self.buttonOutputMakeGeometries.grid(column='0', columnspan='2', row='1')
		self.buttonOutputMakeGeometries.configure(command=self.VisualizeButtonPress)

		self.tabHolder.add(self.frameVisualize, text='Visualize')

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
		fileToVisualize = self.pathchooserVisualize.entry.get()

		toVisualize = o3d.io.read_point_cloud(fileToVisualize)
		toVisualize = o3d.geometry.VoxelGrid.create_from_point_cloud(toVisualize, voxel_size=5)
		o3d.visualization.draw_geometries([toVisualize])

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

			with open('temp.yaml','w') as file:
				yaml.dump(config, file)

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
				t = threading.Thread(target=trainThreadWorker, args=('temp.yaml', self.textTrainOutputStream, self.buttonTrainTrain))
				t.setDaemon(True)
				t.start()
		except:
			self.buttonTrainTrain['state'] = 'normal'
		self.RefreshVariables()

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

	def UseModelLabelButtonPress(self):
		print('Pressed')
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
				print('Not Cluster')
				checkpointFiles = os.listdir('Data' + sep + 'models' + sep + model)
				biggestCheckpoint = 0

				for subFile in checkpointFiles:
					try:
						checkpointNumber = int(subFile.split('_')[1][:-8])
						#print(subFile, checkpointNumber)
					except:
						pass
						#print(subFile)
					if checkpointNumber > biggestCheckpoint:
						biggestCheckpoint = checkpointNumber
					#print('biggest checkpoint',biggestCheckpoint)

				checkpoint = 'Data' + sep + 'models' + sep + model + sep + 'checkpoint_' + str(biggestCheckpoint).zfill(5) + '.pth.tar'
				t = threading.Thread(target=useThreadWorker, args=('temp.yaml', self.textUseOutputStream, self.buttonUseLabel, checkpoint))
				t.setDaemon(True)
				t.start()
		except:
			print('excepting the thing')
			traceback.print_exc()
			self.buttonTrainTrain['state'] = 'normal'

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
			self.buttonImageCombine['state'] = 'disabled'
			pathToCombine = self.pathchooserinputImageImageFolder.entry.get()
			t = threading.Thread(target=ImageToolsCombineImageThread, args=(pathToCombine, self.buttonImageCombine, self.textUseImageToolsStream))
			t.setDaemon(True)
			t.start()
		except:
			traceback.print_exc()
			self.buttonTrainTrain['state'] = 'normal'

	def OutputToolsModelOutputStatsButtonPress(self):
		pass

	def OutputToolsMakeGeometriesButtonPress(self):
		try:
			self.buttonOutputMakeGeometries['state'] = 'disabled'
			h5Path = self.pathchooserinputOutputModelOutput.entry.get()
			makeMeshs = self.checkbuttonOutputMeshs.instate(['selected'])
			makePoints = self.checkbuttonOutputPointClouds.instate(['selected'])
			t = threading.Thread(target=OutputToolsMakeGeometriesThread, args=(h5Path, makeMeshs, makePoints, self.buttonOutputMakeGeometries, self.textOutputOutputStream))
			t.setDaemon(True)
			t.start()
		except:
			traceback.print_exc()
			self.buttonOutputMakeGeometries['state'] = 'normal'

	def RefreshVariables(self):
		configs = sorted(listdir('Data' + sep + 'configs'))
		for file in configs:
			if not file[-5:] == '.yaml':
				configs.remove(file)
		configs.remove('default.yaml')
		configs.insert(0,'default.yaml')
		self.configs = configs

		modelList = []
		models = listdir('Data' + sep + 'models')
		for model in models:
			if os.path.isdir('Data' + sep + 'models' + sep + model) and not 'log' in model:
				modelList.append(model)
		if len(modelList) == 0:
			modelList.append('No Models Yet')
		self.models = modelList

	def SaveConfigButtonPress(self):
		name = self.entryConfigName.get()

		architecture = self.entryConfigArchitecture.get()
		inputSize = self.entryConfigInputSize.get()
		outputSize = self.entryConfigOutputSize.get()
		inplanes = self.numBoxConfigInPlanes.get()
		outplanes = inplanes
		lossOption = self.entryConfigLossOption.get()
		lossWeight = self.entryConfigLossWeight.get()
		targetOpt = self.entryConfigTargetOpt.get()
		weightOpt = self.entryConfigWeightOpt.get()
		padSize = self.entryConfigPadSize.get()
		LR_Scheduler = self.entryConfigLRScheduler.get()
		baseLR = self.numBoxTrainBaseLR.get()
		steps = self.entryConfigSteps.get()

		with open('Data' + sep + 'configs' + sep + 'default.yaml','r') as file:
			defaultConfig = yaml.load(file, Loader=yaml.FullLoader)
		defaultConfig['MODEL']['ARCHITECTURE'] = architecture
		defaultConfig['MODEL']['INPUT_SIZE'] = inputSize
		defaultConfig['MODEL']['OUTPUT_SIZE'] = outputSize
		defaultConfig['MODEL']['IN_PLANES'] = inplanes
		defaultConfig['MODEL']['OUT_PLANES'] = outplanes
		defaultConfig['MODEL']['LOSS_OPTION'] = lossOption
		defaultConfig['MODEL']['LOSS_WEIGHT'] = lossWeight
		defaultConfig['MODEL']['TARGET_OPT'] = targetOpt
		defaultConfig['MODEL']['WEIGHT_OPT'] = weightOpt
		defaultConfig['DATASET']['PAD_SIZE'] = padSize
		defaultConfig['SOLVER']['LR_SCHEDULER_NAME'] = LR_Scheduler
		defaultConfig['SOLVER']['BASE_LR'] = baseLR
		defaultConfig['SOLVER']['STEPS'] = steps

		with open('Data' + sep + 'configs' + sep + name + '.yaml','w') as file:
			yaml.dump(defaultConfig, file)

		MessageBox('Config Saved')
		self.RefreshVariables()


	def run(self):
		self.mainwindow.mainloop()

# print(sys.argv)
# print(type(sys.argv))
# # print('=======================================')
# # print(get_args())
# # print('=======================================')
# # with redirect_argv(['--config-file ' + '/media/aaron/External/Spectroscopy_Images/_Official_GUI/Data/configs/MitoEM-R-BC.yaml', '--inference', '--checkpoint ' + '/media/aaron/External/Spectroscopy_Images/_Official_GUI/Data/models/TestModel/checkpoint_00010.pth.tar']):
# # 	#print('--config-file ' + cfg + ' --inference' + ' --checkpoint ' + checkpoint)
# # 	trainFromMain()
# # exit()

if __name__ == '__main__':
	import tkinter as tk
	root = tk.Tk()
	app = TabguiApp(root)
	app.run()