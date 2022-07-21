import ast
import json

import h5py
import open3d as o3d
from skimage import measure
from PIL import Image, ImageSequence, ImageColor
import numpy as np

import matplotlib as mpl
defaultMatplotlibBackend = mpl.get_backend()
defaultMatplotlibBackend = 'TkAgg'
from matplotlib import pyplot as plt

from utils import *

def getImageFromDataset(inputDataset, zindex):
	"""When given a dataset filename of either a .tif, .json, or .txt dataset and a zindex, returns the image at that index of the dataset

	Parameters
	----------
	inputDataset : str
		The filepath of the dataset you want to get an image from

	zindex : int
		The z index at which you want to get a 2D image slice from

	Returns
	-------
	PIL.Image
		A PIL Image that is a 2D slice of `inputDataset` perpendicular to the z plan at location `zindex`

	Raises
	------
	Exception
		If the filename passed does not end in '.tif', '.tiff', '.json', or '.txt'
	"""

	if inputDataset[-4:] == '.tif' or inputDataset[-5:] == '.tiff':
		img = Image.open(inputDataset)
		img.seek(zindex)
		return img

	elif inputDataset[-5:] == '.json':
		with open(inputDataset, 'r') as inFile:
			d = json.load(inFile)
		img = Image.open(d['image'][zindex])
		return img

	elif inputDataset[-4:] == '.txt':
		filteredImages = []
		with open(inputDataset, 'r') as inFile:
			images = inFile.readlines()
		for image in images:
			if len(image.strip() > 3):
				filteredImages.append(image)
		img = Image.open(filteredImages[zindex].strip())
		return img

	else:
		raise Exception('Unknown Dataset filetype ' + inputDataset[inputDataset.rindex('.'):] + ' Passed to getImageFromDataset')

# def getShapeOfDataset(inputDataset): # DUPLICATE in utils, commented out for now
# 	if inputDataset[-4:] == '.tif': # https://stackoverflow.com/questions/46436522/python-count-total-number-of-pages-in-group-of-multi-page-tiff-files
# 		img = Image.open(inputDataset)
# 		width, height = img.size
# 		count = 0

# 		while True:
# 			try:   
# 				img.seek(count)
# 			except EOFError:
# 				break       
# 			count += 1

# 		return count, height, width

# 	elif inputDataset[-5:] == '.json':
		
# 		with open(inputDataset, 'r') as inFile:
# 			d = json.load(inFile)
# 		# numFiles = len(d['image'])
# 		# img = Image.open(d['image'][0])
# 		# width, height = img.size
# 		return d['depth'], d['height'], d['width']

# 	elif inputDataset[-4:] == '.txt':
# 		filteredImages = []
# 		with open(inputDataset, 'r') as inFile:
# 			images = inFile.readlines()
# 		for image in images:
# 			if len(image.strip()) > 3:
# 				filteredImages.append(image)
# 		img = Image.open(filteredImages[0].strip())
# 		width, height = img.size
# 		count = len(filteredImages)
# 		return count, height, width

# 	else:
# 		raise Exception('Unknown Dataset filetype ' + inputDataset[inputDataset.rindex('.'):] + ' Passed to getShapeOfDataset')

def create2DLabelCheckSemanticImage(inputDataset, modelOutputDataset, z, colors = ['#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9', '#ffffff', '#000000']):
	"""Creates an image with the raw image from `inputDataset` in the background, and the model's predictions (semantic models only) highlighted in different colors on top of it

	Note: this is very similar to create2DLabelCheckSemanticImageForIndex. The difference is as follows:
	create2DLabelCheckSemanticImage - Creates an image that has all semantic planes highlighted using the list of colors
	create2DLabelCheckSemanticImageForIndex - Creates an image that only highlights one specified plane in red

	Parameters
	----------
	inputDataset : str
		The filepath of the dataset you want to use for the raw images

	modelOutputDataset : H5 Dataset
		The H5 Dataset from the a model's prediction h5 file

	z : int
		The z index to grab the 2D Images from

	colors : list of strings of colors encoded in hex, ex: white is '#ffffff', optional
		These colors will be used for the highlighting in order.
		The first is used for the second semantic prediction plane (the first is presumably null)
		The second is used for the third semantic prediction plane, etc.
		The default list was one I found online that was supposed to be very accessible including different types of color blindness

	Returns
	-------
	numpy.ndarray
		A 5 dimensional array (x, y, r, g, b) that represents the image with model predictions highlighted different colors
	"""

	rawImage = getImageFromDataset(inputDataset, z)
	background = np.array(rawImage.convert('RGB'))

	maskList = []
	modelPrediction = modelOutputDataset[:,z,:,:]
	numPlanes = modelOutputDataset.shape[0]
	for plane in range(1, numPlanes):

		indexesToCheck = []
		for i in range(numPlanes):
			if i == plane:
				continue
			indexesToCheck.append(i)

		mask = modelPrediction[indexesToCheck[0]] < modelPrediction[plane]
		for i in indexesToCheck[1:]:
			mask = mask & modelPrediction[i] < modelPrediction[plane]

		maskList.append(mask)
		colorToUse = ImageColor.getcolor(colors[plane-1], "RGB")
		background[mask] = colorToUse

	return background

def create2DLabelCheckSemanticImageForIndex(inputDataset, modelOutputDataset, z, plane):
	"""Creates an image with the raw image from `inputDataset` in the background, and the model's predictions for the specified `plane` in red

	Note: this is very similar to create2DLabelCheckSemanticImage. The difference is as follows:
	create2DLabelCheckSemanticImage - Creates an image that has all semantic planes highlighted using the list of colors
	create2DLabelCheckSemanticImageForIndex - Creates an image that only highlights one specified plane in red

	Parameters
	----------
	inputDataset : str
		The filepath of the dataset you want to use for the raw images

	modelOutputDataset : H5 Dataset
		The H5 Dataset from the a model's prediction h5 file

	z : int
		The z index to grab the 2D Images from

	plane : int
		The index of the plane to highlight on the image

	Returns
	-------
	numpy.ndarray
		A 5 dimensional array (x, y, r, g, b) that represents the image at index `z` with the semantic plane `plane` highlighted in red
	"""

	rawImage = getImageFromDataset(inputDataset, z)
	background = np.array(rawImage.convert('RGB'))

	maskList = []
	modelPrediction = modelOutputDataset[:,z,:,:]
	numPlanes = modelOutputDataset.shape[0]
	indexesToCheck = []

	for planeIter in range(0, numPlanes):
		if planeIter == plane:
			continue
		indexesToCheck.append(planeIter)

	mask = modelPrediction[indexesToCheck[0]] < modelPrediction[plane]
	for i in indexesToCheck[1:]:
		mask = mask & modelPrediction[i] < modelPrediction[plane]

	maskList.append(mask)
	colorToUse = ImageColor.getcolor("#ff0000", "RGB")
	background[mask] = colorToUse

	return background

def create2DLabelCheckInstanceImage(inputDataset, modelOutputDataset, z):
	"""Creates an image with the raw image from `inputDataset` in the background, and the model's predictions (instance models only) are highlighted  in various colors

	Parameters
	----------
	inputDataset : str
		The filepath of the dataset you want to use for the raw images

	modelOutputDataset : H5 Dataset
		The H5 Dataset from the a model's prediction h5 file (must be an instance model)

	z : int
		The z index to grab the 2D Images from

	Returns
	-------
	numpy.ndarray
		A 5 dimensional array (x, y, r, g, b) that represents the image at index `z` with the model's predictions highlighted in random colors
	"""

	rawImage = getImageFromDataset(inputDataset, z)
	background = np.array(rawImage.convert('RGB'))

	modelPrediction = modelOutputDataset[z,:,:]

	for unique in np.unique(modelPrediction):
		if unique == 0:
			continue
		mask = modelPrediction == unique
		colorToUse = np.random.rand(3,) * 255
		background[mask] = colorToUse

	return background

def create2DLabelCheckSemantic(inputDataset, modelOutput, numberOfImages, colors = ['#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9', '#ffffff', '#000000']):
	"""Similar to create2DLabelCheckSemanticImage, however it opens an interactive matplotlib window that shows several examples

	Parameters
	----------
	inputDataset : str
		The filepath of the dataset you want to use for the raw images

	modelOutputDataset : H5 Dataset
		The H5 Dataset from the a model's prediction h5 file

	numberOfImages : int
		How many interactive windows to show

	colors : list of hexadecimal strings. Ex. white = '#ffffff', optional

	Returns
	-------
	None
		Opens an interactive window that shows several images from create2DLabelCheckSemanticImage evenly spaced throughout the dataset. Shows `numberOfImages` images
	
	Raises
	------
	Assert
		If the shape of `inputDataset` and `modelOutput` are not the same
	"""

	mpl.use(defaultMatplotlibBackend)
	h5File = h5py.File(modelOutput, 'r')
	dataset = h5File['vol0']
	datasetShape = dataset.shape
	imageShape = getShapeOfDataset(inputDataset)
	numPlanes = dataset.shape[0]
	stride = int(datasetShape[1] / numberOfImages)

	assert datasetShape[1:] == imageShape, "in create2DLabelCheckSemantic, inputDataset and model output do not have the same shape"

	for z in range(0, datasetShape[1], stride):
		rawImage = getImageFromDataset(inputDataset, z)
		background = np.array(rawImage.convert('RGB'))

		maskList = []
		modelPrediction = dataset[:,z,:,:]

		for plane in range(1, numPlanes):

			background = create2DLabelCheckSemanticImage(inputDataset, dataset, z)

		rawImage = np.array(rawImage)
		plt.figure(figsize=(20,10))
		plt.suptitle('Index: ' + str(z))
		plt.subplot(121)
		plt.imshow(255-rawImage, cmap='binary')
		plt.title('Raw Image')
		plt.subplot(122)
		plt.imshow(background)
		plt.title('Labelled')
		plt.show()
		plt.close()
	h5File.close()
	mpl.use('Agg')

def create2DLabelCheckInstance(inputDataset, modelOutput, numberOfImages):
	"""Similar to create2DLabelCheckInstanceImage, however it opens an interactive matplotlib window that shows several examples

	Parameters
	----------
	inputDataset : str
		The filepath of the dataset you want to use for the raw images

	modelOutputDataset : H5 Dataset
		The H5 Dataset from the a model's prediction h5 file (must be from an instance model)

	numberOfImages : int
		How many interactive windows to show

	Returns
	-------
	None
		Opens an interactive window that shows several images from create2DLabelCheckInstanceImage evenly spaced throughout the dataset. Shows `numberOfImages` images
	
	Raises
	------
	Assert
		If the shape of `inputDataset` and `modelOutput` are not the same
	"""

	mpl.use(defaultMatplotlibBackend)
	h5File = h5py.File(modelOutput, 'r')
	dataset = h5File['processed']
	datasetShape = dataset.shape
	imageShape = getShapeOfDataset(inputDataset)
	stride = int(datasetShape[0] / numberOfImages)

	assert datasetShape == imageShape, "in create2DLabelCheckInstance, inputDataset and model output do not have the same shape"

	for z in range(0, datasetShape[1], stride):
		rawImage = getImageFromDataset(inputDataset, z)
		background = np.array(rawImage.convert('RGB'))

		maskList = []
		background = create2DLabelCheckInstanceImage(inputDataset, dataset, z)

		rawImage = np.array(rawImage)
		plt.figure(figsize=(20,10))
		plt.suptitle('Index: ' + str(z))
		plt.subplot(121)
		plt.imshow(255-rawImage, cmap='binary')
		plt.title('Raw Image')
		plt.subplot(122)
		plt.imshow(background)
		plt.title('Labelled')
		plt.show()
		plt.close()
	mpl.use('Agg')

def getMetadataForH5(h5filename):
	"""Gets the metadata in a models H5 File

	Parameters
	----------
	h5filename : str
		The filepath of the dataset you want to get the metadata for

	Returns
	-------
	dictionary
		Returns a dictionary of the metadata of the dataset in `h5filename`
	"""

	f = h5py.File(h5filename, 'r')
	metadata = f['vol0'].attrs['metadata']
	return ast.literal_eval(metadata)

def create3DLabelAnimationSemantic(inputDataset, modelOutput, outputDir, scaleFactor=10, colors = ['#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9', '#ffffff', '#000000']):
	# TODO add docstring
	mpl.use(defaultMatplotlibBackend)
	h5File = h5py.File(modelOutput, 'r')
	dataset = h5File['vol0']
	datasetShape = dataset.shape
	imageShape = getShapeOfDataset(inputDataset)

	assert datasetShape[1:] == imageShape, "in create3DLabelAnimationSemantic, inputDataset and model output do not have the same shape"

	for z in range(0, datasetShape[1]):
		tc = TimeCounter(dataset.shape[1], 'minutes')
		twoDImage = create2DLabelCheckSemanticImage(inputDataset, dataset, z)

		maskList = []
		modelPrediction = dataset[:,:z+1,:,:]
		numPlanes = modelPrediction.shape[0]
		for plane in range(1, numPlanes):

			indexesToCheck = []
			for i in range(numPlanes):
				if i == plane:
					continue
				indexesToCheck.append(i)

			mask = modelPrediction[indexesToCheck[0]] < modelPrediction[plane]
			for i in indexesToCheck[1:]:
				mask = mask & modelPrediction[i] < modelPrediction[plane]

			maskList.append(mask)
			colorToUse = ImageColor.getcolor(colors[plane-1], "RGB")

		maskToPlot = maskList[0]
		maskToPlot = maskToPlot[::1,::scaleFactor,::scaleFactor]

		fig = plt.figure()
		plt.title('Frame ' + str(z))
		plt.axis('off')

		ax1 = fig.add_subplot(121)
		ax1.imshow(twoDImage)

		ax2 = fig.add_subplot(122, projection='3d')
		ax2.voxels(maskToPlot)

		ax2.axes.set_zlim3d(bottom=0, top=int(dataset.shape[3] / scaleFactor + 1))
		ax2.set_zlabel('Z')

		ax2.axes.set_ylim3d(bottom=0, top=int(dataset.shape[2] / scaleFactor + 1))
		ax2.set_ylabel('Y')

		ax2.axes.set_xlim3d(left=0, right=int(dataset.shape[1] / 1 + 1))
		ax2.set_xlabel('X')

		# ax2.view_init(-140, 60)

		plt.savefig(os.path.join(outputDir, 'frame' + str(z).zfill(4) + '.png'))
		plt.close()
		tc.tick()
		tc.print()

	h5File.close()
	mpl.use('Agg')

def createTxtFileFromImageList(imageList, outputFile):
	"""Given a list of images and an output filepath, creates a .txt dataset

	Parameters
	----------
	imageList : list of strings
		list of string filepaths of images to be put in the .txt dataset. Ex ['test1.tif', 'test2.tif']

	outputFile : str
		filepath to save the .txt dataset to

	Returns
	-------
	None
		saves a .txt file
	"""

	with open(outputFile, 'w') as out:
		for image in imageList:
			print('Writing', image)
			out.write(image.strip() + '\n')
	print('Completely Done, output at:', outputFile)

def createTifFromImageList(imageList, outputFile):
	"""Combines a list of images into one page .tiff file

	Parameters
	----------
	imageList : list of strings
		list of string filepaths of .tif images to combine

	outputFile : str
		output path to save the combined .tif file to

	Returns
	-------
	None
		Saves a paged .tif file
	"""

	images = []

	for image in imageList:
		print("Reading image:", image)
		if not image == '_combined.tif':
			im = Image.open(image)
			images.append(im)
	print("Writing Combined image:", outputFile)
	images[0].save(outputFile, save_all=True, append_images=images[1:])
	print("Finished Combining Images")

def writeJsonForImages(imageList, outputJsonPath):
	"""Creates a .json dataset from a list of images

	Parameters
	----------
	imageList : list of strings
		list of string filepaths of .tif images to combine

	outputJsonPath : str
		output path to save the .json file to

	Returns
	-------
	None
		Saves a .json file
	"""

	jsonD = {}
	im1 = Image.open(imageList[0])
	width, height = im1.size

	jsonD["dtype"] = "uint8"
	jsonD['ndim'] = 1
	jsonD['tile_ratio'] = 1
	jsonD['tile_st'] = [0, 0]
	jsonD["image"] = imageList
	jsonD["height"] = height
	jsonD["width"] = width
	jsonD["tile_size"] = width
	jsonD["depth"] = len(imageList)

	if not outputJsonPath[-5:] == '.json':
		outputJsonPath = outputJsonPath + '.json'

	prettyJson = json.dumps(jsonD, indent=4)

	with open(outputJsonPath, 'w') as outJson:
		outJson.write(prettyJson)

	print('Done, writing json file output at:', outputJsonPath)

def getWeightsFromLabels(labelStack):
	"""Computes the weights to use in a weighted semantic model

	Parameters
	----------
	labelStack : str
		filepath of the dataset that are labels to be used for training

	Returns
	-------
	list
		a list of normalized weights to use for training
	"""

	listOfImages = []
	countDic = {}

	if labelStack[-3:] == '.h5':
		h5File = h5py.File(labelStack)
		data = np.array(h5File['dataset_1'])
		h5File.close()
		for subImage in data:
			listOfImages.append(subImage)

	elif labelStack[-4:] == '.txt':
		with open(labelStack, 'r') as f:
			labelStack = f.readlines()
			for label in labelStack:
				if len(label.rstrip()) > 3:
					im = Image.open(label.rstrip())
					listOfImages.append(im)

	elif labelStack[-5:] == '.json': #TODO add support for .json
		pass

	elif labelStack[-4:] == '.tif' or labelStack[-5:] == '.tiff':
		labelStackFile = Image.open(labelStack)
		for i, page in enumerate(ImageSequence.Iterator(labelStackFile)):
		    listOfImages.append(page)

	for im in listOfImages:
		data = np.array(im)
		unique, nSamples = np.unique(data, return_counts=True)
		for i in range(len(unique)):
			if not unique[i] in countDic:
				countDic[unique[i]] = 0
			countDic[unique[i]] += nSamples[i]

	sortedKeys = list(sorted(list(countDic.keys())))
	nSamples = []
	for key in sortedKeys:
		nSamples.append(countDic[key])
	m = max(nSamples)
	normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
	# normedWeights = [x / sum(nSamples) for x in nSamples]
	return normedWeights

def instanceArrayToMesh(d, uniqueList=None):
	"""Turns an instance array into a 3D triangle mesh.

	Instance Array - 3D array representing space, 0s encode background and other numbers represent a unique id of an instance
	Transformation of array to mesh is based on the marching cubes algorithm, and then smoothed
	Each instance is painted a random color

	Parameters
	----------
	d : numpy.ndarray
		an instance array to turn into a 3D mesh

	uniqueList : list, optional
		not neccisary, if defined may make this process slightly faster, but is probably negligable. Wouldn't worry about using

	Returns
	-------
	open3d.geometry.TriangleMesh
		Triangle Mesh that represents the passed in instance array
	"""

	if not uniqueList:
		uniqueList, countList = np.unique(d, return_counts=True)

	maxIndex = list(countList).index(max(countList))
	maxValueProb0 = uniqueList[maxIndex]
	fullMesh = o3d.geometry.TriangleMesh()
	deltaTracker = TimeCounter(len(uniqueList), timeUnits='minutes', prefix='Creating Mesh: ')
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
	"""Turns an instance array into a 3D point cloud

	Instance Array - 3D array representing space, 0s encode background and other numbers represent a unique id of an instance
	Each instance is painted a random color

	Parameters
	----------
	d : numpy.ndarray
		an instance array to turn into a 3D mesh

	uniqueList : list, optional
		not neccisary, if defined may make this process slightly faster, but is probably negligable. Wouldn't worry about using

	Returns
	-------
	open3d.geometry.PointCloud
		Point Cloud that represents the passed in instance array
	"""

	if not uniqueList:
		uniqueList, countList = np.unique(d, return_counts=True)

	maxIndex = list(countList).index(max(countList))
	maxValueProb0 = uniqueList[maxIndex]
	fullCloud = o3d.geometry.PointCloud()
	deltaTracker = TimeCounter(len(uniqueList), timeUnits='minutes', prefix='Creating Point Cloud: ')
	for subUnique in uniqueList:
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

def getMultiClassImage(imageFilepath, uniquePixels=[]):
	"""Turns an image into an array where each unique color gets a value

	Parameters
	----------
	imageFilepath : str
		filepath of the image to analyze

	uniquePixels : list, optional
		not neccisary, if defined may make this process slightly faster, but is probably negligable. Wouldn't worry about using. Mostly exists just for getMultiClassImageStack

	Returns
	-------
	numpy.ndarray
		An array that represents the input image in `imageFilepath` where each unique color gets an integer value in this array
	"""

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
	return d, uniquePixels

def getMultiClassImageStack(imageFilepath,uniquePixels=[]):
	"""Turns a paged image (from filename, probably .tif), into a 3D Class Image

	Essentially, this function calls getMultiClassImage on every page in a paged .tif file
	Returns a 3D array representing all pages of a paged image

	Parameters
	----------
	imageFilepath : str
		filepath of the paged .tif image that you want to analyze

	uniquePixels : list, optional
		not neccisary, if defined may make this process slightly faster, but is probably negligable. Wouldn't worry about using

	Returns
	-------
	open3d.geometry.TriangleMesh
		Triangle Mesh that represents the passed in instance array
	"""

	labelStack = []
	unique = []
	im = Image.open(imageFilepath)
	for i, imageSlice in enumerate(ImageSequence.Iterator(im)):
		labels, unique = getMultiClassImage(imageSlice, uniquePixels=unique)
		labelStack.append(labels)
	return np.array(labelStack)

def createH5FromNumpy(npArray, filename):
	"""Saves a numpy array as an H5 File

	Parameters
	----------
	npArray : numpy.ndarray
		the array you want to save

	filename : str
		the filename that you are saving the H5 file at

	Returns
	-------
	None
		Saves an H5 file at `filename`
	"""

	h5f = h5py.File(filename, 'w')
	h5f.create_dataset('dataset_1', data=npArray)
	h5f.close()

def getImagesForLabels(d, index): #TODO check if this method is being called. I'm not sure why this is here.
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

def getPointCloudForIndex(d, index):
	"""Turns a semantic numpy array into a point cloud

	Semantic Array - Numpy array in the shape of (number of planes, z, x, y)
	Essentially, wherever  the plane selected by `index` is greater in value than all other planes, a point is placed in the point cloud.

	Parameters
	----------
	d : numpy.ndarray
		a semantic array to turn into a 3D mesh

	index : int
		which plane to turn into a point cloud

	Returns
	-------
	open3d.geometry.PointCloud
		Point Cloud that represents the passed in semantic array
	"""

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
	"""Turns a semantic array into a 3D triangle mesh.

	Semantic Array - Np array in the shape of (number of planes, z, x, y)
	A 'mask' array is created that is a 3D array of True/False. True where that plane has the greatest value, false otherwise
	Transformation of the mask array to mesh is based on the marching cubes algorithm, and then smoothed

	Parameters
	----------
	d : numpy.ndarray
		a semantic array to turn into a 3D mesh

	index : int
		which plane of the semantic array to create a mesh for

	Returns
	-------
	open3d.geometry.TriangleMesh
		Triangle Mesh that represents the passed in instance array
	"""

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

def subSampled3DH5(dataset, sampleFactor, cubeSize = 1000):
	"""Reads an H5 Dataset, and returns a numpy array of it that is downsampled

	Cube Size should probably be divisible by sampleFactor, or there may be some skipping of points on the boundaries of cubes
	Essentially, a cube of `cubeSize` scans through the entire dataset, and adds a downsampled version to the array to be returned
	

	Parameters
	----------
	dataset : h5py dataset
		dataset to read

	sampleFactor : int
		how much to downsample. Ex, 1 would be no downsampling, 2 would have half the points on each axis (1/8 total size, 2^3=8)

	cubeSize : int
		Size of the cube that scans through the larger dataset. This is adjustable, ex computers with larger amounts of ram can use a larger cube for scanning, and smaller computers can use a smaller size

	Returns
	-------
	numpy.ndarray
		numpy array that is a downsampled version of the H5 dataset, `dataset`
	"""

	newX = float(dataset.shape[0]) / float(sampleFactor)
	newY = float(dataset.shape[1]) / float(sampleFactor)
	newZ = float(dataset.shape[2]) / float(sampleFactor)
	if newX - int(newX) > 0:
		newX = int(newX + 1)
	if newY - int(newY) > 0:
		newY = int(newY + 1)
	if newZ - int(newZ) > 0:
		newZ = int(newZ + 1)

	newShape = (int(newX), int(newY), int(newZ))
	toReturn = np.empty(shape=newShape, dtype=dataset.dtype)

	for xiteration in range(0,dataset.shape[0], int(cubeSize)):
		for yiteration in range(0, dataset.shape[1], int(cubeSize)):
			for ziteration in range(0, dataset.shape[2], int(cubeSize)):
				xmin = xiteration
				xmax = min(xiteration + cubeSize, dataset.shape[0])
				ymin = yiteration
				ymax = min(yiteration + cubeSize, dataset.shape[1])
				zmin = ziteration
				zmax = min(ziteration + cubeSize, dataset.shape[2])

				startSlice = dataset[xmin:xmax:sampleFactor, ymin:ymax:sampleFactor, zmin:zmax:sampleFactor]

				xpad, ypad, zpad = 0, 0, 0

				if xmax == dataset.shape[0]:
					xpad = 1
				if ymax == dataset.shape[1]:
					ypad = 1
				if zmax == dataset.shape[2]:
					zpad = 1

				toReturn[int(xmin/sampleFactor):int(xmax/sampleFactor + xpad), int(ymin/sampleFactor):int(ymax/sampleFactor + ypad), int(zmin/sampleFactor):int(zmax/sampleFactor + zpad)] = startSlice

	return toReturn

# def getMultiClassImage(imageFilepath, uniquePixels=[]): #TODO check why this is duplicate and get rid of one if possible
# 	if type(imageFilepath) == type('Test'):
# 		im = Image.open(imageFilepath).convert('RGB')
# 	else:
# 		im = imageFilepath.convert('RGB')
# 	#print('imtype',type(im))
# 	data = np.array(im)
# 	#print(data)
# 	info = np.iinfo(data.dtype) # Get the information of the incoming image type
# 	data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
# 	data = 255 * data # Now scale by 255
# 	a = data.astype(np.uint8)
# 	b = np.zeros((a.shape[0],a.shape[1]))
# 	c = list(b)
# 	#print('atype',type(a),a.shape)
# 	#print(np.unique(a))
# 	for i in range(a.shape[0]):
# 		for j in range(a.shape[1]):
# 			#print(a[i,j])
# 			value = tuple(a[i,j])
# 			if value in uniquePixels:
# 				c[i][j] = uniquePixels.index(value)
# 			else:
# 				uniquePixels.append(value)
# 				c[i][j] = uniquePixels.index(value)
# 	d = np.array(c)
# 	#toReturn = Image.fromarray(d)
# 	return d, uniquePixels

# def getMultiClassImageStack(imageFilepath,uniquePixels=[]): #TODO check why this is duplicate and get rid of one if possible
# 	labelStack = []
# 	unique = []
# 	im = Image.open(imageFilepath)
# 	for i, imageSlice in enumerate(ImageSequence.Iterator(im)):
# 		labels, unique = getMultiClassImage(imageSlice, uniquePixels=unique)
# 		labelStack.append(labels)
# 	return np.array(labelStack), unique

def makeLabels(filename):
	"""Calls getMultiClassImageStack, and returns 'labels', discarding 'unique' which is also returned
	"""
	labels, unique = getMultiClassImageStack(filename)
	return labels

def getImageForLabelNaming(images, labelArray, index, filename):
	"""Shows an interactive matplotlib window with two images. One is a raw image, and the other is the image with a particular set of labels for a semantic model highlighted in red. This could help identify the different planes of a semantic model

	Parameters
	----------
	images : str or PIL Image
		these are the raw microscope images

	labelArray : str or numpy.ndarray
		this is a numpy array containing labels corresponding to `images`

	index : int
		selects which plane of the semantic labels to highlight

	filename : str
		saves the resulting matplotlib window as a file at this filename

	Returns
	-------
	None
		Opens an interactive matplotlib window and saves it to a file
	"""

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