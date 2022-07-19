import ast
import json

from PIL import ImageColor
import h5py
import open3d as o3d
from skimage import measure
from PIL import Image, ImageSequence
import numpy as np

import matplotlib as mpl
defaultMatplotlibBackend = mpl.get_backend()
defaultMatplotlibBackend = 'TkAgg'
from matplotlib import pyplot as plt

from utils import *

def getImageFromDataset(inputDataset, zindex):
	if inputDataset[-4:] == '.tif':
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

def getShapeOfDataset(inputDataset):
	if inputDataset[-4:] == '.tif': # https://stackoverflow.com/questions/46436522/python-count-total-number-of-pages-in-group-of-multi-page-tiff-files
		img = Image.open(inputDataset)
		width, height = img.size
		count = 0

		while True:
			try:   
				img.seek(count)
			except EOFError:
				break       
			count += 1

		return count, height, width

	elif inputDataset[-5:] == '.json':
		
		with open(inputDataset, 'r') as inFile:
			d = json.load(inFile)
		# numFiles = len(d['image'])
		# img = Image.open(d['image'][0])
		# width, height = img.size
		return d['depth'], d['height'], d['width']

	elif inputDataset[-4:] == '.txt':
		filteredImages = []
		with open(inputDataset, 'r') as inFile:
			images = inFile.readlines()
		for image in images:
			if len(image.strip()) > 3:
				filteredImages.append(image)
		img = Image.open(filteredImages[0].strip())
		width, height = img.size
		count = len(filteredImages)
		return count, height, width

	else:
		raise Exception('Unknown Dataset filetype ' + inputDataset[inputDataset.rindex('.'):] + ' Passed to getShapeOfDataset')

def create2DLabelCheckSemanticImage(inputDataset, modelOutputDataset, z, colors = ['#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9', '#ffffff', '#000000']):
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

def create2DLabelCheckSemanticImageForIndex(inputDataset, modelOutputDataset, z, plane, colors = ['#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9', '#ffffff', '#000000']):
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
	mpl.use(defaultMatplotlibBackend)

	print(modelOutput)
	h5File = h5py.File(modelOutput, 'r')
	print(h5File.keys())
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
	f = h5py.File(h5filename, 'r')
	metadata = f['vol0'].attrs['metadata']
	return ast.literal_eval(metadata)

def create3DLabelAnimationSemantic(inputDataset, modelOutput, outputDir, scaleFactor=10, colors = ['#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9', '#ffffff', '#000000']):
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
	with open(outputFile, 'w') as out:
		for image in imageList:
			print('Writing', image)
			out.write(image.strip() + '\n')
	print('Completely Done, output at:', outputFile)

def createTifFromImageList(imageList, outputFile):
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

def getWeightsFromLabels(labelStack): #TODO, make it look at more than just the first image
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

	elif labelStack[-5:] == '.json':
		pass

	elif labelStack[-4:] == '.tif':
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

def subSampled3DH5(dataset, sampleFactor, cubeSize = 1000):
	newX = float(dataset.shape[0]) / float(sampleFactor)
	newY = float(dataset.shape[1]) / float(sampleFactor)
	newZ = float(dataset.shape[2]) / float(sampleFactor)
	if newX - int(newX) > 0:
		newX = int(newX + 1)
	if newY - int(newY) > 0:
		newY = int(newY + 1)
	if newZ - int(newZ) > 0:
		newZ = int(newZ + 1)
	#newShape = (int(dataset.shape[0] / sampleFactor + 1), int(dataset.shape[1] / sampleFactor + 1), int(dataset.shape[2] / sampleFactor + 1))
	newShape = (int(newX), int(newY), int(newZ))
	toReturn = np.empty(shape=newShape, dtype=dataset.dtype)

	for xiteration in range(0,dataset.shape[0], int(cubeSize)):
		for yiteration in range(0, dataset.shape[1], int(cubeSize)):
			for ziteration in range(0, dataset.shape[2], int(cubeSize)):
				xmin = xiteration
				xmax = min(xiteration + cubeSize, dataset.shape[0]) #TODO should the -1 be here?
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