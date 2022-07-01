import time
from io import StringIO

import numpy as np
from PIL import Image, ImageColor
import open3d as o3d

from tkinter import colorchooser
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import StringVar
from tkinter import filedialog as fd

# from connectomics.config import *



# Methods

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

def getNumpyFromDataset(inputDataset):
	shape = getShapeOfDataset(inputDataset)
	imList = []
	if inputDataset[-4:] == '.tif' or inputDataset[-5:] == '.tiff':
		img = Image.open(inputDataset)
		for i in range(shape[0]):
			img.seek(i)
			imList.append(np.asarray(img))
	elif inputDataset[-5:] == '.json':
		with open(inputDataset, 'r') as inFile:
			d = json.load(inFile)
		for file in d['image']:
			img = Image.open(file)
			imList.append(np.asarray(img))
	elif inputDataset[-4:] == '.txt':
		filteredImages = []
		with open(inputDataset, 'r') as inFile:
			images = inFile.readlines()
		for image in images:
			if len(image.strip()) > 3:
				img = Image.open(image.strip())
				imList.append(np.asarray(img))
	imList = np.array(imList)
	return imList

def getPointCloudImageSliceFromDataset(dataset, axis, index, sampleFactor=1):
	print('Getting Cloud')
	dataset = getNumpyFromDataset(dataset)

	if axis == 'x':
		slice_ = dataset[index,:,:]
	elif axis == 'y':
		slice_ = dataset[:,index,:]
	elif axis == 'z':
		slice_ = dataset[:,:,index]
	else:
		raise 'Unknown axis passed to getPointCloudImageSliceFromDataset. Options are "x", "y", or "z"'

	im = Image.fromarray(slice_).convert('RGB')
	im = np.asarray(im)
	im = im[::1,::1]

	points = []
	colors = []
	pcd = o3d.geometry.PointCloud()
	for i in range(im.shape[0] ):
		for j in range(im.shape[1]):
			color = np.array(im[i,j]) / 255
			point = [0, i, j]
			points.append(point)
			colors.append(color)
	points = o3d.utility.Vector3dVector(points)
	colors = o3d.utility.Vector3dVector(colors)
	pcd.points = points
	pcd.colors = colors
	if axis == 'x':
		pcd.translate((index, 0, 0))
	elif axis == 'y':
		s1 = dataset.shape[0]
		R = pcd.get_rotation_matrix_from_xyz((0, 0, -np.pi/2))
		pcd.rotate(R)
		pcd.translate((s1/2, -s1/2 + index, 0))
	elif axis == 'z':
		s0 = dataset.shape[0]
		s1 = dataset.shape[1]
		s2 = dataset.shape[2]
		print(dataset.shape)
		R = pcd.get_rotation_matrix_from_xyz((-np.pi/2, 0, -np.pi/2))
		pcd.rotate(R)
		pcd.translate((s0/2, s1/2 - s0/2, index - s1/2)) # First and Third Index are correct
	return pcd

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

def rgb2hex(colorTuple):
	r, g, b = colorTuple
	return "#{:02x}{:02x}{:02x}".format(r,g,b)

def MessageBox(message, title=None):
	print(message)
	tk.messagebox.showinfo(title=title, message=message)

# Classes

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
		self.remainingTime = None

	def tick(self):
		self.index += 1
		currentTime = time.time()

		fractionComplete = self.index / self.numSteps
		timeSofar = currentTime - self.startTime
		totalTime = 1/fractionComplete * timeSofar
		self.remainingTime = totalTime - timeSofar

	def print(self):
		if self.remainingTime:
			print(self.prefix, self.remainingTime / self.scaleFactor, self.timeUnits, 'left')
		else:
			print(self.prefix, "Cannot calculate time left, either just started or close to end")

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

class MemoryStream(StringIO):
	def __init__(self):
		super().__init__()
		self.text = ''

	def write(self, string):
		self.text = self.text + string

class TextboxStream(StringIO): # Replaced by MemoryStream, works a lot nicer with the threads. This was causing issues.
	def __init__(self, widget, maxLen = None):
		super().__init__()
		self.widget = widget

	def write(self, string):
		self.widget.insert("end", string)
		self.widget.see('end')

class ScrollableFrame(ttk.Frame):
	# https://blog.teclado.com/tkinter-scrollable-frames/
	# Modified a bit, but this is where the inspiration came from

	def __init__(self, container, scrollMin = -250, scrollMax = 750, *args, **kwargs):
		super().__init__(container, *args, **kwargs)
		ContainerOne = ttk.Frame(container)
		ContainerOne.pack(fill=tk.BOTH, expand=True)
		canvas1 = tk.Canvas(ContainerOne, width=500, height=1000, bg='white')
		scroll = ttk.Scrollbar(ContainerOne, command=canvas1.yview)
		canvas1.config(yscrollcommand=scroll.set, scrollregion=(0,scrollMin,100,scrollMax))
		canvas1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		scroll.pack(side=tk.RIGHT, fill=tk.Y)
		frameOne = ttk.Frame(canvas1, width=800, height=450)
		canvas1.create_window(250, 125, window=frameOne)
		self.scrollable_frame = frameOne
		self.container = ContainerOne
		canvas1.yview_moveto(0)