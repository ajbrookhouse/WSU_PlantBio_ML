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

def getShapeOfDataset(inputDataset):
	"""When given a dataset filename of either a .tif, .json, or .txt dataset, returns the shape as a tuple

	Parameters
	----------
	inputDataset : str
		The filepath of the dataset you want to get the shape of

	Returns
	-------
	tuple
		A tuple of ints representing the shape of the dataset

	Raises
	------
	Exception
		If the filename passed does not end in '.tif', '.tiff', '.json', or '.txt'
	"""

	if inputDataset[-4:] == '.tif' or inputDataset[-5:] == '.tiff': # https://stackoverflow.com/questions/46436522/python-count-total-number-of-pages-in-group-of-multi-page-tiff-files
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
	"""When given a dataset filename of either a .tif, .json, or .txt dataset, returns the dataset as a numpy array

	Parameters
	----------
	inputDataset : str
		The filepath of the dataset you want to get as a numpy array

	Returns
	-------
	numpy.ndarray
		A numpy array representation of the dataset

	Raises
	------
	Exception
		If the filename passed does not end in '.tif', '.tiff', '.json', or '.txt'
	"""

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
	else:
		raise Exception('Unknown Dataset filetype ' + inputDataset[inputDataset.rindex('.'):] + ' Passed to getNumpyFromDataset')

	imList = np.array(imList)
	return imList

def getPointCloudImageSliceFromDataset(dataset, axis, index):
	"""Returns a two dimensional slice of a dataset, represented as a point cloud

	Parameters
	----------
	dataset : str
		The filepath of the dataset you want to get a slice of

	axis : {'x', 'y', 'z'}
		The axis along with you want to get a slice

	index : int
		How far along the axis to get the slice from. Can range from 0 to the end of the axis

	Returns
	-------
	o3d.geometry.PointCloud
		A point cloud that when rendered looks like a 2D image from the dataset. Can be rendered in 3D space

	Raises
	------
	Exception
		If `axis` is not equal to 'x', 'y', or 'z'
	"""

	#TODO add option to scale down the image (use half the total number of points, 1/4, etc.) Could load faster this way

	#TODO make it so you don't have to load the whole array as a dataset to get the slice

	dataset = getNumpyFromDataset(dataset)

	if axis == 'x':
		slice_ = dataset[index,:,:]
	elif axis == 'y':
		slice_ = dataset[:,index,:]
	elif axis == 'z':
		slice_ = dataset[:,:,index]
	else:
		raise Exception('Unknown axis passed to getPointCloudImageSliceFromDataset. Options are "x", "y", or "z"')

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
		pcd.translate((s0/2, s1/2 - s0/2, index - s1/2))

	return pcd

def complimentColor(hexValue=None, rgbTuple=None): #adopted from stackoverflow, https://stackoverflow.com/a/3943023/112731
	"""When given a color, returns either the white or black hex code, whichever would show up better as text over the passed color

	This function is supposed to be used to pick which text color to use over a given color.
	You pass in the background color as either a hex string or rgbTuple.
	The function returns the hex color for either white or black
	This return value is the text color you should use over the background color to have best visibility.

	Parameters
	----------
	hexValue : str
		Hex representation of a color, ex white is #FFFFFF
	rgbTuple : tuple of ints
		Red, green, blue representation of a color, ex white is (255, 255, 255)

	Returns
	-------
	str
		A hex representation of the text color you should use
	"""

	if hexValue and rgbTuple:
		raise Exception("Must provide either hexValue or rgbTuple, not both")

	if not hexValue and not rgbTuple:
		raise Exception("Must provide either hexValue or rgbTuple")

	if rgbTuple:
		r, g, b = rgbTuple
	if hexValue:
		r, g, b = ImageColor.getcolor(hexValue, "RGB")

	if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
		return "#000000"
	else:
		return "#FFFFFF"

def whereToArray(where):
	"""Turns the results of np.where(), back into an array

	np.where(array=value) will return a tuple of lists representing coordinates where the array = that value
	This function takes in this result `where`, and turns it back into a binary array
	This returned array is 1 for the points listed in `where`, and 0 everywhere else

	Parameters
	----------
	where : tuple of lists
		A tuple containing lists. For example, a 3D array would return 3 lists when numpy.where() is used
		The first list would be x coordinates, the second y, and the third z

	Returns
	-------
	numpy.ndarray
		A numpy array that is equal to 1 at the points from the np.where list, and 0 elsewhere
	"""

	maxs = []
	numDimensions = len(where)
	print('numDimenstions',numDimensions)
	for i in range(numDimensions):
		maxs.append(np.max(where[i]) + 1)
		print(len(where[i]))
	toReturn = np.zeros(maxs, dtype=int)
	toReturn[where] = 1
	return toReturn

def cloudToSemanticArray(where):
	"""Very similar to whereToArray.
	The only difference is this is recieving a open3d.geometry.PointCloud.points as input, which is different slightly from the output of np.where
	it retuns

	Parameters
	----------
	where : open3d.geometry.PointCloud.points
		A list of points from an open3d PointCloud object

	Returns
	-------
	numpy.ndarray
		A numpy array that is equal to 1 where there are points in the PointCloud, and 0 elsewhere
	"""

	where = np.array(where, dtype=int)
	where = np.transpose(where)
	whereTup = []
	maxs = []
	numDimensions = where.shape[0]
	for i in range(numDimensions):
		maxs.append(np.max(where[i,:]) + 1)
		whereTup.append(where[i])
	whereTup = tuple(whereTup)
	toReturn = np.zeros(maxs, dtype=int)
	toReturn[whereTup] = 1
	return toReturn

def rgb2hex(colorTuple):
	"""Converts an RGB tuple to a hex string

	Parameters
	----------
	colorTuple : tuple of ints
		Red, green, blue representation of a color, ex white is (255, 255, 255)

	Returns
	-------
	str
		A hex representation of `colorTuple`
	"""

	r, g, b = colorTuple
	return "#{:02x}{:02x}{:02x}".format(r,g,b)

def MessageBox(message, title=None):
	"""Makes a message box pop up in tkinter

	Parameters
	----------
	message : str
		The message to be displayed in the message box
	title : str, optional
		The title of the message box. Can be ommited
	"""
	print(message)
	tk.messagebox.showinfo(title=title, message=message)

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

class LayerVisualizerRow(ttk.Frame): #TODO add doc string
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

class LayerVisualizerContainer(ttk.Frame): #TODO add doc string
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
	"""My Implementation of a file chooser in tkinter
	Since it inherits from ttk.Frame, it is essentially a standalone tkinter widget you can place anywhere you can place a frame
	If you want to get the selected filepath, use the method getFilepath
	If there are multiple, use getMultiFilepahts (I'm aware of the mispelling, but may be used in the code so I don't want to change it until I can do a complete refactor somehow)
	
	Methods
	-------
	getFilepath()
		Returns the filename that has been selected by the FileChooser

	getMultiFilepahts()
		Returns the multiple filepaths as a list if the mode is openMultiple
	"""

	def __init__(self, master=None, labelText='File: ', changeCallback=False, mode='open', title='', buttonText='Choose File', **kw):
		"""Initiallizer of the FileChooser Class

		Parameters
		----------
		master : tk or ttk widget
			The master of this widget in tkinter. What this widget will be inside of

		labelText : str, optional
			Sets the text of the label in the FileChooser. Default is "File: "

		changeCallback : function, optional
			If set, everytime that the entry has its value changed, this function is called. It should not take any parameters

		mode : {'open', 'create', 'openMultiple', 'folder'}
			open - Opens one file
			create - Creates a new file
			openMultiple - can open multiple files
			folder - opens a folder

		title : str, optional
			Sets the title for the popup of the file chooser

		buttonText : str, optional
			Sets the text of the button, default is 'Choose File'
		"""

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
			self.filepaths = filename #TODO check to make sure if you need to ast.literal_eval this or if it is already a list
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
	"""For now, look at examples of where MemoryStream is used in gui.py
	#TODO add better documentation
	"""

	def __init__(self):
		super().__init__()
		self.text = ''

	def write(self, string):
		self.text = self.text + string

class TextboxStream(StringIO):
	"""Replaced by MemoryStream, works a lot nicer with the threads. This was causing issues.
	I did not delete it from the code, however I suggest never using this class
	"""

	def __init__(self, widget, maxLen = None):
		super().__init__()
		self.widget = widget

	def write(self, string):
		self.widget.insert("end", string)
		self.widget.see('end')

class ScrollableFrame(ttk.Frame):
	"""Can be used like a regular tkinter frame, but has a scrollbar on the side

	Origional inspiration came from https://blog.teclado.com/tkinter-scrollable-frames/
	I have modified it a bit, but started using the code from above
	This should be able to be used interchangably with regular ttk or tk frames
	Just be sure that you set scrollMin and scrollMax to values that work well
	"""

	def __init__(self, container, scrollMin = -250, scrollMax = 750, *args, **kwargs):
		"""
		Parameters
		----------
		container : tk widget
			The master of this scroll frame in tkinter

		scrollMin : int
			The minimum value that can be scrolled in this scroll bar. If set too low, the frame will start empty, and you will scroll down to be able to see anything.

		scrollMax : int
			The maximum value that can be scrolled in this scroll bar. If set to high, you can scroll down way past where there is any content in the frame, leaving it empty.
		"""

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