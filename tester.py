import tkinter as tk
import tkinter.ttk as ttk
from tkinter import StringVar
from tkinter.filedialog import askopenfilename
import plyer
from tkinter import colorchooser

import open3d as o3d
from PIL import Image, ImageSequence
import pickle
import h5py
from os import listdir
from os.path import sep
import numpy as np
from matplotlib import pyplot as plt
from numpy import load, stack
from matplotlib.pyplot import subplots
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
import cc3d
from skimage import measure
import colorsys
from tkinter import messagebox as mb

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
    return np.array(labelStack), unique

def createH5FromNumpy(npArray, filename):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dataset_1', data=npArray)
    h5f.close()

stack, un = getMultiClassImageStack('ExampleData/chloroplastSegmentedLabels.tif')
print(un)
print(np.unique(stack, return_counts=True))

createH5FromNumpy(stack, 'test2.h5')

exit()















class FileChooser(ttk.Frame):
    def __init__(self, master=None, labelText='File: ', changeCallback=False, **kw):

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
        self.button.configure(cursor='arrow', text='Choose File')
        self.button.grid(column='3', row='0')
        self.button.configure(command=self.ChooseFileButtonPress)

        self.filepath = self.entry.get()

    def entryChangeCallback(self, sv, three, four):
        self.filepath = self.getFilepath()
        if self.changeCallback != False:
            self.changeCallback()

    def ChooseFileButtonPress(self):
        self.filepath = plyer.filechooser.open_file()
        self.sv.set(self.filepath)

    def getFilepath(self):
        return self.entry.get()

class LayerNamer(ttk.Frame):
    def __init__(self, master=None, nameText='', **kw):
        ttk.Frame.__init__(self, master, **kw)
        self.label = ttk.Label(self)
        self.label.configure(text='Name of Layer: ')
        self.label.grid(column='0', row='0')

        self.sv = StringVar()
        self.sv.trace_add("write", self.entryChangeCallback)
        self.entry = ttk.Entry(self, textvariable=self.sv)
        self.entry.grid(column='1', row='0')

        self.button = ttk.Button(self)
        self.button.configure(cursor='arrow', text='View')
        self.button.grid(column='2', row='0')
        self.button.configure(command=self.ViewButtonPress)

        self.nameText = nameText
        print(nameText)

    def entryChangeCallback(self, sv, three, four):
        # Don't know what sv, three, and four are for, however they are needed
        self.nameText = self.entry.get()
        self.master.update()

    def ViewButtonPress(self):
        pass

class LayerNamersContainer(ttk.Frame):
    def __init__(self, master=None, **kw):
        ttk.Frame.__init__(self, master, **kw)
        self.LayerNamers = []

        self.frameToExpand = LayerNamer(self)
        self.frameToExpand.configure(height='200', width='200')
        self.frameToExpand.grid(side='top')

        LayerNamer1 = LayerNamer(self)
        self.frameToExpand.configure(height='200', width='200')
        self.frameToExpand.pack(side='top')

        self.label = ttk.Label(self)
        self.label.configure(text='Name of Layer: ')
        self.label.grid(column='0', row='0')

        self.sv = StringVar()
        self.sv.trace_add("write", self.entryChangeCallback)
        self.entry = ttk.Entry(self, textvariable=self.sv)
        self.entry.grid(column='1', row='0')

        self.button = ttk.Button(self)
        self.button.configure(cursor='arrow', text='View')
        self.button.grid(column='2', row='0')
        self.button.configure(command=self.ViewButtonPress)

        self.nameText = nameText

    def entryChangeCallback(self, sv, three, four):
        # Don't know what sv, three, and four are for, however they are needed
        self.nameText = self.entry.get()
        print(self.nameText)

    def ViewButtonPress(self):
        pass

class LayerVisualizerRow(ttk.Frame):
    def __init__(self, master, color, index, changeCallback=False, **kw):
        ttk.Frame.__init__(self, master, **kw)

        self.fileChooser = FileChooser(self, changeCallback = changeCallback)
        self.fileChooser.grid(column='0', row='0')

        self.colorButton = ttk.Button(self)
        self.colorButton.configure(cursor='arrow', text='Choose Color')
        self.colorButton.grid(column='2', row='0')
        self.colorButton.configure(command=self.ChooseColor)

        self.master = master
        self.index = index

    def ChooseColor(self):
        self.color = colorchooser.askcolor(title ="Choose Color For Layer " + str(self.index))
        print(self.color)

    def GetColor(self):
        return self.color

    def GetFile(self):
        return self.fileChooser.getFilepath()

class LayerVisualizerContainer(ttk.Frame):
    def __init__(self, master=None, **kw):
        ttk.Frame.__init__(self, master, **kw)

        self.frameToExpand = tk.Frame(self)
        self.frameToExpand.configure(height='200', width='200')
        self.frameToExpand.pack(side='top')

        self.LayerVisualizerRows = []
        firstVisualizerRow = LayerVisualizerRow(master = self.frameToExpand, color = 5, index=0, changeCallback = self.changeCallback)
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
            nextVisualizerRow = LayerVisualizerRow(master = self.frameToExpand, color = self.getSuggestedColor(), index=newIndex, changeCallback = self.changeCallback)
            nextVisualizerRow.grid(column='0', row=str(newIndex))
            self.LayerVisualizerRows[-1] = nextVisualizerRow

    def getSuggestedColor(self):
        return 5

class NewprojectApp:
    def __init__(self, master=None):
        # build ui
        self.MainFrame = ttk.Frame(master)
        self.label1 = ttk.Label(self.MainFrame)
        self.label1.configure(text='label1')
        self.label1.pack(side='top')
        self.label2 = ttk.Label(self.MainFrame)
        self.label2.configure(text='label2')
        self.label2.pack(side='top')
        self.label3 = ttk.Label(self.MainFrame)
        self.label3.configure(text='label3')
        self.label3.pack(side='top')
        self.frameToExpand = LayerVisualizerContainer(self.MainFrame)
        self.frameToExpand.configure(height='200', width='200')
        self.frameToExpand.pack(side='top')
        self.label8 = ttk.Label(self.MainFrame)
        self.label8.configure(text='label8')
        self.label8.pack(side='top')
        self.label9 = ttk.Label(self.MainFrame)
        self.label9.configure(text='label9')
        self.label9.pack(side='top')
        self.label10 = ttk.Label(self.MainFrame)
        self.label10.configure(text='label10')
        self.label10.pack(side='top')
        self.MainFrame.configure(height='200', width='200')
        self.MainFrame.pack(side='top')

        # Main widget
        self.mainwindow = self.MainFrame


    def run(self):
        self.mainwindow.mainloop()

if __name__ == '__main__':
    import tkinter as tk
    root = tk.Tk()
    app = NewprojectApp(root)
    app.run()