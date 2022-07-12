"""

https://wiki.tcl-lang.org/page/List+of+ttk+Themes
adapta is liked
aqua is ok

"""

#######################################
# Imports                             #
#######################################

from utils import *
from MLThreadworkers import *
from Remote import *
from dataManipulation import *

from ttkthemes import ThemedTk
import tkinter as tk
import tkinter.ttk as ttk
from pygubu.widgets.pathchooserinput import PathChooserInput
from connectomics.config import *
import yaml
import os
from os.path import isdir
from os.path import sep
from os import listdir
from os import mkdir
import h5py
from PIL import Image, ImageSequence
import threading
from connectomics.config import *
import random
import traceback
import json
import numpy as np

import matplotlib as mpl
defaultMatplotlibBackend = mpl.get_backend()
defaultMatplotlibBackend = 'TkAgg'
from matplotlib import pyplot as plt

#######################################
# Main Application Class              #
####################################### 

class TabguiApp():
	def __init__(self, master=None):
		self.root = master
		self.root.title("Anatomics MLT")
		self.RefreshVariables(firstTime=True)

		self.tabHolder = ttk.Notebook(master)

		# self.frameTrain = ttk.Frame(self.tabHolder) # Old way no scrollbar
		self.scrollerTrain = ScrollableFrame(self.tabHolder, -350, 800)
		self.frameTrainMaster = self.scrollerTrain.container
		self.frameTrain = self.scrollerTrain.scrollable_frame

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
		# self.frameTrain.configure(height='200', width='200')
		# self.frameTrain.pack(side='top')
		self.tabHolder.add(self.frameTrainMaster, text='Train')


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
		self.labelEvaluateImages = ttk.Label(self.frameEvaluate)
		self.labelEvaluateImages.configure(text='Raw Images (.tif): ')
		self.labelEvaluateImages.grid(column='0', row='2')

		self.pathchooserinputEvaluateLabel = PathChooserInput(self.frameEvaluate)
		self.pathchooserinputEvaluateLabel.configure(type='file')
		self.pathchooserinputEvaluateLabel.grid(column='1', row='1')

		self.pathchooserinputEvaluateModelOutput = PathChooserInput(self.frameEvaluate)
		self.pathchooserinputEvaluateModelOutput.configure(type='file')
		self.pathchooserinputEvaluateModelOutput.grid(column='1', row='0')

		self.pathchooserinputEvaluateImages = PathChooserInput(self.frameEvaluate)
		self.pathchooserinputEvaluateImages.configure(type='file')
		self.pathchooserinputEvaluateImages.grid(column='1', row='2')

		self.buttonEvaluateEvaluate = ttk.Button(self.frameEvaluate)
		self.buttonEvaluateEvaluate.configure(text='Evaluate')
		self.buttonEvaluateEvaluate.grid(column='0', columnspan='2', row='3')
		self.buttonEvaluateEvaluate.configure(command=self.EvaluateModelEvaluateButtonPress)

		self.buttonEvaluateCompareImages = ttk.Button(self.frameEvaluate)
		self.buttonEvaluateCompareImages.configure(text='Visually Compare')
		self.buttonEvaluateCompareImages.grid(column='0', columnspan='2', row='4')
		self.buttonEvaluateCompareImages.configure(command=self.EvaluateModelCompareImagesButtonPress)

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

		self.buttonImageCombineTif = ttk.Button(self.frameImage)
		self.buttonImageCombineTif.configure(text='Combine Into TIF')
		self.buttonImageCombineTif.grid(column='0', row='2')
		self.buttonImageCombineTif.configure(command=self.ImageToolsCombineImageButtonPressTif)

		self.buttonImageCombineTxt = ttk.Button(self.frameImage)
		self.buttonImageCombineTxt.configure(text='Combine Into TXT')
		self.buttonImageCombineTxt.grid(column='0', row='3')
		self.buttonImageCombineTxt.configure(command=self.ImageToolsCombineImageButtonPressTxt)

		self.buttonImageCombineJson = ttk.Button(self.frameImage)
		self.buttonImageCombineJson.configure(text='Combine Into JSON')
		self.buttonImageCombineJson.grid(column='0', row='4')
		self.buttonImageCombineJson.configure(command=self.ImageToolsCombineImageButtonPressJson)

		self.textImageTools = tk.Text(self.frameImage)
		self.textImageTools.configure(height='10', width='75')
		_text_ = '''Image Tools Output Will Be Here'''
		self.textImageTools.insert('0.0', _text_)
		self.textImageTools.grid(column='0', columnspan='2', row='5')

		self.frameImage.configure(height='200', width='200')
		self.frameImage.pack(side='top')
		self.tabHolder.add(self.frameImage, text='Image Tools')

		##################################################################################################################
		# 
		##################################################################################################################

		self.frameOutputTools = ttk.Frame(self.tabHolder)

		self.fileChooserOutputStats = FileChooser(master=self.frameOutputTools, labelText='Model Output (.h5): ', changeCallback=False, mode='open', title='Choose File', buttonText='Choose File')
		self.fileChooserOutputStats.grid(column='1', row='0')

		self.fileChooserOutputToolsOutCSV = FileChooser(master=self.frameOutputTools, labelText='CSV Output:', changeCallback=False, mode='create', title='Create CSV', buttonText='Create CSV')
		self.fileChooserOutputToolsOutCSV.grid(column='1', row='1')

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

		self.buttonOutputGetStats = ttk.Button(self.frameOutputTools)
		self.buttonOutputGetStats.configure(text='Get Model Output Stats')
		self.buttonOutputGetStats.grid(column='0', columnspan='2', row='4')
		self.buttonOutputGetStats.configure(command=self.OutputToolsModelOutputStatsButtonPress)

		self.labelDownscaleGeometry = ttk.Label(self.frameOutputTools)
		self.labelDownscaleGeometry.configure(text='Downscaling Factor: \n1 is no downscaling')
		self.labelDownscaleGeometry.grid(column='0', row='5')

		self.entryDownscaleGeometry = ttk.Entry(self.frameOutputTools)
		self.entryDownscaleGeometry.configure(text='1')
		self.entryDownscaleGeometry.grid(column='1', row='5')

		self.cropToFrame = ttk.Frame(self.frameOutputTools)
		self.cropToFrameTitle = ttk.Label(self.cropToFrame)
		self.cropToFrameTitle.configure(text='Restrict Analysis Area (Crop)')
		self.cropToFrameTitle.grid(column='0', columnspan='6', row='0')

		self.cropToFrameXLabel = ttk.Label(self.cropToFrame)
		self.cropToFrameXLabel.configure(text='X Min: ')
		self.cropToFrameXLabel.grid(column='0', row='1')
		self.cropToFrameXEntry = ttk.Entry(self.cropToFrame)
		self.cropToFrameXEntry.grid(column='1', row='1')

		self.cropToFrameYLabel = ttk.Label(self.cropToFrame)
		self.cropToFrameYLabel.configure(text='Y Min:')
		self.cropToFrameYLabel.grid(column='2', row='1')
		self.cropToFrameYEntry = ttk.Entry(self.cropToFrame)
		self.cropToFrameYEntry.grid(column='3', row='1')

		self.cropToFrameZLabel = ttk.Label(self.cropToFrame)
		self.cropToFrameZLabel.configure(text='Z Min:')
		self.cropToFrameZLabel.grid(column='4', row='1')
		self.cropToFrameZEntry = ttk.Entry(self.cropToFrame)
		self.cropToFrameZEntry.grid(column='5', row='1')


		self.cropToFrameX2Label = ttk.Label(self.cropToFrame)
		self.cropToFrameX2Label.configure(text='X Max: ')
		self.cropToFrameX2Label.grid(column='0', row='2')
		self.cropToFrameX2Entry = ttk.Entry(self.cropToFrame)
		self.cropToFrameX2Entry.grid(column='1', row='2')

		self.cropToFrameY2Label = ttk.Label(self.cropToFrame)
		self.cropToFrameY2Label.configure(text='Y Max:')
		self.cropToFrameY2Label.grid(column='2', row='2')
		self.cropToFrameY2Entry = ttk.Entry(self.cropToFrame)
		self.cropToFrameY2Entry.grid(column='3', row='2')

		self.cropToFrameZ2Label = ttk.Label(self.cropToFrame)
		self.cropToFrameZ2Label.configure(text='Z Max:')
		self.cropToFrameZ2Label.grid(column='4', row='2')
		self.cropToFrameZ2Entry = ttk.Entry(self.cropToFrame)
		self.cropToFrameZ2Entry.grid(column='5', row='2')


		self.cropToFrame.grid(column='0', row='6', columnspan='2')

		self.textOutputOutput = tk.Text(self.frameOutputTools)
		self.textOutputOutput.configure(height='25', width='75')
		_text_ = '''Output Goes Here'''
		self.textOutputOutput.insert('0.0', _text_)
		self.textOutputOutput.grid(column='0', columnspan='2', row='7')

		self.frameOutputTools.configure(height='200', width='200')
		self.frameOutputTools.pack(side='top')
		self.tabHolder.add(self.frameOutputTools, text='Output Tools')

		############################################################################################

		self.frameVisualize = ttk.Frame(self.tabHolder)
		self.frameVisualize.configure(height='200', width='200')
		self.frameVisualize.pack(side='top')

		self.visualizeOpenButton = ttk.Button(self.frameVisualize)
		self.visualizeOpenButton.configure(text="Open Visuzliation\nWindow", command=self.visualizeOpenButtonPress)
		self.visualizeOpenButton.pack(side='top')

		# self.visualizeRowsHolder = LayerVisualizerContainer(self.frameVisualize)
		# self.visualizeRowsHolder.grid(column='0', row='1')

		# self.buttonVisualize = ttk.Button(self.frameVisualize)
		# self.buttonVisualize.configure(text='Visualize')
		# self.buttonVisualize.grid(column='0', row='2')
		# self.buttonVisualize.configure(command=self.VisualizeButtonPress)

		# self.textVisualizeOutput = tk.Text(self.frameVisualize)
		# self.textVisualizeOutput.configure(height='10', width='50')
		# _text_ = '''Output Goes Here'''
		# self.textVisualizeOutput.insert('0.0', _text_)
		# self.textVisualizeOutput.grid(column='0', row='3')

		self.tabHolder.add(self.frameVisualize, text='Visualize')

		############################################################################################

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

		self.root.update()
		print("Width x Height", self.root.winfo_width(), 'x', self.root.winfo_height())

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

	def visualizeOpenButtonPress(self):
		from visualizationGUI import runVisualizationWindow
		runVisualizationWindow()

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

			if image[-5:] == '.json':
				chunkSize = 1000
				with open(image, 'r') as fp:
					jsonData = json.load(fp)

				config['DATASET']['DO_CHUNK_TITLE'] = 1
				config['DATASET']['DATA_CHUNK_NUM'] = [max(1,int(jsonData['depth']/chunkSize + .5)), max(1,int(jsonData['height']/chunkSize + .5)), max(1,int(jsonData['width']/chunkSize + .5))] # split the large volume into chunks [z,y,x order]
				config['DATASET']['DATA_CHUNK_ITER'] = int( itTotal / np.sum(config['DATASET']['DATA_CHUNK_NUM']) ) # (training) number of iterations for a chunk which is iterations / number of chunks

			if not isdir('Data' + sep + 'models' + sep + name):
				mkdir('Data' + sep + 'models' + sep + name)

			with open("Data" + sep + "models" + sep + name + sep + "config.yaml", 'w') as file:
				yaml.dump(config, file)

			metaDictionary = {}
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
		recombine = False
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

			outputPath, outputName = os.path.split(outputFile)

			if not outputName.split('.')[-1] == 'h5':
				outputName += '.h5'

			if image[-5:] == '.json': #Means the dataset is chunked and needs to be recombined at the end
				recombine = True
				chunkSize = 1000
				with open(image, 'r') as fp:
					jsonData = json.load(fp)

				config['DATASET']['DO_CHUNK_TITLE'] = 1
				config['DATASET']['DATA_CHUNK_NUM'] = [max(1,int(jsonData['depth']/chunkSize + .5)), max(1,int(jsonData['height']/chunkSize + .5)), max(1,int(jsonData['width']/chunkSize + .5))] # split the large volume into chunks [z,y,x order]

			if image[-4:] == '.txt': #Means the dataset is 2D and needs to be recombined at the end
				recombine = True
			
			if recombine:
				outputPath = outputPath + sep + model + '_tempOutputChunks_' + str(random.randint(1, 9999))

			print(outputPath)

			config['INFERENCE']['OUTPUT_PATH'] = outputPath
			config['INFERENCE']['OUTPUT_NAME'] = outputName
			config['INFERENCE']['IMAGE_NAME'] = image
			config['INFERENCE']['SAMPLES_PER_BATCH'] = samples

			if not outputPath[-1] == sep:
				outputPath += sep

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
				print('Starting Non Cluster')				
				biggestCheckpoint = self.getLastCheckpointForModel(model)
				metaData = self.getMetadataForModel(model)
				memStream = MemoryStream()
				t = threading.Thread(target=useThreadWorker, args=('temp.yaml', memStream, biggestCheckpoint, metaData, recombine))
				t.setDaemon(True)
				t.start()
				self.longButtonPressHandler(t, memStream, self.textUseOutput, [self.buttonUseLabel])
		except:
			traceback.print_exc()
			self.buttonUseLabel['state'] = 'normal'

	def EvaluateModelCompareImagesButtonPress(self):
		imageStack = self.pathchooserinputEvaluateImages.entry.get()
		predStack = self.pathchooserinputEvaluateModelOutput.entry.get()
		metadata = getMetadataForH5(predStack)
		if 'semantic' in metadata['configType'].lower():
			create2DLabelCheckSemantic(imageStack, predStack, 5)
		elif 'instance' in metadata['configType'].lower():
			create2DLabelCheckInstance(imageStack, predStack, 5)

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

	def ImageToolsCombineImageButtonPressTif(self):
		try:
			memStream = MemoryStream()
			self.buttonImageCombineTif['state'] = 'disabled'
			filesToCombine = self.fileChooserImageToolsInput.getFilepath()
			outputFile = self.fileChooserImageToolsOutput.getFilepath()
			if not outputFile[-4:] == '.tif':
				outputFile = outputFile + '.tif'

			t = threading.Thread(target=ImageToolsCombineImageThreadWorker, args=(filesToCombine, outputFile, memStream))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textImageTools, [self.buttonImageCombineTif])
		except:
			traceback.print_exc()
			self.buttonImageCombineTif['state'] = 'normal'

	def ImageToolsCombineImageButtonPressTxt(self):
		try:
			memStream = MemoryStream()
			self.buttonImageCombineTxt['state'] = 'disabled'
			filesToCombine = self.fileChooserImageToolsInput.getFilepath()
			outputFile = self.fileChooserImageToolsOutput.getFilepath()
			if not outputFile[-4:] == '.txt':
				outputFile = outputFile + '.txt'
				
			t = threading.Thread(target=ImageToolsCombineImageThreadWorker, args=(filesToCombine, outputFile, memStream))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textImageTools, [self.buttonImageCombineTxt])
		except:
			traceback.print_exc()
			self.buttonImageCombineTxt['state'] = 'normal'

	def ImageToolsCombineImageButtonPressJson(self):
		try:
			memStream = MemoryStream()
			self.buttonImageCombineJson['state'] = 'disabled'
			filesToCombine = self.fileChooserImageToolsInput.getFilepath()
			outputFile = self.fileChooserImageToolsOutput.getFilepath()
			if not outputFile[-5:] == '.json':
				outputFile = outputFile + '.json'
				
			t = threading.Thread(target=ImageToolsCombineImageThreadWorker, args=(filesToCombine, outputFile, memStream))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textImageTools, [self.buttonImageCombineJson])
		except:
			traceback.print_exc()
			self.buttonImageCombineJson['state'] = 'normal'

	def OutputToolsModelOutputStatsButtonPress(self):
		try:
			xmin, xmax = self.cropToFrameXEntry.get(), self.cropToFrameX2Entry.get()
			ymin, ymax = self.cropToFrameYEntry.get(), self.cropToFrameY2Entry.get()
			zmin, zmax = self.cropToFrameZEntry.get(), self.cropToFrameZ2Entry.get()

			memStream = MemoryStream()
			self.buttonOutputGetStats['state'] = 'disabled'
			filename = self.fileChooserOutputStats.getFilepath() #TODO get file name #Dont know what this means, pretty sure this todo can be deleted
			csvfilename = self.fileChooserOutputToolsOutCSV.getFilepath()
			if not csvfilename[-4:] == '.csv' and len(csvfilename) > 0:
				csvfilename += '.csv'
			t = threading.Thread(target=OutputToolsGetStatsThreadWorker, args=(filename, memStream, csvfilename))
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
			downScaleFactor = self.entryDownscaleGeometry.get()
			if len(downScaleFactor.strip()) ==len(''):
				downScaleFactor = 1
			else:
				downScaleFactor = int(downScaleFactor)
			t = threading.Thread(target=OutputToolsMakeGeometriesThreadWorker, args=(h5Path, makeMeshs, makePoints, memStream, downScaleFactor))
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

	def run(self):
		self.mainwindow.mainloop()


#######################################
# Boilerplate TK Create & Run Window  #
####################################### 

from tkinter import font

if __name__ == '__main__':
	mpl.use('Agg')
	# root = tk.Tk()
	root = ThemedTk(theme='adapta')

	root.option_add( "*font", "sans_serif 12" )

	print(font.families())
	root.minsize(750, 400)

	# Gets Physical Monitor Dimensions
	# print(root.winfo_screenwidth())
	# print(root.winfo_screenheight())

	sp = os.getcwd()
	imgicon = tk.PhotoImage(file=os.path.join(sp,'icon.png'))
	root.tk.call('wm', 'iconphoto', root._w, imgicon)
	root.geometry('750x750')

	app = TabguiApp(root)
	app.run()