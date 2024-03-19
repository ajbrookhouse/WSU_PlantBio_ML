# Quickstart Guide
Here, we will give quick directions to get your first machine learning model working with the program. First, install using the installation instructions ![here](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/Instructions/installation.md)

> This tutorial will do an example of instance3D segmentation, if you want to do semantic segmentation, change Image Stack parameter to "plasmSemanticImages.tif", Label Stack parameter to "plasmSemanticImages.tif", and Training Config: to "Semantic3D.yaml".

## Open Program

Open up miniconda by clicking start, typing miniconda, and selecting "Anaconda Prompt (Miniconda3)". Then type the following:
```bash
cd Documents
cd WSU_PlantBio_ML
conda activate plantTorch
python gui.py
```
The main program should now show up on your screen (if "python gui.py" does not work, try "python3 gui.py"). The window should look like this:

![screenshot of first screen that opens when you open program](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/trainTab.png)

## Create Training Images and Labels

The first step, is to create a training set of images, and a training label set. Here is an example of a training image:

![training image example](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/exampleTrain.png)

Here is an example of a training label:

![training label example](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/exampleLabel.png)

When using the neural networks in this program, it is important that image stacks are combined into one paged '.tif' file.

Also, because we are running an instance segmentaion task, for input "Labels" images, make sure the background is 0 and instances start from 1. For example, if you have a input label image that has 10 mitochondrias, the background of the image shall be 0, and each mitochondria should have labels from 1 to 10. No overlapping labels.

If they are not, the program does have a tool to combine them, shown here:

![Image Tools Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/imagetoolTab.png)

To created a single file image stack from seperate images:

- [ ] Open the program and click on the "Image Tools" tab on the upper right side of the program.
- [ ] Click on the button where it says "Choose Folder" to choose your desired folder of images to combine.
- [ ] A folder selection dialog should come up. Navigate to "ExampleData/Seperated_Images", and then click open in the bottom right of the dialog. Make sure this path is now what is shown in the text box labeled "Folder of Images:".
- [ ] Click the button that says "Choose File"
- [ ] A file creation dialog should pop up. Navigate to "ExampleData/Seperated_Images", and give it the filename "ExampleData/combinedTrainingImages" in the input box labelled "File name:" . Click save in the bottom right hand corner of the dialog.
- [ ] Click the button that says "Combine Into TIF", and the program will combine the images in the folder you selected into one stack, and save it under the filename you chose on the output filename step.
- [ ] Repeat the previous steps in this section, but with the "ExampleData/Seperated_Labels" folder as input, and "ExampleData/combinedTrainingLabels" as the output name

## Training

Next step, training. To show the training window, click the "Train" tab on the toolbar on top. This is the first window that pops up when you open the program.

The window should look like this:

![Train Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/trainTab.png)

- [ ] Where it says "Image Stack (.tif or .h5)" near the top, click the triangle button to the right, and select "ExampleData/chloroplastInstanceImages.tif"
- [ ] Where it says "Labels (.tif or .h5)" near the top, click the triangle button to the right and select "ExampleData/chloroplastInstanceLabels.tif"
- [ ] For the boxes lebelled X, Y, and Z nm/pixel, type desired values for each of them.
- [ ] If you want to change these boxes that require a number input, please avoid using comma. For example, use 10000 and do not use 10,000
- [ ] In the text box labeled "Name:", give the model a name, for example: "Tutorial Network"
- [ ] Next click the train button near the bottom. Output should start appearing in the text box on the bottom. This process can take a long time. Using the default settings, it should take near one day to complete (this is very variable depending on the computer being used and teh number of iterations and samples per batch being used). While it is working, it should keep printing out lines starting with "[Iteration number]". You know the training is complete when you get the line "Rank: None. Device: cuda. Process is finished!"

## Automatic Labelling

Now that the model is trained, it is time to use the model to do some automatic labelling. Click the tab "Auto-Label" near the top of the program. The window should now look like this:

![Auto Label Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/labelTab.png)

- [ ] In the "Image Stack (.tif)" selector, click the "Choose File" button, and in the dialog that pops up, select ExampleData/chloroplastInstanceImages.tif.
> Note, it is not good practice to test the accuracy of a model on the same data that you trained it on. However, we will use the same training images here as they are a small dataset that can be processed quickly. To test on the full stack, create a .json dataset with the chloroplastInstance dataset ![here](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/ExampleData/readme.md). This dataset will take a significant amount of time to process and autolabel, as each of the individual images is much larger, and there are also 1500 of them. This page also has links to the full stacks of other datasets provided on this GitHub.
- [ ] Click the "Choose File" button for "Output File:" and save the file as "ExampleData/myFirstPrediction".
- [ ] Click the selection box to the right of "Model to Use:", and select "Tutorial Network".
- [ ] Click label. This can also take a while, but should be shorter than training. The output text box below should print lines that start with "progress: {number}/{number} batches, total time {some number} while it is running. When it is finished, it should print out "Rank: None. Device: cuda. Process is finished!"
- [ ] Click Instance3D post-process. This will post-process the initial output and generate the final data output.  

## Get Sample Stats

Now that the prediction has been done, you can use the program to get statistics for the sample. Click the tab "Output Tools" near the top of the program. The window should now look like this:

![Output Tools Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/outputtoolTab.png)

- [ ] In the file chooser labelled "Model Output (.h5)", click on "Choose File", and select "ExampleData/myFirstPrediction.h5"
- [ ] In the file chooser labelled "CSV Output", type an destination or click on "Choose File" for the CSV file with output statistics.
- [ ] Click on the button that says "Get Model Output Stats". The textbox should start printing out data about the model. The program should print out the min, max, mean, median, standard deviation, sum, and count of auto-labelled organells from the sample. It will also generate a .CSV file in your designated path.

<!-- ## Create Geometries

After the prediction is done, you can also use the program to create 3D geometries for the sample. To do this, click the "Output Tools" tab on the top of the program. The screen should now look like this:

![Output Tools Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/outputToolsScreenshot.png)

- [ ] In the "Model Output (.h5):" selector, click on "Choose File", and select "ExampleData/myFirstPrediction.h5"
- [ ] Make sure that "Meshs" is not selected, and "Point Clouds" is selected
- [ ] Click Make Geometries, this could take some time depending on the computer and the sample.
- [ ] The program will print out "Completely Finished" when it is done. -->

## Visualize

<!-- Once you have created geometries, we can now show them in a 3D visualization window. To do this, click the "Visualize" tab on the top of the program. There should only be one button, click it and a new window should pop up.

- [ ] Click the file tab, in the top left corner of the screen, then click open. Select the file of the geometry that you just made in the previous screen. It may take a while to open, but after waiting the geometry should appear on the screen.

![Visualize Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/visualizeScreenshot.png) -->
- [ ] Click the Neuroglancer tab.
- [ ] Select the Raw Image and Model Output by typing the name or using the interactive button.
- [ ] Enter the 'Z scale', 'X scale', and 'Y scale'. 
- [ ] Enter 255 in 'Segmentation Threshold(1-255)'
- [ ] Click the "Launch Neuroglancer" to launch the visualization work. Once the visualization is ready, a blue link will show up in the software window. We can either click the link, or copy&paste it to our browser to view the result. 

