# Quickstart Guide
Here, we will give quick directions to get your first machine learning model working with the program. First, install using the installation instructions ![here](https://github.com/ajbrookhouse/WSU_PlantBio_ML#installation)

## Open Program

Open up miniconda by clicking start, typing miniconda, and selecting "Anaconda Prompt (Miniconda3)". Then type the following:
```bash
cd <path to the WSU_PlantBio_ML folder, example: C:\\Users\\YourUsername\\Documents\\WSU_PlantBio_ML>
conda activate plantTorch
python gui.py
```
The main program should now show up on your screen (if "python gui.py" does not work, try "python3 gui.py"). The window should look like this:

![screenshot of first screen that opens when you open program](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/trainScreenshot.png)



## Create Training Images and Labels

The first step, is to create a training set of images, and a training label set. Here is an example of a training image:

![training image example](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/exampleTrain.png)

Here is an example of a training label:

![training label example](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/exampleLabel.png)

When using the neural networks in this program, it is important that image stacks are combined into one paged '.tif' file. If they are not, the program does have a tool to combine them, shown here:

![Image Tools Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/imageToolsScreenshot.png)

To created a single file image stack from seperate images:

- Open the program and click on the "Image Tools" tab on the upper right side of the program.
- Click on the triangle button where it says "Folder Of Images:".
- A folder selection dialog should come up. Navigate to ExampleData/Seperated_Images, and then click open in the bottom right of the dialog. Make sure this path is now what is shown in the text box labeled "Folder of Images:".
- Click the button that says "Combine Images Into Stack", and the program will combine the images in the folder you selected into one stack, and save it in the same folder as "_combined.tif". Feel free to move and/or rename "_combined.tif" to wherever you want on your computer, as you see later, this tool just combines images and places them here as "_combined.tif". When it comes time to do training or labelling, you will select this file, so name it something meaningful and move it when doing your own images. The combined .tif stacks are already in ExampleData, so no need to rename for the tutorial.

## Training

Next step, training. To show the training window, click the "Train" tab on the toolbar on top. The window should look like this:

![Train Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/trainScreenshot.png)

- Where it says "Image Stack (.tif)" on the top, click the triangle button and select "ExampleData/train.tif"
- Where it says "Labels" on the top, click the triangle button and select "ExampleData/label.tif"
- Next, click the box below labels where it says something.yaml (probably "default.yaml"). Select "MitoEM-R-BC.yaml".
- In the text box labeled "Name:", type in "Tutorial Network"
- Next click the train button near the bottom. Output should start appearing in the text box on the bottom. This process can take a long time. Using the default settings, it should take near one day to complete. While it is working, it should keep printing out lines starting with "[Iteration number]". You know the training is complete when you get the line "Rank: None. Device: cuda. Process is finished!"

## Automatic Labelling

Now that the model is trained, it is time to use the model to do some automatic labelling. Click the tab "Auto-Label" near the top of the program. The window should now look like this:

![Auto Label Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/predictScreenshot.png)

- In the "Image Stack (.tif)" selector, click the triangle button, and in the dialog that pops up, select ExampleData/test.tif.
- Click the triangle button for "Output File:" and save the file as "ExampleData/myFirstPrediction".
- Click the selection box below, and select "Tutorial Network".
- Click label. This can also take a while, but should be shorter than training. The output text box below should print lines that start with "progress: {number}/{number} batches, total time {some number} while it is running. When it is finished, it should print out "Rank: None. Device: cuda. Process is finished!"

## Get Sample Stats

TODO fill out

## Create Geometries

Once you have created geometries, we can now show them in a 3D visualization window. To do this, click the "Output Tools" tab on the top of the program. The screen should now look like this:

![Output Tools Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/outputToolsScreenshot.png)

- In the "Model Output (.h5):" selector, click the triangle button, and select "ExampleData/myFirstPrediction.h5"
- Click the checkboxes for Meshs, and Point Clouds
- Click Make Geometries, this could take some time.

## Visualize

Once you have created geometries, we can now show them in a 3D visualization window. To do this, click the "Visualize" tab on the top of the program. The screen should now look like this:

![Visualize Window](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/visualizeScreenshot.png)

- In the "File to Visualize:" selector, click the triangle button, and find the file to visualize
- Click the visualize button
- This should be relatively quick, and an interactive 3D visualization window should come up.

