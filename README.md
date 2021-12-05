# WSU PlantBio ML

General Description of the project including motivations and also what it can do. Give credit to https://github.com/zudi-lin/pytorch_connectomics

# Quickstart Guide
Here, we will give quick directions to get your first machine learning model working with the program. First, install using the installation instructions ![here](https://github.com/ajbrookhouse/WSU_PlantBio_ML#installation)

## Open Program

Open up miniconda in the folder that you downloaded this program. Then type the following:
```bash
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

# Installation

- [ ] Install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html) following their instructions.

- [ ] Open the anaconda prompt. You should be able to find this in the windows start menu with your other programs. Either search for it, or look in the folder most likely called "Anaconda 3 (64-bit)"

- [ ] Set miniconda's working directory to where you want to install the program by typing the following command with out the <>. You can install the program wherever you want, just remember where you choose to install it. The default is to install it in your C:\\Users\\YourUsername folder. If you are ok with that location, skip this next step.

```bash
cd <path of where you want to install the program folder, example: C:\\Users\\YourUsername\\Documents>
```

- [ ] Install the program using the following commands. You should be able to copy them all by clicking the top right of the code containing box. Then run them by pasting them into the terminal.

```bash
conda create --name plantTorch -y
conda activate plantTorch
git clone https://github.com/ajbrookhouse/WSU_PlantBio_ML.git
cd WSU_PlantBio_ML
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install --editable .
cd ..
pip install open3d
pip install paramiko
pip install pygubu

echo Completely finished with installation. Please run the program by typing 'python gui.py'


```

- [ ] Open up the program by typing "python gui.py". If that does not work, type "python3 gui.py"

The main program should now be up on your screen:

![screenshot of first screen that opens when you open program](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/trainScreenshot.png)

# Using Program

## Opening Program

Open up miniconda in the folder that you downloaded this program. Then type the following:
```bash
conda activate plantTorch
python gui.py
```
The main program should now show up on your screen (if "python gui.py" does not work, try "python3 gui.py"). It is pictured above

## Training

The training screen is the default screen that opens up when you first open the program. It should look like this:

![screenshot of main screen for training networks](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/trainScreenshot.png)

### Input Fields

There are a number of fields on this screen that hold parameters for the model. I will divide them up into Essential, Common, and Uncommon. The essential elements must be given a value every single time that you wish to train a network. Common includes elements that have a sensible default value, but may commonly need some short of adjustment. Uncommon are elements that most likely do not need to be changed at all, but were added to the program just in case you want to change them.

1. Essential Input Fields
  - **Image Stack (.tif)**: This field is where you pick the images that you are training on. Click the triangle button on the right and a file picker button will pop up. Select the image stack that you have prepared for training.
  - **Labels (.h5)**: This field is where you pick the labels that you are using for training. Same as Image stack, click the triangle button on the right and a file picker button will pop up. Select the label stack that you have prepared for training. This can be either a .tif stack of images or an .h5 stack.
  - **default.yaml**: Here is a dropdown menu where you can pick from a list of configs. You must pick a config file to specify what type of model you are training. These files are located in (Location) and include parameters such as type of model (is it instance segmentation, semantic segmentation, etc), as well as parameters that do not need to be changed very often (input size, output size, stride, etc). For more information on config files, visit [this page](https://github.com/zudi-lin/pytorch_connectomics#yacs-configuration)
  - **Name**: You must give your model that you are training some sort of name. This name is completely arbitrary, but should have some sort of meaning to you so you can differentiate between the different models that you have trained

2. Common Input Fields
  - **#GPU**: The number of GPU's on your system that you wish to use for training. The default is 1, but can be set to either zero or a higher number.
  - **#CPU**: Number of CPU cores on your system that you wish to use for training. The default is 1, but can be set to a higher number if you wish, however this will not impact performance nearly as much as GPU
  - **Samples Per Batch**: The number of samples to use for each batch of training. A higher number will allow you to train faster, however if there is not enough memory training will fail with a CUDA memory allocation error. If this happens, you need to change to a lower number

3. Uncommon Input Fields
  - List of the rest here with description of what they do

4. Cluster Information
  -Cluster information here 

### Training Buttons

There are two buttons at the bottom of the screen, along with a text box to show the program output.

1. Training Button
  - Information about Training Button

2. Check Cluster Status Button
  -Information about Status Button
  
## Using Network For Prediction

To use the network for prediction, click the tab on the top of the program that says "Auto-Label". The following screen should then be visible:

![screenshot of auto-label screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/predictScreenshot.png)

This screen allows you use a network that you have trained to automatically label

### Input Fields

1. Essential Input Fields
  - **Image Stack (.tif)**: This field is where you pick the images that you are training on. Click the triangle button on the right and a file picker button will pop up. Select the image stack that you have prepared for training.

2. Common Input Fields
  - **Samples Per Batch**: The number of samples to use for each batch of training. A higher number will allow you to train faster, however if there is not enough memory training will fail with a CUDA memory allocation error. If this happens, you need to change to a lower number

3. Uncommon Input Fields
  - List of the rest here with description of what they do

4. Cluster Information
  -Cluster information here 

### Label Button

There are two buttons at the bottom of the screen, along with a text box to show the program output.

1. Training Button
  - Information about Label Button

2. Check Cluster Status Button
  -Information about Status Button




## Model Evaluation

## Image Tools

## Output Tools

## Visualization
