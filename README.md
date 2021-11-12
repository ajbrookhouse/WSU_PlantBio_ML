# WSU PlantBio ML

General Description of the project including motivations and also what it can do. Give credit to https://github.com/zudi-lin/pytorch_connectomics

# Installation



- [ ] Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) following their instructions.

- [ ] Download this github folder to somewhere in your computer you wish to use it.

- [ ] Open miniconda in this folder, then type the command "conda env create -f theEnvironment.yml" without the quotes. This step may take a while, as many libraries need to be installed by miniconda.

- [ ] Type "conda activate plantTorch" to activate the new environment you just created.

- [ ] Now you have everything installed except for the pytorch_connectomics package. To install this, type the following lines. If there is an issue with this process, see [here](https://github.com/zudi-lin/pytorch_connectomics#installation) for more information for installing this package.

```bash
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install --upgrade pip
pip install --editable .
cd ..
```

- [ ] Now if you type, you should be back in the main folder of your project. Open up the program by typing "python gui.py". If that does not work, type "python3 gui.py"

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
  - **default.yaml**: Here is a dropdown menu where you can pick from a list of configs. You must pick a config file to specify what type of model you are training. These files are located in (Location) and include parameters such as type of model (is it instance segmentation, semantic segmentation, etc), as well as parameters that do not need to be changed very often (input size, output size, stride, etc). For more information on config files, visit https://github.com/zudi-lin/pytorch_connectomics
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

## Make Configs
