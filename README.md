# WSU PlantBio ML

General Description of the project including motivations and also what it can do. Give credit to https://github.com/zudi-lin/pytorch_connectomics

# Installation



- [ ] Insall Miniconda https://docs.conda.io/en/latest/miniconda.html following their instructions.

- [ ] Download this github folder to somewhere in your computer you wish to use it.

- [ ] Open miniconda in this folder, then type the command "conda env create -f theEnvironment.yml" without the quotes. This step may take a while, as many libraries need to be installed by miniconda.

- [ ] Type conda activate plantTorch to activate the new environment you just created.

- [ ] Now you have everything installed except for the pytorch_connectomics package. To install this, type the following lines.

```bash
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install --upgrade pip
pip install --editable .
cd ..
```

- [ ] Now if you type, you should be back in the main folder of your project. Open up the program by typing "python gui.py". If that does not work, type "python3 gui.py"

The main program should now be up on your screen:

![screenshot of main screen for training networks](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/trainScreenshot.png)

# Using Program

## Opening Program

Open up miniconda in the folder that you downloaded this program. Then type the following:
```bash
conda activate plantTorch
```
The main program should now show up on your screen. It is pictured above

## Training

The training screen is the default screen that opens up when you first open the program. It should look like this:

![screenshot of main screen for training networks](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/images/trainScreenshot.png)

There are a number of fields on this screen that hold parameters for the model. I will divide them up into Essential, Common, and Uncommon. The essential elements must be given a value every single time that you wish to train a network. Common includes elements that have a sensible default value, but may commonly need some short of adjustment. Uncommon are elements that most likely do not need to be changed at all, but were added to the program just in case you want to change them.

1. Essential Input Fields
  - Image Stack (.tif): This field is where you pick the images that you are training on. Click the triangle button on the right and a file picker button will pop up. Select the image stack that you have prepared for training.
  - Labels (.h5): This field is where you pick the labels that you are using for training. Same as Image stack, click the triangle button on the right and a file picker button will pop up. Select the label stack that you have prepared for training. This can be either a .tif stack of images or an .h5 stack.
  - default.yaml: Here is a dropdown menu where you can pick from a list of configs. These files are located in (Location) and include parameters such as type of model (is it instance segmentation, semantic segmentation, etc), as well as parameters that do not need to be changed very often. For more information on config files, visit https://github.com/zudi-lin/pytorch_connectomics
