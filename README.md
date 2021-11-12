# WSU_PlantBio_ML

## Installation



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
