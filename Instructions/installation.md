# Installation

- [ ] Since the Program requires "The PyTorch Connectomics package" which was mainly developed on Linux machines with NVIDIA GPUs, we recommend using **Linux** or **Windows** to ensure the compatibility of the latest features with your system. The instructions below are for **WINDOWS**. 

- [ ] Install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html) following their instructions. 

- [ ] Open the “Anaconda Prompt”. You should be able to find this in the windows start menu with your other programs. Either search for it, or look in the folder most likely called "Anaconda 3 (64-bit)" Another way to find it is by clicking the start menu / press the windows key, start typing miniconda, and select "Anaconda Prompt (Miniconda3)" 

- [ ] Install the program using the following commands. Please copy them by highlighting them all with your cursor, and then pressing CTRL+C, or right click and select copy. Then run them by pasting them into the terminal (either using CTRL+V or right clicking and click paste.). After you hit paste, the installation process should occur automatically. This may take a while. When it is done, it should print "Completely finished with installation. And the program will open automatically.

```bash 

cd Documents 

conda create --name plantTorch python=3.8.11 -y 

conda activate plantTorch 

conda install git -y 

git clone https://github.com/ajbrookhouse/WSU_PlantBio_ML.git 

cd WSU_PlantBio_ML 

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia 

conda install cudatoolkit=11.8 -c pytorch 

conda install h5py 

git clone https://github.com/ajbrookhouse/pytorch_connectomics 

cd pytorch_connectomics 

pip install --editable . 

cd .. 

pip install open3d 

pip install scikit-image 

pip install paramiko 

pip install pygubu 

pip install pandas 

pip install plyer 

pip install ttkthemes 

pip install connected-components-3d 

conda install -c conda-forge imagecodecs -y 

pip install neuroglancer 

echo Completely finished with installation. Please run the program by typing 'python gui.py' 

python GPU_test.py 
``` 

- [ ] If the program does not open. Please open the program by typing "python gui.py" in your terminal (Normally, this step should be completed automatically by the previous command copy section) 

The main program should now visibleon your screen: 

![screenshot of first screen that opens when you open program](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/trainTab.png) 

  

# Updating Program

Open miniconda by clicking start, typing miniconda, and selecting "Anaconda Prompt (Miniconda3)". Then type the following: 

```bash 

cd Documents 

cd WSU_PlantBio_ML 

git pull 

``` 

If an error is shown when trying to update, please type ‘git reset –hard'. After that use the command “git pull” to update the program.  

# Uninstalling 

If you need to uninstall the program for some reason (One reason could be getting a fresh install), do the following things. Close miniconda. Delete the WSU_PlantBio_ML folder and everything in it. Then open miniconda and type the following: 

  

```bash 

conda deactivate plantTorch (If your miniconda prompt lines start with (plantTorch). If they say (base), please skip this line/step) 

conda env remove -n plantTorch -y 

``` 

Now, all libraries used for the project will be uninstalled. 

If you no longer need miniconda for other programs, feel free to uninstall it like any other windows program. 

Now, all libraries used for the project will be uninstalled, and so will the reset of the program

If you no longer need miniconda for other programs, feel free to uninstall it like any other windows program.

