# Installation

- [ ] Install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html) following their instructions.

- [ ] Open the anaconda prompt. You should be able to find this in the windows start menu with your other programs. Either search for it, or look in the folder most likely called "Anaconda 3 (64-bit)" Another way to find it is by clicking the start menu / press the windows key, start typing miniconda, and select "Anaconda Prompt (Miniconda3)"

- [ ] Set miniconda's working directory to where you want to install the program by typing the following command with out the <>. You can install the program wherever you want, just remember where you choose to install it. The default is to install it in your C:\\Users\\YourUsername folder. If you are ok with that location, skip this next step.

```bash
cd <path of where you want to install the program folder, example: C:\\Users\\YourUsername\\Documents>
```

- [ ] Install the program using the following commands. Please copy them by highlighting them all with your cursor, and then pressing CTRL+C, or right click and select copy. Then run them by pasting them into the terminal (either using CTRL+V or right clicking and click paste.). After you hit paste, the entire installation process should occur automatically. This may take a while. When it is done, it should print "Completely finished with installation. Please run the program by typing 'python gui.py'" to the screen, and then open the program.

```bash
conda create --name plantTorch python=3.8.11 -y
conda activate plantTorch
conda install git -y
git clone https://github.com/ajbrookhouse/WSU_PlantBio_ML.git
cd WSU_PlantBio_ML
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
git checkout c490dd989864504456ad71a3ca3c99096cba6c1a
pip install --editable .
cd ..
pip install open3d
pip install paramiko
pip install pygubu
pip install plyer
pip install ttkthemes
conda install -c conda-forge imagecodecs -y

echo Completely finished with installation. Please run the program by typing 'python gui.py'
python gui.py
 

```

- [ ] Open up the program by typing "python gui.py" (This step may be completed automatically by the previous command copy section)

The main program should now be up on your screen:

![screenshot of first screen that opens when you open program](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/trainScreenshot.png)

# Updating Program

Open up miniconda by clicking start, typing miniconda, and selecting "Anaconda Prompt (Miniconda3)". Then type the following:
```bash
cd <path to the WSU_PlantBio_ML folder, example: C:\\Users\\YourUsername\\Documents\\WSU_PlantBio_ML>
git pull
```

# Uninstalling

If you need to uninstall the program for some reason (One reason could be getting a fresh install), do the following things. Close miniconda. Delete the WSU_PlantBio_ML folder and everything in it. Then open miniconda and type the following:

```bash
conda deactivate plantTorch #(only needed if your miniconda prompt lines start with (plantTorch). If they say (base) you don't need this step)
conda env remove -n plantTorch -y
```

Now, all libraries used for the project will be uninstalled, and so will the reset of the program

If you no longer need miniconda for other programs, feel free to uninstall it like any other windows program.

