# Datasets

- To create a 3D stack out of individual images, all images should be in a folder and the images need to be named in consecutive order (e.g.0001.tif, 0002.tif, 0003.tif, etc.). The number represents the individual image's location along the z axis. It is important that there is at least one leading 0 on each image, (e.g. 0512.tif is one leading zero, 00512.tif is two leading zeros) this makes certain that the program orders them properly. A prefix is fine before the numbers in the filename, but it must be the same for all images. All images in the folder / stack must both have the same dimensions and spatial resolution. 

  

The ImageTools screen: 

  

![Image Tools Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/imagetoolTab.png) 

  

To create a single file for Training / AutoLabelling, you must fill out the following fields: 

  

- Folder To Combine: Click the "Choose Folder" button to pick the folder containing all images to be combined. 

- Output Filename: Click the "Choose File" button and pick the location and name of your output file. 

  

Next, click one (or more) of the "Combine Into" buttons. 

  

- Combine into TIF: combines the dataset into a single .tif image stack. The 3D tiff file can be opened in most image software packages. However, tiff files are large compared to other formats. 

- Combine into TXT: 2D only. This creates a .txt file for each individual image. 

- Combine into JSON: .json files can process an arbitrarily large 3D dataset, and do not take up much space. They also allow the software to load smaller parts of the dataset at a time. Json files are suitable for processing extremely large datasets. 

Note, both the TXT and JSON files contain the locations of the original images. This information is lost when moving the files. 


## Detailed explanation for each parameter
Folder to Combine: Click "chose folder" and select the folder that contains all files you want to have combined into a single 3D stack.  

Output file: provide the path and name of the file that will be generated. 

Combine into Tif: Click this button if you intend to create a 3D tiff file. 

Combine into TXT: Click this button if you intend to create a TXT file for each individual image. 

Combine into JSON: Click this button if you intend to create a 3D JSON file. 
