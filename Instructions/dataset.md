# Datasets

To create a dataset to use for the Training or Prediction process, you first need to start with a folder of images.

- For a 3D model, these images should be all in a folder named things like 0001.tif, 0002.tif, 0003.tif, etc. The number represents the individual image's location along the z axis. It is important that there is at least one leading 0 on each image, (ex 0512.tif is one leading zero, 00512.tif is two leading zeros) this makes certain that the program orders them properly in all instances. A prefix is fine before the numbers in the filename, but it should be the same for all images.
- For a 2D model, the image names / order doesn't really matter, but you might as well still use the same naming conventions as the 3D model.

> In both instances, all images in the folder / stack should have both the same dimensions and spatial resolution.

Here is the ImageTools screen:

![Image Tools Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/imageToolsScreenshot.png)

To create a single file to use for training / autoLabelling, you must fill out the following fields:

- Folder To Combine: Click the "Choose Folder..." button to pick the folder containing all of the images that you want to combine.
- Output Filename: Click the "Choose Output File" button, and pick the location and name of your output file.

Next, click one (or more) of the "Combine Into..." buttons.

- Combine into TIF: 3D only, combines the dataset into a single .tif image stack. It is nice because it is easy to click on the file and view it in the image viewer of your choosing. However, if the dataset is very big, this will take a lot of extra, unneeded space, and may not work if it is too big.
- Combine into TXT: 2D only. This is the only choice for 2D. Makes a .txt file where each line is the filepath of one of the images.
- Combine into JSON: 3D only. This creates a .json file that describes the dataset. This lets you process an arbitrarily large 3D dataset, and doesn't take up much space, regardless of size since it really just stores pointers to other files. This also allows the model to load in the dataset small 'chunks' at a time, meaning you can process extremely large datasets.

That is all that needs to be done. Note, both the TXT and JSON options contain the locations of the origional images used to make them. These options will have to be remade if you move those origional images.

> All three of these options can be used to make either training images, training labels, and prediction images, and can be selected directly in the corresponding file selectors on those pages.
