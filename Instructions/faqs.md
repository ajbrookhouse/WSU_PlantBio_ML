#Faqs

This page explains a few miscellaneous things about the program and how to use it.

## Semantic vs. Instance Segmentation

### Semantic Segmentation

Semantic segmentation teaches the machine learning model to classify each pixel of the sample as one of several classes. The classes are whatever you labelled them to be in your training data, most likely different organelles. For example, the background of the image (the null class) will be "0", the cell wall can be labelled as "1", the plasmodesmata as "2", and so on (will be different depending on your dataset).

Pros:

- Can handle detecting different types of organneles at one time with the same model.

- Faster / requires less processing

Cons:

- The model does not differentiate between different instances of the same organelle. For example, if all plasmodesmata are labelled as "2", the model does not understand the difference between different plasmodesmatas in the sample, it only understands if each pixel is a plasmodesmata or not. While you can try to seperate individual plasmodesmatas by considering each group of pixels that are touching one instance of a plasmodesmata, this is not a perfect method, and will not work as well as instance segmentation, especially when there are organelles that are touching in the image (or very close to each other).

### Instance Segmentation

Instance segmentation teaches the model to learn what pixels belong to one class that it is trying to detect, and also their boundaries. In this way, it is able to differentiate two different organelles of the same type, even if they are touching each other.

Pros:

- The network itself learns how to differentiate between different instances of the same organelle, so even organelles that are touching each other can be differentiated

Cons:

- Can only analyze one type of organelle at one time. For example, you can train the model to label chloroplasts or plasmodesmata, but not both at the same time. If you really need instance segmentation for both, you need to train two seperate models.

- This model's output is not directly usable, and postproccing needs to be done at inference time. This means that it takes longer to use tha Auto-Labelling feature of the software when using an Instance Segmentation Model, because there are extra processing steps neccisary.

## Filetypes

There are several different filetypes that are used to store data this program. They are .tif, .h5, .yaml, .json

1. ".tif" or ".tiff" files are used to store multi page images. These can be used to represent the 3D images that this program works with. However, there seem to be issues with working with files that get larger than roughly 4GB, due to some limitation in a python image manipulation library. These images should be openable / editable by all major image viewing / manipulation programs.

2. ".h5" files, are similar to the ".tif" files, but more versitale. ".h5" files are able to store arrays of any arbitrary dimension / size. They are even able to save arrays bigger than are able to be loaded in memory at once. This allows you to have massive arrays processed and loaded onto the computer that are way bigger than the size of ram on the computer.

3. ".yaml" files are used to store model configuration data to define different types of models used by the program. We ask that you don't modify the origionals, and if you want to make changes to the parameters to please make a copy of the config you want to change. ADD DETAILED LINK

4. "json" files can be used to make tiled datasets. ADD DETAILED LINK