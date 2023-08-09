# Auto-Labelling

Brief Description of Auto-Labelling

## How to use Auto-Labelling Page

To use the Auto-Labelling Feature, you must first train a model. Once you have a model trained, you can use the Auto-Label page, which looks like this:

![Auto Label Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/predictScreenshot.png)

The most important inputs that you must fill out are the following:

- Image Stack: Must put your stack of images that you want to label (can be created from a folder of seperate images using the ImageTools section)
- Output File: Pick an output file name (this is a new file to be created) to put the predictions in. You need to remember this file, as you select it to make 3D reconstructions and get stats
- Model To Use: Select the name of the model you want to make the predictions. This is a model that you trained previously.

> The rest may have default values that should work just fine. But feel free to tweak them.

Next, click the Label button, the text box below will print out information updating you on progress, and let you know when the auto-labelling is complete.

## Detailed Description of Each Parameter

![Auto Label Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/predictScreenshot.png)

- Image Stack:        Must put your stack of images that you want to label (can be created from a folder of seperate images using the ImageTools section)
- Output File:        Pick an output file name (this is a new file to be created) to put the predictions in. You need to remember this file, as you select it to make 3D reconstructions and get stats.
- Model To Use:       Select the name of the model you want to make the predictions. This is a model that you trained previously.
- Pad Size            The model can add padding around the outside of a section that the model is currently processing. More info can be found ![here](https://deepai.org/machine-learning-glossary-and-terms/padding)
- Aug Mode            ?
- Aug Num             ?
- Stride:             The model does not process your entire dataset at once. It works on small sections at a time. After it is finished with each section, it moves over a little bit and processes again. The amount it moves after each iteration is the stride. If the stride is the exact same size as the model input, then it is processing each pixel once. It shouldn't be bigger, as you will miss sections. If it is smaller, some sections will get repeated, which is fine. Overlapped sections use information from all overlaps to create the label
- Samples Per Batch:  How many data iterations to process at the same time. Has no implication for model inference (Auto-Labelling), other than it may be faster to increase this number a little bit if your hardware can handle it.
