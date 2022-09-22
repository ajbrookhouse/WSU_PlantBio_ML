# Evaluate Model

This section is to allow you to quickly view the results of the training. The evaluate button will let you pick a model output .h5 file, and a ground truth label, and let you know quantitatively how similar they are. The Visually compare button lets you quickly see the outputted labels overlayed on top of the images they are supposed to be labelling so you can quickly qualitatively observe how well the Auto-Label process went.

# Evaluate Button

> To be implemented, will compare a model output .h5 to a ground truth label .tif or .h5 to compare how different they are quantitatively such as percent difference, difference in count, etc.

# Visually Compare Button

- Select your Model Output File
- Select the Raw Images that this model output corresponds with (could be a .tif, .json, or .txt)
- Click the visually compare button
- The program will open a few windows showing the raw image and an image with the predicted labels overlayed over the image. This way you can visually inspect the performance of the model.

> Recently implemented for instance, I will test some more for instance and also models which are 2D, but I think it should work for all of these.
