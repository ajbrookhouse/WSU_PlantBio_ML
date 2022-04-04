# Training

Brief Description of Training

# Types of Training

There are four different types of training currently enabled in the program. These are 2D and 3D instance and semantic segmentation

## Semantic

Semantic segmentation teaches the machine learning model to classify each pixel of the sample as one of several classes. The classes are whatever you labelled them to be in your training data, most likely different organelles. For example, the background of the image (the null class) will be "0", the cell wall can be labelled as "1", the plasmodesmata as "2", and so on (will be different depending on your dataset).

Pros:

- Can handle detecting different types of organneles at one time with the same model.

- Faster / requires less processing

Cons:

- The model does not differentiate between different instances of the same organelle. For example, if all plasmodesmata are labelled as "2", the model does not understand the difference between different plasmodesmatas in the sample, it only understands if each pixel is a plasmodesmata or not. While you can try to seperate individual plasmodesmatas by considering each group of pixels that are touching one instance of a plasmodesmata, this is not a perfect method, and will not work as well as instance segmentation, especially when there are organelles that are touching in the image (or very close to each other).

## Instance

Instance segmentation teaches the model to learn what pixels belong to one class that it is trying to detect, and also their boundaries. In this way, it is able to differentiate two different organelles of the same type, even if they are touching each other.

Pros:

- The network itself learns how to differentiate between different instances of the same organelle, so even organelles that are touching each other can be differentiated

Cons:

- Can only analyze one type of organelle at one time. For example, you can train the model to label chloroplasts or plasmodesmata, but not both at the same time. If you really need instance segmentation for both, you need to train two seperate models.

- This model's output is not directly usable, and postproccing needs to be done at inference time (Done automatically for you). This means that it takes longer to use tha Auto-Labelling feature of the software when using an Instance Segmentation Model, because there are extra processing steps neccisary.

## 2D Vs 3D

3D datasets interpret a stack of images as one 3D volume. The x and y axis are the x and y axis of each image, and the images stack up along the z axis, adding the depth. In this way, an image stack creates on singular 3D volume.

2D datasets interpret a stack of images as seperate 2D images. Because of this, the image stack does not assume any spatial relation between the different images in the stack, and analyzes them all seperately. This means you can have unrelated images together in a stack (however they must all be similar enough to the trained model that the neural network can interperet them)

# How to Train Using the Different Types of Training

![Training Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/trainScreenshot.png)

The "Training Config:" selection box controls which type of training that you are doing. Semantic.yaml does 3D semantic. Instance.yaml does 3D instance. Semantic2D.yaml does 2D semantic segmentation, and Instance2D.yaml does 2D instance segmentation.

The most important inputs that you must fill out are the following:

- Image Stack: Must put your training raw images (can be created from a folder of seperate images using the ImageTools section)
- Labels: Must put your label images. (can be created from a folder of seperate images using the ImageTools section) Note: Your labels must be representative of what type of training you are doing. Make a .txt file for 2D analysis, and .json or .tif for 3D analysis. Also, semantic gives a unique color for each type of organelle being labelled, while instance only analyzes one organelle and gives a unique color to each instance of that organelle
- Training Config: pick the config for the type of training you are doing.
- X nm/pixel, Y nm/pixel, and Z nm/pixel: Sets the resolution for your training images, this is important to get volumes
- Name: You must give your neural network a name so you can use it later

>> These are the only fields that must be filled out. The rest have default values that should work just fine, and will be described in better detail below in case you want to tweak them.

Next, click the train button, the text box below will print out information updating you on progress, and let you know when the training is complete.

# Detailed Description of Each Parameter

![Training Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/trainScreenshot.png)

- Bulleted List Describing Each Input Parameter
