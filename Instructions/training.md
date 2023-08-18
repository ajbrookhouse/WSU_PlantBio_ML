# Training

Before using a neural network, you must train it on some data. This process is like 'learning' for the neural network. For example: one type of neural network can be used for many different types of data, but do the same process. For example, a Resnet can be commonly used for image segmentation / labelling, but this is a broad task. The images could be of a city where you are segmenting buildings, or of cells, where you are segmenting organelles. Even out of cell images, there are many different types of cells and many different organelles that could be labelled for. This is why training is needed. A neural network like a resnet needs to see your data, and 'learn' how to label it. To do this, you need two files, a stack of raw images, and then a stack of corresponding labels. By essentially giving the network both the question (raw images) and the answers (the labels), it can learn how to replicate the labelling process.

## Types of Training

The following two types of training are enabled in the program. 3D data are perferable for both training types. 

### Semantic

Semantic segmentation teaches the machine learning model to classify each pixel of the sample as one of several classes. The classes are whatever you labelled them to be in your training data, most likely different organelles. For example, the background of the image (the null class) will be "0", the cell wall can be labelled as "1", the plasmodesmata as "2", and so on (will be different depending on your dataset).

Pros:

- Can handle detecting different types of organneles at one time with the same model.

- Faster / requires less processing

Cons:

- The model does not differentiate between different instances of the same organelle. For example, if all plasmodesmata are labelled as "2", the model does not understand the difference between different plasmodesmatas in the sample, it only understands if each pixel is a plasmodesmata or not. While you can try to seperate individual plasmodesmatas by considering each group of pixels that are touching one instance of a plasmodesmata, this is not a perfect method, and will not work as well as instance segmentation, especially when there are organelles that are touching in the image (or very close to each other).

### Instance

Instance segmentation teaches the model to learn what pixels belong to one class that it is trying to detect, and also their boundaries. In this way, it is able to differentiate two different organelles of the same type, even if they are touching each other.

Pros:

- The network itself learns how to differentiate between different instances of the same organelle, so even organelles that are touching each other can be differentiated

Cons:

- Can only analyze one type of organelle at one time. For example, you can train the model to label chloroplasts or plasmodesmata, but not both at the same time. If you really need instance segmentation for both, you need to train two seperate models.

- This model's output is not directly usable, and postproccing needs to be done at inference time (Done automatically for you). This means that it takes longer to use tha Auto-Labelling feature of the software when using an Instance Segmentation Model, because there are extra processing steps neccisary.

### 2D Vs 3D

3D datasets interpret a stack of images as one 3D volume. The x and y axis are the x and y axis of each image, and the images stack up along the z axis, adding the depth. In this way, an image stack creates on singular 3D volume.

2D datasets interpret a stack of images as seperate 2D images. Because of this, the image stack does not assume any spatial relation between the different images in the stack, and analyzes them all seperately. This means you can have unrelated images together in a stack (however they must all be similar enough to the trained model that the neural network can interperet them)

## How to Train Using the Different Types of Training

![Training Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/trainScreenshot.png)

<!-- The "Training Config:" selection box controls which type of training that you are doing. Semantic.yaml does 3D semantic. Instance.yaml does 3D instance. Semantic2D.yaml does 2D semantic segmentation, and Instance2D.yaml does 2D instance segmentation. -->

The most important inputs that you must fill out are the following:

- Image Stack: Must put your training raw images (can be created from a folder of seperate images using the ImageTools section)
- Labels: Must put your label images. (can be created from a folder of seperate images using the ImageTools section) Note: Your labels must be representative of what type of training you are doing. Make a .txt file for 2D analysis, and .json or .tif for 3D analysis. Also, semantic gives a unique color for each type of organelle being labelled, while instance only analyzes one organelle and gives a unique color to each instance of that organelle
<!-- - Training Config: pick the config for the type of training you are doing. -->
- X nm/pixel, Y nm/pixel, and Z nm/pixel: Usually we set them to 1

- Window Size: How large the training window is. For exapmle, a 5,225,225 window. Such window will scan through our input training volume and be trained through its sliding scan. Please be aware, a larger window may not result in better result.
- Name Model as: You must give your neural network a name so you can use it later

> These are the only fields that must be filled out. The rest have default values that should work just fine, and will be described in better detail below in case you want to tweak them.

Next, click the train button, the text box below will print out information updating you on progress, and let you know when the training is complete.

## Detailed Description of Each Parameter

![Training Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/trainScreenshot.png)

- Image Stack:        The raw images that you will use for the training process. These should be very similar to the types of images that you will want to automatically label in the future. These images should also correspond to the labels that you provide on this screen.
- Labels:             The labels that you will use for the training process. These should correspond with the images that you provide on this screen. More details about what labels represent can be found ![on the FAQ page](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/Instructions/faqs.md#semantic-vs-instance-segmentation)
<!-- - Training Config:    Here is where you choose which type of model you want to train. By default, this program comes with Instance, Instance2D, Semantic, and Semantic2D. If you wish to create more you can. More details about 2D vs 3D and Instance vs Semantic can be found ![on the FAQ page](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/Instructions/faqs.md#semantic-vs-instance-segmentation) -->
- X nm/pixel:         Here is where you provide the spatial resolution for the image that you are training on (auto labelling images with this model in the future should have the same resolution). This specific line is for the x dimension (horizontal along one image slice), how many nanometers are within one pixel.
- Y nm/pixel:         Same as X, but for the y direction (vertical along one image slice)
- Z nm/pixel:         Same as X, but for the z direction (through the stack of images, perpendicular to one image slice)
- #GPU:               The number of Graphic Processing Units (GPUs) to be used for this training. Most likely the answer will be 1, however if you have more you can increase this number. While the program should be able to run with zero, it will be significantly slower than using even 1 GPU
- #CPU:               The number of Central Processing Units (CPUs) to be used for this training. The default is 1. Your system probably has 4, 6, or 8 in total to use if you want to increase this number. Do not increase this number above the amount of CPU cores your computer has as this will cause slowdown. This number can be increased and should speed up some calculations, however CPU does not affect the speed of training a model nearly as much as the GPU does.
- Base LR:            Learning Rate (LR) is a parameter that should be chosen every time a neural network is trained. Essentially, this changes how quickly the model changes its internal parameters each time it sees data. A higher learning rate causes the model to change more each time it sees data. This can lead to the model learning faster / needing less training, however a higher learning rate can lead to unstable training, or cause the model to not fully optimize. A slower learning rate causes the model to change less each time that it sees data, which can lead to longer training times, but can allow the model to optimize more.
- Iteration Step:     Number of iterations we want to run simultaneously
- Iteration Save:     The program will incrementally save your model as it trains. It does this every multiple of this number. For example if iteration_total is 10, and iteration_save is 2, the model would save at 2, 4, 6, 8, and 10. You want it to be saved a few times over the training process, but too low of a number will use unneccisary amounts of disk space. 5, 10, or 25 thousand are probably all fine options.
- Iteration Total:    The total number of training iterations to go through. A higher number means the model trains over more data points. This is an important parameter to training. If you choose too few iterations, the model will not have enough training to meaningfully learn anything. If you choose too many, you can cause the model to overfit on your training data.
- Samples Per Batch:  How many data iterations to process at the same time. 1 means you run a data point through the model, and then using the output you adjust the network parameters, and then move on to the next data point. If you choose 2, two data points are processed simultaneously and then the network parameters are updated. You can choose any positive number here, but if you choose one too high for your hardware you will get some sort of CUDA out of memory error. There isn't anything wrong with choosing 1, but picking a higher number lets you process more data per training iteration.
- Name Model as:      The name to use for your model. This can be whatever you want. The important thing is that it is meaningful to you and that you will know which model is which. This is the name you select on the Auto-Label screen when you are choosing a model to use. Each model you train must have a unique name.

