# Training

Before using a neural network, you must train it on some data. This process is like 'learning' for the neural network. For example: one type of neural network can be used for many different types of data, but do the same process. For example, a Resnet can be commonly used for image segmentation / labelling, but this is a broad task. The images could be of a city where you are segmenting buildings, or of cells, where you are segmenting organelles. Even out of cell images, there are many different types of cells and many different organelles that could be labelled for. This is why training is needed. A neural network like a resnet needs to see your data, and 'learn' how to label it. To do this, you need two files, a stack of raw images, and then a stack of corresponding labels. By essentially giving the network both the question (raw images) and the answers (the labels), it can learn how to replicate the labelling process.

## Detailed Description of Each Parameter

![Training Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/trainTab.png)
- Image Stack: Image stacks are the raw images that were originally generated on the microscope.  The images should be very similar to the types of images that you want to process and may be images from the same 3D volume that will be used for the auto-label process (e.g. use images 1-50 to generate labels and the training stack and use images 50-1000 for the analysis).  

- Labels:             Labels are essentially a mask that tells Anatomics MLT what the structures of interest so that the algorithm can learn what the structures of interest look like. More details about what labels represent can be found here ![on the FAQ page](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/Instructions/faqs.md#semantic-vs-instance-segmentation) 

- Training Config: Choose which type of model you want to train. By default, this program comes with Instance3D, Instance2D, Semantic2D, and Semantic3D.  

- X nm/pixel:         Provide spatial resolution for the image. In many cases the x, y, z resolutions are different (e.g. 10 nm X, 10 nm Y, 40 nm Z). If labels are not generated from the same stack that will be used for auto labelling, the training files should have the same x, y, z resolution.  

- Y nm/pixel:         Same as X, but for the y direction (vertical along one image slice) 

- Z nm/pixel:         Same as X, but for the z direction (through the stack of images, perpendicular to one image slice) 

- #GPU:               The number of Graphic Processing Units (GPUs) to be used for the training. Most likely the answer will be 1, however if you have more (e.g. in a computer cluster) the number can be increased. While the program can run on the CPU only, an appropriate GPU is highly recommended to process large data sets.  

- #CPU:               The number of Central Processing Units (CPUs) to be used for this training. The default is 1. Your system probably has 4, 6, or 8 in total to use if you want to increase this number. Do not increase this number above the number of CPU cores your computer has as this will cause a slowdown. This number can be increased and should speed up some calculations, however CPU does not affect the speed of training a model nearly as much as the GPU does. 

- Base LR:            Learning Rate (LR) is a parameter that should be chosen every time a model is trained. Essentially, this changes how quickly the model changes its internal parameters each time it processes a subset of data. A higher learning rate causes the model to change faster.  This can lead to the model learning faster / needing less training, however a higher learning rate can also lead to unstable training or cause the model to not fully optimize. We recommend a starting value of 0.001 

- Iteration Step:     Number of iterations the program runs simultaneously. (Typically, 1) 

- Iteration Save:     The program will incrementally save your model as it trains. It does this every multiple of this number. For example, if 100000 iterations are used, and iteration save is 10000, the model would save at 10000, 20000, 30000, etc.... It is recommended to save a few times during the training process, but too often will use unnecessary amounts of disk space.  About 10-20 saves during the entire training process is sufficient. The number must be equal or a fraction of "Iteration Total".  

- Iteration Total: The total number of training iterations. A higher number means the model trains over more data points. This is an important parameter to improve training quality, but it also linearly increases computation time. If too few steps are chosen, the model will not be accurate enough to properly identify structures of interest. If too many iterations are chosen the model may actually “overfit” the training data, meaning the model becomes too biased towards the training data and cannot perform well on test data. Typical iteration numbers for good training are 300000 to 500000.  

- Samples Per Batch: How many data iterations to process at the same time. 1 means the software investigates one data point, and then using the output it adjusts the network parameters and moves on to the next data point. If 2 is chosen, two data points are processed simultaneously and then the network parameters are updated. This may lead to higher quality training but doubling “samples per batch” doubles computation time and may cause problems with the computer memory. Typical numbers are 1 to 4 but may be higher if a computer cluster is available. 

- Window Size: The model or detector operates/trains in a small window or kernel. This window slides over the input data for training. It must have three numbers, separated by commas. The first number represents how much it operates in the Z-axis. The second represents the X-axis and the third the Y-axis. The three numbers will decide how large the window or kernel is. Typically, the window size is determined by what target will be labeled. At a minimum a label should fit a whole structure of interest. For example, if mitochondria are labeled a small window size such as 3,129 ,129 is sufficient because the mitochondria in our example stack are comparably small. In the case of statoliths in our example, a window size of 3, 501, 501 is required because the statolith are much larger and have a diameter of approximately 300-500 pixel.  It is not necessary to increase the window size beyond the size of the structure of interest as this will consume more RAM and lead to longer processing time. 

- Name Model as:       Provide a unique, recognizable name that can be recognized for future auto-labelling processes. 
