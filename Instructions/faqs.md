# FAQs

## Semantic vs Instance Segmentation 

  

### Semantic Segmentation 

  

Semantic segmentation teaches the machine learning model to classify each pixel of the sample as one of several classes.  

  

Pros: 

  

- Can handle detecting different types of organelles at one time with the same model. 

  

- Faster / requires less processing 

  

Cons: 

  

- The model does not differentiate between different instances of the same organelle. For example, if all mitochondria are labelled as "1", the model does not understand the difference between the different mitochondria in the sample, it only understands if each pixel is mitochondria or not. Well separated mitochondria will still appear as individual instances, but mitochondria that touch each other will appear as one.  

  

### Instance Segmentation 

  

Instance segmentation teaches the model to learn what pixels belong to one class, and also their boundaries. In this way, it can differentiate two different organelles of the same type, even if they are touching each other. 

  

Pros: 

  

- The network itself learns how to differentiate between different instances of the same organelle. 

  

Cons: 

  

- Can only analyze one type of organelle at one time. For example, you can train the model to label chloroplasts or plasmodesmata, but not at the same time. If instance segmentation for both is required, training two separate models is necessary. 

  

- This model's output is not directly usable, and postposing needs to be done at inference time. This means that it takes longer to use the Auto-Labelling feature of the software when using an Instance Segmentation Model, because there are extra processing steps necessary. 

  

## Filetypes 

There are several different filetypes that are used to store data in this program such as .tif, .h5, .yaml, .json, and .csv files.  

1. ".tif" or ".tiff" files are used to store multi page images. These can be used to represent the 3D images that this program works with. It achieves 3D by stacking multiple 2D images together. However, please be aware that a large .tif will require a large computer memory space (RAM).  

2. ".h5" files, are like ".tif" files, but more versatile. ".h5" files can store arrays of any arbitrary dimension / size. They're efficient to store large arrays or data and processing. 

3. ".yaml" files are used to store model configuration data/hyperparameters to define different models used by the program. When entering parameters in the interface, parameters will be passed to complete the yaml files. Once it is complete, the complete yaml file will be used by the algorithm to train the model.   

4. ".json" files can be used to make stacked/tiled datasets. However, it is not efficient for storing image data.

5. ".csv" files are used to store statistical data.  A CSV file will be generated once you click the "Get Model Output Stats" button. Then you may use Excel or other spreadsheet software to open, view and process. 
