# Auto-Labelling

Brief Description of Auto-Labelling

## How to use Auto-Labelling Page

To use the Auto-Labelling Feature, you must first train a model that can be chosen for the Auto-Label process. 

![Auto Label Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/labelTab.png) 

  
## Detailed Description of Each Parameter 

- Image Stack:     This file is a 3D image stack of original images (usually) taken with a microscope that will be used for analysis.  

- Output File:        Provide an output file name (this is a new file to be created) that will contain the identified structures of interest and that will be used for statistics and visualization.  

- Model To Use:       Select the name of the model to be used. This is a file that was generated during the training using the labels highlighting your structures of interest.  

- Pad Size:            The model can add padding around the outside of a section that the model is currently processing. More info can be found ![here](https://deepai.org/machine-learning-glossary-and-terms/padding) 

- Aug Mode: Augmentation Mode. We would recommend using a ‘mean’ mode, as it will take the average among these augmented/modified samples.   

- Aug Num:             Numbers of different augmentation techniques, may choose None, 4, 8. By applying augmentation techniques, the model will get more samples to label and may perform better. But this will consume more memory (RAM).  

- Stride:             The model does not process the entire dataset at once. It works on small sections at a time. After it has finished processing each section, it moves on and processes again. The amount it moves after each iteration is the stride. We recommend using a stride that corresponds to the window size that was used during the training. Smaller strides than the window size are ok, but bigger strides will result in non-processed sections.  

- Samples Per Batch: How many data iterations to process at the same time. It has no implication for model quality (Auto-Labelling), other than that processing may be faster when the number is slightly increased. Higher numbers require better hardware. 

Label: Clicking this button will start the labeling process.  

The text box will indicate the labeling progress. Once finished it will indicate which post-processing step is needed. If a post processing step is required, please click the appropriate box below the text box (e.g. Semantic3D Post-Process...) 
