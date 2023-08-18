<!-- # Visualize

This screen lets you see your 3D geometries, as well as do some manipulations like cropping, and placing slices of an image in the 3D model where they correspond.

![Visualize Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/visualizeScreenshot.png)

Here, you can add .ply files to the list of files to visualize by clicking the choose file button. You can choose as many or as few files as you want.

You can also pick the Choose Color button for each file. This will let you choose what color that specific file will be in the visualization

> For layers created by instance segmentation, the Choose color will be ignored, and colors will be assigned randomly to differentiate the different instances of the organelles.

Click the visualize button, and after some brief calculation and file loading, a 3D visualization window should appear that is interactive. -->


# Visualize

This screen lets you use Neuroglancer, a visualization package, to visualize the generate the output result.

Simply select the Raw Image and Model Output by typing the name or using the interactive button.

Enter the Z scale, X scale, and Y scale. 

Enter the min and max for X, Y, Z. To get the best result, we may use the shape of the raw image. For exapmle, if the input image has a shape of 500,1200,1600 for its z,x,y, we can use 0 for z min, 500 for z max; 0 for x min, 1200 for x max; 0 for y min, 1600 for y max.

Simply click the "Launch Neuroglancer" to launch the visualization work. Once the visualization is ready, a blue link will show up in the software window. We can either click the link, or copy&paste it to our browser to view the result. 