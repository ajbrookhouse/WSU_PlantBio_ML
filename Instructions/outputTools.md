# Output Tools

This screen has a few tools that let you take the output model .h5 and further process it. Make Geometries lets you make 3D geometry files from your model output. These can be visualized with this program, or opened up in other popular softwares like Blender. The Get Model Output Stats section lets you measure volumes from your model output.

![Output Tools Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/outputToolsScreenshot.png)

> Both making geometries and getting model output stats can be done on a cropped version of the model output. Just fill out the x, y, and z min and max fields before clicking the button.

> If you want to create a downsampled version of the geometry, type a downscaling factor into the Downscaling Factor field. This factor works on all three axis, for example if you use a downscaling factor of 2, all three axis will be halved in size meaning the total volume will be reduced by 8.

## Make Geometries

This makes 3D files that can be opened up both with this program, and also other popular 3D programs such as Blender.

All you have to do, is select the model output .h5 file you created in the Auto-Label page, select whether you want to generate Meshs, Point Clouds, or both using the check boxes, and click the "Make Geometries" button.
The program will then make .ply files for the meshes and point clouds. the filename will be <theH5FilenameYouMadeItFrom>_instance or semantic_pointCloud or mesh_index if it is semantic_.ply
They will have the same name as the model output file and be in the same location, but will have a number representing what layer of the prediction they came from if the prediction is a semantic prediction.
Each layer represents a different class output. They are seperated so you can make them different colors when you visualize them.

> This process may take a while, but the program will let you know when it is completely finished.

> Please also be aware that the generated Geometries may require addition manual adjustment to show properly in 3D programs.

## Get Model Output Stats

All you have to do, is select the model output .h5 file you created in the Auto-Label page, pick a filename with the "Choose File" button, and click the Get Model Output Stats button.

The stats will be printed into the text box once they are calculated (this could take a while for semantic, but should be very quick for instance). 

A more detailed .csv file for each instance will also be generated in your designated folder entered above.

<!-- - Image Index, only a column on 2D models, is the index of the image that is being measured
- Plane Index, 0 is typically background and not measured, 1 is the first organelle in a semantic model, 2 is the second, etc
- Area, only a column on 2D models, is a measured area
- Volume, only for 3D models is a measured volume

> Currently, since instance only supports one organelle at a time, just outputs a list of measured volumes in one row -->
