# Output Tools

brief description

![Output Tools Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/outputToolsScreenshot.png)

# Make Geometries

This makes 3D files that can be opened up both with this program, and also other popular 3D programs such as Blender.

All you have to do, is select the model output .h5 file you created in the Auto-Label page, select whether you want to generate Meshs, Point Clouds, or both using the check boxes, and click the "Make Geometries" button.
The program will then make .ply files for the meshes, and .pcd files for the point clouds.
They will have the same name as the model output file and be in the same location, but will have a number representing what layer of the prediction they came from if the prediction is a semantic prediction.
Each layer represents a different class output. They are seperated so you can make them different colors when you visualize them.

> This process may take a while, but the program will let you know when it is completely finished.

# Get Model Output Stats

All you have to do, is select the model output .h5 file you created in the Auto-Label page, pick a filename with the "Create CSV" button, and click the Get Model Output Stats button.
The stats will be printed into the text box once they are calculated (this could take a while for semantic, but should be very quick for instance). They will also be written to the csv file that you picked. These measurements may need filtering.

- Image Index, only a column on 2D models, is the index of the image that is being measured
- Plane Index, 0 is typically background and not measured, 1 is the first organelle in a semantic model, 2 is the second, etc
- Area, only a column on 2D models, is a measured area
- Volume, only for 3D models is a measured volume

> Currently, since instance only supports one organelle at a time, just outputs a list of measured volumes in one row
