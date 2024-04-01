# Output Tools

This screen has a few tools that let you take the output model .h5 and further process it. Make Geometries lets you make 3D geometry files from your model output. These can be visualized with this program, or opened up in other popular softwares like Blender. The Get Model Output Stats section lets you measure volumes from your model output.

![Output Tools Screen](https://github.com/ajbrookhouse/WSU_PlantBio_ML/blob/main/screenshots/outputtoolTab.png)

Model Output: Chose an .h5 file of interest that was generated during the auto label process.  

CSV Output: Provide a recognizable name for the statistical data output file to be generated. 

Meshes: highlight this box if you want to receive a 3D mesh that can be imported into Blender, Amira, and other 3D image processing software. 

Point Clouds: highlight this box if you want to receive a 3D point cloud that can be imported into Blender, Amira, and other 3D image processing software  

Make Geometries: Click this button if you want to make either a mesh or a point cloud or both. The program will generate .ply files for the meshes and point clouds that start with the name of the original h5 file followed by .ply 

> Normally, keep the downscaling factor at 1. But if you want to create a down sampled version of the geometry, type a downscaling factor into the Downscaling Factor field. This factor works on all three axes, for example if you use a downscaling factor of 2, all three axes will be halved in size meaning the total volume will be reduced by a factor of 8. 

Get Model Output Stats: Click this button if you want to receive a CSV file containing statistical data. The stats will also be printed into the text box once they are calculated (this could take a while for semantic data but should be very quick for instance data). 
