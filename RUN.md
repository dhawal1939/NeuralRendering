# Appearance Editing of Captured Objects
Our method can edit the appearance of captured objects. This is a guide through different scripts and how to run them

## Basic Installation
+ mitsuba2 : Differentiable renderer for material optimization
+ colmap : Sfm software to recover geometry and poses
+ redner : UV maps are retrieved through redner. Although you can also use Blender smart uv unwrap to unwrap the geometry
+ meshlab : Used to remove the surface of the object as well as other extra points

## Geometry Recovery
### Synthetic: 
In this case we have the geometry and need to unwrap the geometry
### Real: 
We only have the images as input. To retrive the geometry we use COLMAP Sfm software. We first take a subset of images and retrieve geometry from it. We register the remaining images onto the retrieved geometry and get their poses as well.  
To generate geometry run:   
`data/colmap_reconstruction/reconstruct.sh <dataset>` 
To register images run:
`data/colmap_reconstruction/register_images.sh NEW_IMAGES=<remaining_frames> IMAGE_LIST=<path_to_list_of_new_images> RELATIVE_PATH=<relative_to_images_dir> BASE_PATH=<base_dataset_dir> DATASET=<dataset_name> COLMAP_PATH='<colmap_output_dir>'`   
If you are using GUI:
+  Follow https://journal.missiondata.com/lab-notes-a-recipe-for-using-photogrammetry-to-create-3d-model-7c256e38b1fb
+ Set shared intrinsics
+ (For manual as well) Set feature matching to sequential if frames are from a video
