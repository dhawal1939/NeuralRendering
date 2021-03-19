# STEPS TO TRAIN ON SYNTHETIC DATA (TAKING MONKEY_SYNTHETIC AS EXAMPLE)

scene_gt.xml: GT geometry and GT materials
scene_dr_colmap_geometry.xml: Colmap geometry and DR material
scene_dr_perfect_geometry.xml: GT geometry and DR material

- SAVE THE ENVIRONMAP AS 'envmap.exr'

- MAKE 'video_frames' directory

- MAKE 'video_frames_test' directory

- MAKE 'video_frames_mask' directory

- MAKE 'video_frames_test_mask' directory

- MAKE 'video_frames_extrinsics' directory

- MAKE 'video_frames_test_extrinsics' directory

- MAKE 'colmap_capture' directory

- MAKE 'colmap_output' directory

[ GENERATES TRAIN AND TEST VIDEO FRAMES AND THEIR EXTRINSICS, FOLDERS REQUIRED ARE 'video_frames' 'video_frames_extrinsics' 'video_frames_test' 'video_frames_test_extrinsics' ]
- python data/synthetic_generate.py --scene_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/scene_gt.xml --vector_scale 3.5 --output_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/ --img_width 512 --img_height 512 --spp 32

[ GENERATES IMAGES FOR USE WITH COLMAP, SAVES TO 'colmap_output' ]
- python data/synthetic_generate_colmap.py --scene_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/scene_gt.xml --vector_scale 3.5 --output_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/ --img_width 512 --img_height 512 --spp 32

[ GENERATES GT IMAGES FOR EVALUATION (POST NETWORK TRAINING), WITH ORIGINAL MATERIAL ]
- python data/synthetic_generate_eval.py --scene_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/scene_gt.xml --output_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/evaluation_frames/material_original/ --spp 50

[ GENERATES GT IMAGES FOR EVALUATION (POST NETWORK TRAINING), WITH A NEW MATERIAL ]
- python data/synthetic_generate_eval.py --scene_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/scene_material_1_gt.xml --output_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/evaluation_frames/material_1/ --spp 50

- RUN COLMAP MANUAL RECON ON 'colmap_capture' DIRECTORY
  - Follow https://journal.missiondata.com/lab-notes-a-recipe-for-using-photogrammetry-to-create-3d-model-7c256e38b1fb
  - Set shared intrinsics
  - Set feature matching to sequential
  - Follow steps till fusion

- DUPLICATION OF RECONSTRUCTION FOR TEST REGISTRATION
  - CREATE A FOLDER NAMES 'colmap_output_test' INSIDE THE 'colmap_output' FOLDER
  - COPY THE CONTENTS OF 'colmap_output' FOLDER INSIDE THE 'colmap_output_test' FOLDER

- DUPLICATION OF RECONSTRUCTION FOR EVAL REGISTRATION
  - CREATE A FOLDER NAMES 'colmap_output_evaluation' INSIDE THE 'colmap_output' FOLDER
  - COPY THE CONTENTS OF 'colmap_output' FOLDER INSIDE THE 'colmap_output_evaluation' FOLDER

[ REGSITER IMAGES FROM 'video_frames' TO THE RECONSTRUCTED GEOMETRY ]
- ./register_images.sh NEW_IMAGES="video_frames" IMAGE_LIST="dense/0/image-list.txt" RELATIVE_PATH="../../../../" BASE_PATH=/media/aakash/wd1/DATASETS/ DATASET=MONKEY_SYNTHETIC COLMAP_PATH='/media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/colmap_output'

[ REGSITER IMAGES FROM 'video_frames_test' TO THE RECONSTRUCTED GEOMETRY ]
- ./register_images.sh NEW_IMAGES="video_frames_test" IMAGE_LIST="dense/0/image-list.txt" RELATIVE_PATH="../../../../../" BASE_PATH=/media/aakash/wd1/DATASETS/ DATASET=MONKEY_SYNTHETIC COLMAP_PATH='/media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/colmap_output/colmap_output_test/'

[ REGISTER IMAGES FROM 'material_original' DIRECTORY INSIDE 'evaluation_frames' TO THE RECONSTRUCTED GEOMETRY ]
- ./register_images.sh NEW_IMAGES="evaluation_frames/material_original/" IMAGE_LIST="dense/0/image-list.txt" RELATIVE_PATH="../../../../../" BASE_PATH=/media/aakash/wd1/DATASETS/ DATASET=MONKEY_SYNTHETIC COLMAP_PATH='/media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/colmap_output/colmap_output_evaluation/'

- CLEAN MESH IN MESHLAB
  - Quadratic edge collapse decimation
  - Remove non-manifold edges/vertices
  - Close holes

[ UV UNWRAPS THE RECONSTRUCTED MESH ]
- python data/uv_redner.py --input_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/cleaned_colmap.obj --output_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/unwrapped_colmap.obj

- FINAL EXPORT TO PLY
  - Export as .ply with Y up and -Z forward

- MAKE 'optimized_textures' DIRECTORY
  - Make a subdirectory 'colmap_geometry'
  - Make another subdirectory 'perfect_geometry'

- MAKE 'dr_tensorboard' DIRECTORY
  - Make a subdirectory 'colmap_geometry'
  - Make another subdirectory 'perfect_geometry'

- MAKE 'dr_log' DIRECTORY
  - Make a subdirectory 'colmap_geometry'
  - Make another subdirectory 'perfect_geometry'

[ CREATE MASK FOR TRAINING VIDEO FRAMES ]
- python create_synthetic_frame_mask.py --data_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/ --frames_dir video_frames --output_dir video_frames_mask

[ CREATE MASK FOR TEST VIDEO FRAMES ]
- python create_synthetic_frame_mask.py --data_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/ --frames_dir video_frames_test --output_dir video_frames_test_mask

[ OPTIMIZES FOR MATERIAL FROM PERFECT GEOMETRY ] NOT READY
- python data/synthetic_dr_perfect.py --scene_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/scene_dr_perfect_geometry.xml --data_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/ --epochs 20

[ OPTIMIZES FOR MATERIAL FROM COLMAP GEOMETRY ]
- python data/synthetic_dr_colmap.py --scene_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/scene_dr_colmap_geometry.xml --data_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/ --image_list_txt /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/colmap_output/dense/0/image-list.txt --epochs 20

[ GERERATES TRAINING DATA FOR GT GEOMETRY AND GT MATERIAL, SAVES TO 'output_dir' ] NOT READY
- python data/synthetic_extract_perfect.py --scene_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/scene_gt.xml --data_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/ --output_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/B/

[ GERERATES TRAINING DATA FOR GT GEOMETRY AND OPTIMIZED MATERIAL, SAVES TO 'output_dir' ] NOT READY
- python data/synthetic_extract_perfect.py --scene_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/scene_dr_perfect_geometry.xml --data_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/ --output_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/B,Diff/

[ GENERATES EVALUATION DATA FOR PERFECT GEOMETRY AND OPTIMIZED MATERIAL, SAVES TO 'output_dir' ]
- python data/synthetic_extract_perfect_eval.py --scene_file /media/aakash/wd1/DATASETS/BUDDHA_SYNTHETIC_RED/scene_dr_perfect_geometry.xml --data_dir /media/aakash/wd1/DATASETS/BUDDHA_SYNTHETIC_RED/ --output_dir /media/aakash/wd1/DATASETS/BUDDHA_SYNTHETIC_RED/B,Diff/evaluation/material_1/

[ GERERATES TRAINING DATA FOR COLMAP GEOMETRY AND OPTIMIZED MATERIAL, SAVES TO 'output_dir' ]
- python data/synthetic_extract_colmap.py --scene_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/scene_dr_colmap_geometry.xml --data_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/ --output_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/B,Diff,Cm/ --train_image_list_txt /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/colmap_output/dense/0/image-list.txt --test_image_list_txt /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/colmap_output/colmap_output_test/dense/0/image-list.txt

[ GERERATES EVALUATION DATA FOR COLMAP GEOMETRY AND OPTIMIZED MATERIAL, SAVES TO 'output_dir' ]
- python data/synthetic_extract_colmap_eval.py --scene_file /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/scene_material_1.xml --data_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/evaluation_frames/material_1/ --colmap_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/colmap_output/colmap_output_evaluation/ --output_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/B,Diff,Cm/eval_material_1/ --image_list_txt /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/colmap_output/colmap_output_evaluation/dense/0/image-list.txt

[ TRAIN OUR NETWORK (1000 EPOCHS, SAVE PER 5 EPOCHS) ]
- python train_sh.py --data /ssd_scratch/cvit/aakash.kt/B,Diff,Cm/ --checkpoint /scratch/darthgera123/monkey/checkpoints/ --logdir /scratch/darthgera123/monkey/logs/ --epoch 1000 --epoch_per_checkpoint 5

[ EVALUATE NETWORK ON COLMAP GEOMETRY ]
- python render_network_synthetic.py --data /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/B,Diff,Cm/eval_material_1/ --checkpoint /media/aakash/wd1/epoch_1000.pt --output_dir /media/aakash/wd1/DATASETS/MONKEY_SYNTHETIC/B,Diff,Cm/eval_material_1/network_output/








# STEPS TO RUN THE PIPELINE FOR REAL SCENES (TAKING FISH AS EXAMPLE)

- SAVE THE ENVIRONMAP AS 'envmap.exr'

- MAKE 'video_frames' directory

- MAKE 'video_frames_test' directory

- MAKE 'video_frames_mask' directory

- MAKE 'video_frames_test_mask' directory

- MAKE 'colmap_capture' directory

- MAKE 'colmap_output' directory

[ SAMPLE TRAINING FRAMES ]
- python data/extract_frames.py --video_file /media/aakash/wd1/DATASETS/new_videos/FISH.mp4 --output_dir /media/aakash/wd1/DATASETS/FISH/video_frames/ --skip 3 --img_width 960 --img_height 540

[ SAMPLE TEST/EVAL FRAMES ]
- python data/extract_frames.py --video_file /media/aakash/wd1/DATASETS/new_videos/FISH_TEST.mp4 --output_dir /media/aakash/wd1/DATASETS/FISH/video_frames_test/ --skip 3 --img_width 960 --img_height 540

[  SAMPLE COLMAP FRAMES ]
- python data/sample_frames.py --input_dir /media/aakash/wd1/DATASETS/FISH/video_frames --output_dir /media/aakash/wd1/DATASETS/FISH/colmap_capture/ --skip 12 --img_width 960 --img_height 540

- RUN COLMAP MANUAL RECON ON 'colmap_capture' DIRECTORY
  - Follow https://journal.missiondata.com/lab-notes-a-recipe-for-using-photogrammetry-to-create-3d-model-7c256e38b1fb
  - Set shared intrinsics
  - Set feature matching to sequential
  - Follow steps till fusioN

- DUPLICATION OF RECONSTRUCTION FOR TEST REGISTRATION
  - CREATE A FOLDER NAMES 'colmap_output_test' INSIDE THE 'colmap_output' FOLDER
  - COPY THE CONTENTS OF 'colmap_output' FOLDER INSIDE THE 'colmap_output_test' FOLDER

[ REGSITER IMAGES FROM 'video_frames' TO THE RECONSTRUCTED GEOMETRY ]
- ./register_images.sh NEW_IMAGES="video_frames" IMAGE_LIST="dense/0/image-list.txt" RELATIVE_PATH="../../../../" BASE_PATH=/media/aakash/wd1/DATASETS/ DATASET=FISH COLMAP_PATH='/media/aakash/wd1/DATASETS/FISH/colmap_output'

[ REGSITER IMAGES FROM 'video_frames_test' TO THE RECONSTRUCTED GEOMETRY ]
- ./register_images.sh NEW_IMAGES="video_frames_test" IMAGE_LIST="dense/0/image-list.txt" RELATIVE_PATH="../../../../../" BASE_PATH=/media/aakash/wd1/DATASETS/ DATASET=FISH COLMAP_PATH='/media/aakash/wd1/DATASETS/FISH/colmap_output/colmap_output_test/'

- CLEAN MESH IN MESHLAB
  - Quadratic edge collapse decimation
  - Remove non-manifold edges/vertices
  - Close holes
  - Export as obj (FISH.obj)

[ UV UNWRAPS THE RECONSTRUCTED MESH ]
- python data/uv_redner.py --input_file /media/aakash/wd1/DATASETS/FISH/cleaned.obj --output_file /media/aakash/wd1/DATASETS/FISH/unwrapped.obj

- FINAL EXPORT TO PLY
  - Export as .ply with Y up and -Z forward

- MAKE MITSUBA SCENE FILES
  - Make 'scene_dr.xml' mitsuba scene description file

- MAKE 'optimized_textures' DIRECTORY

- MAKE 'dr_tensorboard' DIRECTORY

- MAKE 'dr_log' DIRECTORY

[ CREATE MASK ]
[IN CASE OF HUBER PHOTO SCAN USE THE FOLLOWING]
[ TRAIN ]
- python threshold.py --th .9 --input_dir /media/aakash/wd1/DATASETS/HUBER_PHOTOSCAN/video_frames_test --output_dir /media/aakash/wd1/DATASETS/HUBER_PHOTOSCAN/video_frames_test_mask/
[ TEST ]
- python threshold.py --th .9 --input_dir /media/aakash/wd1/DATASETS/HUBER_PHOTOSCAN/video_frames --output_dir /media/aakash/wd1/DATASETS/HUBER_PHOTOSCAN/video_frames_mask/

[ IN CASE OF HEAD THE WHOLE SCENE IS POINT OF INTEREST ]
- python head_masks.py --input_dir /media/aakash/wd1/DATASETS/HEAD/video_frames --output_dir /media/aakash/wd1/DATASETS/HEAD/video_frames_mask/

- python head_masks.py --input_dir /media/aakash/wd1/DATASETS/HEAD/video_frames_test --output_dir /media/aakash/wd1/DATASETS/HEAD/video_frames_test_mask/


[ CREATE MASK FOR TRAINING VIDEO FRAMES ]
- python create_frame_mask.py --data_dir /media/aakash/wd1/DATASETS/FISH --frames_dir video_frames --output_dir video_frames_mask --model_path U2Net/saved_models --model_name u2net

[ CREATE MASK FOR TEST VIDEO FRAMES ]
- python create_frame_mask.py --data_dir /media/aakash/wd1/DATASETS/FISH --frames_dir video_frames_test --output_dir video_frames_test_mask --model_path U2Net/saved_models --model_name u2net

[ OPTIMIZES FOR MATERIAL FROM COLMAP GEOMETRY ]
- python data/real_dr.py --scene_file /media/aakash/wd1/DATASETS/FISH/scene_dr.xml --data_dir /media/aakash/wd1/DATASETS/FISH/ --image_list_txt /media/aakash/wd1/DATASETS/FISH/colmap_output/dense/0/image-list.txt --epochs 20 --img_width 480 --img_height 270

- MAKE TRAIN DIRECTORY (EXAMPLE 'B,Diff,Cm')
  - Make 'train' and 'test' subdirectories
  - In both 'train' and 'test', create 7 directories
    - extrinsics
    - frames
    - forward
    - uv
    - uv_png
    - mask
    - sh
  - Make 'evaluation' subdirectory with the following sub-subdirectories
    - extrinsics
    - frames
    - forward
    - uv
    - uv_png
    - mask
    - sh
    - env_sh

[ GERERATES TRAINING DATA, SAVES TO 'output_dir' ]
- python data/real_extract.py --scene_file /media/aakash/wd1/DATASETS/FISH/scene_dr.xml --data_dir /media/aakash/wd1/DATASETS/FISH/ --output_dir /media/aakash/wd1/DATASETS/FISH/B,Diff,Cm/ --train_image_list_txt /media/aakash/wd1/DATASETS/FISH/colmap_output/dense/0/image-list.txt --test_image_list_txt /media/aakash/wd1/DATASETS/FISH/colmap_output/colmap_output_test/dense/0/image-list.txt --img_width 960 --img_height 540

[ TRAIN NETWORK ]
- python train_sh.py --data /ssd_scratch/cvit/darthgera123/FISH/B,Diff,Cm/ --checkpoint /scratch/darthgera123/FISH/checkpoints/ --logdir /scratch/darthgera123/FISH/logs/ --epoch 50 --epoch_per_checkpoint 5

[ EXTRACT NEW INFERENCE DATA ]
- python data/real_extract_eval.py --scene_file /media/aakash/wd1/DATASETS/FISH/scene_material_1.xml --data_dir /media/aakash/wd1/DATASETS/FISH/video_frames_test/ --data_mask_dir /media/aakash/wd1/DATASETS/FISH/video_frames_test_mask/ --output_dir /media/aakash/wd1/DATASETS/FISH/B,Diff,Cm/evaluation/material_1/ --image_list_txt /media/aakash/wd1/DATASETS/FISH/colmap_output/colmap_output_test/dense/0/image-list.txt --colmap_dir /media/aakash/wd1/DATASETS/FISH/colmap_output/colmap_output_test/ --img_width 960 --img_height 540






# STEPS TO RUN THE PIPELINE (TAKING BUDDHA AS EXAMPLE)

- python data/extract_frames.py --video_file /media/aakash/wd1/DATASETS/BUDDHA/video.mp4 --output_dir /media/aakash/wd1/DATASETS/BUDDHA/video_frames/

- python data/sample_frames.py --input_dir /media/aakash/wd1/DATASETS/BUDDHA/video_frames/ --output_dir /media/aakash/wd1/DATASETS/BUDDHA/colmap_capture/ --skip 3

- RUN COLMAP AUTOMATIC RECON
  - Set shared intrinsics
  - Set quality to medium
  - Set image type to "Video Frames"

- CLEAN MESH IN MESHLAB
  - Quadratic edge collapse decimation
  - Remove non-manifold edges/vertices
  - Export as .obj

- python data/uv_redner.py --input_file /media/aakash/wd1/DATASETS/BUDDHA/cleaned.obj --output_file /media/aakash/wd1/DATASETS/BUDDHA/unwrapped.obj

- TRIANGULATE FACES IN BLENDER
  - Export as .ply with Y up and -Z forward
