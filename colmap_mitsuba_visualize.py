import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_rgb')

from mitsuba.core import Thread, Color3f,Float, Bitmap, Struct
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam
import time
from poses.read_model import camera_pose
import cv2
import numpy as np 
import os
import pyexr


BASE_DIR = '/media/aakash/wd1/DATASETS/PINECONE'

scene_file = os.path.join(BASE_DIR,'scene_dr.xml')
diffuse_init_file = BASE_DIR+'diffuse_opt.png'
alpha_init_file = BASE_DIR+'alpha_opt.png'
mesh_file = BASE_DIR+'triangulation.ply'
focal_length = '28.0mm' # Set to focal length of iPhone X, wide angle camera

IMG_WIDTH = int(504.0)
IMG_HEIGHT = int(378)

Thread.thread().file_resolver().append(BASE_DIR)

def load_image(image_name):
    global BASE_DIR

    path = BASE_DIR+'/colmap_output/dense/0/images/'
    print(len(os.listdir(path)))
    image_ref = cv2.imread(os.path.join(path,image_name))
    
    if image_ref.shape[0] > image_ref.shape[1]:
        image_ref = cv2.resize(image_ref,(IMG_HEIGHT,IMG_WIDTH))
    else:
        image_ref = cv2.resize(image_ref,(IMG_WIDTH,IMG_HEIGHT))

    image_ref = cv2.cvtColor(image_ref,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    image_ref = image_ref**2.2

    return image_ref

if __name__ == '__main__':

    for i in range(0,99):
        print("Image No:",i)

        image_name = 'images'+str(i)+'.JPG'

        image_ref = load_image(image_name)
        h,w,_ = image_ref.shape
        if h>w:
            image_ref = cv2.rotate(image_ref, cv2.cv2.ROTATE_90_CLOCKWISE)
        pose_path = BASE_DIR+'/colmap_output/'
        p = camera_pose(pose_path,image_name)
        pose = ' '.join([str(elem) for elem in p])

        scene = load_file(scene_file, integrator='path', focal_length=focal_length, poses=pose, envmap_pose=pose, \
                spp=10, width=IMG_WIDTH, height=IMG_HEIGHT)
        
        sensor = scene.sensors()[0]
        scene.integrator().render(scene, sensor)
        film = sensor.film()

        img = film.bitmap(raw=True).convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
        image_np = np.array(img)

        final = np.zeros((image_ref.shape[0], image_ref.shape[1]*2, image_ref.shape[2]), dtype=np.float)
        final[:, :image_ref.shape[1], :] = image_ref
        final[:, image_ref.shape[1]:, :] = image_np

        final = ( final ** (1.0/2.2) ) * 255.0
        final = final.astype(np.uint8)

        cv2.imwrite(BASE_DIR+'/dr_log/%s.png' % (str(i).zfill(5)), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))


        