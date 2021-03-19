import torch, os, sys, cv2, json, argparse, random, glob, struct, math, time
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import scipy.ndimage as ndimage
import torchvision.transforms as transforms
import numpy as np 
import os.path as osp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sh
import pyexr

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_rgb')

from mitsuba.core import Vector3f, Float, Float32, Float64, Thread, xml, Spectrum, depolarize, RayDifferential3f, Frame3f, warp, Bitmap, Struct
from mitsuba.core import math as m_math
from mitsuba.core.xml import load_string, load_file
from mitsuba.render import BSDF, Emitter, BSDFContext, BSDFSample3f, SurfaceInteraction3f, ImageBlock, register_integrator, register_bsdf, MonteCarloIntegrator, SamplingIntegrator, has_flag, BSDFFlags, DirectionSample3f

def sph_dir(theta, phi):
    """ Map spherical to Euclidean coordinates """
    st, ct = ek.sincos(theta)
    sp, cp = ek.sincos(phi)
    return Vector3f(cp*st, sp*st, ct)

def sph_convert(v):
    x2 = ek.pow(v.x, 2)
    y2 = ek.pow(v.y, 2)
    z2 = ek.pow(v.z, 2)

    r = ek.sqrt(x2+y2+z2)
    phi = ek.atan2(v.y, v.x)
    theta = ek.atan2(ek.sqrt(x2+y2), v.z)

    return r, theta, phi


if __name__ == '__main__':

    # Load desired BSDF plugin
    bsdf = load_string("""<bsdf version='2.0.0' type='roughconductor'>
                            <float name="alpha" value="0.5"/>
                        </bsdf>""")

    # Create a (dummy) surface interaction to use for the evaluation
    si = SurfaceInteraction3f()

    # Specify an incident direction with 45 degrees elevation
    si.wi = sph_dir(ek.pi * 45 / 180, 0.0)

    # Create grid in spherical coordinates and map it onto the sphere
    res = 300
    theta_o, phi_o = ek.meshgrid(
        ek.linspace(Float, 0,     ek.pi,     res),
        ek.linspace(Float, 0, 2 * ek.pi, 2 * res)
    )
    wo = sph_dir(theta_o, phi_o)
    N = float( wo.numpy().shape[0] )

    # _, theta_o, phi_o = sph_convert(wo)

    # Evaluate the whole array (18000 directions) at once
    values = bsdf.eval(BSDFContext(), si, wo)
    # values = values.numpy()
    # values = np.ones(values.shape, dtype=np.float)
    # values = Vector3f(values)

    # SH computation
    y_0_0 = (4*np.pi/N) * np.sum( sh.y_0_0(values, theta_o, phi_o).numpy(), axis=0, keepdims=True )
    
    y_1_n1 = (4*np.pi/N) * np.sum( sh.y_1_n1(values, theta_o, phi_o).numpy(), axis=0, keepdims=True )
    y_1_0 = (4*np.pi/N) * np.sum( sh.y_1_0(values, theta_o, phi_o).numpy(), axis=0, keepdims=True )
    y_1_p1 = (4*np.pi/N) * np.sum( sh.y_1_p1(values, theta_o, phi_o).numpy(), axis=0, keepdims=True )

    y_2_n2 = (4*np.pi/N) * np.sum( sh.y_2_n2(values, theta_o, phi_o).numpy(), axis=0, keepdims=True )
    y_2_n1 = (4*np.pi/N) * np.sum( sh.y_2_n1(values, theta_o, phi_o).numpy(), axis=0, keepdims=True )
    y_2_0 = (4*np.pi/N) * np.sum( sh.y_2_0(values, theta_o, phi_o).numpy(), axis=0, keepdims=True )
    y_2_p1 = (4*np.pi/N) * np.sum( sh.y_2_p1(values, theta_o, phi_o).numpy(), axis=0, keepdims=True )
    y_2_p2 = (4*np.pi/N) * np.sum( sh.y_2_p2(values, theta_o, phi_o).numpy(), axis=0, keepdims=True )

    # Reconstruct BRDF
    y_0_0 = np.repeat(y_0_0, int(N), axis=0)

    y_1_n1 = np.repeat(y_1_n1, int(N), axis=0)
    y_1_0 = np.repeat(y_1_0, int(N), axis=0)
    y_1_p1 = np.repeat(y_1_p1, int(N), axis=0)

    y_2_n2 = np.repeat(y_2_n2, int(N), axis=0)
    y_2_n1 = np.repeat(y_2_n1, int(N), axis=0)
    y_2_0 = np.repeat(y_2_0, int(N), axis=0)
    y_2_p1 = np.repeat(y_2_p1, int(N), axis=0)
    y_2_p2 = np.repeat(y_2_p2, int(N), axis=0)

    values_recon = sh.y_0_0( Vector3f(y_0_0), theta_o, phi_o) + \
                    sh.y_1_n1( Vector3f(y_1_n1), theta_o, phi_o) + sh.y_1_0( Vector3f(y_1_0), theta_o, phi_o) + \
                    sh.y_1_p1( Vector3f(y_1_p1), theta_o, phi_o) + sh.y_2_n2( Vector3f(y_2_n2), theta_o, phi_o) + \
                    sh.y_2_n1( Vector3f(y_2_n1), theta_o, phi_o) + sh.y_2_0( Vector3f(y_2_0), theta_o, phi_o) + \
                    sh.y_2_p1( Vector3f(y_2_p1), theta_o, phi_o) + sh.y_2_p2( Vector3f(y_2_p2), theta_o, phi_o)
    
    # Extract red channel of BRDF values and reshape into 2D grid
    values_r = np.array(values)[:, 0]
    values_r = values_r.reshape(2 * res, res).T

    # Plot values for spherical coordinates
    fig, ax = plt.subplots(2, figsize=(12, 7))

    im = ax[0].imshow(values_r, extent=[0, 2 * np.pi, np.pi, 0],
                cmap='jet', interpolation='bicubic')

    ax[0].set_xlabel(r'$\phi_o$', size=14)
    ax[0].set_xticks([0, np.pi, 2 * np.pi])
    ax[0].set_xticklabels(['0', '$\\pi$', '$2\\pi$'])
    ax[0].set_ylabel(r'$\theta_o$', size=14)
    ax[0].set_yticks([0, np.pi / 2, np.pi])
    ax[0].set_yticklabels(['0', '$\\pi/2$', '$\\pi$'])

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # Extract red channel of BRDF values and reshape into 2D grid
    values_r = np.array(values_recon)[:, 0]
    values_r = values_r.reshape(2 * res, res).T


    im = ax[1].imshow(values_r, extent=[0, 2 * np.pi, np.pi, 0],
                cmap='jet', interpolation='bicubic')

    ax[1].set_xlabel(r'$\phi_o$', size=14)
    ax[1].set_xticks([0, np.pi, 2 * np.pi])
    ax[1].set_xticklabels(['0', '$\\pi$', '$2\\pi$'])
    ax[1].set_ylabel(r'$\theta_o$', size=14)
    ax[1].set_yticks([0, np.pi / 2, np.pi])
    ax[1].set_yticklabels(['0', '$\\pi/2$', '$\\pi$'])

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # fig.savefig("bsdf_eval.jpg", dpi=150, bbox_inches='tight', pad_inches=0)
    plt.show()