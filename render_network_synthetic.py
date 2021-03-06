import argparse
import numpy as np
import os
import random
import tensorboardX
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from dataset.uv_dataset import UVDataset, UVDatasetSHEval
from model.pipeline import PipeLine, PipeLineSH

import cv2, imageio

from PIL import Image
import scipy.io as io
import external_sh_func as esh

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--texturew', type=int, default=config.TEXTURE_W)
    parser.add_argument('--textureh', type=int, default=config.TEXTURE_H)
    parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
    parser.add_argument('--use_pyramid', type=bool, default=config.USE_PYRAMID)
    parser.add_argument('--view_direction', type=bool, default=config.VIEW_DIRECTION)
    parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
    parser.add_argument('--lif_checkpoint', type=str, default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
    parser.add_argument('--mask_checkpoint', type=str, default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
    parser.add_argument('--logdir', type=str, default=config.LOG_DIR, help='directory to save checkpoint')
    parser.add_argument('--train', default=config.TRAIN_SET)
    parser.add_argument('--epoch', type=int, default=config.EPOCH)
    parser.add_argument('--cropw', type=int, default=config.CROP_W)
    parser.add_argument('--croph', type=int, default=config.CROP_H)
    parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--betas', type=str, default=config.BETAS)
    parser.add_argument('--l2', type=str, default=config.L2_WEIGHT_DECAY)
    parser.add_argument('--eps', type=float, default=config.EPS)
    parser.add_argument('--load', type=str, default=config.LOAD)
    parser.add_argument('--load_step', type=int, default=config.LOAD_STEP)
    parser.add_argument('--epoch_per_checkpoint', type=int, default=config.EPOCH_PER_CHECKPOINT)

    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--pixel_x', type=int, default=-1)
    parser.add_argument('--pixel_y', type=int, default=-1)

    args = parser.parse_args()

    dataset = UVDatasetSHEval(args.data, args.train, args.croph, args.cropw, args.view_direction)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model_lif = torch.load(args.lif_checkpoint)
    model_lif = model_lif.to('cuda')
    model_lif.eval()

    model_mask = torch.load(args.mask_checkpoint)
    model_mask = model_mask.to('cuda')
    model_mask.eval()

    torch.set_grad_enabled(False)

    for idx, samples in enumerate(dataloader):
        print(idx)
        images, uv_maps, extrinsics, masks, sh, env_sh, forward = samples

        RGB_texture_lif, preds = model_lif(uv_maps.cuda(), extrinsics.cuda())
        RGB_texture_masks, masks = model_mask(uv_maps.cuda(), extrinsics.cuda())

        mask_sigmoid = nn.Sigmoid()(masks)
        mask_sigmoid[mask_sigmoid >= 0.5] = 1
        mask_sigmoid[mask_sigmoid < 0.5 ] = 0

        # gt_env = env_sh[0, :, args.pixel_y, args.pixel_x].cpu().numpy()
        # gt_env = gt_env.reshape(-1, 3)

        # f_sh = sh[0, :, args.pixel_y, args.pixel_x].cpu().numpy()
        # f_sh = f_sh.reshape(-1, 3)

        sh = sh.view(-1, 25, 3, sh.shape[2], sh.shape[3])
        preds_ = preds.view(-1, 25, 3, preds.shape[2], preds.shape[3])

        preds_ = preds_ * sh.cuda()
        preds_final = torch.sum(preds_, dim=1, keepdim=False)
        preds_final = torch.clamp(preds_final, 0, 1)   
        
        preds_final *= mask_sigmoid
        preds_final = preds_final.clamp(0, 1)

        mask_sigmoid = mask_sigmoid.cpu().numpy()

        for j in range(0, preds_final.shape[0]):
            output = np.clip(preds_final[j, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0/2.2)
            output = output * 255.0
            output = output.astype(np.uint8)
            output = np.transpose(output, (1, 2, 0))

            gt = np.clip(images[j, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0/2.2)
            gt *= mask_sigmoid[j, :, :, :]
            gt = gt * 255.0
            gt = gt.astype(np.uint8)
            gt = np.transpose(gt, (1, 2, 0))

            for_rend = np.clip(forward[j, :, :, :].cpu().numpy(), 0, 1) ** (1.0/2.2)
            for_rend *= mask_sigmoid[j, :, :, :]
            for_rend = for_rend * 255.0
            for_rend = for_rend.astype(np.uint8)
            for_rend = np.transpose(for_rend, (1, 2, 0))

            if args.pixel_x != -1 and args.pixel_y != -1:
                network_env = preds[j, :, args.pixel_y, args.pixel_x].cpu().numpy()
                network_env = network_env.reshape(-1, 3)

                re_network_env = esh.reconstImageFromSH(network_env, 100, 200, isClip = True)
                re_network_env = np.clip(re_network_env, 0, 1)
                re_network_env = (255 *(np.clip(re_network_env, 0, 1) ** (1.0/2.2) ) ).astype(np.uint8)
                re_network_env = Image.fromarray(re_network_env)
                re_network_env.save(args.output_dir+'/%s_network_env.png' % str(idx).zfill(5))

                # re_gt_env = esh.reconstImageFromSH(gt_env, 100, 200, isClip = True)
                # re_gt_env = np.clip(re_gt_env, 0, 1)
                # re_gt_env = (255 *(np.clip(re_gt_env, 0, 1) ** (1/2.2) ) ).astype(np.uint8)
                # re_gt_env = Image.fromarray(re_gt_env)
                # re_gt_env.save(args.output_dir+'/%s_gt_env.png' % str(idx).zfill(5))
                
                # re_f_sh = esh.reconstImageFromSH(f_sh, 100, 200, isClip = True)
                # re_f_sh = np.clip(re_f_sh, 0, 1)
                # re_f_sh = (255 *(np.clip(re_f_sh, 0, 1) ** (1/2.2) ) ).astype(np.uint8)
                # re_f_sh = Image.fromarray(re_f_sh)
                # re_f_sh.save(args.output_dir+'/%s_f_env.png' % str(idx).zfill(5))

            cv2.imwrite(args.output_dir+'/%s_output.png' % str(idx).zfill(5), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            cv2.imwrite(args.output_dir+'/%s_gt.png' % str(idx).zfill(5), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
            cv2.imwrite(args.output_dir+'/%s_forward.png' % str(idx).zfill(5), cv2.cvtColor(for_rend, cv2.COLOR_RGB2BGR))
