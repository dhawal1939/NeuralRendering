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
from dataset.uv_dataset import UVDataset, UVDatasetSH, UVDatasetSHEvalReal
from model.pipeline import PipeLine, PipeLineSH

import cv2

import external_sh_func as esh

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--texturew', type=int, default=config.TEXTURE_W)
    parser.add_argument('--textureh', type=int, default=config.TEXTURE_H)
    parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
    parser.add_argument('--use_pyramid', type=bool, default=config.USE_PYRAMID)
    parser.add_argument('--view_direction', type=bool, default=config.VIEW_DIRECTION)
    parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
    parser.add_argument('--checkpoint', type=str, default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
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
    parser.add_argument('--pixel_x', type=int, default=0)
    parser.add_argument('--pixel_y', type=int, default=0)

    args = parser.parse_args()

    dataset = UVDatasetSHEvalReal(args.data, args.train, args.croph, args.cropw, args.view_direction)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model = torch.load(args.checkpoint)
    model = model.to('cuda')
    model.eval()
    torch.set_grad_enabled(False)

    for idx, samples in enumerate(dataloader):
        print(idx)
        images, uv_maps, extrinsics, masks, sh, forward = samples

        RGB_texture, preds = model(uv_maps.cuda(), extrinsics.cuda())

        preds = preds*masks.cuda()
        sh = sh.cuda()*masks.cuda()

        preds_final = torch.zeros((preds.shape[0], 3, preds.shape[2], preds.shape[3]), dtype=torch.float, device='cuda:0')
        preds = preds * sh.cuda()
        for z in range(0, 25):
            preds_final[:, 0, :, :] += preds[:, z*3, :, :]
            preds_final[:, 1, :, :] += preds[:, z*3+1, :, :]
            preds_final[:, 2, :, :] += preds[:, z*3+2, :, :]

        output = np.clip(preds_final[0, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0/2.2)
        output = output * 255.0
        output = output.astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))

        gt = np.clip(images[0, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0/2.2)
        gt = gt * 255.0
        gt = gt.astype(np.uint8)
        gt = np.transpose(gt, (1, 2, 0))

        for_rend = np.clip(forward[0, :, :, :].numpy(), 0, 1) ** (1.0/2.2)
        for_rend *= masks[0, :, :, :].numpy()
        for_rend = for_rend * 255.0
        for_rend = for_rend.astype(np.uint8)
        for_rend = np.transpose(for_rend, (1, 2, 0))

        sh = np.clip(sh[0, :, :, :].cpu().numpy(), 0, 1) ** (1.0/2.2)
        sh *= masks[0, :, :, :].numpy()
        sh = sh * 255.0
        sh = sh.astype(np.uint8)
        sh = np.transpose(sh, (1, 2, 0))

        nt = np.clip(RGB_texture[0, :, :, :].cpu().numpy(), 0, 1) ** (1.0/2.2)
        nt *= masks[0, :, :, :].numpy()
        nt = nt * 255.0
        nt = nt.astype(np.uint8)
        nt = np.transpose(nt, (1, 2, 0))

        mask = np.clip(masks[0, :, :, :].cpu().numpy(), 0, 1) ** (1.0/2.2)
        mask = mask * 255.0
        mask = mask.astype(np.uint8)
        mask = np.transpose(mask, (1, 2, 0))
        mask = np.repeat(mask, 3, axis=2)

        uv = np.ones((uv_maps.shape[1], uv_maps.shape[2], 3), dtype=np.float)
        uv[:, :, :2] = np.clip(uv_maps[0, :, :, :].cpu().numpy(), 0, 1) ** (1.0/2.2)
        uv = uv * 255.0
        uv = uv.astype(np.uint8)

        cv2.imwrite(args.output_dir+'/%s_output.png' % str(idx).zfill(5), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        cv2.imwrite(args.output_dir+'/%s_gt.png' % str(idx).zfill(5), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
        cv2.imwrite(args.output_dir+'/%s_forward.png' % str(idx).zfill(5), cv2.cvtColor(for_rend, cv2.COLOR_RGB2BGR))
