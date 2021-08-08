import argparse, cv2
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
from tqdm import tqdm

import config
from dataset.uv_dataset_new import UVDataset
from model.pipeline_new import PipeLine
from loss import PerceptualLoss

parser = argparse.ArgumentParser()
parser.add_argument('--texturew', type=int, default=config.TEXTURE_W)
parser.add_argument('--textureh', type=int, default=config.TEXTURE_H)
parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
parser.add_argument('--use_pyramid', type=bool, default=config.USE_PYRAMID)
parser.add_argument('--view_direction', type=bool, default=config.VIEW_DIRECTION)
parser.add_argument('--data', type=str, default='/media/dhawals/Data/DATASETS/new_pipeline/WOMAN/B,Diff,Cm/',
                    help='directory to data')
parser.add_argument('--checkpoint', type=str, default='/media/dhawals/Data/DATASETS/new_pipeline/WOMAN/checkpoints/',
                    help='directory to save checkpoint')
parser.add_argument('--logdir', type=str, default='/media/dhawals/Data/DATASETS/new_pipeline/WOMAN/checkpoints/',
                    help='directory to save checkpoint')
parser.add_argument('--output_dir', type=str, default='/media/dhawals/Data/DATASETS/new_pipeline/WOMAN/checkpoints/',
                    help='directory to save log etc.')
parser.add_argument('--train', default=config.TRAIN_SET)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--cropw', type=int, default=config.CROP_W)
parser.add_argument('--croph', type=int, default=config.CROP_H)
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
parser.add_argument('--betas', type=str, default=config.BETAS)
parser.add_argument('--l2', type=str, default=config.L2_WEIGHT_DECAY)
parser.add_argument('--eps', type=float, default=config.EPS)
parser.add_argument('--load', type=str, default=config.LOAD)
parser.add_argument('--load_step', type=int, default=config.LOAD_STEP)
parser.add_argument('--epoch_per_checkpoint', type=int, default=5)
parser.add_argument('--samples', type=int, default=config.SAMPLES)
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, original_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= 5:
        lr = original_lr * 0.2 * epoch
    elif epoch < 50:
        lr = original_lr
    elif epoch < 100:
        lr = 0.1 * original_lr
    else:
        lr = 0.01 * original_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    named_tuple = time.localtime()
    time_string = time.strftime("%m_%d_%Y_%H_%M", named_tuple)
    log_dir = os.path.join(args.logdir, time_string)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = tensorboardX.SummaryWriter(logdir=log_dir)

    checkpoint_dir = args.checkpoint + time_string
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    dataset = UVDataset(args.data + '/train/', args.train, args.croph, args.cropw, view_direction=args.view_direction,
                        samples=args.samples)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)

    test_dataset = UVDataset(args.data + '/test/', args.train, args.croph, args.cropw,
                             view_direction=args.view_direction, samples=args.samples)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_step = 0

    if args.load:
        print('Loading Saved Model')
        model = torch.load(os.path.join(args.checkpoint, args.load))
        step = args.load_step
    else:
        # model = PipeLineTex(args.texturew, args.textureh, args.texture_dim, args.use_pyramid, samples=args.samples,
        #                  view_direction=args.view_direction)
        model = PipeLine(args.texturew, args.textureh, args.texture_dim, args.use_pyramid, samples=args.samples,
                         view_direction=args.view_direction)
        step = 0

    l2 = args.l2.split(',')
    l2 = [float(x) for x in l2]
    betas = args.betas.split(',')
    betas = [float(x) for x in betas]
    betas = tuple(betas)

    optimizer = Adam([
        {'params': model.texture.layer1, 'weight_decay': l2[0], 'lr': args.lr},
        {'params': model.texture.layer2, 'weight_decay': l2[1], 'lr': args.lr},
        {'params': model.texture.layer3, 'weight_decay': l2[2], 'lr': args.lr},
        {'params': model.texture.layer4, 'weight_decay': l2[3], 'lr': args.lr},
        {'params': model.unet.parameters(), 'lr': 0.1 * args.lr},
        {'params': model.albedo_tex.layer1, 'lr': args.lr}],
        betas=betas, eps=args.eps)
    # optimizer = Adam([
    #     {'params': model.albedo_tex.layer1, 'lr': args.lr}],
    #     betas=betas, eps=args.eps)

    model = model.to('cuda')
    criterion = nn.L1Loss()

    print('Training started', flush=True)
    for i in range(1, 1 + args.epoch):
        print()
        adjust_learning_rate(optimizer, i, args.lr)

        model.train()
        torch.set_grad_enabled(True)

        for samples in tqdm(dataloader, desc=f'Train: Epoch {i}'):
            # images, uv_maps, wi, cos_t, envmap = samples
            images, uv_maps, extrinsics, wi, cos_t, envmap = samples

            step += images.shape[0]
            optimizer.zero_grad()

            # RGB_texture, RGB_texture_proj, preds, cos_t = model(wi.cuda(), cos_t.cuda(), envmap.cuda(), uv_maps.cuda())
            RGB_texture, RGB_texture_proj, preds, preds_, cos_t = model(wi.cuda(), cos_t.cuda(), envmap.cuda(), uv_maps.cuda(),
                                                                        extrinsics.cuda())
            
            loss = criterion(preds, images.cuda())
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', loss.item(), step)

        model.eval()
        torch.set_grad_enabled(False)
        test_loss = 0
        all_preds = []
        all_gt = []
        all_uv = []
        all_albedo = []
        all_T = []
        idx = 0
        for samples in tqdm(test_dataloader, desc=f'Test: Epoch {i}'):
            if idx == 20:
                break

            # images, uv_maps, wi, cos_t, envmap = samples
            images, uv_maps, extrinsics, wi, cos_t, envmap = samples

            # RGB_texture, RGB_texture_proj, preds, cos_t = model(wi.cuda(), cos_t.cuda(), envmap.cuda(), uv_maps.cuda())
            RGB_texture, RGB_texture_proj, preds, preds_, cos_t = model(wi.cuda(), cos_t.cuda(), envmap.cuda(), uv_maps.cuda(),
                                                                        extrinsics.cuda())

            loss = criterion(preds, images.cuda())
            test_loss += loss.item()

            output = np.clip(preds[0, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0 / 2.2)
            output = output * 255.0
            output = output.astype(np.uint8)

            T = np.clip(preds_[0, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0 / 2.2)
            T = T * 255.0
            T = T.astype(np.uint8)

            gt = np.clip(images[0, :, :, :].numpy(), 0, 1) ** (1.0 / 2.2)
            gt = gt * 255.0
            gt = gt.astype(np.uint8)

            albedo = np.clip(RGB_texture[0, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0 / 2.2)
            albedo = albedo * 255.0
            albedo = albedo.astype(np.uint8)

            uv_maps = uv_maps.permute(0, 3, 1, 2)
            uv = np.clip(uv_maps[0, :, :, :].numpy(), 0, 1)
            uv_final = np.ones((3, uv.shape[1], uv.shape[2]))
            uv_final[0:2, :, :] = uv
            uv_final = uv_final * 255.0
            uv_final = uv_final.astype(np.uint8)

            all_preds.append(output)
            all_gt.append(gt)
            all_uv.append(uv_final)
            all_albedo.append(albedo)
            all_T.append(T)

            idx += 1

        ridx = i % 20

        writer.add_scalar('test/loss', test_loss / 20, test_step)
        writer.add_image('test/output', all_preds[ridx], test_step)
        writer.add_image('test/gt', all_gt[ridx], test_step)
        writer.add_image('test/albedo', all_albedo[ridx], test_step)
        writer.add_image('test/transport', all_T[ridx], test_step)

        test_step += 1

        # save checkpoint
        if i % args.epoch_per_checkpoint == 0:
            print('Saving checkpoint')
            torch.save(model, args.checkpoint + time_string + '/epoch_{}.pt'.format(i))

            albedo = np.transpose(all_albedo[0], (1, 2, 0))
            albedo = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)
            cv2.imwrite('%s/dr_log/%s.png' % (args.output_dir, str(test_step).zfill(5)), albedo)


if __name__ == '__main__':
    main()
