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
from tqdm import tqdm

import config
from dataset.uv_dataset import UVDataset, UVDatasetSH
from model.pipeline import PipeLine, PipeLineSH,PipeLineSHMaskChannel
from loss import PerceptualLoss


parser = argparse.ArgumentParser()
parser.add_argument('--texturew', type=int, default=config.TEXTURE_W)
parser.add_argument('--textureh', type=int, default=config.TEXTURE_H)
parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
parser.add_argument('--use_pyramid', type=bool, default=config.USE_PYRAMID)
parser.add_argument('--view_direction', type=bool, default=config.VIEW_DIRECTION)
parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
parser.add_argument('--checkpoint', type=str, default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
parser.add_argument('--logdir', type=str, default=config.LOG_DIR, help='directory to save logs')
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

    dataset = UVDatasetSH(args.data+'/train/', args.train, args.croph, args.cropw, args.view_direction)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    test_dataset = UVDatasetSH(args.data+'/test/', args.train, args.croph, args.cropw, args.view_direction)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_step = 0

    if args.load:
        print('Loading Saved Model')
        model = torch.load(os.path.join(args.checkpoint, args.load))
        step = args.load_step
    else:
        model = PipeLineSHMaskChannel(args.texturew, args.textureh, args.texture_dim, args.use_pyramid, args.view_direction)
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
        {'params': model.unet.parameters(), 'lr': 0.1 * args.lr}],
        betas=betas, eps=args.eps)
    model = model.to('cuda')
    # criterion = nn.L1Loss()
    criterion = PerceptualLoss()
    m_criterion = nn.BCEWithLogitsLoss()

    print('Training started')
    for i in range(1, 1+args.epoch):
        print('Epoch {}'.format(i))
        adjust_learning_rate(optimizer, i, args.lr)

        model.train()
        torch.set_grad_enabled(True)

        for samples in tqdm(dataloader):
            
            images, uv_maps, extrinsics, gt_masks, sh, forward = samples
            # print(np.unique(gt_masks[0, :, :, :].numpy()))
            step += images.shape[0]
            optimizer.zero_grad()
            RGB_texture, preds,masks = model(uv_maps.cuda(), extrinsics.cuda())
            mask_sigmoid = nn.Sigmoid()(masks)
            if i>=50:
                mask_sigmoid[mask_sigmoid >= 0.5] = 1
                mask_sigmoid[mask_sigmoid <0.5 ] = 0
            # sh = sh.cuda()*mask_sigmoid
            # preds = preds * mask_sigmoid
            preds = preds * sh.cuda()
            
            preds_final = torch.zeros((preds.shape[0], 3, preds.shape[2], preds.shape[3]), dtype=torch.float, device='cuda:0')
            
            for z in range(0, 25):
                preds_final[:, 0, :, :] += preds[:, z*3, :, :]
                preds_final[:, 1, :, :] += preds[:, z*3+1, :, :]
                preds_final[:, 2, :, :] += preds[:, z*3+2, :, :]

            
            preds_final *= mask_sigmoid
            preds_final = preds_final.clamp(0, 1)
            # images_mask = images*gt_masks

            if i <= 50:
                m_loss = m_criterion(masks,gt_masks.cuda())
                loss = 0.75*criterion.calculate(preds_final, images.cuda()) + 0.25*m_loss
            else:
                loss = criterion.calculate(preds_final, images.cuda())
            
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss_mask', m_loss.item(), step)
            writer.add_scalar('train/loss', loss.item(), step)
        
        model.eval()
        torch.set_grad_enabled(False)
        test_loss = 0
        all_preds = []
        all_gt = []
        all_uv = []
        all_error = []
        all_gt_masks = []
        all_masks = []
        all_error_masks = []
        for samples in tqdm(test_dataloader):
            images, uv_maps, extrinsics, gt_masks, sh, forward = samples

            RGB_texture, preds,masks = model(uv_maps.cuda(), extrinsics.cuda())
            # print(masks)
            mask_sigmoid = nn.Sigmoid()(masks)
            if i>=50:
                mask_sigmoid[mask_sigmoid >= 0.5] = 1
                mask_sigmoid[mask_sigmoid <0.5 ] = 0
            # preds = preds*mask_sigmoid
            # sh = sh.cuda()*mask_sigmoid

            preds_final = torch.zeros((preds.shape[0], 3, preds.shape[2], preds.shape[3]), dtype=torch.float, device='cuda:0')
            preds = preds * sh.cuda()

            for z in range(0, 25):
                preds_final[:, 0, :, :] += preds[:, z*3, :, :]
                preds_final[:, 1, :, :] += preds[:, z*3+1, :, :]
                preds_final[:, 2, :, :] += preds[:, z*3+2, :, :]
            
            preds_final *= mask_sigmoid
            preds_final = preds_final.clamp(0, 1)
            # images_mask = images*gt_masks           
            
            if i <= 50:
                m_loss = m_criterion(masks,gt_masks.cuda())
                loss = 0.75*criterion.calculate(preds_final, images.cuda()) + 0.25*m_loss
            else:
                loss = criterion.calculate(preds_final, images.cuda())
            test_loss += loss.item()

            output = np.clip(preds_final[0, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0/2.2)
            output = output * 255.0
            output = output.astype(np.uint8)

            gt = np.clip(images[0, :, :, :].numpy(), 0, 1) ** (1.0/2.2)
            gt = gt * 255.0
            gt = gt.astype(np.uint8)

            error = np.abs(gt-output)

            out_masks = np.clip(masks[0, :, :, :].detach().cpu().numpy(), 0, 1)
            # print(np.max(out_masks),np.min(out_masks))
            # out_masks += 0.5
            out_masks = out_masks * 255.0
            out_masks = out_masks.astype(np.uint8)
            

            gt_masks1 = np.clip(gt_masks[0, :, :, :].numpy(), 0, 1) ** (1.0/2.2)
            gt_masks1 = gt_masks1 * 255.0
            gt_masks1 = gt_masks1.astype(np.uint8)
            

            mask_error = np.abs(gt_masks1-out_masks)

            all_preds.append(output)
            all_gt.append(gt)
            all_masks.append(out_masks)
            all_error.append(error)
            all_gt_masks.append(gt_masks1)
            all_error_masks.append(mask_error)

        # ridx = random.randint(0, len(test_dataset)-1)
        ridx = i%len(test_dataset)

        writer.add_scalar('test/loss', test_loss/len(test_dataset), test_step)
        writer.add_image('test/output', all_preds[ridx], test_step)
        writer.add_image('test/gt', all_gt[ridx], test_step)
        writer.add_image('test/masks', all_masks[ridx], test_step)
        writer.add_image('test/error_masks', all_error_masks[ridx], test_step)
        writer.add_image('test/error', all_error[ridx], test_step)
        writer.add_image('test/gt_masks', all_gt_masks[ridx], test_step)
        print(np.unique(all_masks[ridx]))
        print(np.unique(all_gt_masks[ridx]))
        test_step += 1

        # save checkpoint
        
        if i % args.epoch_per_checkpoint == 0:
            print('Saving checkpoint')
            torch.save(model, args.checkpoint+time_string+'/epoch_{}.pt'.format(i))

if __name__ == '__main__':
    main()
