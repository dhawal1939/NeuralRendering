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
from dataset.uv_dataset import UVDatasetSH, UVDatasetMask
from model.pipeline import PipeLineSH,PipeLineMask
from loss import PerceptualLoss


parser = argparse.ArgumentParser()
parser.add_argument('--texturew', type=int, default=config.TEXTURE_W)
parser.add_argument('--textureh', type=int, default=config.TEXTURE_H)
parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
parser.add_argument('--mask_texture_dim', type=int, default=config.TEXTURE_DIM)
parser.add_argument('--use_pyramid', type=bool, default=config.USE_PYRAMID)
parser.add_argument('--view_direction', type=bool, default=config.VIEW_DIRECTION)
parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
parser.add_argument('--checkpoint', type=str, default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
parser.add_argument('--logdir', type=str, default=config.LOG_DIR, help='directory to save logs')
parser.add_argument('--train', default=config.TRAIN_SET)
parser.add_argument('--epoch', type=int, default=config.EPOCH)
parser.add_argument('--mask_epoch', type=int, default=config.MASK_EPOCH)
parser.add_argument('--cropw', type=int, default=config.CROP_W)
parser.add_argument('--croph', type=int, default=config.CROP_H)
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
parser.add_argument('--betas', type=str, default=config.BETAS)
parser.add_argument('--l2', type=str, default=config.L2_WEIGHT_DECAY)
parser.add_argument('--eps', type=float, default=config.EPS)
parser.add_argument('--load_mask', type=str, default=config.LOAD)
parser.add_argument('--load_lif', type=str, default=config.LOAD)
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

    image_dataset = UVDatasetSH(args.data+'/train/', args.train, args.croph, args.cropw, args.view_direction)
    image_dataloader = DataLoader(image_dataset, batch_size=2, shuffle=True, num_workers=4)

    image_test_dataset = UVDatasetSH(args.data+'/test/', args.train, args.croph, args.cropw, args.view_direction)
    image_test_dataloader = DataLoader(image_test_dataset, batch_size=1, shuffle=True, num_workers=4)
    lif_test_step = 0

    mask_dataset = UVDatasetMask(args.data+'/train/', args.train, args.croph, args.cropw, args.view_direction)
    mask_dataloader = DataLoader(mask_dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    mask_test_dataset = UVDatasetMask(args.data+'/test/', args.train, args.croph, args.cropw, args.view_direction)
    mask_test_dataloader = DataLoader(mask_test_dataset, batch_size=1, shuffle=True, num_workers=4)
    mask_test_step = 0


    model_mask = PipeLineMask(256, 256, args.mask_texture_dim, args.use_pyramid, args.view_direction)
    mask_step = 0
    model_lif = PipeLineSH(args.texturew, args.textureh, args.texture_dim, args.use_pyramid, args.view_direction)
    lif_step = 0


    l2 = args.l2.split(',')
    l2 = [float(x) for x in l2]
    betas = args.betas.split(',')
    betas = [float(x) for x in betas]
    betas = tuple(betas)
    optimizer_mask = Adam([
        {'params': model_mask.texture.layer1, 'weight_decay': l2[0], 'lr': args.lr},
        {'params': model_mask.texture.layer2, 'weight_decay': l2[1], 'lr': args.lr},
        {'params': model_mask.texture.layer3, 'weight_decay': l2[2], 'lr': args.lr},
        {'params': model_mask.texture.layer4, 'weight_decay': l2[3], 'lr': args.lr},
        {'params': model_mask.unet.parameters(), 'lr': 0.1 * args.lr}],
        betas=betas, eps=args.eps)
    
    # optimizer_lif = Adam([
    #     {'params': model_lif.texture.layer1, 'weight_decay': l2[0], 'lr': args.lr},
    #     {'params': model_lif.texture.layer2, 'weight_decay': l2[1], 'lr': args.lr},
    #     {'params': model_lif.texture.layer3, 'weight_decay': l2[2], 'lr': args.lr},
    #     {'params': model_lif.texture.layer4, 'weight_decay': l2[3], 'lr': args.lr},
    #     {'params': model_lif.unet.parameters(), 'lr': 0.1 * args.lr}],
    #     betas=betas, eps=args.eps)
    optimizer_lif = Adam([
        {'params': model_lif.texture.parameters(), 'weight_decay': l2[1], 'lr': args.lr},
        {'params': model_lif.unet.parameters(), 'lr': 0.1 * args.lr},
        ],betas=betas, eps=args.eps)

    model_mask = torch.load('/scratch/aakash/new/WOMAN/checkpoints/05_30_2021_07_35/mask_epoch_10.pt')
    model_mask = model_mask.to('cuda')
    model_lif = model_lif.to('cuda')
    criterion_lif = PerceptualLoss()
    criterion_mask = nn.BCEWithLogitsLoss()

    # print('Mask Training started')
    # for i in range(1, 1+args.mask_epoch):
    #     print('Epoch {}'.format(i))

    #     model_mask.train()
    #     torch.set_grad_enabled(True)

    #     for samples in tqdm(mask_dataloader):
            
    #         uv_maps, extrinsics, gt_masks = samples
    #         mask_step += gt_masks.shape[0]
    #         optimizer_mask.zero_grad()
    #         RGB_texture, masks = model_mask(uv_maps.cuda(), extrinsics.cuda())
            
    #         m_loss = criterion_mask(masks,gt_masks.cuda())
    #         m_loss.backward()
            
    #         optimizer_mask.step()
    #         writer.add_scalar('train/loss_mask', m_loss.item(), mask_step)
            
    #     model_mask.eval()
    #     torch.set_grad_enabled(False)
    #     test_loss = 0
        
    #     all_gt_masks = []
    #     all_masks = []
    #     all_error_masks = []
        
    #     for samples in tqdm(mask_test_dataloader):
    #         uv_maps, extrinsics, gt_masks = samples

    #         RGB_texture, masks = model_mask(uv_maps.cuda(), extrinsics.cuda())
    #         m_loss = criterion_mask(masks,gt_masks.cuda())
    #         loss = m_loss
            
    #         test_loss += loss.item()

    #         out_masks = np.clip(masks[0, :, :, :].detach().cpu().numpy(), 0, 1)
    #         out_masks = out_masks * 255.0
    #         out_masks = out_masks.astype(np.uint8)
    #         all_masks.append(out_masks)
                

    #         gt_masks1 = np.clip(gt_masks[0, :, :, :].numpy(), 0, 1) ** (1.0/2.2)
    #         gt_masks1 = gt_masks1 * 255.0
    #         gt_masks1 = gt_masks1.astype(np.uint8)
    #         all_gt_masks.append(gt_masks1)
            
    #         mask_error = np.abs(gt_masks1-out_masks)
    #         all_error_masks.append(mask_error)

    #     ridx = i%len(mask_test_dataset)
    #     writer.add_scalar('test/mask_loss', test_loss/len(mask_test_dataset), mask_test_step)
    #     writer.add_image('test/masks', all_masks[ridx], mask_test_step)
    #     writer.add_image('test/error_masks', all_error_masks[ridx], mask_test_step)
    #     writer.add_image('test/gt_masks', all_gt_masks[ridx], mask_test_step)
    #     print(np.unique(all_masks[ridx]))
    #     print(np.unique(all_gt_masks[ridx]))
    #     mask_test_step += 1

    #     # save checkpoint
        
    #     if i % args.epoch_per_checkpoint == 0:
    #         print('Saving checkpoint')
    #         torch.save(model_mask, args.checkpoint+time_string+'/mask_epoch_{}.pt'.format(i))

    
    print("Output Training started")
    model_mask.eval()
    for i in range(1, 1+args.epoch):
        print('Epoch {}'.format(i))
        model_lif.train()
        torch.set_grad_enabled(True)

        for samples in tqdm(image_dataloader):
            
            images, uv_maps, extrinsics, gt_masks, sh  = samples
            lif_step += images.shape[0]
            optimizer_lif.zero_grad()
            
            RGB_texture_lif, preds = model_lif(uv_maps.cuda(), extrinsics.cuda())
            RGB_texture_masks, masks = model_mask(uv_maps.cuda(), extrinsics.cuda())
            
            mask_sigmoid = nn.Sigmoid()(masks).clone().detach()
            mask_sigmoid[mask_sigmoid >= 0.5] = 1
            mask_sigmoid[mask_sigmoid <0.5 ] = 0


            sh = sh.view(-1, 9, 3, sh.shape[2], sh.shape[3])
            preds = preds.view(-1, 9, 3, preds.shape[2], preds.shape[3])

            preds = preds * sh.cuda()
            preds_final = torch.sum(preds, dim=1, keepdim=False)
            preds_final = torch.clamp(preds_final, 0, 1)     

            preds_final *= mask_sigmoid
            preds_final = preds_final.clamp(0, 1)

            
            loss = criterion_lif.calculate(preds_final, images.cuda())            
            loss.backward()
            optimizer_lif.step()

            writer.add_scalar('train/loss', loss.item(), lif_step)
        
        model_lif.eval()
        torch.set_grad_enabled(False)
        test_loss = 0
        all_preds = []
        all_gt = []
        all_uv = []
        all_error = []
        all_gt_masks = []
        all_masks = []
        all_error_masks = []
        for samples in tqdm(image_test_dataloader):
            images, uv_maps, extrinsics, gt_masks, sh = samples

            RGB_texture_lif, preds = model_lif(uv_maps.cuda(), extrinsics.cuda())
            RGB_texture_masks, masks = model_mask(uv_maps.cuda(), extrinsics.cuda())
            
            mask_sigmoid = nn.Sigmoid()(masks)
            mask_sigmoid[mask_sigmoid >= 0.5] = 1
            mask_sigmoid[mask_sigmoid <0.5 ] = 0

            sh = sh.view(-1, 9, 3, sh.shape[2], sh.shape[3])
            preds = preds.view(-1, 9, 3, preds.shape[2], preds.shape[3])

            preds = preds * sh.cuda()
            preds_final = torch.sum(preds, dim=1, keepdim=False)
            preds_final = torch.clamp(preds_final, 0, 1)   
        
            preds_final *= mask_sigmoid
            preds_final = preds_final.clamp(0, 1)
 
            loss = criterion_lif.calculate(preds_final, images.cuda())
            
            test_loss += loss.item()

            output = np.clip(preds_final[0, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0/2.2)
            output = output * 255.0
            output = output.astype(np.uint8)
            all_preds.append(output)

            gt = np.clip(images[0, :, :, :].numpy(), 0, 1) ** (1.0/2.2)
            gt = gt * 255.0
            gt = gt.astype(np.uint8)
            all_gt.append(gt)

            error = np.abs(gt-output)
            all_error.append(error)

            # out_masks = np.clip(masks[0, :, :, :].detach().cpu().numpy(), 0, 1)
            # out_masks = out_masks * 255.0
            # out_masks = out_masks.astype(np.uint8)
            # all_masks.append(out_masks)
         
            # gt_masks1 = np.clip(gt_masks[0, :, :, :].numpy(), 0, 1) ** (1.0/2.2)
            # gt_masks1 = gt_masks1 * 255.0
            # gt_masks1 = gt_masks1.astype(np.uint8)
            # all_gt_masks.append(gt_masks1)
         
            # mask_error = np.abs(gt_masks1-out_masks)
            # all_error_masks.append(mask_error)

        # ridx = random.randint(0, len(test_dataset)-1)
        ridx = i%len(image_test_dataset)
        
        writer.add_scalar('test/loss', test_loss/len(image_test_dataset), lif_test_step)
        writer.add_image('test/output', all_preds[ridx], lif_test_step)
        writer.add_image('test/gt', all_gt[ridx], lif_test_step)
        writer.add_image('test/error', all_error[ridx], lif_test_step)
        
        lif_test_step += 1

        
        if i % args.epoch_per_checkpoint == 0:
            print('Saving checkpoint')
            torch.save(model_lif, args.checkpoint+time_string+'/lif_epoch_{}.pt'.format(i))

if __name__ == '__main__':
    main()
