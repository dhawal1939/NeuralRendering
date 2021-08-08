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
from dataset.uv_dataset import UVDatasetSH
from model.pipeline import PipeLineSH
from loss import PerceptualLoss


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
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)

    test_dataset = UVDatasetSH(args.data+'/test/', args.train, args.croph, args.cropw, args.view_direction)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_step = 0

    if args.load:
        print('Loading Saved Model')
        model = torch.load(os.path.join(args.checkpoint, args.load))
        step = args.load_step
    else:
        model = PipeLineSH(args.texturew, args.textureh, args.texture_dim, args.use_pyramid, args.view_direction)
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
    criterion = nn.L1Loss()
    # criterion = PerceptualLoss()

    print('Training started')
    for i in range(1, 1+args.epoch):
        print('Epoch {}'.format(i))
        adjust_learning_rate(optimizer, i, args.lr)

        model.train()
        torch.set_grad_enabled(True)

        for samples in tqdm(dataloader):
            if args.view_direction:
                images, env, forward, uv_maps, extrinsics, masks, sh = samples
                env = env.cuda()

                step += images.shape[0]
                optimizer.zero_grad()
                RGB_texture, preds = model(uv_maps.cuda(), extrinsics.cuda())
            else:
                images, uv_maps, masks, sh, forward = samples
                
                step += images.shape[0]
                optimizer.zero_grad()
                RGB_texture, preds = model(uv_maps.cuda())
            
            # create a volume of the same size of preds by replicating envSH.npy
            # multiply each channel with mask
            # take l1 loss with preds output of the model
            sh = sh.view(-1, 9, 3, sh.shape[2], sh.shape[3])
            preds = preds.view(-1, 9, 3, preds.shape[2], preds.shape[3])

            preds = preds * sh.cuda()
            preds_final = torch.sum(preds, dim=1, keepdim=False)
            # preds_final = torch.zeros((preds.shape[0], 3, preds.shape[2], preds.shape[3]), dtype=torch.float, device='cuda:0')
            # preds = preds * sh.cuda()
            # for z in range(0, 25):
            #     preds_final[:, 0, :, :] += preds[:, z*3, :, :]
            #     preds_final[:, 1, :, :] += preds[:, z*3+1, :, :]
            #     preds_final[:, 2, :, :] += preds[:, z*3+2, :, :]

            # loss1 = criterion(RGB_texture, rgb.cuda())
            # loss2 = criterion(preds_final, images.cuda())
            # loss = loss1 + loss2
            # loss.backward()
            # optimizer.step()

            # loss = criterion(preds_final, images.cuda())
            # preds_final = nn.Sigmoid()(preds_final)
            # preds_final = torch.clamp(preds_final,0,1)
            # preds_final = preds_final*masks.cuda()
            # loss = criterion.calculate(preds_final, images.cuda())
            loss = criterion(preds_final, images.cuda())
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', loss.item(), step)
        
        model.eval()
        torch.set_grad_enabled(False)
        test_loss = 0
        all_preds = []
        all_gt = []
        all_uv = []
        all_error = []
        all_forward = []
        for samples in tqdm(test_dataloader):
            images, env, forward, uv_maps, extrinsics, masks, sh = samples
            env = env.cuda()

            RGB_texture, preds = model(uv_maps.cuda(), extrinsics.cuda())

            sh = sh.view(-1, 9, 3, sh.shape[2], sh.shape[3])
            preds = preds.view(-1, 9, 3, preds.shape[2], preds.shape[3])

            preds = preds * sh.cuda()
            preds_final = torch.sum(preds, dim=1, keepdim=False)
            
            # preds = preds*masks.cuda()
            # sh = sh.cuda()*masks.cuda()

            # preds_final = torch.zeros((preds.shape[0], 3, preds.shape[2], preds.shape[3]), dtype=torch.float, device='cuda:0')
            # preds = preds * sh.cuda()

            # for z in range(0, 25):
            #     preds_final[:, 0, :, :] += preds[:, z*3, :, :]
            #     preds_final[:, 1, :, :] += preds[:, z*3+1, :, :]
            #     preds_final[:, 2, :, :] += preds[:, z*3+2, :, :]
            
            # loss1 = criterion(RGB_texture, rgb.cuda())
            # loss2 = criterion(preds_final, images.cuda())
            # loss = loss1 + loss2
            # test_loss += loss.item()

            # loss = criterion(preds_final, images.cuda())
            # preds_final = nn.Sigmoid()(preds_final)
            # preds_final = torch.clamp(preds_final,0,1)
            # preds_final = preds_final*masks.cuda()
            # loss = criterion.calculate(preds_final, images.cuda())
            loss = criterion(preds_final, images.cuda())
            test_loss += loss.item()

            # env = np.clip(env[0, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0/2.2)

            output = np.clip(preds_final[0, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0/2.2)
            output = output * 255.0
            output = output.astype(np.uint8)

            gt = np.clip(images[0, :, :, :].numpy(), 0, 1) ** (1.0/2.2)
            gt = gt * 255.0
            gt = gt.astype(np.uint8)

            error = np.abs(gt-output)

            for_rend = np.clip(forward[0, :, :, :].numpy(), 0, 1) ** (1.0/2.2)
            for_rend = for_rend * 255.0
            for_rend = for_rend.astype(np.uint8)

            # uv_maps = uv_maps.permute(0, 3, 1, 2)
            # uv = np.clip(uv_maps[0, :, :, :].numpy(), 0, 1)
            # uv_final = np.ones((3, uv.shape[1], uv.shape[2]))
            # uv_final[0:2, :, :] = uv
            # uv_final = uv_final * 255.0
            # uv_final = uv_final.astype(np.uint8)

            all_preds.append(output)
            all_gt.append(gt)
            # all_uv.append(uv_final)
            all_forward.append(for_rend)
            all_error.append(error)

        # ridx = random.randint(0, len(test_dataset)-1)
        ridx = i%len(test_dataset)

        writer.add_scalar('test/loss', test_loss/len(test_dataset), test_step)
        writer.add_image('test/output', all_preds[ridx], test_step)
        writer.add_image('test/gt', all_gt[ridx], test_step)
        writer.add_image('test/forward', all_forward[ridx], test_step)
        # writer.add_image('test/uv', all_uv[ridx], test_step)
        writer.add_image('test/error', all_error[ridx], test_step)

        test_step += 1

        # save checkpoint
        
        if i % args.epoch_per_checkpoint == 0:
            print('Saving checkpoint')
            torch.save(model, args.checkpoint+time_string+'/epoch_{}.pt'.format(i))

if __name__ == '__main__':
    main()
