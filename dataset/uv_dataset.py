import numpy as np
import os
from PIL import Image,ImageOps
from torch.utils.data import Dataset
import torch

from util import augment, augment_eval,augment_og, augment_center_crop,augment_center_crop_mask

class UVDatasetSHEval(Dataset):

    def __init__(self, dir, idx_list, H, W, view_direction=False):
        self.idx_list = os.listdir(dir+'/frames/')
        for i in range(len(self.idx_list)):
            self.idx_list[i] = self.idx_list[i].replace('.png', '').replace('image', '')

        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, 'frames/'+self.idx_list[idx]+'.png'), 'r')
        mask = Image.open(os.path.join(self.dir, 'mask/'+self.idx_list[idx]+'.png'), 'r')
        mask = ImageOps.grayscale(mask)
        forward = Image.open(os.path.join(self.dir, 'forward/'+self.idx_list[idx]+'.png'), 'r')

        sh = np.transpose( np.load(os.path.join(self.dir, 'sh/'+self.idx_list[idx]+'.npy')), (2, 0, 1) )
        env_sh = np.transpose( np.load(os.path.join(self.dir, 'env_sh/'+self.idx_list[idx]+'.npy')), (2, 0, 1) )

        uv_map = np.load(os.path.join(self.dir, 'uv/'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        uv_map = uv_map[:, :, :2]
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')
        img, uv_map, sh, env_sh, mask, forward = augment_eval(img, mask, uv_map, sh, env_sh, self.crop_size, forward)
        img = img ** (2.2)
        forward = forward ** (2.2)

        if self.view_direction:
            extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))
            return img.type(torch.float), uv_map.type(torch.float), torch.from_numpy(extrinsics).type(torch.float), \
                    mask.type(torch.float), sh.type(torch.float), env_sh.type(torch.float), forward.type(torch.float)
        else:
            return img, uv_map, mask, sh, forward

class UVDatasetSHEvalReal(Dataset):

    def __init__(self, dir, idx_list, H, W, view_direction=False):
        self.idx_list = os.listdir(dir+'/frames/')
        for i in range(len(self.idx_list)):
            self.idx_list[i] = self.idx_list[i].replace('.png', '').replace('image', '')

        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, 'frames/'+self.idx_list[idx]+'.png'), 'r')
        mask = Image.open(os.path.join(self.dir, 'mask/'+self.idx_list[idx]+'.png'), 'r')
        mask = ImageOps.grayscale(mask)
        forward = Image.open(os.path.join(self.dir, 'forward/'+self.idx_list[idx]+'.png'), 'r')
        sh = np.transpose( np.load(os.path.join(self.dir, 'sh/'+self.idx_list[idx]+'.npy')), (2, 0, 1) )
        uv_map = np.load(os.path.join(self.dir, 'uv/'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        uv_map = uv_map[:, :, :2]
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')
        img, uv_map, sh, mask, forward = augment_center_crop(img, mask, uv_map, sh, self.crop_size, forward)
        img = img ** (2.2)
        forward = forward ** (2.2)

        # print('frames/'+self.idx_list[idx]+'.png')
        # print(img.shape)
        # print(sh.shape)
        # print(uv_map.shape)
        # print(mask.shape)

        if self.view_direction:
            # view_map = np.load(os.path.join(self.dir, 'view_normal/'+self.idx_list[idx]+'.npy'))
            extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))
            return img.type(torch.float), uv_map.type(torch.float), torch.from_numpy(extrinsics).type(torch.float), \
                    mask.type(torch.float), sh.type(torch.float), forward.type(torch.float)
        else:
            return img, uv_map, mask, sh, forward

class UVDatasetSHAakash(Dataset):

    def __init__(self, dir, idx_list, H, W, view_direction=False):
        self.idx_list = os.listdir(dir+'/frames/')
        for i in range(len(self.idx_list)):
            self.idx_list[i] = self.idx_list[i].replace('.png', '').replace('image', '')

        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, 'frames/'+self.idx_list[idx]+'.png'), 'r')
        mask = Image.open(os.path.join(self.dir, 'mask/'+self.idx_list[idx]+'.png'), 'r')
        mask = ImageOps.grayscale(mask)
        mask = mask.point(lambda p: p > 0.8*255 and 255)
        forward = Image.open(os.path.join(self.dir, 'forward/'+self.idx_list[idx]+'.png'), 'r')
        sh = np.transpose( np.load(os.path.join(self.dir, 'sh/'+self.idx_list[idx]+'.npy')), (2, 0, 1) )
        uv_map = np.load(os.path.join(self.dir, 'uv/'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        uv_map = uv_map[:, :, :2]
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')
        img, uv_map, sh, mask, forward = augment(img, mask, uv_map, sh, self.crop_size, forward)
        img = img ** (2.2)
        forward = forward ** (2.2)

        # print('frames/'+self.idx_list[idx]+'.png')
        # print(img.shape)
        # print(sh.shape)
        # print(uv_map.shape)
        # print(mask.shape)

        if self.view_direction:
            # view_map = np.load(os.path.join(self.dir, 'view_normal/'+self.idx_list[idx]+'.npy'))
            extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))
            return img.type(torch.float), uv_map.type(torch.float), torch.from_numpy(extrinsics).type(torch.float), \
                    mask.type(torch.float), sh.type(torch.float), forward.type(torch.float)
        else:
            return img, uv_map, mask, sh, forward

class UVDatasetSH(Dataset):

    def __init__(self, dir, idx_list, H, W, view_direction=False):
        self.idx_list = os.listdir(dir+'/frames/')
        for i in range(len(self.idx_list)):
            self.idx_list[i] = self.idx_list[i].replace('.png', '').replace('image', '')

        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, 'frames/'+self.idx_list[idx]+'.png'), 'r')
        mask = Image.open(os.path.join(self.dir, 'mask/'+self.idx_list[idx]+'.png'), 'r')
        mask = ImageOps.grayscale(mask)
        mask = mask.point(lambda p: p > 0.8*255 and 255)
        forward = Image.open(os.path.join(self.dir, 'forward/'+self.idx_list[idx]+'.png'), 'r')
        sh = np.transpose( np.load(os.path.join(self.dir, 'sh/'+self.idx_list[idx]+'.npy')), (2, 0, 1) )
        uv_map = np.load(os.path.join(self.dir, 'uv/'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        uv_map = uv_map[:, :, :2]
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')
        img, uv_map, sh, mask, forward = augment(img, mask, uv_map, sh, self.crop_size, forward)
        img = img ** (2.2)
        forward = forward ** (2.2)

        # print('frames/'+self.idx_list[idx]+'.png')
        # print(img.shape)
        # print(sh.shape)
        # print(uv_map.shape)
        # print(mask.shape)

        if self.view_direction:
            # view_map = np.load(os.path.join(self.dir, 'view_normal/'+self.idx_list[idx]+'.npy'))
            extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))
            return img.type(torch.float), uv_map.type(torch.float), torch.from_numpy(extrinsics).type(torch.float), \
                    mask.type(torch.float), sh.type(torch.float), forward.type(torch.float)
        else:
            return img, uv_map, mask, sh, forward

class UVDataset(Dataset):

    def __init__(self, dir, idx_list, H, W, view_direction=False):
        self.idx_list = os.listdir(dir+'/frames/')
        for i in range(len(self.idx_list)):
            self.idx_list[i] = self.idx_list[i].replace('.png', '')
            
        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, 'frames/'+self.idx_list[idx]+'.png'), 'r')
        uv_map = np.load(os.path.join(self.dir, 'uv/'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        uv_map = uv_map[:, :, :2]
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')
        img, uv_map, mask = augment_og(img, uv_map, self.crop_size)
        img = img ** (2.2)
        
        if self.view_direction:
            # view_map = np.load(os.path.join(self.dir, 'view_normal/'+self.idx_list[idx]+'.npy'))
            extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))
            return img.type(torch.float), uv_map.type(torch.float), torch.from_numpy(extrinsics).type(torch.float), \
                mask.type(torch.float)
        else:
            return img, uv_map, mask

class UVDatasetMask(Dataset):

    def __init__(self, dir, idx_list, H, W, view_direction=False):
        self.idx_list = sorted(os.listdir(dir+'/mask/'))
        for i in range(len(self.idx_list)):
            self.idx_list[i] = self.idx_list[i].replace('.png', '')
            
        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, 'frames/'+self.idx_list[idx]+'.png'), 'r')
        uv_map = np.load(os.path.join(self.dir, 'uv/'+self.idx_list[idx]+'.npy'))
        mask = Image.open(os.path.join(self.dir, 'mask/'+self.idx_list[idx]+'.png'), 'r')
        mask = ImageOps.grayscale(mask)
        mask = mask.point(lambda p: p > 0.8*255 and 255)
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        uv_map = uv_map[:, :, :2]
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')
        img, mask, uv_map = augment_center_crop_mask(img, mask, uv_map, self.crop_size)
        img = img ** (2.2)
        
        if self.view_direction:
            # view_map = np.load(os.path.join(self.dir, 'view_normal/'+self.idx_list[idx]+'.npy'))
            extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))
            # return img.type(torch.float), uv_map.type(torch.float), torch.from_numpy(extrinsics).type(torch.float), \
            #     mask.type(torch.float)
            return uv_map.type(torch.float), torch.from_numpy(extrinsics).type(torch.float), \
                mask.type(torch.float)
        else:
            return uv_map, mask