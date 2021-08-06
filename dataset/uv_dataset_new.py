import numpy as np
import os, cv2
from PIL import Image,ImageOps
from torch.utils.data import Dataset
import torch

from util import augment_new, augment_eval,augment_og, augment_center_crop,augment_center_crop_mask

class UVDataset(Dataset):

    def __init__(self, dir, idx_list, H, W, view_direction=False):
        self.idx_list = sorted(os.listdir(dir+'/frames/'))
        for i in range(len(self.idx_list)):
            self.idx_list[i] = self.idx_list[i].replace('.png', '').replace('image', '')

        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction

        self.envmap = cv2.cvtColor( cv2.imread('%s/envmap.jpg' % dir), cv2.COLOR_BGR2RGB ).astype(np.float)
        self.envmap = (self.envmap / 255.0) ** (2.2)

    def __len__(self):
        return len(self.idx_list)
    
    def sample_hemishpere(self, n):
        s = np.random.random((n, 3))*2.0 - 1.0
        s[:, 2] = np.abs(s[:, 2]+0.01)
        s /= np.linalg.norm(s, axis=1, keepdims=True)

        return torch.from_numpy(s).type(torch.float)

    def convert_spherical(self, wi):
        theta = np.arccos(wi[:, :, 2:3])
        phi = np.arctan2(wi[:, :, 1:2], wi[:, :, 0:1])

        return torch.from_numpy( np.concatenate((theta, phi), axis=2) ).type(torch.float)

    def sample_envmap(self, wi):
        wi[:, :, 0] = wi[:, :, 0] / np.pi * self.envmap.shape[0]
        wi[:, :, 1] = wi[:, :, 1] / (2*np.pi) * self.envmap.shape[1]
        wi = wi.type(torch.uint8)

        return self.envmap[ wi[:, :, 0].reshape(-1), wi[:, :, 1].reshape(-1), : ].reshape(wi.shape[0], 3, -1)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, 'frames/'+self.idx_list[idx]+'.png'), 'r')

        transform = np.load(os.path.join(self.dir, 'transform/'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(transform)
        transform[nan_pos] = 0

        uv_map = np.load(os.path.join(self.dir, 'uv/'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        uv_map = uv_map[:, :, :2]

        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')

        img, uv_map, transform = augment_new(img, uv_map, transform, self.crop_size)
        img = img ** (2.2)

        extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))

        transform = torch.reshape(transform, (-1, 3, 3))
        wi = self.sample_hemishpere(10)
        wi = np.tile(wi, (transform.shape[0], 1, 1))
        cos_t = torch.from_numpy(wi[:, :, 2]).type(torch.float)
        wi = np.transpose( transform @ np.transpose(wi, (0, 2, 1)), (0, 2, 1) )
        wi = self.convert_spherical(wi)

        sampled_env = self.sample_envmap(wi)

        wi = wi.reshape(self.crop_size[0], self.crop_size[1], 10, 2).permute(3, 2, 0, 1)
        cos_t = cos_t.reshape(self.crop_size[0], self.crop_size[1], 10).permute(2, 0, 1)
        sampled_env = sampled_env.reshape(self.crop_size[0], self.crop_size[1], 10, 3)
        sampled_env = np.transpose(sampled_env, (3, 2, 0, 1))

        return img.type(torch.float), uv_map.type(torch.float), torch.from_numpy(extrinsics).type(torch.float), \
                wi, cos_t, torch.from_numpy(sampled_env).type(torch.float)
