import numpy as np
import os, cv2
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torch
import time

from util import augment_new, augment_eval, augment_og, augment_center_crop, augment_center_crop_mask


class UVDataset(Dataset):

    def __init__(self, dir, idx_list, H, W, samples=10, view_direction=False):
        self.idx_list = sorted(os.listdir(dir + '/frames/'))
        for i in range(len(self.idx_list)):
            self.idx_list[i] = self.idx_list[i].replace('.png', '').replace('image', '')

        self.tiny_number = 1e-8
        self.dir = dir
        self.samples = samples
        self.crop_size = (H, W)
        self.view_direction = view_direction

        self.envmap = cv2.cvtColor(cv2.imread('%s/envmap.jpg' % dir), cv2.COLOR_BGR2RGB).astype(np.float)
        self.envmap = (self.envmap / 255.0) ** (2.2)
        # self.envmap = torch.from_numpy(self.envmap).cuda()

    def __len__(self):
        return len(self.idx_list)

    def sample_hemishpere(self, n):
        eta_1, eta_2 = torch.rand(n, dtype=torch.float32).cuda(), torch.rand(n, dtype=torch.float32).cuda()

        z = eta_1 + self.tiny_number
        phi = 2 * np.pi * eta_2
        sin_theta = torch.clip(1 - torch.pow(z, 2), min=0., max=np.inf)

        x, y = torch.cos(phi) * sin_theta, torch.sin(phi) * sin_theta
        # s = np.stack((x, y, z), axis=1)

        s = torch.stack((x, y, z), dim=2)

        # s /= np.linalg.norm(s, axis=1).reshape(-1, 1)
        # return torch.from_numpy(s).type(torch.float)

        s /= torch.linalg.norm(s, dim=2, keepdims=True)
        return s

    def convert_spherical(self, wi):
        # print(wi[:, :, 2:3].min(), wi[:, :, 2:3].max())
        theta = torch.arccos(wi[:, :, 2:3])
        phi = torch.atan2(wi[:, :, 1:2], wi[:, :, 0:1])

        return torch.cat((theta, phi), dim=2)


    def sample_envmap(self, wi):
        wi[:, :, 0] = wi[:, :, 0] / np.pi * self.envmap.shape[0]
        wi[:, :, 1] = wi[:, :, 1] / (2 * np.pi) * self.envmap.shape[1]
        wi = wi.type(torch.uint8)

        return torch.from_numpy(self.envmap[wi[:, :, 0].reshape(-1).cpu().numpy(), wi[:, :, 1].reshape(-1).cpu().numpy(), :].reshape(wi.shape[0], 3, -1)).cuda()

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, 'frames/' + self.idx_list[idx] + '.png'), 'r')

        transform = np.load(os.path.join(self.dir, 'transform/' + self.idx_list[idx] + '.npy'))  # [540, 960, 9]
        nan_pos = np.isnan(transform)
        transform[nan_pos] = 0
        inf_pos = np.isinf(transform)
        transform[inf_pos] = 0

        uv_map = np.load(os.path.join(self.dir, 'uv/' + self.idx_list[idx] + '.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        uv_map = uv_map[:, :, :2]

        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')

        img, uv_map, transform = augment_new(img, uv_map, transform, self.crop_size)
        img = img ** (2.2)

        extrinsics = np.load(os.path.join(self.dir, 'extrinsics/' + self.idx_list[idx] + '.npy'))

        transform = torch.reshape(transform, (-1, 3, 3)).cuda()  # [hxw, 3, 3]

        # wi = self.sample_hemishpere(self.samples)  # [self.samples, 3]
        # wi = np.tile(wi, (transform.shape[0], 1, 1))  # [hxw, self.samples, 3]

        wi = self.sample_hemishpere((transform.shape[0], self.samples))
        cos_t = wi[:, :, 2].type(torch.float32)  # [hxw, self.samples]

        wi = (transform @ wi.permute(0, 2, 1)).permute(0, 2, 1)  # [hxw, samples, 3]
        wi /= (torch.linalg.norm(wi, dim=2, keepdims=True) + self.tiny_number)
        wi = self.convert_spherical(wi)  # [hxw, samples, 2]

        sampled_env = self.sample_envmap(wi)  # [hxw, self.samples, 3]

        wi = wi.reshape(self.crop_size[0], self.crop_size[1], self.samples, 2).permute(3, 2, 0, 1)
        cos_t = cos_t.reshape(self.crop_size[0], self.crop_size[1], self.samples).permute(2, 0, 1)
        sampled_env = sampled_env.reshape(self.crop_size[0], self.crop_size[1], self.samples, 3)
        sampled_env = sampled_env.permute(3, 2, 0, 1)

        return img.type(torch.float), uv_map.type(torch.float), torch.from_numpy(extrinsics).type(torch.float), \
               wi, cos_t, sampled_env