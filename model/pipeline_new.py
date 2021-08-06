import math, cv2
import numpy as np
import sys
import torch
import torch.nn as nn

import torch.nn.functional as F

sys.path.append('..')
from model.texture import Texture,TextureMapper
from model.unet_og import UNet


class PipeLine(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=True, view_direction=True):
        super(PipeLine, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.view_direction = view_direction
        # self.texture = Texture(W, H, feature_num, use_pyramid)
        self.texture = Texture(W, H, feature_num, use_pyramid)
        self.albedo_tex = Texture(W, H, 3, False)
        # self.texture = TextureMapper(texture_size=W,texture_num_ch=16,mipmap_level=4)
        self.unet = UNet(feature_num+20, 10*3)

    def _spherical_harmonics_basis(self, extrinsics):
        '''
        extrinsics: a tensor shaped (N, 3)
        output: a tensor shaped (N, 9)
        '''
        batch = extrinsics.shape[0]
        sh_bands = torch.ones((batch, 9), dtype=torch.float)
        # coff_0 = 1 / (2.0*math.sqrt(np.pi))
        # coff_1 = math.sqrt(3.0) * coff_0
        # coff_2 = math.sqrt(15.0) * coff_0
        # coff_3 = math.sqrt(1.25) * coff_0
        # # l=0
        # sh_bands[:, 0] = coff_0
        # # l=1
        # sh_bands[:, 1] = extrinsics[:, 1] * coff_1
        # sh_bands[:, 2] = extrinsics[:, 2] * coff_1
        # sh_bands[:, 3] = extrinsics[:, 0] * coff_1
        # # l=2
        # sh_bands[:, 4] = extrinsics[:, 0] * extrinsics[:, 1] * coff_2
        # sh_bands[:, 5] = extrinsics[:, 1] * extrinsics[:, 2] * coff_2
        # sh_bands[:, 6] = (3.0 * extrinsics[:, 2] * extrinsics[:, 2] - 1.0) * coff_3
        # sh_bands[:, 7] = extrinsics[:, 2] * extrinsics[:, 0] * coff_2
        # sh_bands[:, 8] = (extrinsics[:, 0] * extrinsics[:, 0] - extrinsics[:, 2] * extrinsics[:, 2]) * coff_2
        coff_0 = 1 / (2.0*math.sqrt(np.pi))
        coff_1 = math.sqrt(3.0) * coff_0
        coff_2 = math.sqrt(15.0) * coff_0
        coff_3 = math.sqrt(1.25) * coff_0
        # l=0
        sh_bands[:, 0] = coff_0
        # l=1
        sh_bands[:, 1] = extrinsics[:, 1] * coff_1 * math.sqrt(0.5)
        sh_bands[:, 2] = extrinsics[:, 2] * coff_1
        sh_bands[:, 3] = extrinsics[:, 0] * coff_1 * (-1)*math.sqrt(0.5)
        # l=2
        sh_bands[:, 4] = extrinsics[:, 0] * extrinsics[:, 1] * coff_2 * 0.5* math.sqrt(0.5)
        sh_bands[:, 5] = extrinsics[:, 1] * extrinsics[:, 2] * coff_2 * math.sqrt(0.5)
        sh_bands[:, 6] = (3.0 * extrinsics[:, 2] * extrinsics[:, 2] - 1.0) * coff_3
        sh_bands[:, 7] = extrinsics[:, 2] * extrinsics[:, 0] * coff_2 *(-1)*math.sqrt(0.5)
        sh_bands[:, 8] = (extrinsics[:, 0] * extrinsics[:, 0] - extrinsics[:, 2] * extrinsics[:, 2]) * coff_2*0.5*math.sqrt(0.5)
        return sh_bands

    def forward(self, *args):
        wi, cos_t, envmap, uv_map, extrinsics = args
        wi = wi.view(-1, 20, wi.shape[3], wi.shape[4])
        cos_t = torch.unsqueeze(cos_t, dim=1)
        cos_t = torch.tile(cos_t, (1, 3, 1, 1, 1))

        # Make Diffuse BRDF
        albedo = self.albedo_tex(uv_map)
        albedo_ = torch.unsqueeze(albedo, dim=2)
        albedo_ = torch.tile(albedo_, (1, 1, cos_t.shape[2], 1, 1))

        nt = self.texture(uv_map)
        basis = self._spherical_harmonics_basis(extrinsics).cuda()
        basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)
        nt[:, 3:12, :, :] = nt[:, 3:12, :, :] * basis

        inp = torch.cat((nt, wi), dim=1)
        vis = self.unet(inp)
        vis = vis.reshape(-1, 3, 10, cos_t.shape[3], cos_t.shape[4])
        # vis = torch.unsqueeze(vis, dim=1)
        # vis = torch.tile(vis, (1, 3, 1, 1, 1))

        final = albedo_ * cos_t * envmap * vis
        final = 2.0 / 10.0 * torch.sum(final, dim=2)

        return albedo, final
