import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, voxel_size, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.voxel_x, self.voxel_y, self.voxel_z = voxel_size
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        
        import matplotlib.pyplot as plt
        import numpy as np
        import time
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            
            #~ fig = plt.figure(figsize=(16, 16))
            #~ fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
            #~ for i in range(0, 32):
                #~ img = (spatial_feature[i:i + 1].view(1, self.ny, self.nx).permute(1, 2, 0).cpu().numpy() * 255).astype(np.int)
                #~ img = img[::-1, :, :] # y -> -y
                #~ plt.subplot(4, 8, i + 1)
                #~ plt.imshow(img)
                #~ plt.axis('off')
            #~ plt.show()
            #~ fig.savefig(time.asctime(time.localtime(time.time())), dpi=200)
            
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        
        import matplotlib.pyplot as plt
        import numpy as np
        import time
        
        points = batch_dict['points']
        batch_semantic_features = []
        batch_size = points[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            semantic_features = torch.zeros(
                self.num_bev_features, self.ny, self.nx, dtype=pillar_features.dtype, device=pillar_features.device)
            
            batch_mask = points[:, 0] == batch_idx
            this_points = points[batch_mask, :]
            xs = (this_points[:, 1] / self.voxel_x).type(torch.long)
            ys = (this_points[:, 2] / self.voxel_y + self.ny / 2).type(torch.long)
            zs = ((this_points[:, 3] + 3.0) / (self.voxel_z / self.num_bev_features)).type(torch.long)
            xs = torch.clamp(xs, min=0, max=self.nx - 1)
            ys = torch.clamp(ys, min=0, max=self.ny - 1)
            zs = torch.clamp(zs, min=0, max=self.num_bev_features - 1)
            semantic_features[zs, ys, xs] = (this_points[:, -3] + this_points[:, -2] + this_points[:, -1]) / 3
            #~ semantic_features[zs, ys, xs] = this_points[:, 4]
            #~ semantic_features[zs, ys, xs] = 1
            
            #~ fig = plt.figure(figsize=(16, 16))
            #~ fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
            #~ for i in range(0, 32):
                #~ img = (semantic_features[i:i + 1, :, :].permute(1, 2, 0).cpu().numpy() * 255).astype(np.int)
                #~ img = img[::-1, :, :] # y -> -y
                #~ plt.subplot(4, 8, i + 1)
                #~ plt.imshow(img)
                #~ plt.axis('off')
            #~ plt.show()
            #~ fig.savefig(time.asctime(time.localtime(time.time())), dpi=200)
            
            batch_semantic_features.append(semantic_features)
        
        batch_semantic_features = torch.stack(batch_semantic_features, 0)
        batch_semantic_features = batch_semantic_features.view(batch_size, self.num_bev_features, self.ny, self.nx)
        batch_dict['semantic_features'] = batch_semantic_features
        
        return batch_dict
