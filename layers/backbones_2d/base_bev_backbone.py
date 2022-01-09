import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        #~ self.fusion_layers = nn.Sequential(
            #~ nn.Conv2d(input_channels + 3, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #~ nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            #~ nn.ReLU()
        #~ ) # 20220103
        
        #~ self.fusion_layers = nn.Sequential(
            #~ nn.Conv2d(10, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #~ nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            #~ nn.ReLU()
        #~ ) # 20220104
        
        #~ self.fusion_layers = nn.Sequential(
            #~ nn.Conv2d(3, input_channels, kernel_size=7, stride=1, padding=3, bias=False),
            #~ nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            #~ nn.ReLU()
        #~ ) # 20220105
        
        #~ self.fusion_layers = nn.Sequential(
            #~ nn.Conv2d(10, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            #~ nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            #~ nn.ReLU()
        #~ ) # 20220106
        
        #~ self.fusion_layers = nn.Sequential(
            #~ nn.Conv2d(20, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #~ nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            #~ nn.ReLU()
        #~ ) # 20220107
        
        #~ self.fusion_layers = nn.Sequential(
            #~ nn.Conv2d(40, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #~ nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            #~ nn.ReLU()
        #~ ) # 20220108
        
        #~ self.fusion_layers = nn.Sequential(
            #~ nn.Conv2d(40, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #~ nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            #~ nn.ReLU()
        #~ ) # 20220109
        
        #~ self.fusion_layers = nn.Sequential(
            #~ nn.Conv2d(input_channels + 40, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #~ nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            #~ nn.ReLU()
        #~ ) # 20220110
        
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(40, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        ) # 20220111
        
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        semantic_features = data_dict['semantic_features']
        
        #~ features = self.fusion_layers(torch.cat([spatial_features, semantic_features], dim=1)) # 20220103, c=64+3, Car 3d AP: 86.9950, 77.4302, 75.3491
        #~ features = self.fusion_layers(semantic_features) # 20220104, c=10, kernel_size=1, (r+g+b) / 3, Car 3d AP: 85.1289, 74.7592, 71.3553
        #~ features = self.fusion_layers(semantic_features) # 20220105, c=3, kernel_size=7, (r,g,b), Car 3d AP: 57.3326, 45.4200, 43.6049
        #~ features = self.fusion_layers(semantic_features) # 20220106, c=10, kernel_size=3, (r+g+b) / 3, Car 3d AP: 85.9584, 75.9415, 71.7329
        #~ features = self.fusion_layers(semantic_features) # 20220107, c=20, kernel_size=1, (r+g+b) / 3, Car 3d AP: 86.4965, 76.8295, 72.0208
        #~ features = self.fusion_layers(semantic_features) # 20220108, c=40, kernel_size=1, (r+g+b) / 3, Car 3d AP: 87.0098, 76.9380, 73.1823
        #~ features = self.fusion_layers(semantic_features) # 20220109, c=40, kernel_size=1, intensity, Car 3d AP: 74.7418, 58.3303, 56.4951
        #~ features = self.fusion_layers(torch.cat([spatial_features, semantic_features], dim=1)) # 20220110, c=64+40, Car 3d AP: 86.1564, 76.6295, 74.6659
        
        features = self.fusion_layers(semantic_features) # 20220111, c=40, kernel_size=1, occupancy, Car 3d AP: 87.6261, 77.5537, 74.9875 -> this shows colored_pointpillars is shit!!!
        
        #~ features = self.fusion_layers(torch.cat([spatial_features, semantic_features], dim=1)) # 20220112, c=64+10, Car 3d AP: 86.4325, 76.9217, 75.4987
        #~ features = self.fusion_layers(torch.cat([spatial_features, semantic_features], dim=1)) # 20220113, c=64+20, Car 3d AP: 86.7387, 77.1374, 75.2964
        #~ features = self.fusion_layers(semantic_features) # 20220114, c=10, kernel_size=7, (r+g+b) / 3, Car 3d AP: 85.9258, 75.2121, 71.6240
        #~ features = self.fusion_layers(semantic_features) # 20220115, c=30, kernel_size=1, (r,g,b), Car 3d AP: 84.4311, 75.5213, 71.4392
        
        #~ features = spatial_features # 20220116, torch.max(), Linear(in_features=10, out_features=64), Car 3d AP: 86.5273, 76.9653, 75.6938
        #~ features = spatial_features # 20220117, torch.mean(), Linear(in_features=10, out_features=64), Car 3d AP: 87.0298, 77.0740, 75.5394
        #~ features = spatial_features # 20220118, torch.max(), Linear(in_features=13, out_features=64), Car 3d AP: 85.2592, 76.3921, 73.0607
        #~ features = spatial_features # 20220119, torch.mean(), Linear(in_features=13, out_features=64), Car 3d AP: 85.2050, 76.1199, 72.4744
        
        ups = []
        x = features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
