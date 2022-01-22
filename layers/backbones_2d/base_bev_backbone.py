import numpy as np
import torch
import torch.nn as nn


class AttentionalFusionModule(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.w_1 = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )
        self.w_2 = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )
        
    def forward(self, x_1, x_2):
        weight_1 = self.w_1(x_1)
        weight_2 = self.w_2(x_2)
        aw = torch.softmax(torch.cat([weight_1, weight_2], dim=1), dim=1)
        y = x_1 * aw[:, 0:1, :, :] + x_2 * aw[:, 1:2, :, :]
        return y.contiguous()


class SharpeningFusionModule(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.fusion_layer_1x1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        )
        self.fusion_layer_3x3 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        )
        
    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)
        w = torch.sigmoid(self.fusion_layer_3x3(self.fusion_layer_1x1(x)))
        f_1 = w * x_1
        f_2 = (1 - w) * x_2
        sum_f = f_1 + f_2
        max_f = torch.max(f_1, f_2)
        mean_threshold = F.adaptive_avg_pool2d(sum_f, output_size=(1, 1)) # [B, C, 1, 1]
        y = torch.where(max_f > mean_threshold, max_f * 2, sum_f)
        return y.contiguous()


class GatedFusionModule(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        )
        
    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)
        f_1 = torch.sigmoid(self.conv_1(x)) * x_1
        f_2 = torch.sigmoid(self.conv_2(x)) * x_2
        y = self.fusion_layer(torch.cat([f_1, f_2], dim=1))
        return y.contiguous()


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
        
        #~ self.fusion_layers = nn.Sequential(
            #~ nn.Conv2d(40, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #~ nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            #~ nn.ReLU()
        #~ ) # 20220111
        
        self.spatial_conv1x1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.semantic_conv1x1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks_1 = nn.ModuleList()
        self.blocks_2 = nn.ModuleList()
        self.deblocks_1 = nn.ModuleList()
        self.deblocks_2 = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers_1 = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            cur_layers_2 = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers_1.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
                cur_layers_2.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks_1.append(nn.Sequential(*cur_layers_1))
            self.blocks_2.append(nn.Sequential(*cur_layers_2))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks_1.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_2.append(nn.Sequential(
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
                    self.deblocks_1.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_2.append(nn.Sequential(
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
            self.deblocks_1.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            self.deblocks_2.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        
        self.fusion_layers = AttentionalFusionModule(c_in)
        #~ self.fusion_layers = SharpeningFusionModule(c_in)
        #~ self.fusion_layers = GatedFusionModule(c_in)

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
        
        # This is incredible, but I don't think this proves anything.
        #~ features = self.fusion_layers(semantic_features) # 20220111, c=40, kernel_size=1, occupancy, Car 3d AP: 87.6261, 77.5537, 74.9875
        
        #~ features = self.fusion_layers(torch.cat([spatial_features, semantic_features], dim=1)) # 20220112, c=64+10, Car 3d AP: 86.4325, 76.9217, 75.4987
        #~ features = self.fusion_layers(torch.cat([spatial_features, semantic_features], dim=1)) # 20220113, c=64+20, Car 3d AP: 86.7387, 77.1374, 75.2964
        #~ features = self.fusion_layers(semantic_features) # 20220114, c=10, kernel_size=7, (r+g+b) / 3, Car 3d AP: 85.9258, 75.2121, 71.6240
        #~ features = self.fusion_layers(semantic_features) # 20220115, c=30, kernel_size=1, (r,g,b), Car 3d AP: 84.4311, 75.5213, 71.4392
        
        #~ features = spatial_features # 20220116, torch.max(), Linear(in_features=10, out_features=64), Car 3d AP: 86.5273, 76.9653, 75.6938
        #~ features = spatial_features # 20220117, torch.mean(), Linear(in_features=10, out_features=64), Car 3d AP: 87.0298, 77.0740, 75.5394
        #~ features = spatial_features # 20220118, torch.max(), Linear(in_features=13, out_features=64), Car 3d AP: 85.2592, 76.3921, 73.0607
        #~ features = spatial_features # 20220119, torch.mean(), Linear(in_features=13, out_features=64), Car 3d AP: 85.2050, 76.1199, 72.4744
        
        #~ features = spatial_features # 20220120, double points, torch.max(), Linear(in_features=10, out_features=64), Car 3d AP: AP:86.2017, 76.7041, 73.9883
        #~ features = self.fusion_layers(semantic_features) # 20220121, double points, c=40, kernel_size=1, occupancy, Car 3d AP: 86.5430, 76.6896, 72.1760
        #~ features = self.fusion_layers(semantic_features) # 20220122, double points, c=40, kernel_size=1, (r+g+b) / 3, Car 3d AP: 86.0676, 76.4452, 71.8087
        
        #~ features = semantic_features # 20220123, c=64, (r+g+b) / 3, Car 3d AP: 85.9842, 76.4812, 72.2739
        
        spatial_features = self.spatial_conv1x1(spatial_features)
        semantic_features = self.semantic_conv1x1(semantic_features)
        
        ups = []
        x = spatial_features
        for i in range(len(self.blocks_1)):
            x = self.blocks_1[i](x)
            if len(self.deblocks_1) > 0:
                ups.append(self.deblocks_1[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks_1) > len(self.blocks_1):
            x = self.deblocks_1[-1](x)
        final_spatial_features = x
        
        ups = []
        x = semantic_features
        for i in range(len(self.blocks_2)):
            x = self.blocks_2[i](x)
            if len(self.deblocks_2) > 0:
                ups.append(self.deblocks_2[i](x))
            else:
                ups.append(x)
        
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        
        if len(self.deblocks_2) > len(self.blocks_2):
            x = self.deblocks_2[-1](x)
        final_semantic_features = x

        import matplotlib.pyplot as plt
        import numpy as np
        import time
        
        #~ batch_size = x.shape[0]
        #~ for batch_idx in range(batch_size):
            #~ feature_map = x[batch_idx]
            #~ fig = plt.figure(figsize=(16, 16))
            #~ fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
            #~ for i in range(0, 32):
                #~ img = feature_map[i + 256:i + 1 + 256].permute(1, 2, 0).cpu().numpy()
                #~ img = img[::-1, :, :] # y -> -y
                #~ pmin = np.min(img)
                #~ pmax = np.max(img)
                #~ img = (((img - pmin) / (pmax - pmin + 0.000001)) * 255).astype(np.int)
                #~ plt.subplot(4, 8, i + 1)
                #~ plt.imshow(img)
                #~ plt.axis('off')
            #~ plt.show()
            #~ fig.savefig(time.asctime(time.localtime(time.time())), dpi=200)
        
        # Car 3d AP: 88.4615, 78.1517, 77.0567 -> Better than `intensity` model and `occupancy` model in Car class.
        #~ final_features = self.fusion_layers(final_spatial_features, final_semantic_features) # 20220124, c=64, (r+g+b) / 3, AttentionalFusionModule, 39ms
        # Car 3d AP: 87.5320, 77.3184, 75.2125 -> This model is better than pointpillars in all categories, which is embarrassing.
        #~ final_features = self.fusion_layers(final_spatial_features, final_semantic_features) # 20220125, c=64, intensity, AttentionalFusionModule, 39ms
        # Car 3d AP: 88.4228, 77.9727, 76.6534 -> Like I said, `occupancy` model shouldn't be the best.
        #~ final_features = self.fusion_layers(final_spatial_features, final_semantic_features) # 20220126, c=64, occupancy, AttentionalFusionModule, 39ms
        
        # Car 3d AP: 88.6761, 78.3303, 76.9806
        #~ final_features = self.fusion_layers(final_spatial_features, final_semantic_features) # 20220127, c=64, (r+g+b) / 3, semantic_conv1x1, AttentionalFusionModule, 38ms
        # Car 3d AP: 88.1927, 77.8198, 76.4382
        final_features = self.fusion_layers(final_spatial_features, final_semantic_features) # 20220128, c=64, (r+g+b) / 3, spatial_conv1x1, semantic_conv1x1, AttentionalFusionModule, 38ms
        
        data_dict['spatial_features_2d'] = final_features

        return data_dict
