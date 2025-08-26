# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_experts=4):
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            [ResBlock3D(in_channels, out_channels) for _ in range(num_experts)]
        )
        

        self.gating_network = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.gating_network(x)

        expert_outputs = [expert(x) for expert in self.experts]

        stacked_outputs = torch.stack(expert_outputs, dim=0)

        stacked_outputs = stacked_outputs.permute(1, 0, 2, 3, 4, 5)

        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        mixed_output = (stacked_outputs * weights).sum(dim=1)
        
        return mixed_output

class ResBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class Classifier(nn.Module):

    def __init__(self, in_channels=1, num_classes=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class CoarseUNet(nn.Module): 
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.net(x)

class FineUNet_MoE(nn.Module):

    def __init__(self, in_channels=2, out_channels=1, base_filters=4, num_experts=2):
        super().__init__()
        
        self.enc1 = MoELayer(in_channels, base_filters, num_experts)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = MoELayer(base_filters, base_filters * 2, num_experts)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = MoELayer(base_filters * 2, base_filters * 4, num_experts)

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = MoELayer(base_filters * 4, base_filters * 2, num_experts)
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = MoELayer(base_filters * 2, base_filters, num_experts)

        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(self.pool1(enc1_out))

        bottleneck_out = self.bottleneck(self.pool2(enc2_out))

        dec2_in = self.up2(bottleneck_out)
        dec2_in = torch.cat([dec2_in, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_in)
        
        dec1_in = self.up1(dec2_out)
        dec1_in = torch.cat([dec1_in, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_in)

        final_out = self.final_conv(dec1_out)
        
        return final_out



class MultiTask_FineUNet_MoE(nn.Module):

    def __init__(self, in_channels=2, out_channels_seg=1, num_classes=8, base_filters=16, num_experts=4):
        super().__init__()

        self.enc1 = MoELayer(in_channels, base_filters, num_experts)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = MoELayer(base_filters, base_filters * 2, num_experts)
        self.pool2 = nn.MaxPool3d(2)
        
        self.bottleneck = MoELayer(base_filters * 2, base_filters * 4, num_experts)

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_filters * 4, num_classes)
        )

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = MoELayer(base_filters * 4, base_filters * 2, num_experts) 
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = MoELayer(base_filters * 2, base_filters, num_experts)
        
        self.final_conv = nn.Conv3d(base_filters, out_channels_seg, kernel_size=1)

    def forward(self, x):

        enc1_out = self.enc1(x)
        enc2_out = self.enc2(self.pool1(enc1_out))
        bottleneck_out = self.bottleneck(self.pool2(enc2_out))

        cls_logits = self.classification_head(bottleneck_out)

        dec2_in = self.up2(bottleneck_out)
        dec2_in = torch.cat([dec2_in, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_in)
        
        dec1_in = self.up1(dec2_out)
        dec1_in = torch.cat([dec1_in, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        
        seg_logits = self.final_conv(dec1_out)
        
        return seg_logits, cls_logits


class CoarseUNet_Medium(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, base_filters=8):
        super().__init__()

        def conv_block(ic, oc):
            return nn.Sequential(
                nn.Conv3d(ic, oc, kernel_size=3, padding=1),
                nn.BatchNorm3d(oc),
                nn.ReLU(inplace=True),
                nn.Conv3d(oc, oc, kernel_size=3, padding=1),
                nn.BatchNorm3d(oc),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)

        self.bottleneck = conv_block(base_filters, base_filters * 2)

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_filters * 2, base_filters) # + skip connection

        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        enc1_out = self.enc1(x)
        bottleneck_in = self.pool1(enc1_out)
        
        bottleneck_out = self.bottleneck(bottleneck_in)
        
        dec1_in = self.up1(bottleneck_out)
        dec1_in = torch.cat([dec1_in, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        
        final_out = self.final_conv(dec1_out)
        return final_out


class PatchedFineModel(nn.Module):

    def __init__(self, num_classes, patch_size, patch_overlap):
        super().__init__()
        self.internal_model = MultiTask_FineUNet_MoE(num_classes = num_classes)

        assert all(s > 0 for s in patch_size), "Размеры патча должны быть > 0"
        assert all(0 <= o < s for o, s in zip(patch_overlap, patch_size)), "Перекрытие должно быть меньше размера патча"
        
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

        self.stride = [s - o for s, o in zip(patch_size, patch_overlap)]

    def forward(self, x_full, x_coarse_full):

        fine_input_full = torch.cat([x_full, x_coarse_full], dim=1)

        *_, D, H, W = fine_input_full.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

        pad_d = (sd - (D - pd) % sd) % sd
        pad_h = (sh - (H - ph) % sh) % sh
        pad_w = (sw - (W - pw) % sw) % sw

        padded_input = F.pad(fine_input_full, (0, pad_w, 0, pad_h, 0, pad_d))
        
        *_, D_pad, H_pad, W_pad = padded_input.shape

        padded_canvas = torch.zeros((x_full.shape[0], 1, D_pad, H_pad, W_pad), device=x_full.device)
        padded_norm_map = torch.zeros_like(padded_canvas)
        
        list_cls_logits = []

        for d in range(0, D_pad - pd + 1, sd):
            for h in range(0, H_pad - ph + 1, sh):
                for w in range(0, W_pad - pw + 1, sw):
                    patch = padded_input[:, :, d:d+pd, h:h+ph, w:w+pw]
                    seg_logits_patch, cls_logits_patch = self.internal_model(patch)
                    
                    padded_canvas[:, :, d:d+pd, h:h+ph, w:w+pw] += seg_logits_patch
                    padded_norm_map[:, :, d:d+pd, h:h+ph, w:w+pw] += 1
                    
                    list_cls_logits.append(cls_logits_patch)
        

        final_seg_logits = (padded_canvas / padded_norm_map)[:, :, :D, :H, :W]
    
        final_cls_logits = torch.stack(list_cls_logits).mean(dim=0)
        
        return final_seg_logits, final_cls_logits