# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerfulResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, 
                 width_multiplier=2, use_se=True, dropout_rate=0.1):
        super().__init__()
        mid_channels = max(in_channels, out_channels) * width_multiplier
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.dropout1 = nn.Dropout3d(dropout_rate)
        
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.dropout2 = nn.Dropout3d(dropout_rate)
        
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.use_se = use_se
        if use_se:
            reduction = max(1, out_channels // 16)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(out_channels, reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(reduction, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
        

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout2(out)
        
        out = self.bn3(self.conv3(out))
        
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        
        shortcut_out = self.shortcut(residual)
        
        out += shortcut_out
        out = F.relu(out)
        return out















class MoELayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)

        self.experts = nn.ModuleList([
            PowerfulResBlock3D(in_channels, out_channels) for _ in range(num_experts)
        ])


        self.gating_network =   nn.Sequential(
                                nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool3d(1),
                                nn.Flatten(),
                                nn.Linear(in_channels // 2, num_experts)
                            )

        self.register_buffer('expert_usage_count', torch.zeros(num_experts))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
    def forward(self, x):
        raw_weights = self.gating_network(x)
        if self.training:
            noise = torch.randn_like(raw_weights) * 1e-2
            raw_weights += noise
            
        top_k_weights, top_k_indices = torch.topk(raw_weights, k=self.k, dim=1)
        top_k_weights = F.softmax(top_k_weights, dim=1)
        
        if self.training:
            self._update_expert_usage(top_k_indices)
        
        with torch.no_grad():
            sample_input = torch.zeros(1, *x.shape[1:], device=x.device, dtype=x.dtype)
            sample_output = self.experts[0](sample_input)
            output_channels = sample_output.shape[1]

        output_shape = list(x.shape)
        output_shape[1] = output_channels
        output = torch.zeros(output_shape, device=x.device, dtype=x.dtype)
        
        for i in range(self.k):
            expert_indices = top_k_indices[:, i]
            expert_weights = top_k_weights[:, i:i+1]
            
            unique_experts = torch.unique(expert_indices)
            
            for expert_idx in unique_experts:
                mask = (expert_indices == expert_idx)
                if not mask.any():
                    continue
                
                expert_input = x[mask]
                expert_weight = expert_weights[mask]
                
                expert_output = self.experts[expert_idx](expert_input)
                weighted_output = expert_output * expert_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                output[mask] += weighted_output
        
        return output
    
    def _update_expert_usage(self, top_k_indices):
        batch_size = top_k_indices.shape[0]
        for expert_idx in range(self.num_experts):
            usage = (top_k_indices == expert_idx).float().sum()
            self.expert_usage_count[expert_idx] += usage
        
        self.total_samples += batch_size * self.k
    
    def get_load_balancing_loss(self):
        if self.total_samples == 0:
            return torch.tensor(0.0, device=self.expert_usage_count.device)

        usage_freq = self.expert_usage_count / self.total_samples

        ideal_freq = 1.0 / self.num_experts

        load_balancing_loss = torch.sum((usage_freq - ideal_freq) ** 2)
        
        return load_balancing_loss
    
    def reset_usage_stats(self):
        """Сбрасывает статистику использования"""
        self.expert_usage_count.zero_()
        self.total_samples.zero_()

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

class MoETrainingManager:
    """Помощник для управления load balancing loss во время обучения"""

    def __init__(self, model, load_balance_weight=0.01):
        self.model = model
        self.load_balance_weight = load_balance_weight
        
    def compute_total_loss(self, main_loss):
        """Вычисляет общий loss включая load balancing"""
        total_loss = main_loss
        total_load_balance_loss = 0.0
        
        for module in self.model.modules():
            if isinstance(module, MoELayer):
                lb_loss = module.get_load_balancing_loss()
                total_load_balance_loss += lb_loss
        
        if total_load_balance_loss > 0:
            total_loss = total_loss + self.load_balance_weight * total_load_balance_loss
            
        return total_loss, total_load_balance_loss
    
    def reset_stats_if_needed(self, epoch, reset_frequency=10):

        if epoch % reset_frequency == 0:
            for module in self.model.modules():
                if isinstance(module, MoELayer):
                    module.reset_usage_stats()

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

