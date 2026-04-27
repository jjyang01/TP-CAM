import torch
import torch.nn as nn
import torch.nn.functional as F

class BCARModule(nn.Module):
    def __init__(self, in_channels, num_classes, lambda_val=0.5):
        super(BCARModule, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lambda_factor = lambda_val
        
        self.q_t = nn.Linear(in_channels, in_channels)
        self.k_i = nn.Conv2d(in_channels, in_channels, 1)
        self.v_i = nn.Conv2d(in_channels, in_channels, 1)
        
        self.q_i = nn.Conv2d(in_channels, in_channels, 1)
        self.k_t = nn.Linear(in_channels, in_channels)
        self.v_t = nn.Linear(in_channels, in_channels)
        
        self.corr_proj = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, f_img, f_txt_classes, cam_base):
        B, C, H, W = f_img.shape
        N = H * W
        
        qt = self.q_t(f_txt_classes)
        ki = self.k_i(f_img).view(B, C, N)
        vi = self.v_i(f_img).view(B, C, N)
        
        attn_t2i = torch.bmm(qt, ki) / (C ** 0.5)
        attn_t2i = F.softmax(attn_t2i, dim=-1)
        feat_t2i = torch.bmm(attn_t2i, vi.transpose(1, 2)).transpose(1, 2)
        
        qi = self.q_i(f_img).view(B, C, N).permute(0, 2, 1)
        kt = self.k_t(f_txt_classes).permute(0, 2, 1)
        vt = self.v_t(f_txt_classes)
        
        attn_i2t = torch.bmm(qi, kt) / (C ** 0.5)
        attn_i2t = F.softmax(attn_i2t, dim=-1)
        feat_i2t = torch.bmm(attn_i2t, vt).permute(0, 2, 1)
        
        f_bcar = feat_i2t.view(B, C, H, W)
        f_corr = self.corr_proj(f_bcar)
        
        cam_refined = cam_base + self.lambda_factor * f_corr
        
        return cam_refined