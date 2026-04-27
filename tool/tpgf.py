import torch
import torch.nn as nn
import torch.nn.functional as F

class TPGFModule(nn.Module):
    def __init__(self, in_channels):
        super(TPGFModule, self).__init__()
        self.proj_img = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.proj_txt = nn.Linear(in_channels, in_channels, bias=False)
        
    def forward(self, f_img, f_txt):
        B, C, H, W = f_img.shape
        f_img_proj = self.proj_img(f_img)
        f_txt_proj = self.proj_txt(f_txt).view(B, C, 1, 1)
        
        f_img_norm = F.normalize(f_img_proj, p=2, dim=1)
        f_txt_norm = F.normalize(f_txt_proj, p=2, dim=1)
        
        sim_map = torch.sum(f_img_norm * f_txt_norm, dim=1, keepdim=True)
        attn_weight = torch.sigmoid(sim_map)
        
        f_img_filtered = f_img * attn_weight
        
        return f_img_filtered, attn_weight