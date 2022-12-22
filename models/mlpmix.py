import torch

class MLP(torch.nn.Module):
    def __init__(self, dim, inter_dim, dropout=0.0):
        super().__init__()
        self.l = torch.nn.Linear(dim, inter_dim)
        self.l2 = torch.nn.Linear(inter_dim, dim)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.l(x)
        x = self.gelu(x)
        x = self.l2(x)
        # x = self.dropout(x)
        return x



class MLPBlock(torch.nn.Module):
    def __init__(self, dim, hw, inter_dim_c=2048, inter_dim_s=256, dropout=0.0):
        super().__init__()
        self.mlp_c = MLP(dim, inter_dim_c, dropout)
        self.mlp_s = MLP(hw, inter_dim_s, dropout)
        self.norm_c = torch.nn.LayerNorm((hw, dim))
        self.norm_s = torch.nn.LayerNorm((dim, hw))
        self.dropout = torch.nn.Dropout(dropout)
        
        
    def forward(self, x):
        x += self.mlp_c(self.norm_c(x))
        x = x.permute(0, 2, 1)
        x += self.mlp_s(self.norm_s(x))
        x = x.permute(0, 2, 1)
        # x = self.dropout(x)
        return x
        


        
class MLPMix(torch.nn.Module):
    def __init__(self, patch_size=16, depth=8, num_cls=2, dim=512, inter_dim_c=2048, inter_dim_s=256, hw=196, dropout=0.0):
        super().__init__()
        self.pos_stem = torch.nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = torch.nn.ModuleList([
            MLPBlock(dim, hw, inter_dim_c, inter_dim_s, dropout)
            for _ in range(depth)
        ])
        self.norm = torch.nn.LayerNorm((hw, dim))
        self.fc = torch.nn.Linear(dim, num_cls)
    
    def forward(self, x):
        x = self.pos_stem(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1)
        x = x.permute(0, 2, 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.fc(x)
        

        
