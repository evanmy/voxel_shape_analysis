import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.bn2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.activation(out)
        return out

def make_resnet(cfg):
    encoder = []
    for i, ch in enumerate(cfg['enc'], 2):
        if ch == 'D2':
            encoder += [nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)]
        elif ch == 'D4':
            encoder += [nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=4, padding=0)]            
        else:
            if i>(len(cfg['enc'])-1):
                i = -2
            encoder += [res_block(ch,ch)]
            in_ch = ch
            out_ch = cfg['enc'][i]
            
    
    decoder = []
    in_ch = out_ch
    out_ch = cfg['dec'][1]
    for i, ch in enumerate(cfg['dec'], 2):
        if ch == 'U2':
            decoder += [nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)]
        elif ch == 'U4':
            decoder += [nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=4, output_padding=1)]            
        else:
            if i>(len(cfg['dec'])-1):
                i = -1
            decoder += [res_block(ch,ch)]
            in_ch = ch
            out_ch = cfg['dec'][i]

            
    return nn.Sequential(*encoder), nn.Sequential(*decoder), cfg

class Descriptor(nn.Module):
    def __init__(self, cae, z_map):
        super(Descriptor, self).__init__()
        self.encoder = cae[0]
        self.decoder = cae[1]
        cfg = cae[2]
        bottleneck_dim = int(np.clip(cfg['enc'][-2]//4,4,None))

        self.conv1 =  nn.Conv3d(1, cfg['enc'][0], kernel_size=3, stride=1, padding=1)
        self.conv2 =  nn.Conv3d(cfg['dec'][-1], 1, kernel_size=3, stride=1, padding=1)
        self.z_map = z_map
        self.sequential = nn.Sequential(nn.Linear(cfg['enc'][-2], bottleneck_dim),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(bottleneck_dim, bottleneck_dim))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def siamese(self,x):
        out = self.conv1(x)
        z = self.encoder(out)
        out = self.decoder(z)
        out = self.conv2(out)
        out = F.sigmoid(out)
        return z, out
    
    def fc(self,x):
        out = self.sequential(x)
        return out
    
    def forward(self, x1, x2):
        z1, out1 = self.siamese(x1)
        z2, out2 = self.siamese(x2)
        
        if self.z_map:
            z1 = z1.view(z1.size(0),-1)
            z2 = z2.view(z2.size(0),-1)
           
            z1 = self.fc(z1)
            z2 = self.fc(z2)
        else: 
            z1 = z1.view(z1.size(0),-1)[:,:z1.size(1)//2]
            z2 = z2.view(z2.size(0),-1)[:,:z2.size(1)//2]

        return z1, z2, out1, out2

    
class CAE(nn.Module):
    def __init__(self, cae):
        super(CAE, self).__init__()
        self.encoder = cae[0]
        self.decoder = cae[1]
        cfg = cae[2]

        self.conv1 =  nn.Conv3d(1, cfg['enc'][0], kernel_size=3, stride=1, padding=1)
        self.conv2 =  nn.Conv3d(cfg['dec'][-1], 1, kernel_size=3, stride=1, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def decode(self,z):
        return self.decoder(z)
    
    def forward(self,x):
        out = self.conv1(x)
        z = self.encoder(out)
        out = self.decode(z)
        out = self.conv2(out)
        out = F.sigmoid(out)
        return z, out