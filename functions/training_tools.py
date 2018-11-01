import torch
import numpy as np
import torch.nn.functional as F

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def global_loss(w, z1, z2, out1, out2, template, lmdb):
    z_loss = torch.mean(w*torch.mean(torch.abs(z1-z2),1))
    recon_loss = dice_loss(out1, template) + dice_loss(out2, template)
    return z_loss + lmdb*recon_loss


def Zeta(epoch):
    u = (-3+10)*epoch/30-10
    u = torch.tensor(u).float().cuda()
    return torch.clamp(10**u,0.0,1e-3)
