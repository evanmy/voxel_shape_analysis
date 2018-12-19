import torch
import numpy as np
import torch.nn.functional as F

def dice_loss(pred, target):
    eps = 1.
    
    if len(target.size())==4:
        n,c,x,y = target.size()
    if len(target.size())==5:
        n,c,x,y,z = target.size()

    target = target.view(n,c,-1)
    pred = pred.view(n,c,-1)
    num = torch.sum(2*(target*pred),2) + eps
    den = (pred*pred).sum(2) + (target*target).sum(2) + eps
    dice_loss = 1-num/den
    total_avg = torch.mean(dice_loss)
    
    return total_avg


def global_loss(w, z1, z2, out1, out2, template, lmdb):
    z_loss = torch.mean(w*torch.mean(torch.abs(z1-z2),1))
    recon_loss = dice_loss(out1, template) + dice_loss(out2, template)
    return z_loss + lmdb*recon_loss


def Zeta(epoch):
    u = (-3+10)*epoch/30-10
    u = torch.tensor(u).float().cuda()
    return torch.clamp(10**u,0.0,1e-3)
