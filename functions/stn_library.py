import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Grid_Generator:
    def __init__(self, batch_size, height, width, depth):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.depth = depth
        
    def normal_grid(self):
        x = np.linspace(-1, 1, self.height)
        y = np.linspace(-1, 1, self.width)
        z = np.linspace(-1, 1, self.depth)
        x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z)
        x_mesh = np.expand_dims(x_mesh,3)
        y_mesh = np.expand_dims(y_mesh,3)
        z_mesh = np.expand_dims(z_mesh,3)
        ones = np.ones_like(x_mesh)
        grid = np.concatenate((z_mesh, x_mesh, y_mesh, ones),3)
        grid = np.expand_dims(grid,0)
        grid = np.tile(grid,[self.batch_size,1,1,1,1])

        grid = torch.from_numpy(grid).float().cuda()
        grid = grid.view(-1, int(self.height*self.width*self.depth), 4)
        
        return grid

    def deformed_grid(self, theta, euclidean, rot_only=False):        
        
        '''Create Matrix'''
        Mx = np.array([[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]])
        Mx = torch.from_numpy(Mx).repeat(self.batch_size,1,1).float().cuda()
        My = np.array([[[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,1]]])
        My = torch.from_numpy(My).repeat(self.batch_size,1,1).float().cuda()
        Mz = np.array([[[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]]])
        Mz = torch.from_numpy(Mz).repeat(self.batch_size,1,1).float().cuda()

        '''Fill-in the Matrix'''
        # Rotation in x-axis
        Mx[:,1,1] = theta[:,0].cos()
        Mx[:,1,2] = -theta[:,0].sin()
        Mx[:,2,1] = theta[:,0].sin()
        Mx[:,2,2] = theta[:,0].cos()

        #Rotation in y-axis
        My[:,0,0] = theta[:,1].cos()
        My[:,0,2] = theta[:,1].sin()
        My[:,2,0] = -theta[:,1].sin()
        My[:,2,2] = theta[:,1].cos()

        #Rotation in z-axis
        Mz[:,0,0] = theta[:,2].cos()
        Mz[:,0,1] = -theta[:,2].sin()
        Mz[:,1,0] = theta[:,2].sin()
        Mz[:,1,1] = theta[:,2].cos()
        
        if rot_only:
            _M = torch.bmm(Mz, torch.bmm(My,Mx))
            
        elif euclidean:
            Mt = np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]])
            Mt = torch.from_numpy(Mt).repeat(self.batch_size,1,1).float().cuda()
            
            #Fill in Translation
            Mt[:,0,3] = theta[:,3]
            Mt[:,1,3] = theta[:,4]
            Mt[:,2,3] = theta[:,5]
            
            _M = torch.bmm(Mz, torch.bmm(My, torch.bmm(Mx,Mt)) )
        
        else:
            Mt = np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]])            
            Ms = np.array([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]])
            Mt = torch.from_numpy(Mt).repeat(self.batch_size,1,1).float().cuda()
            Ms = torch.from_numpy(Ms).repeat(self.batch_size,1,1).float().cuda()            

            '''Fill in Translation'''
            Mt[:,0,3] = theta[:,3]
            Mt[:,1,3] = theta[:,4]
            Mt[:,2,3] = theta[:,5]
            _M = torch.bmm(Mz, torch.bmm(My, torch.bmm(Mx,Mt)) )            
            
            '''Fill in Scale'''
            Ms[:,0,0] = theta[:,6]
            Ms[:,1,1] = theta[:,7]
            Ms[:,2,2] = theta[:,8]
            
            _M = torch.bmm(Ms, _M)
        
        M = _M[:,:-1,:]
        M = torch.transpose(M, 1, 2)

        grid = self.normal_grid()
        flow = torch.bmm(grid, M)
        flow = flow.view(-1, self.height, self.width, self.depth, 3)
        
        return flow
    
class Spatial_Transformer(nn.Module):
    def __init__(self):
        super(Spatial_Transformer, self).__init__()
        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 8, 8, 3]
        
        self.block0 = conv_block(1, enc_nf[0], 2)
        self.block1 = conv_block(enc_nf[0], enc_nf[1], 2)
        self.block2 = conv_block(enc_nf[1], enc_nf[2], 2)
        self.block3 = conv_block(enc_nf[2], enc_nf[3], 2)

        self.block4 = conv_block(enc_nf[3], dec_nf[0], 1)   #1
        self.block5 = conv_block(dec_nf[0]*2, dec_nf[1], 1) #2
        self.block6 = conv_block(dec_nf[1]*2, dec_nf[2], 1) #3
        self.block7 = conv_block(dec_nf[2]+enc_nf[0], dec_nf[3], 1) #4
        self.block8 = conv_block(dec_nf[3], dec_nf[4], 1)   #5
        self.block9 = conv_block(dec_nf[4]+1, dec_nf[5], 1)
        
        self.flow = nn.Conv3d(dec_nf[5], dec_nf[6], kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, images):
        x_in = images
        x0 = self.block0(x_in)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        
        x = self.block4(x3)
        x = self.upsample(x)
        x = torch.cat([x, x2], 1)
        
        x = self.block5(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], 1)

        x = self.block6(x)
        x = self.upsample(x)
        x = torch.cat([x, x0], 1)
        
        x = self.block7(x)
        x = self.block8(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_in], 1)
        x = self.block9(x)
        flow = self.flow(x)
        flow = flow.view(-1,80,80,80,3)
        out = F.grid_sample(images, flow)
        
        return out, flow
    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(conv_block, self).__init__()
        self.bn = nn.InstanceNorm3d(out_channels)
    
        if stride == 1:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride= stride, padding=0)
            self.activation = nn.LeakyReLU(0.2)
            self.padder = nn.ConstantPad3d((1,1,1,1,1,1),0)
            
        elif stride == 2:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride= stride, padding=0)
            self.activation = nn.LeakyReLU(0.2)
            self.padder = nn.ConstantPad3d((0,1,0,1,0,1),0)

            
    def forward(self, x):
        out = self.conv(self.padder(x))
        out = self.bn(out)
        out = self.activation(out)
        return out
    
'''VGG'''
class VGG(nn.Module):
    def __init__(self, euclidean):
        super(VGG, self).__init__()
        
        self.convBlock = nn.Sequential(
            nn.Conv3d(in_channels= 1, out_channels= 32, kernel_size= 3, stride=1, padding=1),
            nn.BatchNorm3d(num_features= 32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels= 32, out_channels= 32, kernel_size= 3, stride=1, padding=1),
            nn.BatchNorm3d(num_features= 32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels= 32, out_channels= 32, kernel_size= 3, stride=1, padding=1),
            nn.BatchNorm3d(num_features= 32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels= 32, out_channels= 32, kernel_size= 3, stride=1, padding=1),
            nn.BatchNorm3d(num_features= 32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels= 32, out_channels= 32, kernel_size= 3, stride=1, padding=1),
            nn.BatchNorm3d(num_features= 32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
                
        if euclidean:
            out_chs = 6
        else:
            out_chs = 9
            
        self.fc =  nn.Sequential(
            nn.Linear(in_features= 256, out_features= out_chs))
            
                
    def forward(self, x):
        out = self.convBlock(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
def theta_norm(theta, euclidean):
    Theta = []
    '''Rotation'''
    Theta += [theta[:,:3] - torch.mean(theta[:,:3])]
    
    '''Translation'''
    Theta += [theta[:,3:6] - torch.mean(theta[:,3:6])]
    
    if not euclidean:
        '''Scale'''
        eps = 1e-8
        
        n_reg = theta.size(0)//3
        
        _theta0 = torch.abs(theta[0:n_reg,6:9]+eps)
        _theta0 = torch.log(_theta0) - torch.mean(torch.log(_theta0))
        _theta0 = torch.exp(_theta0)
        
        _theta1 = torch.abs(theta[n_reg:2*n_reg,6:9]+eps)
        _theta1 = torch.log(_theta1) - torch.mean(torch.log(_theta1))
        _theta1 = torch.exp(_theta1)
        
        _theta2 = torch.abs(theta[2*n_reg:,6:9]+eps)
        _theta2 = torch.log(_theta2) - torch.mean(torch.log(_theta2))
        _theta2 = torch.exp(_theta2)        
        
        _theta = torch.cat((_theta0, _theta1, _theta2), 0)
        
        Theta += [_theta]
    
    return torch.cat(Theta,1)