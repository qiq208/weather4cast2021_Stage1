#from fastai.basics import *
import torch
from torch import nn
import torch.nn.functional as F

############################################################################################

def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    '''Conv2d + ELU'''
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.ELU(inplace=True))

def ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    '''Conv2d + ELU + GroupNorm'''
    return nn.Sequential(ConvLayer(in_channels, out_channels, kernel_size, stride, padding),
                         nn.GroupNorm(num_groups=8, num_channels=out_channels, eps=1e-6),
                         nn.Dropout2d(p=0.2))


# VDrawBlock
class VDrawBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin_mean = nn.Linear(in_features=in_features, out_features=out_features)
        self.lin_var = nn.Linear(in_features=in_features, out_features=out_features)

    # Reparameterisation trick
    def forward(self, x):
        z_mean = self.lin_mean(x)
        z_var = self.lin_var(x)
        
        if self.training:
            std = torch.exp(0.5 * z_var)
            epsilon = torch.empty_like(z_var, device=z_mean.device).normal_()
            x_out =  z_mean + std*epsilon
        else:
            x_out =  z_mean
        
        return x_out, z_mean, z_var

    
############################################################################################

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_layers=4):
        super().__init__()
        self.first_layer = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.mid_layers = nn.ModuleList([ConvBlock(in_channels + i*out_channels, out_channels, kernel_size=3, stride=1, padding=1) 
                                         for i in range(1, nb_layers)])
        self.last_layer = ConvLayer(in_channels + nb_layers*out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        layers_concat = list()
        layers_concat.append(x)
        layers_concat.append(self.first_layer(x))
        
        for mid_layer in self.mid_layers:
            layers_concat.append(mid_layer(torch.cat(layers_concat, dim=1)))
            
        return self.last_layer(torch.cat(layers_concat, dim=1))

def AvgPoolDenseBlock(in_channels, out_channels, nb_layers=4):
    return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
                         DenseBlock(in_channels, out_channels, nb_layers))

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convtrans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.elu = nn.ELU(inplace=True)
    
    def forward(self, x, x_enc):
        x = self.convtrans(x, output_size=x_enc.shape[-2:])
        x = self.elu(x)
        return torch.cat([x, x_enc], dim=1)

############################################################################################

class Net1(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.counter = (torch.zeros(1, requires_grad=False)-1).cuda()
        
        self.enc1 = DenseBlock(in_channels, 64, 4)
        self.enc2 = AvgPoolDenseBlock(64, 96, 4)
        self.enc3 = AvgPoolDenseBlock(96, 128, 4)
        self.enc4 = AvgPoolDenseBlock(128, 128, 4)
        self.enc5 = AvgPoolDenseBlock(128, 128, 4)
        self.enc6 = AvgPoolDenseBlock(128, 128, 4)
        self.enc7 = AvgPoolDenseBlock(128, 128, 4)
        self.enc8 = AvgPoolDenseBlock(128, 128, 4)
        
        self.bridge = ConvBlock(128, 128)
        
        self.var = VDrawBlock(128*2*2, 128*2*2)
        
        #self.dec8 = ConvBlock(128, 128)
        
        self.dec7_1 = UpConvBlock(128, 128)
        self.dec7_2 = ConvBlock(128+128, 128)
        self.dec6_1 = UpConvBlock(128, 128)
        self.dec6_2 = ConvBlock(128+128, 128)
        self.dec5_1 = UpConvBlock(128, 128)
        self.dec5_2 = ConvBlock(128+128, 128)
        self.dec4_1 = UpConvBlock(128, 128)
        self.dec4_2 = ConvBlock(128+128, 128)
        self.dec3_1 = UpConvBlock(128, 128)
        self.dec3_2 = ConvBlock(128+128, 128)
        self.dec2_1 = UpConvBlock(128, 128)
        self.dec2_2 = ConvBlock(128+96, 128)
        self.dec1_1 = UpConvBlock(128, 128)
        self.dec1_2 = ConvBlock(128+64, 128)
        
        self.out_1 = nn.Conv2d(128, out_channels, 3, stride=1, padding=1)
        #self.out_2 = nn.Sigmoid()
          
    def forward(self, x):
        #self.counter = self.counter +1.0
        #x0 = x[:,:108,...]               # 4, 108, 495, 436
        #s = x[:,108:115,...]             # 4, 7, 495, 436
        #t = x[:,115,...].unsqueeze(1)    # 4, 1, 495, 436
        
        N, C, H, W = x.shape            # 4, 108, 495, 436
        #x0r = x0.reshape(N, 9, 12, H, W) # 4, 9, 12, 495, 436
        #x0_mean = x0r.mean(dim=2)        # 4, 9, 495, 436
        #x0_std = x0r.std(dim=2)
        #x0_rng = x0r.max(dim=2).values - x0r.min(dim=2).values
        
        #x = torch.cat([x0, x0_mean, x0_std, x0_rng, s, t], dim=1)
        #x = x / 255.
        
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        
        x100 = self.bridge(x8)
        
        x100 = torch.flatten(x100, start_dim=1)
        x100, z_mean, z_logvar  = self.var(x100)
        x100 = x100.view(N, -1, 2, 2)
        #x100 = self.dec8(x100)
        
        x107 = self.dec7_2(self.dec7_1(x100, x7))
        x106 = self.dec6_2(self.dec6_1(x107, x6))
        x105 = self.dec5_2(self.dec5_1(x106, x5))
        x104 = self.dec4_2(self.dec4_1(x105, x4))
        x103 = self.dec3_2(self.dec3_1(x104, x3))
        x102 = self.dec2_2(self.dec2_1(x103, x2))
        x101 = self.dec1_2(self.dec1_1(x102, x1))
        
        #out = self.out_2(self.out_1(x101))*255.
        out = self.out_1(x101)
        
        # recover time dimension
        out = out.view(-1, 32, 4, out.shape[-2], out.shape[-1])
        
        # clamp 'asii_turb_trop_prob'
        out[:,:,2,...] = 0.003 + (0.997 - 0.003) * torch.sigmoid(out[:,:,2,...])
        
        return out, z_mean, z_logvar
        #return out.view(N, self.out_channels, H, W), z_mean, z_logvar, self.counter

def logit(x):
    return torch.log(x / (1. - x))

def norm_logit(x, M=0.997, m=0.003):
    
    x = logit(x)
    
    M = -logit(torch.tensor(m))
    x += M
    x /= 2*M
    
    return x
    
def leaderboard_loss(output, target):
    
    target[:,:,2,...] = target[:,:,2,...].clamp(0.003, 0.997)
    
    temp_mse = (1/0.03163512)*F.mse_loss(output[:,:,0,...],target[:,:,0,...])
    crr_mse  = (1/0.00024158)*F.mse_loss(output[:,:,1,...],target[:,:,1,...])
    prob_mse = (1/0.00703378)*F.mse_loss(norm_logit(output[:,:,2,...]), norm_logit(target[:,:,2,...]))
    cma_mse  = (1/0.19160305)*F.mse_loss(output[:,:,3,...],target[:,:,3,...])
    
    return (temp_mse + crr_mse + prob_mse + cma_mse)/4

def temp_loss(output, target):
    mask = (target>0)
    loss = F.mse_loss(torch.masked_select(output, mask), torch.masked_select(target, mask))
    return loss

def leaderboard_loss2(output, target):
    
    target[:,:,2,...] = target[:,:,2,...].clamp(0.003, 0.997)
    
    #temp_mse = (1/0.03163512)*F.mse_loss(output[:,:,0,...],target[:,:,0,...])
    temp_mse = (1/0.03163512)*temp_loss(output[:,:,0,...],target[:,:,0,...])
    crr_mse  = (1/0.00024158)*F.mse_loss(output[:,:,1,...],target[:,:,1,...])
    prob_mse = (1/0.00703378)*F.mse_loss(norm_logit(output[:,:,2,...]), norm_logit(target[:,:,2,...]))
    cma_mse  = (1/0.19160305)*F.mse_loss(output[:,:,3,...],target[:,:,3,...])
    
    return (temp_mse + crr_mse + prob_mse + cma_mse)/4

def VUNetLoss(output, target):
    
    # L2 loss
    loss_L2 = leaderboard_loss(output[0], target)
    
    # VAE penalty
    N = target.shape[-1]
    
    mu = output[1]
    log_var = output[2]
    #loss_KL = ((1 / N) * (torch.exp(z_var) + torch.square(z_mean) - 1. - z_var).sum(dim=-1)).mean()
    loss_KL = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    #if verbose: print(loss_L2, loss_KL)
        
    return loss_L2 + 80*loss_KL

def valid_leaderboard(output, target):
    # L2 loss
    return leaderboard_loss(output[0], target)

def VUNetLoss2(output, target):
    
    # L2 loss
    loss_L2 = leaderboard_loss2(output[0], target)
    
    # VAE penalty
    N = target.shape[-1]
    
    mu = output[1]
    log_var = output[2]
    #loss_KL = ((1 / N) * (torch.exp(z_var) + torch.square(z_mean) - 1. - z_var).sum(dim=-1)).mean()
    loss_KL = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    #if verbose: print(loss_L2, loss_KL)
    
    return loss_L2 + 80*loss_KL
    #return loss_L2 + 4*loss_KL

def valid_leaderboard2(output, target):
    # L2 loss
    return leaderboard_loss2(output[0], target)

def kl_loss(output, target):
    kl_anneal = torch.sigmoid(-12 + 24*(output[3]/(1384*6)))
    #print(kl_anneal)
    # L2 loss
    mu = output[1]
    log_var = output[2]
    #loss_KL = ((1 / N) * (torch.exp(z_var) + torch.square(z_mean) - 1. - z_var).sum(dim=-1)).mean()
    loss_KL = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    return loss_KL