import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channel=1, dim_feature=32, dim_latent=128):
        super(Encoder, self).__init__()
        self.in_channel     = in_channel
        self.dim_feature    = dim_feature 
        self.dim_latent     = dim_latent
        
        self.conv1  = nn.Conv2d(in_channels=in_channel, out_channels=dim_feature * 1, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2  = nn.Conv2d(in_channels=dim_feature * 1, out_channels=dim_feature * 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3  = nn.Conv2d(in_channels=dim_feature * 2, out_channels=dim_feature * 4, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4  = nn.Conv2d(in_channels=dim_feature * 4, out_channels=dim_feature * 8, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5  = nn.Conv2d(in_channels=dim_feature * 8, out_channels=dim_feature * 16, kernel_size=3, stride=2, padding=1, bias=True)
        
        self.bn1    = nn.BatchNorm2d(dim_feature * 1)
        self.bn2    = nn.BatchNorm2d(dim_feature * 2)
        self.bn3    = nn.BatchNorm2d(dim_feature * 4)
        self.bn4    = nn.BatchNorm2d(dim_feature * 8)
        self.bn5    = nn.BatchNorm2d(dim_feature * 16)

        self.linear_mean    = nn.Linear(in_features=dim_feature * 16, out_features=dim_latent, bias=True)
        self.linear_var     = nn.Linear(in_features=dim_feature * 16, out_features=dim_latent, bias=True)
        
        self.activation = nn.LeakyReLU()
        self.flatten    = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation(x)
        
        x = self.flatten(x)
        
        mu      = self.linear_mean(x)
        log_var = self.linear_var(x)
        
        return mu, log_var
    
# ======================================================================
# decoder 
# ======================================================================
class Decoder(nn.Module):
    def __init__(self, dim_latent=128, dim_feature=32, out_channel=1):
        super(Decoder, self).__init__()
        self.dim_latent     = dim_latent
        self.dim_feature    = dim_feature
        self.out_channel    = out_channel 
        
        self.input      = nn.Linear(dim_latent, dim_feature * 16)
        self.unflatten  = nn.Unflatten(1, (dim_feature * 16, 1, 1))
        self.upsample   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1      = nn.Conv2d(in_channels=dim_feature * 16, out_channels=dim_feature * 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2      = nn.Conv2d(in_channels=dim_feature * 8, out_channels=dim_feature * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3      = nn.Conv2d(in_channels=dim_feature * 4, out_channels=dim_feature * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4      = nn.Conv2d(in_channels=dim_feature * 2, out_channels=dim_feature * 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5      = nn.Conv2d(in_channels=dim_feature * 1, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn1        = nn.BatchNorm2d(dim_feature * 8)
        self.bn2        = nn.BatchNorm2d(dim_feature * 4)
        self.bn3        = nn.BatchNorm2d(dim_feature * 2)
        self.bn4        = nn.BatchNorm2d(dim_feature * 1)

        self.activation = nn.LeakyReLU()
        self.output     = nn.Tanh()
        # self.output     = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.output(x)
        
        return x
    

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder    = Encoder
        self.Decoder    = Decoder
        self.dim_latent = Encoder.dim_latent
        
    def reparameterization(self, mu, log_var):
        sigma   = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z       = mu + sigma * epsilon
        
        return z

    def forward(self, x):
        z, mu, log_var = self.compute_latent(x)
        y = self.Decoder(z)
        
        return y, mu, log_var, z

    def compute_loss(self, prediction, mu, log_var, target):
        criterion   = nn.MSELoss(reduction='sum')
        loss_recon  = criterion(prediction, target)
        loss_kld    = -0.5 * torch.sum(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
        loss        = loss_recon + loss_kld        
    
        return loss, loss_recon, loss_kld
    
    def sample(self, number_sample, device):
        z = torch.randn(number_sample, self.dim_latent).to(device)
        y = self.Decoder(z)
        
        return y
   
    def compute_prediction(self, x):
        y = self.forward(x)
        
        return y
    
    def compute_latent(self, x):
        mu, log_var = self.Encoder(x)
        z = self.reparameterization(mu, log_var)

        return z, mu, log_var