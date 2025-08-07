import torch
import torchvision
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from tqdm import tqdm
import os
import copy
import argparse
import configparser
import sys
import datetime 
from tqdm import tqdm
import csv
from torchmetrics import PeakSignalNoiseRatio
# from .types_ import *
import pytorch_lightning as pl



# ======================================================================
# options
# ======================================================================
if len(sys.argv) > 1:
    dir_work                = sys.argv[1]
    cuda_device             = int(sys.argv[2])
    batch_size              = int(sys.argv[3])
    number_epoch            = int(sys.argv[4])
    lr_vae                  = float(sys.argv[5])
else:
    dir_work                = '/home/hong/work/vae_energy_model/'
    cuda_device             = 0
    batch_size              = 100
    number_epoch            = 100 
    lr_vae                  = 0.001

# ======================================================================

pl.seed_everything(0)

# ======================================================================
# options
# ======================================================================
in_channel  = 1
dim_latent  = 128
dim_feature = 32
dim_output  = 1
sigma_noise = 0.5

# ======================================================================
# device
# ======================================================================
device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'mps')
# torch.cuda.set_device(device)

# ======================================================================
# dataset 
# ======================================================================
dir_data    = os.path.join(dir_work, 'data')
dir_dataset = 'MNIST'
use_subset  = True

kwargs = {'num_workers': 1, 'pin_memory': True}

transform = torchvision.transforms.Compose([ 
    torchvision.transforms.Resize([32, 32]),
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
])

dataset_train   = torchvision.datasets.MNIST(dir_data, transform=transform, train=True, download=True)
dataset_test    = torchvision.datasets.MNIST(dir_data, transform=transform, train=False, download=True)

if use_subset:
    use_label   = 4

    idx_label_train         = (dataset_train.targets == use_label)
    dataset_train.data      = dataset_train.data[idx_label_train]
    dataset_train.targets   = dataset_train.targets[idx_label_train]

    idx_label_test          = (dataset_test.targets == use_label)
    dataset_test.data       = dataset_test.data[idx_label_test]
    dataset_test.targets    = dataset_test.targets[idx_label_test]

 
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True, shuffle=True)
dataloader_test  = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=True, shuffle=True)

    
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
    
 
# ======================================================================
# evaluation 
# ======================================================================
psnr = PeakSignalNoiseRatio().to(device)


# ======================================================================
# optimizer 
# ======================================================================
encoder     = Encoder(in_channel, dim_feature, dim_latent).to(device)
decoder     = Decoder(dim_latent, dim_feature, in_channel).to(device)
vae         = VAE(encoder, decoder).to(device)
optim_vae   = torch.optim.Adam(vae.parameters(), lr=lr_vae)


# ======================================================================
# variables for the results 
# ======================================================================
val_loss_vae_mean                   = np.zeros(number_epoch)
val_loss_vae_std                    = np.zeros(number_epoch)
val_loss_vae_data_fidelity_mean     = np.zeros(number_epoch)
val_loss_vae_data_fidelity_std      = np.zeros(number_epoch)
val_loss_vae_regularization_mean    = np.zeros(number_epoch)
val_loss_vae_regularization_std     = np.zeros(number_epoch)
val_psnr_mean                       = np.zeros(number_epoch)
val_psnr_std                        = np.zeros(number_epoch)

# ======================================================================
# training
# ======================================================================
vae.train()

for i in range(number_epoch):

    val_loss_vae                = []
    val_loss_vae_data_fidelity  = []
    val_loss_vae_regularization = []
    val_psnr                    = []

    for j, (image, _) in enumerate(dataloader_train):
        noise       = torch.randn_like(image)
        image_noise = image + sigma_noise * noise 
     
        image       = image.to(device) 
        image_noise = image_noise.to(device)
        
        optim_vae.zero_grad()

        prediction, mu, log_var, z = vae.forward(image_noise)
        loss_vae, loss_vae_recon, loss_vae_kld = vae.compute_loss(prediction, mu, log_var, image)
        loss_vae.backward() 
            
        value_psnr = psnr(prediction, image).detach().cpu().numpy().mean()

        # -------------------------------------------------------------------
        # update networks
        # -------------------------------------------------------------------        
        optim_vae.step()
        
        # -------------------------------------------------------------------
        # save results for each batch iteration
        # -------------------------------------------------------------------        
        val_loss_vae.append(loss_vae.item()) 
        val_loss_vae_data_fidelity.append(loss_vae_recon.item()) 
        val_loss_vae_regularization.append(loss_vae_kld.item())
        val_psnr.append(value_psnr)


    # -------------------------------------------------------------------
    # save results for each epoch
    # -------------------------------------------------------------------        
    val_loss_vae_mean[i]                = np.mean(val_loss_vae)
    val_loss_vae_std[i]                 = np.std(val_loss_vae)
    val_loss_vae_data_fidelity_mean[i]  = np.mean(val_loss_vae_data_fidelity)
    val_loss_vae_data_fidelity_std[i]   = np.std(val_loss_vae_data_fidelity)
    val_loss_vae_regularization_mean[i] = np.mean(val_loss_vae_regularization)
    val_loss_vae_regularization_std[i]  = np.std(val_loss_vae_regularization)
    val_psnr_mean[i]                    = np.mean(val_psnr)
    val_psnr_std[i]                     = np.std(val_psnr)

    log = '[%4d/%4d] loss=%8.5f, psnr=%6.3f' % (i, number_epoch, val_loss_vae_mean[i], val_psnr_mean[i])
    print(log)
    
    if (np.isnan(val_psnr_mean[i])) or ((i > 10) and (val_psnr_mean[i] < 5)) or ((i > 20) and (val_psnr_mean[i] < 10)) or ((i > 100) and (val_psnr_mean[i] < 13)):
        print('exit')
        exit()
    
# ======================================================================
# path for the results
# ======================================================================
vae.eval()

# ======================================================================
# training results
# ======================================================================
it = iter(dataloader_train)
(image_train, _) = next(it)

noise               = torch.randn_like(image_train)
image_noise_train   = image_train + sigma_noise * noise
        
image_train         = image_train.to(device)
image_noise_train   = image_noise_train.to(device)

y_train, mu_train, log_var_train, z_train = vae.forward(image_noise_train)

z_train             = z_train.detach().cpu().numpy().squeeze()
image_noise_train   = image_noise_train.detach().cpu().numpy().squeeze()
y_train             = y_train.detach().cpu().numpy().squeeze()

# ======================================================================
# testing results
# ======================================================================
it = iter(dataloader_test)
(image_test, _) = next(it)

noise               = torch.randn_like(image_test)
image_noise_test    = image_test + sigma_noise * noise
        
image_test          = image_test.to(device)
image_noise_test    = image_noise_test.to(device)
       
y_test, mu_test, log_var_test, z_test = vae.forward(image_noise_test)

z_test              = z_test.detach().cpu().numpy().squeeze()
image_noise_test    = image_noise_test.detach().cpu().numpy().squeeze()
y_test              = y_test.detach().cpu().numpy().squeeze()

# ======================================================================
# results from latent
# ======================================================================
y = vae.sample(batch_size, device)
y = y.detach().cpu().numpy().squeeze()

# ======================================================================
# path for the results
# ======================================================================
dir_figure  = os.path.join(dir_work, 'figure')
dir_option  = os.path.join(dir_work, 'option')
dir_result  = os.path.join(dir_work, 'result')
dir_model   = os.path.join(dir_work, 'model')

dir_dataset = 'MNIST'
use_subset  = True

now         = datetime.datetime.now()
date_stamp  = now.strftime('%Y_%m_%d') 
time_stamp  = now.strftime('%H_%M_%S') 

path_figure = os.path.join(dir_figure, dir_dataset)
path_option = os.path.join(dir_option, dir_dataset)
path_result = os.path.join(dir_result, dir_dataset)
path_model  = os.path.join(dir_model, dir_dataset)

date_figure = os.path.join(path_figure, date_stamp)
date_option = os.path.join(path_option, date_stamp)
date_result = os.path.join(path_result, date_stamp)
date_model  = os.path.join(path_model, date_stamp)

file_figure = os.path.join(date_figure, '{}.png'.format(time_stamp))
file_option = os.path.join(date_option, '{}.ini'.format(time_stamp))
file_result = os.path.join(date_result, '{}.csv'.format(time_stamp))
file_model  = os.path.join(date_model, '{}.pth'.format(time_stamp))

if not os.path.exists(dir_figure):
    os.mkdir(dir_figure)

if not os.path.exists(dir_option):
    os.mkdir(dir_option)

if not os.path.exists(dir_result):
    os.mkdir(dir_result)

if not os.path.exists(dir_model):
    os.mkdir(dir_model)

if not os.path.exists(path_figure):
    os.mkdir(path_figure)

if not os.path.exists(path_option):
    os.mkdir(path_option)

if not os.path.exists(path_result):
    os.mkdir(path_result)

if not os.path.exists(path_model):
    os.mkdir(path_model)

if not os.path.exists(date_figure):
    os.mkdir(date_figure)

if not os.path.exists(date_option):
    os.mkdir(date_option)

if not os.path.exists(date_result):
    os.mkdir(date_result)
    
if not os.path.exists(date_model):
    os.mkdir(date_model)

save_fig = True

if save_fig:
    # -------------------------------------------------------------------
    # save the options
    # -------------------------------------------------------------------         
    with open(file_option, 'w') as f:
        f.write('{} : {}\n'.format('batch size', batch_size))
        f.write('{} : {}\n'.format('number epoch', number_epoch))
        f.write('{} : {}\n'.format('lr vae', lr_vae))

    f.close()

    # -------------------------------------------------------------------
    # save the results
    # -------------------------------------------------------------------         
    with open(file_result, 'w', newline='') as f:
        writer  = csv.writer(f, delimiter=',')
        writer.writerow(val_loss_vae_mean)
        writer.writerow(val_loss_vae_std)
        writer.writerow(val_loss_vae_data_fidelity_mean)
        writer.writerow(val_loss_vae_data_fidelity_std)
        writer.writerow(val_loss_vae_regularization_mean)
        writer.writerow(val_loss_vae_regularization_std)
        writer.writerow(val_psnr_mean)
        writer.writerow(val_psnr_std)
    
    f.close()

    # -------------------------------------------------------------------
    # save the figures from training
    # -------------------------------------------------------------------
    
    nRow    = 6 
    nCol    = 6 
    fSize   = 3
     
    fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

    ax[0][0].set_title('loss vae')
    ax[0][0].plot(val_loss_vae_mean, color='red', label='loss', linewidth=3)
    ax[0][0].plot(val_loss_vae_data_fidelity_mean, color='green', label='data fidelity')
    ax[0][0].plot(val_loss_vae_regularization_mean, color='blue', label='regularization')
    ax[0][0].legend()

    ax[0][1].set_title('accuracy (psnr)')
    ax[0][1].plot(val_psnr_mean, color='red', label='vae')
    ax[0][1].legend()

    ax[0][2].set_title('latent')
    ax[0][2].plot(z_train[0], color='red', label='train')
    ax[0][2].plot(z_test[0], color='blue', label='test')
    ax[0][2].legend()
    
    for i in range(nCol):
        ax[1][i].set_title('training')
        ax[1][i].imshow(image_noise_train[i,:,:], cmap='gray')

    for i in range(nCol):
        ax[2][i].set_title('training (vae)')
        ax[2][i].imshow(y_train[i,:,:], cmap='gray')

    for i in range(nCol):
        ax[3][i].set_title('testing')
        ax[3][i].imshow(image_noise_test[i,:,:], cmap='gray')

    for i in range(nCol):
        ax[4][i].set_title('testing (vae)')
        ax[4][i].imshow(y_test[i,:,:], cmap='gray')

    for i in range(nCol):
        ax[5][i].set_title('latent (vae)')
        ax[5][i].imshow(y[i,:,:], cmap='gray')


    plt.tight_layout()

    fig.savefig(file_figure, bbox_inches='tight', dpi=600)
    plt.close(fig)