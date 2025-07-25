import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_norm import spectral_norm

class Generator(nn.Module):
    """Generate images from EEG features"""
    def __init__(self, eeg_dim=512, noise_dim=100, n_classes=1654):
        super(Generator, self).__init__()
        
        self.class_embedding = nn.Embedding(n_classes, 50)
        
        input_dim = eeg_dim + noise_dim + 50
        
        self.fc = nn.Linear(input_dim, 8 * 8 * 1024)
        
        self.convt1 = spectral_norm(nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(512)
        
        self.convt2 = spectral_norm(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False))
        self.bn2 = nn.BatchNorm2d(256)
        
        self.convt3 = spectral_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False))
        self.bn3 = nn.BatchNorm2d(128)
        
        self.convt4 = spectral_norm(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False))
        self.bn4 = nn.BatchNorm2d(64)
        
        self.convt5 = spectral_norm(nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        
    def forward(self, eeg_features, noise, class_labels):
        class_emb = self.class_embedding(class_labels)
        
        x = torch.cat([eeg_features, noise, class_emb], dim=1)
        
        x = self.fc(x)
        x = x.view(-1, 1024, 8, 8)
        
        x = self.leaky_relu(self.bn1(self.convt1(x)))  # 16x16
        x = self.leaky_relu(self.bn2(self.convt2(x)))  # 32x32
        x = self.leaky_relu(self.bn3(self.convt3(x)))  # 64x64
        x = self.leaky_relu(self.bn4(self.convt4(x)))  # 128x128
        x = self.tanh(self.convt5(x))                   # 128x128x3
        
        return x

class Discriminator(nn.Module):
    """Discriminate between real and generated images"""
    def __init__(self, n_classes=1654):
        super(Discriminator, self).__init__()
        
        self.class_embedding = nn.Embedding(n_classes, 50)
        self.class_projection = nn.Linear(50, 128 * 128)
        
        self.conv1 = nn.Conv2d(4, 64, 3, 2, 1, bias=False)  # 4 channels (3 RGB + 1 class)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.conv6 = nn.Conv2d(1024, 1, 3, 1, 1, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, class_labels):
        batch_size = x.size(0)
        
        class_emb = self.class_embedding(class_labels)
        class_map = self.class_projection(class_emb)
        class_map = class_map.view(batch_size, 1, 128, 128)
        
        x = torch.cat([x, class_map], dim=1)
        
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        
        return x.view(batch_size, -1).mean(dim=1)
