import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# class ImageEncoder(nn.Module):
#     """Encode images to latent features for contrastive learning"""
#     def __init__(self, latent_dim=512):
#         super(ImageEncoder, self).__init__()
        
#         resnet = resnet50(pretrained=True)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])  
        
#         self.projection_head = nn.Sequential(
#             nn.Linear(2048, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, latent_dim)
#         )
        
#         for param in list(self.backbone.parameters())[:-20]:
#             param.requires_grad = False
    
#     def forward(self, x):
#         # x shape: (batch_size, 3, 128, 128)
#         features = self.backbone(x)
#         features = features.view(features.size(0), -1) 
#         projected = self.projection_head(features)
#         return F.normalize(projected, dim=1)  # L2 normalize for contrastive learning

# class EEGEncoder(nn.Module):
#     """Encode EEG signals to latent features"""
#     def __init__(self, input_channels=17, sequence_length=100, latent_dim=512):
#         super(EEGEncoder, self).__init__()
        
#         self.lstm1 = nn.LSTM(input_channels, 128, batch_first=True, bidirectional=True)
#         self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        
#         self.fc1 = nn.Linear(128 * sequence_length, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, latent_dim)
        
#         self.contrastive_head = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, latent_dim)
#         )
        
#         self.dropout = nn.Dropout(0.3)
#         self.leaky_relu = nn.LeakyReLU(0.2)
        
#     def forward(self, x, return_contrastive=False):
#         # x shape: (batch_size, 4, 17, 100) -> (batch_size, 100, 17)
#         x = x.mean(dim=1)  
#         x = x.permute(0, 2, 1)  # (batch_size, 100, 17)
        
#         x, _ = self.lstm1(x)
#         x = self.dropout(x)
#         x, _ = self.lstm2(x)
        
#         x = x.flatten(1)
#         x = self.leaky_relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.leaky_relu(self.fc2(x))
#         x = self.dropout(x)
#         features = self.fc3(x)
        
#         if return_contrastive:
#             contrastive_features = self.contrastive_head(features)
#             contrastive_features = F.normalize(contrastive_features, dim=1)
#             return features, contrastive_features
        
#         return features




# Thoguhtviz
# class ImageEncoder(nn.Module):
#     """Encode images to latent features for contrastive learning"""
#     def __init__(self, latent_dim=512):
#         super(ImageEncoder, self).__init__()
        
#         resnet = resnet50(pretrained=True)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])  
        
#         self.projection_head = nn.Sequential(
#             nn.Linear(2048, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, latent_dim)
#         )
        
#         for param in list(self.backbone.parameters())[:-20]:
#             param.requires_grad = False
    
#     def forward(self, x):
#         # x shape: (batch_size, 3, 128, 128)
#         features = self.backbone(x)
#         features = features.view(features.size(0), -1) 
#         projected = self.projection_head(features)
#         return F.normalize(projected, dim=1)  # L2 normalize for contrastive learning


class EEGEncoder(nn.Module):
    """EEG Encoder for shape (batch, 14, 32, 1)"""
    def __init__(self, input_channels=14, sequence_length=32, latent_dim=512):
        super(EEGEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.lstm1 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        
        self.fc1 = nn.Linear(128 * sequence_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, latent_dim)
        
        self.contrastive_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)
        )
        
        self.dropout = nn.Dropout(0.3)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, return_contrastive=False):
        # x shape: (batch_size, 14, 32, 1) -> (batch_size, 14, 32)
        x = x.squeeze(-1)  
        
        x = self.leaky_relu(self.conv1(x))  # (batch, 64, 32)
        x = self.dropout(x)
        x = self.leaky_relu(self.conv2(x))  # (batch, 128, 32)
        x = self.dropout(x)
        x = self.leaky_relu(self.conv3(x))  # (batch, 256, 32)
        
        # Transpose for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, 32, 256)
        
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        
        x = x.flatten(1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        features = self.fc3(x)
        
        if return_contrastive:
            contrastive_features = self.contrastive_head(features)
            contrastive_features = F.normalize(contrastive_features, dim=1)
            return features, contrastive_features
        
        return features


## Thoughtviz MNIST
class ImageEncoder(nn.Module):
    """Encode MNIST images (28x28 grayscale) to latent features for contrastive learning"""
    def __init__(self, latent_dim=512):
        super(ImageEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1) 
        projected = self.projection_head(features)
        return F.normalize(projected, dim=1)  # L2 normalize for contrastive learning
