import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ImageEncoder(nn.Module):
    """Encode images to latent features for contrastive learning"""
    def __init__(self, latent_dim=512):
        super(ImageEncoder, self).__init__()
        
        # Use pretrained ResNet50 as backbone
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)
        )
        
        # Freeze early layers of ResNet
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
    
    def forward(self, x):
        # x shape: (batch_size, 3, 128, 128)
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        projected = self.projection_head(features)
        return F.normalize(projected, dim=1)  # L2 normalize for contrastive learning

class EEGEncoder(nn.Module):
    """Encode EEG signals to latent features"""
    def __init__(self, input_channels=17, sequence_length=100, latent_dim=512):
        super(EEGEncoder, self).__init__()
        
        # LSTM layers for temporal processing
        self.lstm1 = nn.LSTM(input_channels, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * sequence_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, latent_dim)
        
        # Projection head for contrastive learning
        self.contrastive_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)
        )
        
        self.dropout = nn.Dropout(0.3)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, return_contrastive=False):
        # x shape: (batch_size, 4, 17, 100) -> (batch_size, 100, 17)
        # Average across trials dimension and permute for LSTM
        x = x.mean(dim=1)  # Average across 4 trials
        x = x.permute(0, 2, 1)  # (batch_size, 100, 17)
        
        # LSTM processing
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        
        # Flatten and process through FC layers
        x = x.flatten(1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        features = self.fc3(x)
        
        if return_contrastive:
            # Return both generator features and contrastive features
            contrastive_features = self.contrastive_head(features)
            contrastive_features = F.normalize(contrastive_features, dim=1)
            return features, contrastive_features
        
        return features
