import torch
import torch.nn.functional as F

def info_nce_loss(eeg_features, img_features, temperature=0.07):
    """
    InfoNCE loss for contrastive learning
    
    Args:
        eeg_features: EEG feature embeddings (batch_size, feature_dim)
        img_features: Image feature embeddings (batch_size, feature_dim)
        temperature: Temperature parameter for scaling
    
    Returns:
        InfoNCE loss value
    """
    batch_size = eeg_features.size(0)
    
    similarity_matrix = torch.matmul(eeg_features, img_features.T) / temperature
    labels = torch.arange(batch_size).to(eeg_features.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet loss for contrastive learning
    
    Args:
        anchor: Anchor embeddings (EEG features)
        positive: Positive embeddings (corresponding image features)
        negative: Negative embeddings (non-corresponding image features)
        margin: Margin for triplet loss
    
    Returns:
        Triplet loss value
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def contrastive_loss(anchor, positive, negative, margin=1.0):
    """
    Alternative contrastive loss implementation
    
    Args:
        anchor: Anchor embeddings
        positive: Positive embeddings  
        negative: Negative embeddings
        margin: Margin for contrastive loss
    
    Returns:
        Contrastive loss value
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    
    pos_loss = torch.mean(pos_dist ** 2)    
    neg_loss = torch.mean(F.relu(margin - neg_dist) ** 2)
    
    return pos_loss + neg_loss

def cosine_similarity_loss(eeg_features, img_features):
    """
    Cosine similarity loss for feature alignment
    
    Args:
        eeg_features: EEG feature embeddings
        img_features: Image feature embeddings
    
    Returns:
        Negative cosine similarity (to minimize)
    """
    eeg_norm = F.normalize(eeg_features, dim=1)
    img_norm = F.normalize(img_features, dim=1)    
    similarity = F.cosine_similarity(eeg_norm, img_norm, dim=1)
    
    return -similarity.mean()
