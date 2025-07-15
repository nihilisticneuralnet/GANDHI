import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import pickle
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from torchvision.models import resnet50
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def pairwise_cosine_similarity(A, B, dim=1, eps=1e-8):
    """Pairwise cosine similarity between all pairs"""
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

def batchwise_pearson_correlation(Z, B):
    """Batch-wise Pearson correlation"""
    Z_mean = torch.mean(Z, dim=1, keepdim=True)
    B_mean = torch.mean(B, dim=1, keepdim=True)
    Z_centered = Z - Z_mean
    B_centered = B - B_mean
    numerator = Z_centered @ B_centered.T
    Z_centered_norm = torch.linalg.norm(Z_centered, dim=1, keepdim=True)
    B_centered_norm = torch.linalg.norm(B_centered, dim=1, keepdim=True)
    denominator = Z_centered_norm @ B_centered_norm.T
    pearson_correlation = (numerator / denominator)
    return pearson_correlation

def batchwise_cosine_similarity(Z, B):
    """Batch-wise cosine similarity"""
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def topk_accuracy(similarities, labels, k=5):
    """Top-k accuracy metric"""
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = 0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities, axis=1)[:, -(i+1)] == labels) / len(labels)
    return topsum

def mixco(voxels, beta=0.15, s_thresh=0.5):
    """Mixup for contrastive learning"""
    perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device, dtype=voxels.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device, dtype=voxels.dtype)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, bidirectional=True):
    """MixCo NCE loss"""
    brain_clip = (preds @ targs.T) / temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas
        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2) / 2
        return loss
    else:
        loss = F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2) / 2
        return loss

def info_nce_loss(eeg_features, img_features, temperature=0.07):
    """InfoNCE loss"""
    batch_size = eeg_features.size(0)
    similarity_matrix = torch.matmul(eeg_features, img_features.T) / temperature
    labels = torch.arange(batch_size).to(eeg_features.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def supervised_contrastive_loss(features, labels, temperature=0.07):
    """Supervised contrastive loss"""
    batch_size = features.size(0)
    device = features.device
    
    features = F.normalize(features, dim=1)
    
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask    
    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)    
    loss = -mean_log_prob_pos.mean()
    return loss

def barlow_twins_loss(z1, z2, lambda_param=0.005):
    """Barlow Twins loss"""
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)    
    c = torch.mm(z1_norm.T, z2_norm) / z1.size(0)    
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[1:].view(c.size(0)-1, c.size(0)+1)[:, :-1].pow_(2).sum()
    
    loss = on_diag + lambda_param * off_diag
    return loss

def vicreg_loss(z1, z2, sim_coeff=25., var_coeff=25., cov_coeff=1.):
    """VICReg loss"""
    batch_size = z1.size(0)
    repr_loss = F.mse_loss(z1, z2)    
    std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
    std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (batch_size - 1)
    cov_z2 = (z2.T @ z2) / (batch_size - 1)
    
    cov_loss = off_diagonal(cov_z1).pow_(2).sum().div(z1.size(1)) + \
               off_diagonal(cov_z2).pow_(2).sum().div(z2.size(1))
    
    loss = sim_coeff * repr_loss + var_coeff * std_loss + cov_coeff * cov_loss
    return loss

def off_diagonal(x):
    """Return off-diagonal elements of a matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def swav_loss(features, prototypes, temperature=0.1, sinkhorn_iterations=3):
    """SwAV loss"""
    q = sinkhorn(torch.exp(features @ prototypes.T / temperature), sinkhorn_iterations)
    
    p = F.softmax(features @ prototypes.T / temperature, dim=1)
    
    loss = -torch.mean(torch.sum(q * torch.log(p), dim=1))
    return loss

def sinkhorn(out, iterations=3):
    """Sinkhorn-Knopp algorithm"""
    Q = torch.exp(out / 0.05).t()
    B = Q.shape[1]
    K = Q.shape[0]
    
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for _ in range(iterations):
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K
        
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    
    Q *= B
    return Q.t()
