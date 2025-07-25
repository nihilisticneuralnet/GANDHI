import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from typing import Tuple, List, Optional, Dict, Any

import seaborn as sns
plt.style.use('seaborn-v0_8')  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def info_nce_loss(eeg_features, img_features, temperature=0.07):
    """
    InfoNCE loss for contrastive learning
    """
    batch_size = eeg_features.size(0)
    
    similarity_matrix = torch.matmul(eeg_features, img_features.T) / temperature
    labels = torch.arange(batch_size).to(eeg_features.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet loss for contrastive learning
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

# Thoughtviz
def load_test_data(eeg_pickle_path):
    with open(eeg_pickle_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
    
    x_test = data_dict['x_test']  # (5706, 14, 32, 1)
    y_test = data_dict['y_test']  # (5706, 10)
    test_class_labels = np.argmax(y_test, axis=1)
    
    return x_test, test_class_labels

def discriminator_hinge_loss(real_output, fake_output):
    real_loss = torch.mean(F.relu(1.0 - real_output))
    fake_loss = torch.mean(F.relu(1.0 + fake_output))
    return (real_loss + fake_loss) / 2.0

def generator_hinge_loss(fake_output):
    return -torch.mean(fake_output)

def diff_augment(x, policy="color,translation"):
    if "color" in policy:
        x = x + torch.randn_like(x) * 0.1
        x = torch.clamp(x, -1, 1)
    
    if "translation" in policy:
        if random.random() > 0.5:
            shift = random.randint(-4, 4)
            x = torch.roll(x, shift, dims=2)
            x = torch.roll(x, shift, dims=3)
    
    return x


def visualize_generated_samples(eeg_encoder, generator, dataset, num_samples=8, save_path='generated_samples.png'):
    eeg_encoder.eval()
    generator.eval()
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
    fig.suptitle('Results', fontsize=16, fontweight='bold')
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            eeg_sample, real_image, _, class_label = dataset[idx]
            
            eeg_batch = eeg_sample.unsqueeze(0).to(device)
            class_batch = class_label.to(device)
            noise = torch.randn(1, 100).to(device)
            
            eeg_features = eeg_encoder(eeg_batch)
            generated_image = generator(eeg_features, noise, class_batch)
            
            real_img = (real_image + 1) / 2  
            real_img = torch.clamp(real_img, 0, 1)
            
            gen_img = (generated_image.squeeze().cpu() + 1) / 2
            gen_img = torch.clamp(gen_img, 0, 1)
            
            eeg_plot = eeg_sample.mean(dim=0).mean(dim=0).numpy()  
            
            axes[0, i].plot(eeg_plot)
            axes[0, i].set_title(f'EEG Signal\nClass: {class_label.item()}', fontsize=10)
            axes[0, i].set_xlabel('Time')
            axes[0, i].set_ylabel('Amplitude')
            axes[0, i].grid(True, alpha=0.3)
            
            axes[1, i].imshow(real_img.permute(1, 2, 0))
            axes[1, i].set_title('Real Image', fontsize=10)
            axes[1, i].axis('off')
            
            axes[2, i].imshow(gen_img.permute(1, 2, 0))
            axes[2, i].set_title('Generated Image', fontsize=10)
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as {save_path}")

def plot_training_history_with_contrastive(contrastive_losses, g_losses, d_losses, save_path='training_history.png'):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(contrastive_losses, label='Contrastive Loss', color='green')
    plt.title('Contrastive Learning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(g_losses, label='Generator Loss', color='blue')
    plt.plot(d_losses, label='Discriminator Loss', color='red')
    plt.title('GAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    window = max(1, len(g_losses) // 20)
    cont_smooth = np.convolve(contrastive_losses, np.ones(window)/window, mode='valid')
    g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
    d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
    
    plt.plot(cont_smooth, label='Contrastive Loss (Smoothed)', color='green')
    plt.plot(g_smooth, label='Generator Loss (Smoothed)', color='blue')
    plt.plot(d_smooth, label='Discriminator Loss (Smoothed)', color='red')
    plt.title('Smoothed Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training history saved as {save_path}")

def evaluate_contrastive_alignment(eeg_encoder, image_encoder, dataset, num_samples=100):
    eeg_encoder.eval()
    image_encoder.eval()
    
    similarities = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            eeg_sample, real_image, _, class_label = dataset[i]
            
            eeg_batch = eeg_sample.unsqueeze(0).to(device)
            img_batch = real_image.unsqueeze(0).to(device)
            
            _, eeg_features = eeg_encoder(eeg_batch, return_contrastive=True)
            img_features = image_encoder(img_batch)
            
            similarity = F.cosine_similarity(eeg_features, img_features, dim=1)
            similarities.append(similarity.item())
    
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    
    print(f"Average EEG-Image similarity: {avg_similarity:.4f} ± {std_similarity:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(avg_similarity, color='red', linestyle='--', 
                label=f'Mean: {avg_similarity:.4f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of EEG-Image Feature Similarities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('similarity_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return avg_similarity, similarities


def visualize_feature_space(eeg_encoder, image_encoder, dataset, num_samples=500):
        from sklearn.manifold import TSNE
        import matplotlib.colors as mcolors
        
        eeg_encoder.eval()
        image_encoder.eval()
        
        eeg_features_list = []
        img_features_list = []
        labels_list = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                eeg_sample, real_image, _, class_label = dataset[i]
                
                eeg_batch = eeg_sample.unsqueeze(0).to(device)
                img_batch = real_image.unsqueeze(0).to(device)
                
                _, eeg_features = eeg_encoder(eeg_batch, return_contrastive=True)
                img_features = image_encoder(img_batch)
                
                eeg_features_list.append(eeg_features.cpu().numpy())
                img_features_list.append(img_features.cpu().numpy())
                labels_list.append(class_label.item())
        
        eeg_features_array = np.vstack(eeg_features_list)
        img_features_array = np.vstack(img_features_list)
        all_features = np.vstack([eeg_features_array, img_features_array])
        
        feature_types = ['EEG'] * len(eeg_features_array) + ['Image'] * len(img_features_array)
        class_labels = labels_list + labels_list
        
        print("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(all_features)
        
        plt.figure(figsize=(12, 8))
        
        unique_classes = sorted(list(set(class_labels)))[:10]  
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        
        for i, class_id in enumerate(unique_classes):
            eeg_mask = np.array([(ft == 'EEG' and cl == class_id) 
                                for ft, cl in zip(feature_types, class_labels)])
            if np.any(eeg_mask):
                plt.scatter(features_2d[eeg_mask, 0], features_2d[eeg_mask, 1], 
                           c=[colors[i]], marker='o', s=50, alpha=0.6, 
                           label=f'EEG Class {class_id}' if i < 5 else "")
            
            img_mask = np.array([(ft == 'Image' and cl == class_id) 
                                for ft, cl in zip(feature_types, class_labels)])
            if np.any(img_mask):
                plt.scatter(features_2d[img_mask, 0], features_2d[img_mask, 1], 
                           c=[colors[i]], marker='^', s=50, alpha=0.6,
                           label=f'Image Class {class_id}' if i < 5 else "")
        
        plt.title('t-SNE Visualization of EEG and Image Features\n(Circles: EEG, Triangles: Images)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('feature_space_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Feature space visualization saved as 'feature_space_visualization.png'")
