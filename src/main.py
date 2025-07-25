import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

from encoders import EEGEncoder, ImageEncoder
from gan import Generator, Discriminator
from contrastive_learning import info_nce_loss, triplet_loss
from dataset import EEGImageDataset
from utils import (
    discriminator_hinge_loss, 
    generator_hinge_loss, 
    diff_augment,
    visualize_generated_samples,
    plot_training_history_with_contrastive,
    evaluate_contrastive_alignment,
    visualize_feature_space
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# def train(eeg_file_path, images_dir, num_epochs=100, batch_size=32, lr=0.0002):
#     dataset = EEGImageDataset(eeg_file_path, images_dir)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
#     eeg_encoder = EEGEncoder().to(device)
#     image_encoder = ImageEncoder().to(device)
#     generator = Generator(n_classes=len(dataset.class_dirs)).to(device)
#     discriminator = Discriminator(n_classes=len(dataset.class_dirs)).to(device)
    
#     contrastive_optimizer = optim.Adam(
#         list(eeg_encoder.parameters()) + list(image_encoder.parameters()), 
#         lr=lr, betas=(0.9, 0.999)
#     )
#     g_optimizer = optim.Adam(
#         list(eeg_encoder.parameters()) + list(generator.parameters()), 
#         lr=lr, betas=(0.0, 0.9)
#     )
#     d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
    
#     contrastive_losses = []
#     g_losses = []
#     d_losses = []
    
#     print("Starting training...")
    
#     for epoch in range(num_epochs):
#         contrastive_loss_total = 0
#         g_loss_total = 0
#         d_loss_total = 0
        
#         pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
#         for batch_idx, (eeg_batch, real_images, negative_images, class_labels) in enumerate(pbar):
#             batch_size_actual = eeg_batch.size(0)
            
#             eeg_batch = eeg_batch.to(device)
#             real_images = real_images.to(device)
#             negative_images = negative_images.to(device)
#             class_labels = class_labels.squeeze().to(device)
            
#             # =============== Contrastive Learning Phase ===============
#             contrastive_optimizer.zero_grad()
            
#             _, eeg_contrastive_features = eeg_encoder(eeg_batch, return_contrastive=True)
#             img_contrastive_features = image_encoder(real_images)
#             negative_img_features = image_encoder(negative_images)
            
#             info_nce = info_nce_loss(eeg_contrastive_features, img_contrastive_features)
#             triplet = triplet_loss(eeg_contrastive_features, img_contrastive_features, negative_img_features)
            
#             contrastive_loss = info_nce + 0.5 * triplet
#             contrastive_loss.backward()
#             contrastive_optimizer.step()
            
#             # =============== GAN Training Phase ===============
#             noise = torch.randn(batch_size_actual, 100).to(device)
            
#             d_optimizer.zero_grad()
            
#             eeg_features = eeg_encoder(eeg_batch).detach()
            
#             fake_images = generator(eeg_features, noise, class_labels)
            
#             real_images_aug = diff_augment(real_images)
#             fake_images_aug = diff_augment(fake_images.detach())
            
#             real_output = discriminator(real_images_aug, class_labels)
#             fake_output = discriminator(fake_images_aug, class_labels)
            
#             d_loss = discriminator_hinge_loss(real_output, fake_output)
#             d_loss.backward()
#             d_optimizer.step()
            
#             g_optimizer.zero_grad()
            
#             noise2 = torch.randn(batch_size_actual, 100).to(device)
#             eeg_features = eeg_encoder(eeg_batch)
#             fake_images = generator(eeg_features, noise, class_labels)
#             fake_images2 = generator(eeg_features, noise2, class_labels)
            
#             fake_images_aug = diff_augment(fake_images)
#             fake_images2_aug = diff_augment(fake_images2)
            
#             fake_output = discriminator(fake_images_aug, class_labels)
#             fake_output2 = discriminator(fake_images2_aug, class_labels)
            
#             g_loss = generator_hinge_loss(fake_output) + generator_hinge_loss(fake_output2)
            
#             mode_loss = torch.mean(torch.abs(fake_images2 - fake_images)) / (
#                 torch.mean(torch.abs(noise2 - noise)) + 1e-5)
#             mode_loss = 1.0 / (mode_loss + 1e-5)
            
#             with torch.no_grad():
#                 real_img_features = image_encoder(real_images)
#             fake_img_features = image_encoder(fake_images)
#             alignment_loss = torch.nn.functional.mse_loss(fake_img_features, real_img_features)
            
#             total_g_loss = g_loss + 1.0 * mode_loss + 0.1 * alignment_loss
#             total_g_loss.backward()
#             g_optimizer.step()
            
#             contrastive_loss_total += contrastive_loss.item()
#             g_loss_total += total_g_loss.item()
#             d_loss_total += d_loss.item()
            
#             pbar.set_postfix({
#                 'Cont_Loss': f'{contrastive_loss_total/(batch_idx+1):.4f}',
#                 'G_Loss': f'{g_loss_total/(batch_idx+1):.4f}',
#                 'D_Loss': f'{d_loss_total/(batch_idx+1):.4f}'
#             })
        
#         if (epoch + 1) % 10 == 0:
#             with torch.no_grad():
#                 sample_eeg = eeg_batch[:8]
#                 sample_noise = torch.randn(8, 100).to(device)
#                 sample_classes = class_labels[:8]
#                 sample_features = eeg_encoder(sample_eeg)
#                 sample_images = generator(sample_features, sample_noise, sample_classes)
                
#                 plt.figure(figsize=(12, 8))
#                 for i in range(8):
#                     plt.subplot(2, 4, i+1)
#                     img = sample_images[i].cpu()
#                     img = (img + 1) / 2 
#                     img = torch.clamp(img, 0, 1)
#                     plt.imshow(img.permute(1, 2, 0))
#                     plt.axis('off')
#                     plt.title(f'Class: {sample_classes[i].item()}')
                
#                 plt.tight_layout()
#                 plt.savefig(f'generated_samples_epoch_{epoch+1}.png')
#                 plt.close()
        
#         contrastive_losses.append(contrastive_loss_total/len(dataloader))
#         g_losses.append(g_loss_total/len(dataloader))
#         d_losses.append(d_loss_total/len(dataloader))
        
#         print(f'Epoch {epoch+1}: Contrastive_Loss = {contrastive_loss_total/len(dataloader):.4f}, '
#               f'G_Loss = {g_loss_total/len(dataloader):.4f}, '
#               f'D_Loss = {d_loss_total/len(dataloader):.4f}')
    
#     plot_training_history_with_contrastive(contrastive_losses, g_losses, d_losses)
    
#     return eeg_encoder, image_encoder, generator, discriminator



# Thoughtviz
def train(eeg_pickle_path, images_dir, num_epochs=100, batch_size=32, lr=0.0002):
    dataset = EEGImageDataset(eeg_pickle_path, images_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    eeg_encoder = EEGEncoder(input_channels=14, sequence_length=32).to(device)
    image_encoder = ImageEncoder().to(device)
    generator = Generator(n_classes=10).to(device) 
    discriminator = Discriminator(n_classes=10).to(device) 
    
    contrastive_optimizer = optim.Adam(
        list(eeg_encoder.parameters()) + list(image_encoder.parameters()), 
        lr=lr, betas=(0.9, 0.999)
    )
    g_optimizer = optim.Adam(
        list(eeg_encoder.parameters()) + list(generator.parameters()), 
        lr=lr, betas=(0.0, 0.9)
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
    
    contrastive_losses = []
    g_losses = []
    d_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        contrastive_loss_total = 0
        g_loss_total = 0
        d_loss_total = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (eeg_batch, real_images, negative_images, class_labels) in enumerate(pbar):
            batch_size_actual = eeg_batch.size(0)
            
            eeg_batch = eeg_batch.to(device)
            real_images = real_images.to(device)
            negative_images = negative_images.to(device)
            class_labels = class_labels.squeeze().to(device)
            
            # =============== Contrastive Learning Phase ===============
            contrastive_optimizer.zero_grad()
            
            _, eeg_contrastive_features = eeg_encoder(eeg_batch, return_contrastive=True)
            img_contrastive_features = image_encoder(real_images)
            negative_img_features = image_encoder(negative_images)
            
            info_nce = info_nce_loss(eeg_contrastive_features, img_contrastive_features)
            triplet = triplet_loss(eeg_contrastive_features, img_contrastive_features, negative_img_features)
            
            contrastive_loss = info_nce + 0.5 * triplet
            contrastive_loss.backward()
            contrastive_optimizer.step()
            
            # =============== GAN Training Phase ===============
            noise = torch.randn(batch_size_actual, 100).to(device)
            
            d_optimizer.zero_grad()
            
            eeg_features = eeg_encoder(eeg_batch).detach()
            
            fake_images = generator(eeg_features, noise, class_labels)
            
            real_images_aug = diff_augment(real_images)
            fake_images_aug = diff_augment(fake_images.detach())
            
            real_output = discriminator(real_images_aug, class_labels)
            fake_output = discriminator(fake_images_aug, class_labels)
            
            d_loss = discriminator_hinge_loss(real_output, fake_output)
            d_loss.backward()
            d_optimizer.step()
            
            g_optimizer.zero_grad()
            
            noise2 = torch.randn(batch_size_actual, 100).to(device)
            eeg_features = eeg_encoder(eeg_batch)
            fake_images = generator(eeg_features, noise, class_labels)
            fake_images2 = generator(eeg_features, noise2, class_labels)
            
            fake_images_aug = diff_augment(fake_images)
            fake_images2_aug = diff_augment(fake_images2)
            
            fake_output = discriminator(fake_images_aug, class_labels)
            fake_output2 = discriminator(fake_images2_aug, class_labels)
            
            g_loss = generator_hinge_loss(fake_output) + generator_hinge_loss(fake_output2)
            
            mode_loss = torch.mean(torch.abs(fake_images2 - fake_images)) / (
                torch.mean(torch.abs(noise2 - noise)) + 1e-5)
            mode_loss = 1.0 / (mode_loss + 1e-5)
            
            with torch.no_grad():
                real_img_features = image_encoder(real_images)
            fake_img_features = image_encoder(fake_images)
            alignment_loss = F.mse_loss(fake_img_features, real_img_features)
            
            total_g_loss = g_loss + 1.0 * mode_loss + 0.1 * alignment_loss
            total_g_loss.backward()
            g_optimizer.step()
            
            contrastive_loss_total += contrastive_loss.item()
            g_loss_total += total_g_loss.item()
            d_loss_total += d_loss.item()
            
            pbar.set_postfix({
                'Cont_Loss': f'{contrastive_loss_total/(batch_idx+1):.4f}',
                'G_Loss': f'{g_loss_total/(batch_idx+1):.4f}',
                'D_Loss': f'{d_loss_total/(batch_idx+1):.4f}'
            })
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_eeg = eeg_batch[:8]
                sample_noise = torch.randn(8, 100).to(device)
                sample_classes = class_labels[:8]
                sample_features = eeg_encoder(sample_eeg)
                sample_images = generator(sample_features, sample_noise, sample_classes)
                
                plt.figure(figsize=(12, 8))
                for i in range(8):
                    plt.subplot(2, 4, i+1)
                    img = sample_images[i].cpu()
                    img = (img + 1) / 2  
                    img = torch.clamp(img, 0, 1)
                    plt.imshow(img.permute(1, 2, 0))
                    plt.axis('off')
                    plt.title(f'Class: {sample_classes[i].item()}')
                
                plt.tight_layout()
                plt.savefig(f'generated_samples_epoch_{epoch+1}.png')
                plt.close()
        
        contrastive_losses.append(contrastive_loss_total/len(dataloader))
        g_losses.append(g_loss_total/len(dataloader))
        d_losses.append(d_loss_total/len(dataloader))
        
        print(f'Epoch {epoch+1}: Contrastive_Loss = {contrastive_loss_total/len(dataloader):.4f}, '
              f'G_Loss = {g_loss_total/len(dataloader):.4f}, '
              f'D_Loss = {d_loss_total/len(dataloader):.4f}')
    
    plot_training_history_with_contrastive(contrastive_losses, g_losses, d_losses)
    
    return eeg_encoder, image_encoder, generator, discriminator

def main():
    eeg_file_path = "/kaggle/input/dongyangli-deleeg-image-decode/sub-01/sub-01/preprocessed_eeg_training.npy"
    images_dir = "/kaggle/input/dongyangli-deleeg-image-decode/osfstorage-archive/training_images/training_images"
    
    print("Training EEG-to-Image GAN with Contrastive Learning...")
    print("="*60)
    
    eeg_encoder, image_encoder, generator, discriminator = train(
        eeg_file_path=eeg_file_path,
        images_dir=images_dir,
        num_epochs=80,  
        batch_size=64,  
        lr=0.0002
    )
    
    torch.save({
        'eeg_encoder': eeg_encoder.state_dict(),
        'image_encoder': image_encoder.state_dict(),
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()
    }, 'eeg_to_image_gan_contrastive.pth')
    
    print("\nTraining completed and models saved!")
    
    dataset = EEGImageDataset(eeg_file_path, images_dir)
    
    print("\nEvaluating EEG-Image feature alignment...")
    avg_similarity, similarities = evaluate_contrastive_alignment(
        eeg_encoder, image_encoder, dataset, num_samples=200
    )
    
    print("\nGenerating visualizations...")
    visualize_generated_samples(eeg_encoder, generator, dataset, num_samples=8)
    
    visualize_feature_space(eeg_encoder, image_encoder, dataset, num_samples=500)
    print(f"Average feature alignment score: {avg_similarity:.4f}")

if __name__ == "__main__":
    main()




