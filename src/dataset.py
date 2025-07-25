import torch
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms

# .npy file
# class EEGImageDataset(Dataset):
#     """Dataset for EEG-Image pairs with support for contrastive learning"""
#     def __init__(self, eeg_file_path, images_dir, transform=None):
#         data_dict = np.load(eeg_file_path, allow_pickle=True).item()
#         self.eeg_data = data_dict['preprocessed_eeg_data']
        
#         self.images_dir = Path(images_dir)
#         self.class_dirs = sorted([d for d in self.images_dir.iterdir() if d.is_dir()])
#         self.class_to_idx = {d.name: i for i, d in enumerate(self.class_dirs)}
        
#         self.image_paths = []
#         self.image_labels = []
        
#         for class_idx, class_dir in enumerate(self.class_dirs):
#             image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
#             for img_path in image_files:
#                 self.image_paths.append(img_path)
#                 self.image_labels.append(class_idx)
        
#         self.class_to_images = {}
#         for i, label in enumerate(self.image_labels):
#             if label not in self.class_to_images:
#                 self.class_to_images[label] = []
#             self.class_to_images[label].append(i)
        
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((128, 128)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
        
#         print(f"Dataset: {len(self.eeg_data)} EEG samples, {len(self.image_paths)} images")
#         print(f"Classes: {len(self.class_dirs)}")
        
#     def __len__(self):
#         return min(len(self.eeg_data), len(self.image_paths))
    
#     def __getitem__(self, idx):
#         eeg = torch.FloatTensor(self.eeg_data[idx])
        
#         if idx < len(self.image_paths):
#             img_idx = idx
#         else:
#             img_idx = random.randint(0, len(self.image_paths) - 1)
            
#         img_path = self.image_paths[img_idx]
#         img_class = self.image_labels[img_idx]
        
#         image = Image.open(img_path).convert('RGB')
#         image = self.transform(image)
        
#         negative_class = random.choice([c for c in self.class_to_images.keys() if c != img_class])
#         negative_img_idx = random.choice(self.class_to_images[negative_class])
#         negative_img_path = self.image_paths[negative_img_idx]
#         negative_image = Image.open(negative_img_path).convert('RGB')
#         negative_image = self.transform(negative_image)
        
#         return eeg, image, negative_image, torch.LongTensor([img_class])



# .pkl file (Thoughtviz)
class EEGImageDataset(Dataset):
    def __init__(self, eeg_pickle_path, images_dir, transform=None):
        with open(eeg_pickle_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
        
        self.eeg_data = data_dict['x_train']  # (45390, 14, 32, 1)
        self.eeg_labels = data_dict['y_train']  # (45390, 10) - one-hot encoded
        
        self.class_labels = np.argmax(self.eeg_labels, axis=1)
        
        self.images_dir = Path(images_dir)
        self.class_dirs = sorted([d for d in self.images_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {d.name: i for i, d in enumerate(self.class_dirs)}
        
        self.image_paths = []
        self.image_labels = []
        
        for class_idx, class_dir in enumerate(self.class_dirs):
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpeg'))
            for img_path in image_files:
                self.image_paths.append(img_path)
                self.image_labels.append(class_idx)
        
        self.class_to_images = {}
        for i, label in enumerate(self.image_labels):
            if label not in self.class_to_images:
                self.class_to_images[label] = []
            self.class_to_images[label].append(i)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Dataset: {len(self.eeg_data)} EEG samples, {len(self.image_paths)} images")
        print(f"Classes: {len(self.class_dirs)}")
        print(f"EEG shape: {self.eeg_data.shape}")
        print(f"EEG labels shape: {self.eeg_labels.shape}")
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = torch.FloatTensor(self.eeg_data[idx])  # (14, 32, 1)
        eeg_class = self.class_labels[idx]
        
        if eeg_class in self.class_to_images and len(self.class_to_images[eeg_class]) > 0:
            img_idx = random.choice(self.class_to_images[eeg_class])
        else:
            img_idx = random.randint(0, len(self.image_paths) - 1)
            
        img_path = self.image_paths[img_idx]
        img_class = self.image_labels[img_idx]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        available_classes = [c for c in self.class_to_images.keys() if c != img_class and len(self.class_to_images[c]) > 0]
        if available_classes:
            negative_class = random.choice(available_classes)
            negative_img_idx = random.choice(self.class_to_images[negative_class])
            negative_img_path = self.image_paths[negative_img_idx]
            negative_image = Image.open(negative_img_path).convert('RGB')
            negative_image = self.transform(negative_image)
        else:
            negative_img_idx = random.choice([i for i in range(len(self.image_paths)) if self.image_labels[i] != img_class])
            negative_img_path = self.image_paths[negative_img_idx]
            negative_image = Image.open(negative_img_path).convert('RGB')
            negative_image = self.transform(negative_image)
        
        return eeg, image, negative_image, torch.LongTensor([img_class])
