# GANDHI: Generative Adversarial Network for Decoding High-level Images from EEG signals

**GANDHI** combines Generative Adversarial Networks with contrastive learning to decode and reconstruct high-level visual content from EEG brain signals.

## Overview

The project implements a sophisticated pipeline that:
- **Encodes EEG signals** into meaningful feature representations using CNN-LSTM architectures
- **Aligns neural and visual features** through contrastive learning (InfoNCE and triplet loss)
- **Generates high-quality images** using a conditional GAN with spectral normalization
- **Supports multiple datasets** including ThoughtViz and EEG_Image_decode

## Architecture

### Core Components

1. **EEG Encoder**: Processes multi-channel EEG signals (14 channels, 32 time points)
   - 1D Convolutional layers for spatial feature extraction
   - Bidirectional LSTM for temporal modeling
   - Contrastive projection head for feature alignment

2. **Image Encoder**: Extracts features from visual stimuli
   - CNN architecture optimized for target image types (MNIST/Natural images)
   - Projection head for contrastive learning compatibility

3. **Generator**: Conditional image generation from EEG features
   - Spectral normalization for training stability
   - Class-conditional generation with embedding layers
   - Transposed convolutional layers for image synthesis

4. **Discriminator**: Adversarial training component
   - Multi-scale feature discrimination
   - Class-conditional architecture

### Training Strategy

- **Phase 1**: Contrastive learning for EEG-image feature alignment
- **Phase 2**: Adversarial training with hinge loss and mode collapse prevention
- **Regularization**: Spectral normalization, dropout, and data augmentation

## Datasets

### 1. ThoughtViz Dataset
- **Source**: [ThoughtViz Repository](https://github.com/ptirupat/ThoughtViz)
- **Content**: EEG recordings paired with MNIST digit and ImageNet images
- **Format**: Pickle files with preprocessed EEG data
- **Classes**: 10 classes

### 2. EEG_Image_decode Dataset
- **Source**: [EEG_Image_decode Repository](https://github.com/dongyangli-del/EEG_Image_decode)
- **Content**: EEG recordings with natural image stimuli
- **Format**: NumPy arrays and image directories
- **Classes**: Multiple object categories from ImageNet

## Installation

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/GANDHI.git
cd GANDHI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the model**
```bash
cd src
python main.py
```


### Training Example

```python
from src.main import train_gan_with_mnist

# Train on ThoughtViz dataset
eeg_encoder, image_encoder, generator, discriminator = train_gan_with_mnist(
    eeg_pickle_path="data/thoughtviz/data.pkl",
    num_epochs=100,
    batch_size=64,
    lr=0.0002
)
```

## Results and Evaluation

### Metrics
- **Contrastive Alignment**: Cosine similarity between EEG and image features
- **Generation Quality**: FID (Fréchet Inception Distance) scores
- **Classification Accuracy**: Downstream task performance



## Technical Details

### Key Innovations

1. **Spectral Normalization**: Ensures training stability and prevents mode collapse
2. **Contrastive Learning**: Aligns EEG and visual feature spaces using InfoNCE and triplet loss
3. **Multi-scale Architecture**: Processes EEG signals at multiple temporal resolutions
4. **Mode Seeking Loss**: Prevents generator collapse and improves diversity

### Loss Functions

- **Contrastive Loss**: InfoNCE + Triplet Loss
- **Adversarial Loss**: Hinge loss for stable training
- **Feature Alignment Loss**: MSE between generated and real image features
- **Mode Seeking Loss**: Encourages output diversity

## References

- https://medarc-ai.github.io/mindeye/
- https://github.com/prajwalsingh/EEG2Image
- https://github.com/ptirupat/ThoughtViz
- https://github.com/dongyangli-del/EEG_Image_decode
