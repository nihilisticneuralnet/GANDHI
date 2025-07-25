# GANDHI: Generative Adversarial Network for Decoding High-level Images from EEG signals

**GANDHI** combines Generative Adversarial Networks with contrastive learning to decode and reconstruct high-level visual content from EEG brain signals.

## Overview

The project implements a sophisticated pipeline that:
- **Encodes EEG signals** into meaningful feature representations using CNN-LSTM architectures
- **Aligns neural and visual features** through contrastive learning (InfoNCE and triplet loss)
- **Generates high-quality images** using a conditional GAN with spectral normalization
- **Supports multiple datasets** including ThoughtViz and EEG_Image_decode

## Architecture
```mermaid

graph TB
    subgraph "Input Data"
        EEG["EEG Signals<br/>(Batch, 14, 32, 1)"]
        IMG_POS["Positive Images<br/>(Same class as EEG)"]
        IMG_NEG["Negative Images<br/>(Different class)"]
        LABELS["Class Labels<br/>(0-9)"]
    end
    
    subgraph "Feature Encoders"
        EEG_ENC["🧠 EEG Encoder<br/>Conv1D(14→64→128→256)<br/>+ BiLSTM(256→128→64)<br/>+ FC(4096→1024→512)"]
        IMG_ENC["🖼️ Image Encoder<br/>CNN Feature Extractor<br/>+ Projection Head<br/>→ 512D features"]
    end
    
    subgraph "Contrastive Projection"
        EEG_PROJ["EEG Projection Head<br/>FC(512→512→512)<br/>+ L2 Normalization"]
        IMG_PROJ["Image Projection Head<br/>FC(512→512→512)<br/>+ L2 Normalization"]
    end
    
    subgraph "Contrastive Learning"
        SIM_MATRIX["Similarity Matrix<br/>EEG_features @ IMG_features^T<br/>/ temperature(0.07)"]
        INFO_NCE["InfoNCE Loss<br/>Cross-entropy on<br/>similarity matrix"]
        TRIPLET["Triplet Loss<br/>||anchor-positive||²<br/>- ||anchor-negative||²<br/>+ margin"]
    end
    
    subgraph "Output"
        ALIGNED_SPACE["Aligned Feature Space<br/>EEG ↔ Image correspondence<br/>Cosine similarity ≈ 0.7+"]
    end
    
    %% Data Flow - Contrastive Learning
    EEG --> EEG_ENC
    IMG_POS --> IMG_ENC
    IMG_NEG --> IMG_ENC
    
    EEG_ENC --> EEG_PROJ
    IMG_ENC --> IMG_PROJ
    
    EEG_PROJ --> SIM_MATRIX
    IMG_PROJ --> SIM_MATRIX
    
    SIM_MATRIX --> INFO_NCE
    EEG_PROJ --> TRIPLET
    IMG_PROJ --> TRIPLET
    
    INFO_NCE --> ALIGNED_SPACE
    TRIPLET --> ALIGNED_SPACE
    
    %% Styling
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef encoder fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef projection fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef contrastive fill:#fff3e0,stroke:#f57900,stroke-width:2px
    classDef output fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    
    class EEG,IMG_POS,IMG_NEG,LABELS input
    class EEG_ENC,IMG_ENC encoder
    class EEG_PROJ,IMG_PROJ projection
    class SIM_MATRIX,INFO_NCE,TRIPLET contrastive
    class ALIGNED_SPACE output

```

```mermaid
graph TB
    subgraph "Inputs"
        EEG_FEAT["EEG Features<br/>(from trained encoder)<br/>512D - Fixed/Frozen"]
        NOISE["Random Noise<br/>Z ~ N(0,1)<br/>100D"]
        CLASS["Class Labels<br/>(0-9)"]
    end
    
    subgraph "Generator Network"
        CLS_EMB["Class Embedding<br/>Embedding(10, 50)<br/>50D vectors"]
        G_CONCAT["Concatenation<br/>EEG(512) + Noise(100) + Class(50)<br/>= 662D input"]
        G_FC["FC Layer<br/>662 → 12544<br/>Reshape to (256, 7, 7)"]
        G_CONV1["ConvTranspose2D<br/>256→128, K=4, S=2, P=1<br/>+ BatchNorm + LeakyReLU<br/>Output: (128, 14, 14)"]
        G_CONV2["ConvTranspose2D<br/>128→64, K=4, S=2, P=1<br/>+ BatchNorm + LeakyReLU<br/>Output: (64, 28, 28)"]
        G_CONV3["ConvTranspose2D<br/>64→1, K=3, S=1, P=1<br/>+ Tanh Activation<br/>Output: (1, 28, 28)"]
    end
    
    subgraph "Discriminator Network"
        D_CLS_EMB["Class Embedding<br/>Embedding(10, 50)<br/>Project to 28×28"]
        D_CONCAT["Channel Concatenation<br/>Image(1) + Class Map(1)<br/>= 2 channels"]
        D_CONV1["Conv2D<br/>2→64, K=3, S=2, P=1<br/>+ LeakyReLU<br/>Output: (64, 14, 14)"]
        D_CONV2["Conv2D<br/>64→128, K=3, S=2, P=1<br/>+ BatchNorm + LeakyReLU<br/>Output: (128, 7, 7)"]
        D_CONV3["Conv2D<br/>128→256, K=3, S=2, P=1<br/>+ BatchNorm + LeakyReLU<br/>Output: (256, 3, 3)"]
        D_CONV4["Conv2D<br/>256→1, K=3, S=1, P=1<br/>Global Average Pooling<br/>Output: Real/Fake Score"]
    end
    
    subgraph "Loss Functions"
        G_LOSS["Generator Loss<br/>-E[D(G(z))] (Hinge)<br/>+ Mode Seeking Loss<br/>+ Feature Alignment Loss"]
        D_LOSS["Discriminator Loss<br/>E[ReLU(1-D(x))] + E[ReLU(1+D(G(z)))]<br/>(Hinge Loss)"]
    end
    
    subgraph "Output"
        FAKE_IMG["Generated Images<br/>(1, 28, 28)<br/>Reconstructed from EEG"]
        REAL_IMG["Real Images<br/>(Ground truth for comparison)"]
    end
    
    %% Data Flow - GAN
    EEG_FEAT --> G_CONCAT
    NOISE --> G_CONCAT
    CLASS --> CLS_EMB
    CLS_EMB --> G_CONCAT
    
    G_CONCAT --> G_FC
    G_FC --> G_CONV1
    G_CONV1 --> G_CONV2
    G_CONV2 --> G_CONV3
    G_CONV3 --> FAKE_IMG
    
    FAKE_IMG --> D_CONCAT
    REAL_IMG --> D_CONCAT
    CLASS --> D_CLS_EMB
    D_CLS_EMB --> D_CONCAT
    
    D_CONCAT --> D_CONV1
    D_CONV1 --> D_CONV2
    D_CONV2 --> D_CONV3
    D_CONV3 --> D_CONV4
    
    D_CONV4 --> G_LOSS
    D_CONV4 --> D_LOSS
    
    G_LOSS -.-> G_FC
    G_LOSS -.-> G_CONV1
    G_LOSS -.-> G_CONV2
    G_LOSS -.-> G_CONV3
    
    D_LOSS -.-> D_CONV1
    D_LOSS -.-> D_CONV2
    D_LOSS -.-> D_CONV3
    D_LOSS -.-> D_CONV4
    
    %% Styling
    classDef input fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    classDef generator fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef discriminator fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef loss fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class EEG_FEAT,NOISE,CLASS input
    class CLS_EMB,G_CONCAT,G_FC,G_CONV1,G_CONV2,G_CONV3 generator
    class D_CLS_EMB,D_CONCAT,D_CONV1,D_CONV2,D_CONV3,D_CONV4 discriminator
    class G_LOSS,D_LOSS loss
    class FAKE_IMG,REAL_IMG output

```


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
git clone https://github.com/nihilisticneuralnet/GANDHI.git
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
