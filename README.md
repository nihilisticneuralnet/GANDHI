# GANDHI

## Generative Adversarial Network for Decoding High-level Images from EEG signals
# GANDHI: Generative Adversarial Network for Decoding High-level Images from EEG signals

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art deep learning framework that reconstructs visual images from EEG (electroencephalography) signals using generative adversarial networks enhanced with contrastive learning.

## 🧠 Overview

GANDHI represents a breakthrough in brain-computer interface technology, enabling the direct translation of neural activity into visual content. The system combines multiple advanced techniques:

- *Dual-encoder architecture* for EEG and image feature extraction
- *Contrastive learning* for robust cross-modal alignment
- *Spectral normalization* for training stability
- *Class-conditional generation* for multi-category image synthesis
- *Advanced regularization* techniques to prevent mode collapse

## 🔥 Key Features

### Core Capabilities
- *Multi-modal Learning*: Simultaneous training on EEG signals and visual images
- *Contrastive Alignment*: InfoNCE and triplet loss for robust feature matching
- *High-Resolution Generation*: 128×128 RGB image synthesis
- *Class-Conditional*: Support for 1,600+ image categories
- *Real-time Inference*: Optimized for fast EEG-to-image conversion

### Technical Innovations
- *Spectral Normalization*: Ensures discriminator Lipschitz continuity
- *Differential Augmentation*: Prevents discriminator overfitting
- *Mode Seeking Loss*: Reduces mode collapse in generation
- *Feature Alignment Loss*: Bridges EEG and image feature spaces
- *Bidirectional LSTM*: Captures temporal EEG dynamics

## 🏗 Architecture

### System Components


┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   EEG Encoder   │    │ Image Encoder   │    │   Generator     │
│                 │    │                 │    │                 │
│ • BiLSTM layers │    │ • ResNet-50     │    │ • Transposed    │
│ • FC projection │    │ • Projection    │    │   convolutions  │
│ • Contrastive   │    │   head          │    │ • Spectral norm │
│   features      │    │ • L2 normalize  │    │ • Class embed   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Discriminator   │
                    │                 │
                    │ • Conv layers   │
                    │ • Class embed   │
                    │ • Hinge loss    │
                    └─────────────────┘


### Model Specifications

| Component | Architecture | Parameters | Output |
|-----------|-------------|------------|--------|
| EEG Encoder | BiLSTM (128→64) + FC (512) | ~2.1M | 512-dim features |
| Image Encoder | ResNet-50 + MLP | ~25.6M | 512-dim features |
| Generator | 5-layer TransConv | ~15.3M | 128×128×3 images |
| Discriminator | 6-layer Conv | ~8.7M | Real/fake prediction |

## 📊 Dataset Requirements

### EEG Data Format
- *Shape*: (n_samples, 4, 17, 100)
  - n_samples: Number of EEG recordings
  - 4: Number of trials per stimulus
  - 17: EEG channels (electrode positions)
  - 100: Time points (typically 1000ms at 100Hz)

### Image Data Structure

training_images/
├── class_001/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── class_002/
│   ├── image_001.jpg
│   └── ...
└── ...


### Preprocessing Requirements
- *EEG*: Bandpass filtered (0.5-50Hz), baseline corrected, artifacts removed
- *Images*: RGB format, any resolution (auto-resized to 128×128)
- *Synchronization*: EEG trials must correspond to image presentations

## 🚀 Installation

### Prerequisites
bash
# Python 3.7 or higher
python --version

# CUDA-compatible GPU (recommended)
nvidia-smi


### Environment Setup
bash
# Clone the repository
git clone https://github.com/yourusername/gandhi.git
cd gandhi

# Create virtual environment
python -m venv gandhi_env
source gandhi_env/bin/activate  # On Windows: gandhi_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt


### Dependencies
txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
tqdm>=4.60.0
Pillow>=8.0.0


## 🔧 Usage

### Basic Training
python
from gandhi import train_gan_with_contrastive

# Define data paths
eeg_file_path = "path/to/preprocessed_eeg_training.npy"
images_dir = "path/to/training_images/"

# Train the model
eeg_encoder, image_encoder, generator, discriminator = train_gan_with_contrastive(
    eeg_file_path=eeg_file_path,
    images_dir=images_dir,
    num_epochs=100,
    batch_size=32,
    lr=0.0002
)


### Advanced Configuration
python
# Custom training parameters
config = {
    'num_epochs': 150,
    'batch_size': 64,
    'learning_rate': 0.0001,
    'beta1': 0.0,
    'beta2': 0.9,
    'contrastive_weight': 1.0,
    'mode_seeking_weight': 1.0,
    'alignment_weight': 0.1,
    'spectral_norm_iterations': 1
}

# Train with custom configuration
models = train_gan_with_contrastive(
    eeg_file_path=eeg_file_path,
    images_dir=images_dir,
    **config
)


### Image Generation
python
# Load trained models
checkpoint = torch.load('eeg_to_image_gan_contrastive.pth')
eeg_encoder.load_state_dict(checkpoint['eeg_encoder'])
generator.load_state_dict(checkpoint['generator'])

# Generate images from EEG
with torch.no_grad():
    eeg_features = eeg_encoder(eeg_batch)
    noise = torch.randn(batch_size, 100).to(device)
    generated_images = generator(eeg_features, noise, class_labels)


### Evaluation
python
# Evaluate feature alignment
avg_similarity, similarities = evaluate_contrastive_alignment(
    eeg_encoder, image_encoder, dataset, num_samples=500
)

# Visualize results
visualize_generated_samples(eeg_encoder, generator, dataset, num_samples=8)

# Feature space analysis
visualize_feature_space(eeg_encoder, image_encoder, dataset, num_samples=1000)


## 📈 Performance Metrics

### Quantitative Evaluation
- *Feature Alignment*: Cosine similarity between EEG and image features
- *Generation Quality*: Inception Score (IS) and Fréchet Inception Distance (FID)
- *Classification Accuracy*: Downstream task performance
- *Reconstruction Fidelity*: SSIM and LPIPS metrics

### Training Monitoring
- *Contrastive Loss*: InfoNCE + Triplet loss convergence
- *Adversarial Loss*: Generator vs. discriminator dynamics
- *Mode Collapse*: Diversity metrics and sample quality
- *Feature Alignment*: Cross-modal similarity trends

## 🔬 Technical Details

### Loss Functions

#### Contrastive Learning
python
# InfoNCE Loss
L_InfoNCE = -log(exp(sim(eeg_i, img_i)/τ) / Σ_j exp(sim(eeg_i, img_j)/τ))

# Triplet Loss
L_triplet = max(0, ||f_eeg - f_img_pos||² - ||f_eeg - f_img_neg||² + margin)


#### Adversarial Training
python
# Discriminator Hinge Loss
L_D = E[max(0, 1 - D(x_real))] + E[max(0, 1 + D(x_fake))]

# Generator Hinge Loss
L_G = -E[D(x_fake)]


### Architecture Innovations

#### Spectral Normalization
python
# Constrains discriminator to be 1-Lipschitz
W_spectral = W / σ(W)  # σ(W) is largest singular value


#### Differential Augmentation
python
# Prevents discriminator overfitting
x_aug = DiffAugment(x, policy="color,translation")


## 🎯 Applications

### Research Applications
- *Neuroscience*: Understanding visual processing in the brain
- *Brain-Computer Interfaces*: Direct neural control of visual systems
- *Cognitive Science*: Studying mental imagery and perception
- *Medical Diagnosis*: Analyzing visual processing disorders

### Practical Applications
- *Assistive Technology*: Visual communication for speech-impaired individuals
- *Virtual Reality*: Brain-controlled immersive experiences
- *Creative Tools*: AI-assisted art generation from neural signals
- *Accessibility*: Alternative visual interfaces for disabilities

## 🛣 Future Enhancements

### Planned Features
- [ ] *Higher Resolution*: 256×256 and 512×512 image generation
- [ ] *Temporal Dynamics*: Video generation from EEG sequences
- [ ] *Multi-Subject*: Cross-subject generalization improvements
- [ ] *Real-time Processing*: Streaming EEG-to-image conversion
- [ ] *Mobile Deployment*: Lightweight model variants

### Research Directions
- [ ] *Attention Mechanisms*: Transformer-based EEG encoding
- [ ] *Causal Analysis*: Understanding EEG-image relationships
- [ ] *Few-shot Learning*: Adaptation to new subjects/classes
- [ ] *Multimodal Fusion*: Integration with other neural signals
- [ ] *Interpretability*: Explaining model decisions

## 📚 Citation

If you use GANDHI in your research, please cite:

bibtex
@article{gandhi2024,
  title={GANDHI: Generative Adversarial Network for Decoding High-level Images from EEG signals},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black gandhi/
flake8 gandhi/


### Areas for Contribution
- *Data Preprocessing*: EEG signal processing improvements
- *Model Architecture*: New encoder/decoder designs
- *Evaluation Metrics*: Better assessment methods
- *Documentation*: Tutorials and examples
- *Optimization*: Performance improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- *Dataset*: Thanks to [Original Dataset Authors] for providing EEG-image data
- *Inspiration*: Built upon advances in GAN research and neuroscience
- *Community*: PyTorch and deep learning communities for frameworks
- *Funding*: [Grant/Institution information if applicable]

## 📞 Contact

- *Lead Developer*: [Your Name] - [email@domain.com]
- *Research Group*: [Institution/Lab Name]
- *Project Homepage*: [https://gandhi-project.github.io](https://gandhi-project.github.io)

---

<p align="center">
  <img src="assets/gandhi_logo.png" alt="GANDHI Logo" width="200">
</p>

<p align="center">
  <em>Bridging the gap between neural signals and visual perception</em>
</p>
