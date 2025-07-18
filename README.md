# GANDHI: Generative Adversarial Network for Decoding High-level Images from EEG signals

A deep learning framework that reconstructs visual images from EEG signals using generative adversarial networks enhanced with contrastive learning.


## Architecture



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




## Acknowledgments

- *Dataset*: Thanks to [Original Dataset Authors] for providing EEG-image data
- *Inspiration*: Built upon advances in GAN research and neuroscience
- *Community*: PyTorch and deep learning communities for frameworks
- *Funding*: [Grant/Institution information if applicable]


