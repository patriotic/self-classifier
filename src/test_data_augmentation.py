import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from src import utils

# Define the path to the image
image_path = ('D:\FAU\RL\self-classifier\scratch\sample_images\imagenet.jpeg')

# Load the image using PIL
image = Image.open(image_path)

# Define the data augmentation object
global_crops_scale = (0.4, 1.0)
local_crops_scale = (0.05, 0.4)
local_crops_number = 6
data_augmentation = utils.DataAugmentation(global_crops_scale, local_crops_scale, local_crops_number)

# Apply data augmentation to the image
augmented_crops = data_augmentation(image)

# Create a square figure with three rows
num_cols = len(augmented_crops) + 1
plt.figure(figsize=(8, 8))

# Plot original image in the middle of the first row
plt.subplot(3, 3, 1)
plt.imshow(image)
plt.title('Original Image')

# Plot augmented crops in the second and third rows
for i, crop in enumerate(augmented_crops):
    plt.subplot(3, 3, i + 2)
    plt.imshow(transforms.ToPILImage()(crop))
    plt.title(f'Augmented View {i + 1}')

# Remove axis labels for clarity
for ax in plt.gcf().axes:
    ax.axis('off')

# Adjust spacing between subplots
plt.tight_layout(pad=2)
plt.show()