import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load real images
real_images = [Image.open(f"temp/real/real_{i}.jpg") for i in range(1, 5)]

# Load depth images
depth_images = [Image.open(f"temp/depth/depth_{i}.jpg") for i in range(1, 5)]

# Create a figure with 6 subplots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))

# Loop through each pair of real and depth images
for i in range(4):
    # Plot real image
    real_image_flipped = np.flipud(real_images[i])
    axes[i, 0].imshow(real_image_flipped)
    axes[i, 0].axis("off")
    axes[i, 0].set_title(f"Real Image {i+1}")

    # Plot depth image
    axes[i, 1].imshow(depth_images[i])
    axes[i, 1].axis("off")
    axes[i, 1].set_title(f"Depth Image {i+1}")

# Adjust layout
plt.tight_layout()
plt.savefig('depth_plot.png', dpi = 400)


# Show the plot
plt.show()