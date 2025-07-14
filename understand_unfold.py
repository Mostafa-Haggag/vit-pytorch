import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import make_grid
from torch import nn
'''
i want to understand what does unfold do exactly ?
nn.Unfold is like a sliding window or patch extractor.
 It takes an input tensor of shape:

[b, c, H, W]  # batch size, channels, height, width
and extracts overlapping patches from it using a kernel 
(window) and stride, similar to a convolution operation —
 but instead of computing convolutions,
  it just unrolls the patches into columns.
  [b, c, H, W] = [1, 3, 32, 32]  # For example
Now nn.Unfold(kernel_size=7, stride=4, padding=3) does the following:

1. Pads the input:
Padding of 3 pixels is added on each side:

New height and width = 32 + 2 * 3 = 38

2. Slides a 7×7 window over the image, with stride 4:
It moves 4 pixels at a time in both directions (height and width).

At each position, it extracts a patch of size 7×7×c (flattened).
 Computes the number of patches:
 Using the convolution output formula:
output_dim = floor((input_dim + 2 * padding - kernel_size) / stride + 1)
So for 32×32 input:
H_out = (32 + 6 - 7) // 4 + 1 = 31 // 4 + 1 = 7 + 1 = 8
 You get 8×8 = 64 patches.
 Each patch is of shape:
[3, 7, 7] → flattened to 147 (if c = 3)
So the final output of nn.Unfold will be:
[b, c * kernel_size * kernel_size, num_patches] =
[1, 3*7*7, 64] = [1, 147, 64]
# 
Now you have a sequence of 64 tokens (patches),
 each of dimension 147 — ready for the Transformer to process!

'''
# 1. Load an image
# Replace with your own image path or use a sample
image = read_image("lenna.png")  # shape: [C, H, W]
image = image[:3]  # Keep only RGB channels (if extra)
image = T.Resize((32, 32))(image)  # Resize to 32x32 for simplicity
image = image.unsqueeze(0).float() / 255.  # [1, 3, 32, 32], normalize to [0, 1]

# 2. Define unfold
kernel_size = 7
stride = 4
padding = 3

unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

# 3. Apply unfold
patches = unfold(image)  # shape: [1, 3*7*7, num_patches]
num_patches = patches.shape[-1]
patch_size = kernel_size

# 4. Reshape patches to [num_patches, C, H, W]
patches = patches.squeeze(0).T  # shape: [num_patches, patch_dim]
patches = patches.reshape(num_patches, 3, patch_size, patch_size)
print(f"Extracted {num_patches} patches of size {patch_size}x{patch_size}")
# 5. Visualize
grid = make_grid(patches, nrow=8, padding=1)
plt.figure(figsize=(10, 10))
plt.title(f"Extracted {num_patches} patches of size {patch_size}x{patch_size}")
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.show()
