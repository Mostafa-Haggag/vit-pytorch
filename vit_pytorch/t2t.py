import math
import torch
from torch import nn

from vit_pytorch.vit import Transformer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# Tokens-to-Token Vision Transformer (T2T-ViT)
# helpers

def exists(val):
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding):
    '''
    This equation calculates the output dimension (height or width) of a convolutional layer, given:
    image_size: the input dimension (height or width)
    kernel_size: the size of the filter (e.g., 3 for a 3x3 kernel)
    stride: how many pixels the filter moves at each step
    padding: how many pixels are added to the border of the input (usually 0 or 1)
    # Always take floor (or cast to int) because you can't have partial output pixels — only whole ones.
    '''
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        # This assumes that the input x is of shape:
            # [b, h*w, c]
                # batch size
                # h*w h*w is a flattened image, meaning the image was
                    # originally 2D but has been flattened into a 1D sequence
                # c is the number of channels (e.g., 1 for grayscale, 3 for RGB)
        # The code reshapes this flat image representation back into an image format.
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

# main class
'''
T2T: A smarter patching strategy
Instead of just slicing the image into patches once, T2T-ViT applies multiple local operations (like unfolding) to progressively transform the image into tokens.
Specifically, to avoid information loss in generating tokens from the re-structurizated image, we split it into patches with overlapping.
🔁 How it works:
Iterative Unfolding (like convolution) – Instead of a single patch extraction, the image is unfolded multiple times:

Each unfold extracts overlapping local regions (patches).

Then rearranges and reprojects the data into new tokens.

Local Transformers – Between these unfoldings, a lightweight Transformer block models the local dependencies (relationships between neighboring patches).

Final tokens – After several unfold steps, you get tokens that encode richer local structures (like textures and edges), better than raw 16x16 pixel patches.
As such, each patch is correlated with surrounding patches to establish a prior that there should be
 stronger correlations between surrounding tokens. The tokens in each split patch are concatenated as one token 
(Tokens-to-Token, Fig. 3), and thus the local information can be aggregated from surrounding pixels and patches.

'''
class T2TViT(nn.Module):
    def __init__(self, *, image_size, num_classes,
                 dim,
                 depth = None,
                 heads = None,
                 mlp_dim = None,
                 pool = 'cls',
                 channels = 3,
                 dim_head = 64,
                 dropout = 0.,
                 emb_dropout = 0.,
                 transformer = None,
                 t2t_layers = ((7, 4), (3, 2), (3, 2))):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        layer_dim = channels # in the beginging 3
        output_image_size = image_size # set to the image size
        # Stage 1: Apply Unfold(7, stride=4) → initial tokenization from raw image.
        #
        # Stage 2/3: Do Rearrange -> Unfold -> Rearrange -> Transformer:
        #
        # Transformer models short-range dependencies between the patches before the next unfolding.
        #
        # Helps to make the tokens more expressiv
        #
        for i, (kernel_size, stride) in enumerate(t2t_layers):
            layer_dim *= kernel_size ** 2
            is_first = i == 0 # returns true or False if it is first layer
            is_last = i == (len(t2t_layers) - 1) # if it is last layer
            output_image_size = conv_output_size(image_size=output_image_size,
                                                 kernel_size=kernel_size,
                                                 stride= stride,
                                                 padding=stride // 2)
            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),# Reshape back to image if needed
                nn.Unfold(kernel_size = kernel_size, stride = stride, padding = stride // 2), # Patchify image with kernel & stride
                # it extracts [1, C*kernelsize*kernelsize, num_patches]
                # output_dim = floor((input_dim + 2 * padding - kernel_size) / stride + 1)
                # size of each patch and width of each patch
                # so number of patches hw and width
                Rearrange('b c n -> b n c'),# Flatten patches into tokens
                Transformer(dim = layer_dim, # Add local Transformer
                            heads = 1,
                            depth = 1,
                            dim_head = layer_dim,
                            mlp_dim = layer_dim,
                            dropout = dropout) if not is_last else nn.Identity(),
            ])
        # From Image to Tokens
        # Start with image img of shape [b, 3, H, W]
        #
        # Apply T2T: Progressive unfoldings + local transformer → get final tokens.
        #
        # Add [CLS] token + positional embeddings.
        #
        # Run full Transformer (global attention across tokens).
        #
        # Final classification through MLP head.
        layers.append(nn.Linear(layer_dim, dim))
        self.to_patch_embedding = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        # inside in here is the special part
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
