import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
'''
posemb_sincos_2d generates 
a 2D sinusoidal positional embedding, 
often used in Vision Transformers (ViTs)
and other transformer-like architectures
for image data.
'''
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    '''
    In transformer models,
    there's no inherent sense of position.
    This embedding encodes (x, y) positions
     using sine and cosine functions into a
     vector representation — allowing the mode
    l to understand spatial relationships.

    '''
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    # Creates a grid of y and x coordinates for the image with height h and width w.
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    # The embedding vector must be divisible by 4 to split equally for:
    # sin(x), cos(x), sin(y), cos(y)
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    # Creates frequency scales using a
    # logarithmic temperature-based frequency.
    # Similar to how positional
    # encodings are done in transformers.
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    # Computes the outer product of positions with frequencies for x and y axes.
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    # Concatenates sine and cosine of both x and y positional encodings → final embedding per (x, y) position.
    # pe.shape = (h × w, dim) → one vector of size dim per spatial position
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        # just a linear laayer e basta cosi

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 channels = 3,
                 dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        # exactly the same as the vit trtransformaer
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # instead of using random embedding he is using something else completely
        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )
        # SimpleViT uses non-learnable sinusoidal positional embeddings,

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)
        # SimpleViT doesn't use a [CLS] token. It pools the final tokens using a mean over all patches

        x = self.to_latent(x)
        return self.linear_head(x)
