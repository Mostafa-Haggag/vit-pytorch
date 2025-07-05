import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
#
'''
The FeedForward class you've posted is a position-wise feedforward network used in Transformer architectures.
 This specific implementation includes Layer Normalization before the feedforward block, which differs slightly
  from the original Transformer paper (which used post-norm),
 but is commonly used in Pre-LN Transformers (pre-layer normalization), which have better training
  stability for deeper models.
  # In a Transformer block, the Feedforward Network (FFN) is applied after the self-attention mechanism, 
  and it operates independently on each position (token) in the input sequence. 
'''

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),#  # Normalize the input across feature dimensions
            nn.Linear(dim, hidden_dim),#
            nn.GELU(),# # Expand dimensionality
            nn.Dropout(dropout),# regularization
            nn.Linear(hidden_dim, dim),#  Project back to original dimension
            nn.Dropout(dropout)
        )
        # The FFN introduces non-linearity and allows the model to learn more complex transformations:
        #
        # The first Linear(dim → hidden_dim) increases the representation capacity.
        #
        # GELU (Gaussian Error Linear Unit) adds smooth non-linearity.
        #
        # The second Linear(hidden_dim → dim) brings it back to the original size so residual connections are possible.

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    # This is a multi-head self-attention module, a fundamental part of the Transformer architecture.
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        # dim: input and output embedding dimension per token.
        # heads: number of attention heads.
        # dim_head: size of each head.
        # inner_dim: total concatenated dimension from all heads = heads × dim_head.
        # project_out: if only one head and dim_head == dim, no need for final projection.
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        # Applies layer normalization before attention (Pre-LN Transformer).
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        # Applies softmax over the attention scores (per token).
        # a token refers to a flattened patch of the image that is treated
        # as an element in a sequence—just like a word/token in a sentence for NLP.
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # Projects input into queries (Q), keys (K), and values (V) — all at once.
        # Outputs a tensor of shape [batch, seq_len, inner_dim * 3], which is then split into Q, K, V.
        # this is matrix E that they mentioend when working with multiple heads
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)# Applies layer normalization before attention

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # divide into 3 parts please ## : splits Q, K, V from the same linear layer.
        # what is returned is a tupple ya bro
        # each wiht size of 1,65,1024
        # number of heads is 16

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # From shape [batch, seq_len, inner_dim] to [batch, heads, seq_len, dim_head].

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # Computes attention scores: dot product of Q and K, scaled to stabilize gradients.
        # Shape: [batch, heads, seq_len, seq_len]

        attn = self.attend(dots)# apply the soft max
        # # Convert raw scores to probabilities using softmax.
        attn = self.dropout(attn)
        # Optionally drop some attention weights (dropout).

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # Combine heads: [batch, seq_len, heads * dim_head] = [batch, seq_len, inner_dim].
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool = 'cls',
                 channels = 3,
                 dim_head = 64,
                 dropout = 0.,
                 emb_dropout = 0.):
        '''
         image_size: int of the size of the image ,
         patch_size int of the size of the patch,
         num_classes : number of calsses to be used ,
         dim: Embedding size for each patch/token (e.g. 768)
         depth: Number of Transformer layers (i.e. how "deep" the Transformer is)

         heads: Number of self-attention heads per Transformer block
         mlp_dim: Hidden dimension in the feedforward (MLP) layers inside the Transformer
         pool = 'cls', # How to reduce final token outputs ('cls' or 'mean')
         channels = 3,
         dim_head = 64,Dimension of each attention head (e.g. 64)
         dropout = 0.,
         emb_dropout = 0.

        '''
        super().__init__()
        # create a tuple out of something
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # understandthing number of patches that we will use
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        #  it's a patch embedding layer,
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # Normalizes each patch vector individually to stabilize learning.
            # So the output shape becomes:(b, 64, 3072)
            nn.LayerNorm(patch_dim),# Normalizes each patch vector (of size 3072).
            # Projects each patch from patch_dim (e.g. 3072) to dim (e.g. 768), the model’s embedding size.
            nn.Linear(patch_dim, dim),# Projects each 3072-dim patch vector into a lower (or higher) dimensional embedding space
            nn.LayerNorm(dim),
        )
        # Each of the 64 patches is now a 768-dimensional embedding, ready for input to a transformer.
        # You divide an image into patches → each patch becomes a token.
        '''
        This block is commonly used in Vision Transformers (ViT) to:

        Split an image into patches.
        
        Flatten and project each patch into an embedding.
        
        Normalize the embeddings.
        '''
        # remind me again what is difference between layer norm
        # randn Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1
        # Why do we use layer norm ??
            # To normalize each patch (or token) individually
            # Helps stabilize training and improve convergence
            # Makes model less sensitive to input scale

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
            # Each input token (or patch) must be normalized independently
            # Batch statistics are not reliable due to variable-length sequences or small batch sizes
            # No spatial structure (unlike CNNs)
        # Embedding dimension = dim
        # Batch size = 1
        # Token count = num_patches + 1 → the +1 is usually for a special token like [CLS]
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Then you prepend a [CLS] token → so now you have one more token than patches.
        # Input:       [CLS], patch1, patch2, patch3, ..., patchN
        # Position:    0      , 1     , 2     , 3     , ..., N
        # The [CLS] token interacts (via self-attention) with all other patch tokens.
        # After all layers, the [CLS] token holds a rich summary of the image → it's used for classification.
        # It’s a learnable vector that is prepended to the input sequence.
        # The transformer treats it like any other token, but it's meant to gather information from all other tokens.
        # At the end of the transformer, we use its representation as the final summary of the input.
        #  Transformers don’t have any built-in notion of position (unlike CNNs).
        #  So we add positional information to each token.
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img) # you now have shape of  # (b, 64, 3072)
        # 8*8 h width of patch, 10
        b, n, _ = x.shape
        # n number of patches
        # repeat it per batch
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)
        # x.shape = (b, 65, dim)
        x += self.pos_embedding[:, :(n + 1)]
        # Transformers are position-agnostic — they don’t “see” order unless you add positional encoding.
        #
        # self.pos_embedding is a learnable tensor of shape (1, max_tokens, dim) — here, probably (1, 65, 768)
        x = self.dropout(x)

        x = self.transformer(x)
        # 'cls': use only the [CLS] token (x[:, 0])
        #  ViT uses a learnable class token and returns its embedding.
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # 🔍 Why not just use average (like in SimpleViT)?
        # You can (SimpleViT does this).
        #
        # But [CLS] gives the model more flexibility — it learns what information to extract and focus on.
        #
        # It’s a learnable global representation, rather than a fixed one like average pooling.
        #
        # Think of it like this:
        #
        # Method	How it summarizes image
        # [CLS] token	Learns what to focus on from all patches
        # Mean pooling	Just averages all patch tokens equally
        # How it works inside the Transformer:
        # The [CLS] token starts as a vector of size dim (e.g., 768).
        #
        # It attends to all other patches using self-attention.
        #
        # In each layer, it aggregates information from other patches.
        #
        # At the end: it's like a smart "collector" token that represents the whole image.
        x = self.to_latent(x)
        return self.mlp_head(x)
