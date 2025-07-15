from random import randrange
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    # layers: list of layers (e.g., nn.Modules).
    #
    # dropout: probability (float between 0 and 1) of dropping each layer
    if dropout == 0:
        return layers
    ## If dropout is 0, return the layers unchanged.



    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout
    # Create a tensor of random values and compare with dropout to decide which layers to drop.

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        # set last layer to no dropping
        to_drop[rand_index] = False
    # Randomly prunes layers during training, possibly acting as a regularizer.
    #
    # Ensures at least one layer remains.
    #
    # Simple and efficient
    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    # : Talking-Heads Attention (CaiT-style)
    # Key features:
    # Adds trainable mixing between attention heads, before and after softmax.
    #
    # This allows heads to communicate and mix their outputs — more expressive.
    #
    # Adds flexibility and better performance in deep models.
    # “Let attention heads communicate directly with each other — both before and after the softmax.”
    # It modifies two places:
        # Pre-softmax head mixing: Modify attention logits (the QK^T scores) before softmax.
        # Post-softmax head mixing: Mix attention weights after softmax.
    # You are linearly combining the original h attention heads into g new heads via a mixing matrix — kind of like projecting into a new head space.
    #
    # This mixing can be useful in more advanced Transformer variants (like in CosFormer, Linear Transformers, or efficient attention models)
    # where you want to re-parameterize or compress attention heads before or after applying softmax.
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads # dimension head set to 64
        self.heads = heads# set to 16
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads
        # x has the shape of Batchsize,n,feature channels
        # h is number of head
        x = self.norm(x)
        context = x if not exists(context) else torch.cat((x, context), dim = 1)
        # If no context is provided → just use x (standard self-attention).
        #
        # If context is provided → concatenate x and context along the sequence (token) dimension.

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        # it has size of 1,64,1024 ----> 1024=64*16----- 16 number of head and 64 dim of head
        # this is tupple of 3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # q has size of1,16,64,64
        # k has size of  1,16,64,64
        # v has size of 1,16,64,64
        # h has size of 16
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # b = batch size
        # h = number of attention heads
        # i = number of query positions (sequence length of Q)
        # j = number of key positions (sequence length of K)
        # d = dimensionality of each query/key vector per head (often d = head_dim)
        # You're computing dot products between queries and keys,
        # across their d dimension, for every query position i and key position j
        # This results in a tensor of shape (b, h, i, j), where each entry is the
        # dot product between the i-th query and the j-th key, for a given head and batch.
        # For each attention head and batch:
        #
        # For every query vector (i), compute a dot product with every key vector (j).
        #
        # This gives a similarity score between each query and all keys.

        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        # This mixes the scores across heads before softmax:
        #
        # Instead of each head doing its own thing, they now share information.
        #
        # Matrix mix_heads_pre_attn of shape [heads, heads] controls this mixing.
        # You're mixing attention heads — i.e., transforming the h heads into g new
        # (possibly fewer or more) synthetic heads using a learned mixing matrix.
        attn = self.attend(dots)
        attn = self.dropout(attn)

        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax
        # This mixes the attention weights (after softmax) across heads:
        #
        # Again, matrix mix_heads_post_attn is trainable and of shape [heads, heads].+
            # ✅ Effect: Instead of each head using its own weights blindly,
            # it can borrow attention behavior from other heads.
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # The rest proceeds as usual — get outputs for each head, then concatenate and project.
        out = rearrange(out, 'b h n d -> b n (h d)')# shape is back to 1,64,1024
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = ind + 1),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = ind + 1)
            ]))
    def forward(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x
'''
ChatGPT said:
Absolutely! Let's break down CaiT (Class-Attention in Image Transformers) —
a powerful architecture that improves the Vision Transformer (ViT) for image classification
Train deeper transformer models for vision (e.g., 48+ layers),

Use a smarter attention mechanism focused on the [CLS] token,

Improve gradient flow and generalization.
CaiT builds on ViT, but makes two major improvements:

🔁 1. LayerScale + Talking Heads Attention
    ✅ Problem:
    Deeper transformers are hard to train — gradients vanish, models overfit or diverge.
    
    ✅ Solution:
    LayerScale: Applies learnable scaling (γ) after residual branches:
    x = x + γ1 * MLP(LN(x))
    x = x + γ2 * Attention(LN(x))
    These small learnable weights stabilize training of very deep transformers (e.g., 48–68 layers).
    
    Talking Heads Attention: Improves attention expressiveness by adding linear layers:
    
    Softmax(QK^T) → Linear → Value

⭐ 2. Class-Attention Block (CAB)
    This is CaiT’s main innovation.
    
    📌 Problem:
    In ViT, the [CLS] token attends to all patches throughout the transformer. 
    This is not optimal as patch tokens don’t specialize.
    
    ✅ Solution:
    CaiT delays the use of the [CLS] token:
    
    Runs standard Self-Attention on patch tokens only for many layers.
    
    Then applies Class-Attention afterward, where only the [CLS] token attends to patch tokens — not vice versa.
    
    This decouples patch feature learning and classification — leading to better specialization.
'''
# Image → Patch Embedding
#          ↓
#    +----------------------+
#    | Patch-Only Transformer|
#    +----------------------+
#          ↓
#      [cls] token appended
#          ↓
#    +----------------------+
#    | Class-Attention Blocks |
#    +----------------------+
#          ↓
#      MLP Head → Output
class CaiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        layer_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        # like every normal transformer
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        # this the new stuff
        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # self attention
        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        # cross attention mechnaisms
        x = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(x[:, 0])
