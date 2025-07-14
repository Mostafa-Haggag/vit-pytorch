import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from vit_pytorch.vit import ViT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.efficient import ViT as EfficientViT

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class DistillMixin:
    # Yes — those functions and attributes can come from the ViT class, which is the other parent of DistillableViT.
    # Thanks to Python's multiple inheritance and method resolution order (MRO),
    # methods in DistillMixin can access anything defined in ViT, as long as the final class (DistillableViT)
    # inherits from both.
    def forward(self, img, distill_token = None):
        '''
        img: The input image tensor. Typically shaped like (B, C, H, W).
        distill_token (optional): A learnable token used during distillation
         to carry extra supervision
        '''
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)
        # Converts the image into a sequence of patch embeddings, like in a Vision Transformer.
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        # Adds a learnable [CLS] token at the start, just like in BERT/ViT.
        #
        # Uses einops.repeat to repeat the class token across the batch.
        #
        # After this, x has shape (B, N+1, D).
        x += self.pos_embedding[:, :(n + 1)]
        # Adds position embeddings to the patch + class tokens.
        #
        # This gives the model spatial awareness of token order


        if distilling:
            # if we are ditillling
            distill_tokens = repeat(distill_token, '1 n d -> b n d', b = b)
            x = torch.cat((x, distill_tokens), dim = 1)
            # If a distill_token is provided, it's added after the class + patch tokens.
            #
            # Now x has shape (B, N+2, D).

        x = self._attend(x)
        # This is the transformer block — not shown here,
        # but likely includes attention layers and feedforward layers.
        # x keeps the same shape, now it's the result of transformer processing.

        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]
            # After attention, the distill token is separated from the other tokens.
            # x now has shape (B, N+1, D)
            # distill_tokens has shape (B, D)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # Reduces the sequence of tokens to a single vector for classification.
        # Two options:
            # 'mean' pooling: average all tokens
            # 'cls' token: use just the first token (common in ViT)

        x = self.to_latent(x)
        out = self.mlp_head(x)
        # to_latent: optional layer to change the dimensionality before the head.
        # mlp_head: the final classifier layer.

        if distilling:
            return out, distill_tokens
            # If distillation is active:
                # Returns the classification output and the distill token output
            # Otherwise:
                # Just returns the classification output

        return out

class DistillableViT(DistillMixin, ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        # *args collects positional arguments into a tuple.
        # **kwargs collects keyword arguments (i.e., named arguments) into a dictionary.
        self.args = args# empty tumpy
        self.kwargs = kwargs# a dicitonary
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableT2TViT(DistillMixin, T2TViT):
    def __init__(self, *args, **kwargs):
        super(DistillableT2TViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = T2TViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableEfficientViT(DistillMixin, EfficientViT):
    def __init__(self, *args, **kwargs):
        super(DistillableEfficientViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = EfficientViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        return self.transformer(x)

# knowledge distillation wrapper

class DistillWrapper(Module):
    def __init__(
        self,
        *,
        teacher,
        student,
        temperature = 1.,
        alpha = 0.5,
        hard = False,
        mlp_layernorm = False
    ):
        super().__init__()
        assert (isinstance(student, (DistillableViT, DistillableT2TViT, DistillableEfficientViT))) , 'student must be a vision transformer'

        self.teacher = teacher
        self.student = student

        dim = student.dim# the dim of the transofrmer 1024
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim) if mlp_layernorm else nn.Identity(),
            nn.Linear(dim, num_classes)
        )# final layer for mlp

    def forward(self, img, labels, temperature = None, alpha = None, **kwargs):

        alpha = default(alpha, self.alpha)
        T = default(temperature, self.temperature)

        with torch.no_grad():
            # you pass the iamge by teacher
            teacher_logits = self.teacher(img)

        student_logits, distill_tokens = self.student(img, distill_token = self.distillation_token, **kwargs)
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.cross_entropy(student_logits, labels)

        if not self.hard:
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim = -1),
                F.softmax(teacher_logits / T, dim = -1).detach(),
            reduction = 'batchmean')
            distill_loss *= T ** 2

        else:
            # soft loss
            teacher_labels = teacher_logits.argmax(dim = -1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)

        return loss * (1 - alpha) + distill_loss * alpha
