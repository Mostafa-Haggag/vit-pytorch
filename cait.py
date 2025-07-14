import torch
from vit_pytorch.cait import CaiT
if __name__ == "__main__":
    v = CaiT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 12,             # depth of transformer for patch to patch attention only
        cls_depth = 2,          # depth of cross attention of CLS tokens to patch
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05    # randomly dropout 5% of the layers
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img) # (1, 1000)