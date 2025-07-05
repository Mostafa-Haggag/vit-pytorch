import torch
import matplotlib.pyplot as plt
import seaborn as sns

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    # x and y are h by w   with y has all indiex the same in each while x has the number from 0 to end
    # This creates two 2D tensors of shape (h, w):
    # y[i][j] = i → same row index repeated along each row (top-down)
    # x[i][j] = j → same column index repeated along each column (left-right)
    # 🧠 Now you have pixel coordinates (y, x) for every position in the image grid.
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    # tensor from 0 to dim//4 then you divide
    omega = 1.0 / (temperature ** omega)
    # These control the wavelength of sine/cosine terms: from low frequency (slow waves) to high frequency (tight waves).
    # 🔬 This lets the model encode both global position and local detail.
    # 256,1*1,16
    # outer product of positions with frequencies for x and y axes.
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    # Each row corresponds to a position's y or x multiplied by all frequencies
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    # Take sin and cos of both x and y position encodings
    #
    # Each part gives shape (h*w, dim//4)
    #
    # cat them to shape (h*w, dim)
    #
    # So now each position (pixel) has a full dim-dimensional vector encoding its x and y location using sine and cosine at different frequencies.
    return pe.type(dtype)
if __name__ == "__main__":

    # ---- Parameters
    H, W, DIM = 2, 2, 8  # 16x16 grid, 64-dim embeddings
    pos_embed = posemb_sincos_2d(H, W, DIM)

    # ---- Visualize with heatmap
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title(f"2D SinCos Positional Embedding Heatmap ({H}x{W}, dim={DIM})")
    sns.heatmap(pos_embed, cmap="viridis", cbar=True)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position Index (y * w + x)")
    plt.tight_layout()
    plt.show()
    # ---- Plot: bar plot per patch in a 2x2 layout
    fig, axes = plt.subplots(H, W, figsize=(8, 6))
    fig.suptitle("Sin-Cos Positional Embedding per Patch (2x2 layout)", fontsize=14)

    for i in range(H):
        for j in range(W):
            idx = i * W + j
            ax = axes[i, j]
            ax.bar(range(DIM), pos_embed[idx].numpy())
            ax.set_title(f"Patch ({i},{j})")
            ax.set_xticks(range(DIM))
            ax.set_ylim([-1.2, 1.2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
