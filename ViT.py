import torch
import torch.nn as nn

# 定义 Patch Embedding 层
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        # 确保图像尺寸能被 patch 尺寸整除
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W) -> (B, embed_dim, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2)  # (B, embed_dim, num_patches ** 0.5, num_patches ** 0.5) -> (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        return x

# 定义多头自注意力机制
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, N, 3 * embed_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # (B, num_heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, self.embed_dim)  # (B, N, embed_dim)
        output = self.out_proj(attn_output)
        return output

# 定义 Transformer 块
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x1 = self.norm1(x)
        attn_output = self.attn(x1)
        x = x + self.dropout(attn_output)
        x2 = self.norm2(x)
        mlp_output = self.mlp(x2)
        x = x + self.dropout(mlp_output)
        return x

# 定义 Vision Transformer 模型
class VisionTransformer(nn.Module):
    def __init__(self, image_size=20, patch_size=1, in_channels=1024, num_classes=10,
                 embed_dim=256, num_heads=8, num_layers=4, mlp_dim=256, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.image_size = image_size

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        if not list(x.shape)[1:] == [256, 20, 20]:
            return x
        x1 = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_tokens, x1), dim=1)
        x1 = x1 + self.pos_embed
        x1 = self.dropout(x1)

        for block in self.transformer_blocks:
            x1 = block(x1)

        x1 = self.norm(x1)
        x1 = self.linear(x1)
        x1 = x1[:, 1:, :]
        x1 = torch.transpose(x1, -1, -2)
        x1 = x1.reshape(x1.shape[0], x1.shape[1], self.image_size, self.image_size)

        return x + x1


# 测试代码
if __name__ == "__main__":
    batch_size = 2
    channels = 1024
    model = VisionTransformer(20, 1, 512, 10, 256, 8, 4, 256, 0.1)
    x = torch.randn(2, 512, 20, 20)
    output = model(x)
    print(output.shape)
