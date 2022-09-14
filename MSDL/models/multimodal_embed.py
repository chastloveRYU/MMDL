import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from functools import partial


class MultimodalEmbed(nn.Module):
    def __init__(self, embed_dim=768, norm_layer=None):
        super(MultimodalEmbed, self).__init__()

        self.num_patches = 4
        self.proj = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):

        B, N = x.shape
        x = x.view(B, 1, N)
        x = self.proj(x)
        x = x.transpose(1, 2) # BCN -> BNC
        x = self.norm(x)

        # print(x.shape)
        return x


if __name__ == "__main__":
    x = torch.randn(512, 4)
    print(x.shape)

    embed = MultimodalEmbed()
    x = embed(x)
    print(x.shape)

    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

    print(model)



