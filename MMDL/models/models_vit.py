# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer_MT
from torchsummary import summary

__all__ = ['vit_base']


class VisionTransformer(timm.models.vision_transformer_MT.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        # self.multimodal = multimodal

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x, y=None):
        B = x.shape[0]
        # print(B)
        x = self.patch_embed(x)
        # print(x.shape)

        if self.multimodal is not None:
            y = self.multimodal_embed(y)
            x = torch.cat((x, y), dim=1)

        # print(x.shape)
        cls_tokens_icetype = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens_icecon = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens_icetype, x, cls_tokens_icecon), dim=1)
        # print(x.shape)
        # print(self.pos_embed.shape)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # print(x.shape)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            # print(x.shape)
            outcome_icetype = x[:, 0]
            outcome_icecon = x[:, -1]
            # print(x.shape)
            # print(outcome_icecon.shape)

        return outcome_icetype, outcome_icecon

    def forward_head(self, fea_icetype, fea_icecon):
        # if self.global_pool:
        #     x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(fea_icetype)
        y = self.fc_norm(fea_icecon)
        return self.head_icetype(x), self.head_icecon(y)

    def forward(self, x, y=None):
        fea_icetype, fea_icecon = self.forward_features(x, y)
        icetype, icecon = self.forward_head(fea_icetype, fea_icecon)
        return icetype, icecon


def vit_base(logger, args, **kwargs):
    model = VisionTransformer(img_size=30, in_chans=2,
                              num_classes_icetype=args.num_classes_icetype,
                              num_classes_icecon=args.num_classes_icecon,
                              multimodal=args.metadata,
                              patch_size=6,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=6, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    model = VisionTransformer(img_size=30,
                             in_chans=2,
                              patch_size=6,
                             num_classes_icetype=11,
                              num_classes_icecon=12,
                             multimodal='geo_temporal_inci')
    # summary(model, (3, 224, 224))
    # print(model)

    x = torch.randn(16, 2, 30, 30)
    y = torch.randn(16, 8)
    # print(x.shape)
    out_icetype, out_icecon = model(x, y)
    print(out_icetype.shape)
    print(out_icecon.shape)