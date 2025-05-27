import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from einops import rearrange, repeat
_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class Attention_spa(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # print("进入Attention的大小：",x.shape)
        B, N, C = x.shape
        # x = self.qkv(x)
        # print("qkv后x的大小：",x.shape)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # print(1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# def apply_mask(fft_image, mask):
#     # 应用掩码
#     return fft_image * mask
#
#
# def create_masks(rows, cols, radius):
#     # 创建中心点
#     center_row, center_col = rows // 2, cols // 2
#     # 初始化掩码
#     low_pass_mask = torch.zeros((rows, cols), dtype=torch.float32)
#     high_pass_mask = torch.ones((rows, cols), dtype=torch.float32)
#     # 创建低通和高通掩码
#     for row in range(rows):
#         for col in range(cols):
#             if (row - center_row) ** 2 + (col - center_col) ** 2 <= radius ** 2:
#                 low_pass_mask[row, col] = 1.0
#                 high_pass_mask[row, col] = 0.0
#     return low_pass_mask, high_pass_mask
#
#
# def fft_image(image, radius=3):
#     # 将图像数据转换为复数形式
#     image_fft = torch.fft.fft2(image, dim=(-2, -1))
#
#     # 创建低频和高频掩码
#     rows, cols = image.shape[-2], image.shape[-1]
#     low_pass_mask, high_pass_mask = create_masks(rows, cols, radius)
#     low_pass_mask = low_pass_mask.to(image.device)
#     high_pass_mask = high_pass_mask.to(image.device)
#
#     # 应用掩码分离低频和高频成分
#     low_freq = apply_mask(image_fft, low_pass_mask)
#     high_freq = apply_mask(image_fft, high_pass_mask)
#
#     # 对低频和高频成分执行逆 FFT
#     low_freq_image = torch.fft.ifft2(low_freq, dim=(-2, -1)).real
#     high_freq_image = torch.fft.ifft2(high_freq, dim=(-2, -1)).real
#
#     return low_freq_image, high_freq_image
class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(9, 9, dim, 2, dtype=torch.float32) * 0.02)
        # self.complex_weight2 = nn.Parameter(torch.randn(7, 4, dim, 2, dtype=torch.float32) * 0.02)
        # self.complex_weight3 = nn.Parameter(torch.randn(3, 2, dim, 2, dtype=torch.float32) * 0.02)
        # self.conv1 = nn.Conv2d(in_channels=192, out_channels=96, padding=0,
        #                         kernel_size=1, stride=1)
        # self.conv2 = nn.Conv2d(in_channels=192, out_channels=64, padding=0,
        #                        kernel_size=1, stride=1)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x_fft = torch.fft.fftshift(torch.fft.fftn(x, dim=(1, 2)), dim=(1, 2))

        # 创建掩码
        low_freq_mask = torch.zeros((9, 9), dtype=torch.bool, device='cuda')
        mid_freq_mask = torch.zeros((9, 9), dtype=torch.bool, device='cuda')
        high_freq_mask = torch.ones((9, 9), dtype=torch.bool, device='cuda')

        # 定义低频掩码
        low_freq_mask[3:6, 3:6] = True

        # 定义中频掩码
        mid_freq_mask[2:7, 2:7] = True
        mid_freq_mask[3:6, 3:6] = False  # 去掉低频部分

        # 高频掩码已经定义为全1，后续会减去低频和中频部分
        high_freq_mask = high_freq_mask & ~low_freq_mask & ~mid_freq_mask

        # 扩展掩码以匹配数据的形状
        low_freq_mask = low_freq_mask.unsqueeze(0).unsqueeze(-1)
        mid_freq_mask = mid_freq_mask.unsqueeze(0).unsqueeze(-1)
        high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(-1)
        weight = torch.view_as_complex(self.complex_weight)
        # 提取特征
        low_freq_features = x_fft * low_freq_mask * weight
        mid_freq_features = x_fft * mid_freq_mask * weight
        high_freq_features = x_fft * high_freq_mask * weight

        low_freq_features = low_freq_features.real
        mid_freq_features = mid_freq_features.real
        high_freq_features = high_freq_features.real


        # print(low_freq_features.shape,mid_freq_features.shape,high_freq_features.shape)

        return low_freq_features,mid_freq_features,high_freq_features

  #傅里叶变化块
class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, N, C = x.shape
        a = b = int(math.sqrt(N))
        # x = self.filter(self.norm1(x))
        x1,x2,x3 = self.filter(self.norm1(x))
        x1 = self.norm2(x1)
        x2 = self.norm2(x2)
        x3 = self.norm2(x3)
        x1 = x1 + self.drop_path(self.mlp(x1))
        x2 = x2 + self.drop_path(self.mlp(x2))
        x3 = x3 + self.drop_path(self.mlp(x3))
        x = x.view(B, a,b ,C)
        x = x1 + x2 + x3 + x
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.view(B,N,C)
        return x


  #注意力块
class Block_attention_spa(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        num_heads = 6  # 4 for tiny, 6 for small and 12 for base
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention_spa(dim, num_heads=num_heads, qkv_bias=True, qk_scale=False, attn_drop=drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)
        return x


class SFDCT_Former(nn.Module):

    def __init__(self, img_size=9, patch_size=1,
                 in_chans=30,in_chans_spa=30,
                 num_classes=1000, embed_dim=768,
                 depth=12,mlp_ratio=4.,
                 representation_size=None,
                 uniform_drop=False,
                 drop_rate=0.2, drop_path_rate=0.1,
                 norm_layer=None,
                 dropcls=0):
        super().__init__()
        self.name = 'SFDCT-Former'
        self.conv3d_spa = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
        )
        self.conv2d_spa = nn.Sequential(
            nn.Conv2d(in_channels=in_chans * 8, out_channels=in_chans, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(1, 2)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_spa = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans_spa, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()
        for i in range(depth):
                layer = Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i],
                              norm_layer=norm_layer, h=h, w=w)
                self.blocks.append(layer)


        self.blocks_spa = nn.ModuleList()
        for i in range(depth):
            layer_spa = Block_attention_spa(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i],
                                        norm_layer=norm_layer, h=h, w=w)
            self.blocks_spa.append(layer_spa)

        self.norm = norm_layer(embed_dim*4)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

         # cross attention fusion
        self.cross_attention1 = CrossAttention(input_size=192)
        self.cross_attention2 = CrossAttention(input_size=192)
        self.cross_attention3 = CrossAttention(input_size=192)
        self.cross_attention4 = CrossAttention(input_size=192)
        # Classifier head
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Linear(768, num_classes) if num_classes > 0 else nn.Identity()
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    def forward(self, x):

        if x.dim() == 3:
            x = x.unsqueeze(0)  # 添加批次维度
        x = rearrange(x, 'b 1 h w c -> b 1 c h w')

        x = self.conv3d_spa(x)

        x = self.flatten(x)

        x = self.conv2d_spa(x)

        x = self.patch_embed_spa(x)

        x = x + self.pos_embed

        x = self.pos_drop(x)

        layer1_spa = self.blocks_spa[0]
        layer2_spa = self.blocks_spa[1]
        layer3_spa = self.blocks_spa[2]
        layer4_spa = self.blocks_spa[3]

        layer1 = self.blocks[0]
        layer2 = self.blocks[1]
        layer3 = self.blocks[2]
        layer4 = self.blocks[3]

        spa1 = layer1_spa(x)

        fre1 = layer1(x)

        cross1 = self.cross_attention1(spa1, fre1)

        spa2 = layer2_spa(spa1)


        fre2 = layer2(fre1)

        cross2 = self.cross_attention1(spa2, fre2)

        fre3 = layer3_spa(fre2)

        fre3 = layer3(fre2)

        cross3 = self.cross_attention1(spa3, fre3)
        spa4 = layer4_spa(spa3)
        fre4 = layer4(fre3)

        cross4 = self.cross_attention1(spa4, spe4)
        x = torch.cat((cross1, cross2, cross3, cross4), dim=2)
        x= x + spa4 + fre4
        # print(x.shape)

        x = self.norm(x).mean(1)
        x = self.final_dropout(x)
        # print("经过final_dropout后的大小：",x.shape)
        x = self.head(x)
        # print("经过head后的大小:",x.shape)
        return x


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict
class CrossAttention(nn.Module):
    def __init__(self, input_size):
        super(CrossAttention, self).__init__()
        self.input_size = input_size
        self.W_query = nn.Linear(input_size, input_size)
        self.W_key = nn.Linear(input_size, input_size)
        self.W_value = nn.Linear(input_size, input_size)

    def forward(self, x1, x2):
        # Query和Key分别来自两个分支的特征
        query = self.W_query(x1)
        key = self.W_key(x2)

        # 计算注意力权重
        attention_weights = F.softmax(torch.matmul(query, key.transpose(1, 2)), dim=-1)

        # Value来自两个分支的特征
        value1 = self.W_value(x1)
        value2 = self.W_value(x2)

        # 通过注意力权重加权融合两个分支的特征
        attended1 = torch.matmul(attention_weights, value1)
        attended2 = torch.matmul(attention_weights.transpose(1, 2), value2)

        # 最终融合特征
        fused_feature = attended1 + attended2

        return fused_feature

def sfdct_Former(pretrained=False, **kwargs):
    model = SFDCT_Former(
        img_size=9,
        patch_size=1,
        in_chans=40,
        in_chans_spa=40,
        num_classes=16,
        embed_dim=192,
        depth=12,
        mlp_ratio=4.,
        **kwargs)
    # model.default_cfg = _cfg()
    return model