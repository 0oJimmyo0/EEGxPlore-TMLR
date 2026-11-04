# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange


def resize_time_embedding(time_embed: torch.Tensor, target_length: int) -> torch.Tensor:
    """Deterministically resize [1, source_length, D] temporal embeddings."""
    if time_embed.ndim != 3:
        raise ValueError(f"Expected time embedding [1,S,D], got {tuple(time_embed.shape)}")
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")
    if time_embed.shape[1] == target_length:
        return time_embed
    resized = F.interpolate(
        time_embed.transpose(1, 2),
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return resized.transpose(1, 2)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Keep the separate q/v bias convention used by the original LaBraM
        # checkpoint, while allowing qkv to be replaced by a LoRA wrapper.
        qkv = self.qkv(x)
        if qkv_bias is not None:
            qkv = qkv + qkv_bias
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn
            
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        # EEG_size = to_2tuple(EEG_size)
        # patch_size = to_2tuple(patch_size)
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


class LaBraMNativeAxisResidualAdapter(nn.Module):
    """Lightweight residual correction over LaBraM patch-token grids [B, C, S, D]."""

    def __init__(
        self,
        dim=200,
        bottleneck=64,
        num_heads=4,
        dropout=0.0,
        init_alpha=1e-3,
        use_channel_mixer=True,
        use_patch_mixer=True,
        use_token_mlp=False,
        depth_dim=0,
        patch_variant="full",
        patch_output_dropout=0.0,
        adapter_operator="attention",
    ):
        super().__init__()
        adapter_operator = str(adapter_operator).strip().lower()
        if adapter_operator not in {"attention", "mlp"}:
            raise ValueError(
                "adapter_operator must be 'attention' or 'mlp'; "
                f"got {adapter_operator!r}"
            )
        patch_variant = str(patch_variant).strip().lower()
        if patch_variant not in {"full", "output_dropout", "bottleneck", "low_rank"}:
            raise ValueError(
                "patch_variant must be one of: full, output_dropout, bottleneck, low_rank; "
                f"got {patch_variant!r}"
            )
        if not 0.0 <= float(patch_output_dropout) < 1.0:
            raise ValueError(
                f"patch_output_dropout must be in [0, 1), got {patch_output_dropout!r}"
            )
        if int(bottleneck) <= 0:
            raise ValueError(f"bottleneck must be positive, got {bottleneck!r}")
        if adapter_operator == "attention" and patch_variant == "low_rank" and int(bottleneck) % int(num_heads) != 0:
            raise ValueError(
                "low_rank adapter bottleneck must be divisible by the number of heads; "
                f"got bottleneck={bottleneck}, num_heads={num_heads}"
            )
        if patch_variant != "output_dropout" and float(patch_output_dropout) != 0.0:
            raise ValueError(
                "patch_output_dropout is only valid with patch_variant='output_dropout'"
            )
        # The variant applies to whichever native axis branch is enabled.
        # In particular, low_rank is also the channel-branch variant for a
        # channel-only adapter, so it must not be rejected merely because the
        # patch branch is disabled.
        if patch_variant != "full" and not (use_channel_mixer or use_patch_mixer):
            raise ValueError("A non-full adapter variant requires an enabled native-axis mixer")
        self.patch_variant = patch_variant
        self.adapter_operator = adapter_operator
        self.patch_output_dropout_p = float(patch_output_dropout)
        self.depth_dim = int(depth_dim)
        self._last_raw_patch_ratio = None
        self._last_raw_channel_ratio = None
        self._last_geometry = None

        if use_channel_mixer:
            self.channel_norm = nn.LayerNorm(dim)
            if adapter_operator == "mlp":
                self.channel_down = nn.Linear(dim, int(bottleneck))
                self.channel_activation = nn.GELU()
                self.channel_dropout = nn.Dropout(float(dropout))
                self.channel_up = nn.Linear(int(bottleneck), dim)
            elif patch_variant == "low_rank":
                self.channel_down = nn.Linear(dim, int(bottleneck))
                self.channel_attn = nn.MultiheadAttention(
                    embed_dim=int(bottleneck), num_heads=num_heads,
                    dropout=dropout, batch_first=True)
                self.channel_up = nn.Linear(int(bottleneck), dim)
            else:
                self.channel_attn = nn.MultiheadAttention(
                    embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.alpha_channel = nn.Parameter(torch.tensor(float(init_alpha)))

        if use_patch_mixer:
            if adapter_operator == "mlp":
                self.patch_norm = nn.LayerNorm(dim)
                self.patch_down = nn.Linear(dim, int(bottleneck))
                self.patch_activation = nn.GELU()
                self.patch_dropout = nn.Dropout(float(dropout))
                self.patch_up = nn.Linear(int(bottleneck), dim)
            elif patch_variant in {"full", "output_dropout"}:
                self.patch_norm = nn.LayerNorm(dim)
                self.patch_attn = nn.MultiheadAttention(
                    embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
                self.patch_output_dropout = nn.Dropout(p=float(patch_output_dropout))
            elif patch_variant == "low_rank":
                self.patch_norm = nn.LayerNorm(dim)
                self.patch_down = nn.Linear(dim, int(bottleneck))
                self.patch_attn = nn.MultiheadAttention(
                    embed_dim=int(bottleneck), num_heads=num_heads,
                    dropout=dropout, batch_first=True)
                self.patch_up = nn.Linear(int(bottleneck), dim)
            else:
                self.singleton_patch_residual = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, int(bottleneck)),
                    nn.GELU(),
                    nn.Linear(int(bottleneck), dim),
                )
            self.alpha_patch = nn.Parameter(torch.tensor(float(init_alpha)))

        if use_token_mlp:
            hidden = max(1, int(bottleneck))
            self.token_norm = nn.LayerNorm(dim)
            self.token_mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dim),
            )
            self.alpha_token = nn.Parameter(torch.tensor(float(init_alpha)))

        if self.depth_dim > 0:
            self.depth_gate = nn.Sequential(
                nn.LayerNorm(self.depth_dim),
                nn.Linear(self.depth_dim, dim),
                nn.Tanh(),
            )
            nn.init.zeros_(self.depth_gate[1].weight)
            nn.init.zeros_(self.depth_gate[1].bias)
        else:
            self.depth_gate = None

    def forward(self, x, depth_summary=None):
        if x.dim() != 4:
            raise ValueError(f"Expected [B,C,S,D], got {tuple(x.shape)}")

        batch_size, channels, patches, dim = x.shape
        self._last_geometry = {
            "adapter_batch_size": batch_size,
            "adapter_channel_count": channels,
            "adapter_patch_count": patches,
            "adapter_embed_dim": dim,
            "channel_attention_sequence_length": channels if hasattr(self, "channel_attn") else 0,
            "channel_mlp_sequence_length": channels if hasattr(self, "channel_down") and self.adapter_operator == "mlp" else 0,
            "channel_spatial_interactions_active": int(
                hasattr(self, "channel_attn") and channels > 1
            ),
            "patch_attention_sequence_length": patches if hasattr(self, "patch_attn") else 0,
            "patch_mlp_sequence_length": patches if hasattr(self, "patch_down") and self.adapter_operator == "mlp" else 0,
            "patch_temporal_interactions_active": int(hasattr(self, "patch_attn") and patches > 1),
            "channel_axis_branch_active": int(self.adapter_operator == "mlp" and hasattr(self, "channel_down")),
            "patch_axis_branch_active": int(self.adapter_operator == "mlp" and hasattr(self, "patch_down")),
        }
        delta = torch.zeros_like(x)

        if self.adapter_operator == "mlp" and hasattr(self, "channel_down"):
            xc = x.permute(0, 2, 1, 3).reshape(batch_size * patches, channels, dim)
            yc = self.channel_up(
                self.channel_dropout(self.channel_activation(self.channel_down(self.channel_norm(xc))))
            )
            yc = yc.reshape(batch_size, patches, channels, dim).permute(0, 2, 1, 3)
            self._last_raw_channel_ratio = float(
                yc.detach().float().norm().div(x.detach().float().norm().clamp_min(1e-12)).cpu()
            )
            delta = delta + self.alpha_channel * yc
        elif hasattr(self, "channel_attn"):
            xc = x.permute(0, 2, 1, 3).reshape(batch_size * patches, channels, dim)
            xc = self.channel_norm(xc)
            if self.patch_variant == "low_rank":
                xc = self.channel_down(xc)
            yc, _ = self.channel_attn(xc, xc, xc, need_weights=False)
            if self.patch_variant == "low_rank":
                yc = self.channel_up(yc)
            yc = yc.reshape(batch_size, patches, channels, dim).permute(0, 2, 1, 3)
            self._last_raw_channel_ratio = float(
                yc.detach().float().norm().div(x.detach().float().norm().clamp_min(1e-12)).cpu()
            )
            delta = delta + self.alpha_channel * yc

        if self.adapter_operator == "mlp" and hasattr(self, "patch_down"):
            xp = x.reshape(batch_size * channels, patches, dim)
            yp = self.patch_up(
                self.patch_dropout(self.patch_activation(self.patch_down(self.patch_norm(xp))))
            )
            yp = yp.reshape(batch_size, channels, patches, dim)
            self._last_raw_patch_ratio = float(
                yp.detach().float().norm().div(x.detach().float().norm().clamp_min(1e-12)).cpu()
            )
            delta = delta + self.alpha_patch * yp
        elif hasattr(self, "patch_attn"):
            xp = x.reshape(batch_size * channels, patches, dim)
            xp = self.patch_norm(xp)
            if self.patch_variant == "low_rank":
                xp = self.patch_down(xp)
            yp, _ = self.patch_attn(xp, xp, xp, need_weights=False)
            if self.patch_variant == "low_rank":
                yp = self.patch_up(yp)
            else:
                yp = self.patch_output_dropout(yp)
            yp = yp.reshape(batch_size, channels, patches, dim)
            self._last_raw_patch_ratio = float(
                yp.detach().float().norm().div(x.detach().float().norm().clamp_min(1e-12)).cpu()
            )
            delta = delta + self.alpha_patch * yp

        if hasattr(self, "singleton_patch_residual"):
            yp = self.singleton_patch_residual(x)
            self._last_raw_patch_ratio = float(
                yp.detach().float().norm().div(x.detach().float().norm().clamp_min(1e-12)).cpu()
            )
            delta = delta + self.alpha_patch * yp

        if hasattr(self, "token_mlp"):
            delta = delta + self.alpha_token * self.token_mlp(self.token_norm(x))

        if self.depth_gate is not None and depth_summary is not None:
            if depth_summary.dim() != 2 or depth_summary.shape[0] != batch_size:
                raise ValueError(
                    f"Expected depth_summary [B,{self.depth_dim}], got {tuple(depth_summary.shape)}"
                )
            gate = 1.0 + 0.1 * self.depth_gate(depth_summary).view(batch_size, 1, 1, dim)
            delta = delta * gate

        return delta


class LoRALinear(nn.Module):
    """A faithful LoRA wrapper with a frozen base Linear layer."""

    def __init__(self, base, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base).__name__}")
        if int(rank) <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank!r}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        base = self.base(x)
        update = F.linear(self.dropout(x), self.lora_A)
        update = F.linear(update, self.lora_B) * self.scaling
        return base + update


def inject_lora(model, rank=8, alpha=16.0, dropout=0.0, target="qkv"):
    """Replace selected LaBraM attention projections with independent LoRA."""
    targets = {item.strip() for item in str(target).split(',') if item.strip()}
    valid = {"qkv", "proj"}
    if not targets or not targets.issubset(valid):
        raise ValueError(f"LoRA target must contain only qkv and/or proj, got {target!r}")
    replaced = []
    for module_name, module in list(model.named_modules()):
        if module_name == "" or module_name.rsplit('.', 1)[-1] not in targets:
            continue
        parent_name, child_name = module_name.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target {module_name} is {type(module).__name__}, not Linear")
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(module_name)
    if not replaced:
        raise RuntimeError(f"No LoRA target modules found for target={target!r}")
    print(f"[LaBraM LoRA] injected rank={rank} alpha={alpha} target={target}: {replaced}", flush=True)
    return replaced


class NeuralTransformer(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, adapter_type="none",
                 adapter_bottleneck=64, adapter_num_heads=4, adapter_dropout=0.0,
                 adapter_variant="full", adapter_patch_output_dropout=0.0,
                 adapter_operator="attention",
                 adapter_init_alpha=0.01, adapter_gamma=1.0,
                 adapter_zero_init_output=False,
                 adapter_seed=12345,
                 adapter_use_token_mlp=False, adapter_depth_mode="none",
                 adapter_depth_k=4, adapter_gamma_zero_skip_branch=False,
                 adapter_fixed_alpha=None, isruc_sequence=False,
                 isruc_sequence_length=20, isruc_sequence_dropout=0.1,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.isruc_sequence = bool(isruc_sequence)
        self.isruc_sequence_length = int(isruc_sequence_length)
        if self.isruc_sequence and self.isruc_sequence_length <= 0:
            raise ValueError("isruc_sequence_length must be positive")

        # To identify whether it is neural tokenizer or neural decoder. 
        # For the neural decoder, use linear projection (PatchEmbed) to project codebook dimension to hidden dimension.
        # Otherwise, use TemporalConv to extract temporal features from EEG signals.
        self.patch_embed = TemporalConv(out_chans=out_chans) if in_chans == 1 else PatchEmbed(EEG_size=EEG_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.isruc_sequence:
            self.sequence_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=4,
                    dim_feedforward=4 * embed_dim,
                    dropout=float(isruc_sequence_dropout),
                    batch_first=True,
                    activation=F.gelu,
                    norm_first=True,
                ),
                num_layers=1,
                enable_nested_tensor=False,
            )

        adapter_type = str(adapter_type).strip().lower()
        if adapter_type not in {"none", "generic", "channel", "patch", "channel_patch"}:
            raise ValueError(
                "adapter_type must be one of: none, generic, channel, patch, channel_patch; "
                f"got {adapter_type!r}"
            )
        adapter_depth_mode = str(adapter_depth_mode).strip().lower()
        if adapter_depth_mode not in {"none", "lastk_delta"}:
            raise ValueError("adapter_depth_mode must be 'none' or 'lastk_delta'")
        self.adapter_type = adapter_type
        self.adapter_variant = str(adapter_variant).strip().lower()
        self.adapter_operator = str(adapter_operator).strip().lower()
        if self.adapter_operator not in {"attention", "mlp"}:
            raise ValueError(
                "adapter_operator must be 'attention' or 'mlp'; "
                f"got {adapter_operator!r}"
            )
        if self.adapter_variant not in {"full", "output_dropout", "bottleneck", "low_rank"}:
            raise ValueError(
                "adapter_variant must be one of: full, output_dropout, bottleneck, low_rank; "
                f"got {adapter_variant!r}"
            )
        self.adapter_patch_output_dropout = float(adapter_patch_output_dropout)
        if not 0.0 <= self.adapter_patch_output_dropout < 1.0:
            raise ValueError(
                "adapter_patch_output_dropout must be in [0, 1), "
                f"got {adapter_patch_output_dropout!r}"
            )
        if self.adapter_variant != "output_dropout" and self.adapter_patch_output_dropout != 0.0:
            raise ValueError(
                "adapter_patch_output_dropout is only valid with adapter_variant='output_dropout'"
            )
        if int(adapter_bottleneck) <= 0:
            raise ValueError(f"adapter_bottleneck must be positive, got {adapter_bottleneck!r}")
        self.adapter_gamma = float(adapter_gamma)
        self.adapter_zero_init_output = bool(adapter_zero_init_output)
        self.adapter_seed = int(adapter_seed)
        self.adapter_depth_mode = adapter_depth_mode
        self.adapter_depth_k = max(1, int(adapter_depth_k))
        self.adapter_gamma_zero_skip_branch = bool(adapter_gamma_zero_skip_branch)
        self.adapter_fixed_alpha = None if adapter_fixed_alpha is None else float(adapter_fixed_alpha)
        self.native_axis_adapter = None
        self._adapter_last_delta_ratio = None
        self._adapter_last_raw_patch_ratio = None
        self._adapter_last_raw_channel_ratio = None

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # Keep adapter RNG independent from the common LaBraM initialization.
        # This preserves dense-vs-adapter parity for all shared parameters.
        if self.adapter_type != "none":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.adapter_seed)
                self.native_axis_adapter = LaBraMNativeAxisResidualAdapter(
                    dim=embed_dim,
                    bottleneck=adapter_bottleneck,
                    num_heads=adapter_num_heads,
                    dropout=adapter_dropout,
                    init_alpha=adapter_init_alpha,
                    use_channel_mixer=self.adapter_type in {"channel", "channel_patch"},
                    use_patch_mixer=self.adapter_type in {"patch", "channel_patch"},
                    # generic is an axis-blind token-wise residual control;
                    # it is deliberately not combined with native mixers.
                    use_token_mlp=adapter_use_token_mlp or self.adapter_type == "generic",
                    depth_dim=embed_dim if self.adapter_depth_mode != "none" else 0,
                    patch_variant=self.adapter_variant,
                    patch_output_dropout=self.adapter_patch_output_dropout,
                    adapter_operator=self.adapter_operator,
                )
                self.native_axis_adapter.apply(self._init_weights)
                if self.adapter_zero_init_output:
                    # Start as an exact identity residual.  The previous
                    # default initialized the final residual projections with
                    # random weights, allowing a sizeable correction before
                    # the native branch had learned a useful direction.
                    zero_modules = []
                    for name in ("channel_up", "patch_up"):
                        module = getattr(self.native_axis_adapter, name, None)
                        if module is not None:
                            zero_modules.append(module)
                    patch_attn = getattr(self.native_axis_adapter, "patch_attn", None)
                    if patch_attn is not None and hasattr(patch_attn, "out_proj"):
                        if not hasattr(self.native_axis_adapter, "patch_up"):
                            zero_modules.append(patch_attn.out_proj)
                    singleton = getattr(self.native_axis_adapter, "singleton_patch_residual", None)
                    if singleton is not None and isinstance(singleton[-1], nn.Linear):
                        zero_modules.append(singleton[-1])
                    token_mlp = getattr(self.native_axis_adapter, "token_mlp", None)
                    if token_mlp is not None and isinstance(token_mlp[-1], nn.Linear):
                        zero_modules.append(token_mlp[-1])
                    for module in zero_modules:
                        nn.init.zeros_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                if self.native_axis_adapter.depth_gate is not None:
                    nn.init.zeros_(self.native_axis_adapter.depth_gate[1].weight)
                    nn.init.zeros_(self.native_axis_adapter.depth_gate[1].bias)
                if self.adapter_fixed_alpha is not None:
                    for name, parameter in self.native_axis_adapter.named_parameters():
                        if name.startswith("alpha_"):
                            parameter.data.fill_(self.adapter_fixed_alpha)
                            parameter.requires_grad_(False)

        if self.native_axis_adapter is not None:
            alpha_info = []
            for name in ("alpha_channel", "alpha_patch", "alpha_token"):
                if hasattr(self.native_axis_adapter, name):
                    value = float(getattr(self.native_axis_adapter, name).detach().cpu())
                    alpha_info.append(f"{name}={value:.6g}")
            print(
                "[LaBraM adapter] native structured residual enabled: "
                f"type={self.adapter_type} gamma={self.adapter_gamma} "
                f"variant={self.adapter_variant} patch_output_dropout={self.adapter_patch_output_dropout} "
                f"operator={self.adapter_operator} "
                f"bottleneck={adapter_bottleneck} "
                f"token_mlp={bool(adapter_use_token_mlp or self.adapter_type == 'generic')} "
                f"depth_mode={self.adapter_depth_mode} depth_k={self.adapter_depth_k} "
                f"fixed_alpha={self.adapter_fixed_alpha} "
                f"zero_init_output={self.adapter_zero_init_output} "
                f"{' '.join(alpha_info)}",
                flush=True,
            )

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def get_adapter_diagnostics(self):
        """Return cheap scalar diagnostics for training logs."""
        if self.native_axis_adapter is None:
            return {}
        diagnostics = {}
        for name in ("alpha_channel", "alpha_patch", "alpha_token"):
            if hasattr(self.native_axis_adapter, name):
                diagnostics[name] = float(getattr(self.native_axis_adapter, name).detach().cpu())
        if self._adapter_last_delta_ratio is not None:
            diagnostics["adapter_delta_ratio"] = self._adapter_last_delta_ratio
        if self._adapter_last_raw_patch_ratio is not None:
            diagnostics["raw_patch_ratio"] = self._adapter_last_raw_patch_ratio
        if self._adapter_last_raw_channel_ratio is not None:
            diagnostics["raw_channel_ratio"] = self._adapter_last_raw_channel_ratio
        if self.native_axis_adapter._last_geometry is not None:
            diagnostics.update(self.native_axis_adapter._last_geometry)

        def grad_norm(*parameters):
            squared_norm = 0.0
            for parameter in parameters:
                if parameter is not None and parameter.grad is not None:
                    squared_norm += float(parameter.grad.detach().float().pow(2).sum().cpu())
            return squared_norm ** 0.5

        def grad_slice_norm(parameter, start, end):
            if parameter is None or parameter.grad is None:
                return 0.0
            return float(parameter.grad.detach().float()[start:end].pow(2).sum().sqrt().cpu())

        patch_attn = getattr(self.native_axis_adapter, "patch_attn", None)
        if patch_attn is not None:
            embed_dim = patch_attn.embed_dim
            in_proj_weight = patch_attn.in_proj_weight
            in_proj_bias = patch_attn.in_proj_bias
            diagnostics["patch_q_grad_norm"] = (
                grad_slice_norm(in_proj_weight, 0, embed_dim)
                ** 2
                + grad_slice_norm(in_proj_bias, 0, embed_dim)
                ** 2
            ) ** 0.5
            diagnostics["patch_k_grad_norm"] = (
                grad_slice_norm(in_proj_weight, embed_dim, 2 * embed_dim)
                ** 2
                + grad_slice_norm(in_proj_bias, embed_dim, 2 * embed_dim)
                ** 2
            ) ** 0.5
            diagnostics["patch_v_grad_norm"] = (
                grad_slice_norm(in_proj_weight, 2 * embed_dim, 3 * embed_dim)
                ** 2
                + grad_slice_norm(in_proj_bias, 2 * embed_dim, 3 * embed_dim)
                ** 2
            ) ** 0.5
            diagnostics["patch_output_projection_grad_norm"] = grad_norm(
                patch_attn.out_proj.weight,
                patch_attn.out_proj.bias,
            )
        channel_attn = getattr(self.native_axis_adapter, "channel_attn", None)
        if channel_attn is not None:
            embed_dim = channel_attn.embed_dim
            in_proj_weight = channel_attn.in_proj_weight
            in_proj_bias = channel_attn.in_proj_bias
            diagnostics["channel_attention_sequence_length"] = float(
                self.native_axis_adapter._last_geometry["adapter_channel_count"]
            )
            diagnostics["channel_q_grad_norm"] = (
                grad_slice_norm(in_proj_weight, 0, embed_dim)
                ** 2
                + grad_slice_norm(in_proj_bias, 0, embed_dim)
                ** 2
            ) ** 0.5
            diagnostics["channel_k_grad_norm"] = (
                grad_slice_norm(in_proj_weight, embed_dim, 2 * embed_dim)
                ** 2
                + grad_slice_norm(in_proj_bias, embed_dim, 2 * embed_dim)
                ** 2
            ) ** 0.5
            diagnostics["channel_v_grad_norm"] = (
                grad_slice_norm(in_proj_weight, 2 * embed_dim, 3 * embed_dim)
                ** 2
                + grad_slice_norm(in_proj_bias, 2 * embed_dim, 3 * embed_dim)
                ** 2
            ) ** 0.5
            diagnostics["channel_output_projection_grad_norm"] = grad_norm(
                channel_attn.out_proj.weight,
                channel_attn.out_proj.bias,
            )
        adapter_grad_sq = 0.0
        adapter_core_grad_sq = 0.0
        alpha_grad_sq = 0.0
        for name, parameter in self.native_axis_adapter.named_parameters():
            if parameter.grad is not None:
                grad_sq = float(parameter.grad.detach().float().pow(2).sum().cpu())
                adapter_grad_sq += grad_sq
                if name.startswith("alpha_"):
                    alpha_grad_sq += grad_sq
                else:
                    adapter_core_grad_sq += grad_sq
        diagnostics["adapter_grad_norm"] = adapter_grad_sq ** 0.5
        diagnostics["adapter_core_grad_norm"] = adapter_core_grad_sq ** 0.5
        diagnostics["alpha_grad_norm"] = alpha_grad_sq ** 0.5
        last_block_grad_sq = 0.0
        for parameter in self.blocks[-1].parameters():
            if parameter.grad is not None:
                last_block_grad_sq += float(parameter.grad.detach().float().pow(2).sum().cpu())
        diagnostics["last_block_grad_norm"] = last_block_grad_sq ** 0.5
        classifier_grad_sq = 0.0
        for parameter in self.head.parameters():
            if parameter.grad is not None:
                classifier_grad_sq += float(parameter.grad.detach().float().pow(2).sum().cpu())
        diagnostics["classifier_grad_norm"] = classifier_grad_sq ** 0.5
        return diagnostics

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _select_pos_embed(self, input_chans, num_channels):
        if self.pos_embed is None:
            return None
        if input_chans is not None:
            return self.pos_embed[:, input_chans]
        # Some downstream datasets in this repo preserve the tensor channel order
        # but do not persist a channel-name manifest. In that case, align the
        # positional slots to the stored tensor shape instead of expanding to all
        # 128 available channel slots.
        return self.pos_embed[:, :num_channels + 1]

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        if self.native_axis_adapter is not None:
            return self.forward_features_with_adapter(
                x,
                input_chans=input_chans,
                return_patch_tokens=return_patch_tokens,
                return_all_tokens=return_all_tokens,
                **kwargs,
            )

        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed_used = self._select_pos_embed(input_chans, n)
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = resize_time_embedding(self.time_embed, input_time_window).unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)
        
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward_features_with_adapter(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        self._adapter_last_delta_ratio = None
        self._adapter_last_raw_patch_ratio = None
        self._adapter_last_raw_channel_ratio = None
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed_used = self._select_pos_embed(input_chans, n)
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = resize_time_embedding(self.time_embed, input_time_window).unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        depth_stats = []
        use_depth = self.adapter_depth_mode == "lastk_delta"
        start_idx = max(0, len(self.blocks) - self.adapter_depth_k)
        for idx, blk in enumerate(self.blocks):
            prev_x = x
            x = blk(x, rel_pos_bias=None)
            if use_depth and idx >= start_idx:
                depth_stats.append((x[:, 1:, :] - prev_x[:, 1:, :]).mean(dim=1))

        patch_tokens = x[:, 1:, :]
        expected_tokens = n * input_time_window
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"LaBraM adapter expected {expected_tokens} patch tokens, got {patch_tokens.shape[1]}"
            )
        if not (self.adapter_gamma_zero_skip_branch and abs(self.adapter_gamma) == 0.0):
            token_grid = patch_tokens.reshape(batch_size, n, input_time_window, patch_tokens.shape[-1])
            depth_summary = torch.stack(depth_stats, dim=1).mean(dim=1) if depth_stats else None
            delta_grid = self.native_axis_adapter(token_grid, depth_summary=depth_summary)
            correction = self.adapter_gamma * delta_grid.reshape(batch_size, expected_tokens, -1)
            self._adapter_last_raw_patch_ratio = getattr(
                self.native_axis_adapter, "_last_raw_patch_ratio", None
            )
            self._adapter_last_raw_channel_ratio = getattr(
                self.native_axis_adapter, "_last_raw_channel_ratio", None
            )
            self._adapter_last_delta_ratio = float(
                correction.detach().float().norm().div(patch_tokens.detach().float().norm().clamp_min(1e-12)).cpu()
            )
            patch_tokens = patch_tokens + correction
            x = torch.cat((x[:, :1, :], patch_tokens), dim=1)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            patch_tokens = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(patch_tokens)
            return self.fc_norm(patch_tokens.mean(1))
        if return_all_tokens:
            return x
        if return_patch_tokens:
            return x[:, 1:]
        return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        '''
        if self.isruc_sequence and x.ndim == 5:
            batch_size, sequence_length, channels, patches, patch_size = x.shape
            if sequence_length != self.isruc_sequence_length:
                raise ValueError(
                    f"ISRUC expects {self.isruc_sequence_length} epochs per sequence, "
                    f"got {sequence_length}"
                )
            epoch_features = self.forward_features(
                x.reshape(batch_size * sequence_length, channels, patches, patch_size),
                input_chans=input_chans,
            )
            sequence_features = self.sequence_encoder(
                epoch_features.reshape(batch_size, sequence_length, self.embed_dim)
            )
            return self.head(sequence_features)
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed_used = self._select_pos_embed(None, n)
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = resize_time_embedding(self.time_embed, input_time_window).unsqueeze(1).expand(batch_size, n, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                # use last norm for all intermediate layers
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")
    
    def get_intermediate_layers(self, x, use_last_norm=False):
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed_used = self._select_pos_embed(None, n)
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = resize_time_embedding(self.time_embed, input_time_window).unsqueeze(1).expand(batch_size, n, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)

        return features


@register_model
def labram_base_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def labram_large_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, out_chans=16, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def labram_huge_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, out_chans=32, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
