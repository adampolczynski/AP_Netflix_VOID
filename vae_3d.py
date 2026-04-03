"""
CogVideoX-5B 3D Causal VAE — pure PyTorch implementation
----------------------------------------------------------
Reconstructed from `cogvideo5bvae.safetensors` (436 keys).

Architecture:
  Encoder: conv_in → [DownBlock×4] → MidBlock → norm_out → conv_out
  Decoder: conv_in → MidBlock → [UpBlock×4] → norm_out → conv_out

Key design choices (verified from state-dict shapes):
  • Conv layers: CausalConv3d (causal temporal padding + same spatial)
  • Encoder norms: GroupNorm(32, C)
  • Decoder norms: SpatialNorm3D (conditioning on original latent z)
  • Spatial compression: 8× (3 spatial downsamplers, 3 spatial upsamplers)
  • Temporal compression: 4× (first 2 down/up-blocks; avg-pool1d / F.interpolate)
  • Latent channels: 16  |  scaling_factor: 1.15258426
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def _silu(x):
    return F.silu(x)


class _CausalConvWrapper(nn.Module):
    """Thin wrapper: holds a regular Conv3d but applies causal temporal padding
    and symmetric spatial padding in forward().  The underlying weight key is
    ``<name>.conv.{weight,bias}`` — matching the safetensors key layout.
    """

    def __init__(self, in_c, out_c, kernel_size=(3, 3, 3), stride=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        kt, kh, kw = kernel_size
        self.t_pad  = kt - 1          # causal: pad left only
        self.h_pad  = kh // 2
        self.w_pad  = kw // 2
        # The underlying weight is stored at `self.conv`
        self.conv = nn.Conv3d(in_c, out_c, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # F.pad order (last dim first): w_left w_right h_left h_right t_left t_right
        x = F.pad(x, (self.w_pad, self.w_pad,
                      self.h_pad, self.h_pad,
                      self.t_pad, 0))
        return self.conv(x)


# ─────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────

class _GroupNorm(nn.GroupNorm):
    """GroupNorm with 32 groups.  Forward is identical to nn.GroupNorm."""
    def __init__(self, num_channels):
        super().__init__(num_groups=32, num_channels=num_channels,
                         eps=1e-6, affine=True)


class SpatialNorm3D(nn.Module):
    """Decoder spatial normalisation conditioned on the original latent z.

    Keys (all 1×1×1 CausalConv3d):
      norm_layer : GroupNorm(32, f_channels)
      conv_y     : maps z → scale   [f_channels, z_channels, 1,1,1]
      conv_b     : maps z → bias    [f_channels, z_channels, 1,1,1]

    Forward: norm_layer(h) * conv_y(z_resized) + conv_b(z_resized)
    """

    def __init__(self, f_channels, z_channels=16):
        super().__init__()
        self.norm_layer = _GroupNorm(f_channels)
        self.conv_y = _CausalConvWrapper(z_channels, f_channels, (1, 1, 1))
        self.conv_b = _CausalConvWrapper(z_channels, f_channels, (1, 1, 1))

    def forward(self, h, z):
        # Resize z to the spatial and temporal size of h
        if z.shape[2:] != h.shape[2:]:
            z = F.interpolate(z.float(), size=h.shape[2:], mode="nearest").to(h.dtype)
        norm_h = self.norm_layer(h)
        return norm_h * self.conv_y(z) + self.conv_b(z)


# ─────────────────────────────────────────────────────────────
# Residual blocks
# ─────────────────────────────────────────────────────────────

class _EncoderResBlock(nn.Module):
    """Encoder resnet block: GroupNorm → SiLU → CausalConv3d × 2.
    Keys: norm1, norm2 (plain weight/bias), conv1.conv, conv2.conv,
          optionally conv_shortcut (1×1×1).
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.norm1          = _GroupNorm(in_c)
        self.conv1          = _CausalConvWrapper(in_c, out_c)
        self.norm2          = _GroupNorm(out_c)
        self.conv2          = _CausalConvWrapper(out_c, out_c)
        # shortcut: plain Conv3d 1×1×1 (no causal padding needed)
        self.conv_shortcut  = (nn.Conv3d(in_c, out_c, 1, 1, 0)
                               if in_c != out_c else None)

    def forward(self, x):
        h = _silu(self.norm1(x))
        h = self.conv1(h)
        h = _silu(self.norm2(h))
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class _DecoderResBlock(nn.Module):
    """Decoder resnet block: SpatialNorm3D → SiLU → CausalConv3d × 2.
    Keys: norm1, norm2 (with conv_b/conv_y/norm_layer), conv1.conv, conv2.conv,
          optionally conv_shortcut (1×1×1).
    """

    def __init__(self, in_c, out_c, z_channels=16):
        super().__init__()
        self.norm1          = SpatialNorm3D(in_c,  z_channels)
        self.conv1          = _CausalConvWrapper(in_c, out_c)
        self.norm2          = SpatialNorm3D(out_c, z_channels)
        self.conv2          = _CausalConvWrapper(out_c, out_c)
        # shortcut: plain Conv3d 1×1×1 (no causal padding needed)
        self.conv_shortcut  = (nn.Conv3d(in_c, out_c, 1, 1, 0)
                               if in_c != out_c else None)

    def forward(self, x, z):
        h = _silu(self.norm1(x, z))
        h = self.conv1(h)
        h = _silu(self.norm2(h, z))
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


# ─────────────────────────────────────────────────────────────
# Downsamplers / Upsamplers
# ─────────────────────────────────────────────────────────────

class _Downsampler3D(nn.Module):
    """Spatial 2× downsampler with optional causal temporal 2× compression.
    The weight key is `conv.{weight,bias}` and is a 2D Conv (3×3).
    """

    def __init__(self, channels, compress_time=False):
        super().__init__()
        self.compress_time = compress_time
        # 2D spatial downsample conv (applied per-frame)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        B, C, T, H, W = x.shape

        # Optional temporal compression via causal avg-pool1d
        if self.compress_time:
            # reshape to (B*H*W, C, T)
            xt = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
            if T % 2 == 1:
                # keep first frame, avg-pool the rest (only if non-empty)
                first = xt[:, :, :1]
                rest  = xt[:, :, 1:]
                if rest.shape[-1] > 0:
                    rest = F.avg_pool1d(rest, kernel_size=2, stride=2)
                xt = torch.cat([first, rest], dim=-1)
            else:
                xt = F.avg_pool1d(xt, kernel_size=2, stride=2)
            T = xt.shape[-1]
            x = xt.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2)  # back to [B,C,T,H,W]

        # Spatial downsampling: apply 2D conv per-frame
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.pad(x, (0, 1, 0, 1))                   # asymmetric spatial pad
        x = self.conv(x)
        _, C2, H2, W2 = x.shape
        x = x.reshape(B, T, C2, H2, W2).permute(0, 2, 1, 3, 4)
        return x


class _Upsampler3D(nn.Module):
    """Spatial 2× upsampler with optional causal temporal 2× expansion.
    Weight key: `conv.{weight,bias}` (2D Conv 3×3).
    """

    def __init__(self, channels, compress_time=False):
        super().__init__()
        self.compress_time = compress_time
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, T, H, W = x.shape

        if self.compress_time:
            # compress_time: upsampling handled here (temporal + spatial 2× combined)
            if T > 1 and T % 2 == 1:
                x_first = x[:, :, 0]                              # [B, C, H, W]
                x_rest  = x[:, :, 1:]                             # [B, C, T-1, H, W]
                x_first = F.interpolate(x_first, scale_factor=2.0, mode="nearest")
                x_rest  = F.interpolate(x_rest,  scale_factor=2.0, mode="nearest")
                x_first = x_first.unsqueeze(2)
                x = torch.cat([x_first, x_rest], dim=2)
            elif T > 1:
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            else:                                                  # T == 1
                x = x.squeeze(2)
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
                x = x.unsqueeze(2)
            # apply conv only (no second interpolate)
            B2, C2, T2, H2, W2 = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B2 * T2, C2, H2, W2)
            x = self.conv(x)
            _, C3, H3, W3 = x.shape
            x = x.reshape(B2, T2, C3, H3, W3).permute(0, 2, 1, 3, 4)
        else:
            # spatial-only: interpolate per-frame then conv
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = self.conv(x)
            _, C2, H2, W2 = x.shape
            x = x.reshape(B, T, C2, H2, W2).permute(0, 2, 1, 3, 4)

        return x


# ─────────────────────────────────────────────────────────────
# Encoder building blocks
# ─────────────────────────────────────────────────────────────

class _EncoderDownBlock(nn.Module):
    def __init__(self, in_c, out_c, num_resnet, add_downsample, compress_time):
        super().__init__()
        blocks = []
        for i in range(num_resnet):
            blocks.append(_EncoderResBlock(in_c if i == 0 else out_c, out_c))
        self.resnets = nn.ModuleList(blocks)
        self.downsamplers = (
            nn.ModuleList([_Downsampler3D(out_c, compress_time)])
            if add_downsample else None
        )

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        if self.downsamplers:
            for d in self.downsamplers:
                x = d(x)
        return x


class _EncoderMidBlock(nn.Module):
    def __init__(self, channels, num_resnet=2):
        super().__init__()
        self.resnets = nn.ModuleList(
            [_EncoderResBlock(channels, channels) for _ in range(num_resnet)]
        )

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        return x


class CogVideoX5BEncoder(nn.Module):
    """Encoder: image [B,3,T,H,W] → latent params [B,32,T',H',W']"""

    # (in_c, out_c, num_resnets, add_downsample, compress_time)
    _BLOCKS = [
        (128, 128, 3, True,  True),   # block 0: spatial+temporal 2×
        (128, 256, 3, True,  True),   # block 1: spatial+temporal 2×
        (256, 256, 3, True,  False),  # block 2: spatial only
        (256, 512, 3, False, False),  # block 3: no downsampling
    ]

    def __init__(self):
        super().__init__()
        self.conv_in     = _CausalConvWrapper(3, 128)
        self.down_blocks = nn.ModuleList([
            _EncoderDownBlock(ic, oc, nr, ad, ct)
            for ic, oc, nr, ad, ct in self._BLOCKS
        ])
        self.mid_block   = _EncoderMidBlock(512)
        self.norm_out    = _GroupNorm(512)
        self.conv_out    = _CausalConvWrapper(512, 32)   # → 32 = 2×16

    def forward(self, x):
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.mid_block(x)
        x = _silu(self.norm_out(x))
        x = self.conv_out(x)
        return x                           # [B, 32, T', H', W']


# ─────────────────────────────────────────────────────────────
# Decoder building blocks
# ─────────────────────────────────────────────────────────────

class _DecoderUpBlock(nn.Module):
    def __init__(self, in_c, out_c, num_resnet, add_upsample, compress_time,
                 z_channels=16):
        super().__init__()
        blocks = []
        for i in range(num_resnet):
            blocks.append(_DecoderResBlock(in_c if i == 0 else out_c, out_c, z_channels))
        self.resnets = nn.ModuleList(blocks)
        self.upsamplers = (
            nn.ModuleList([_Upsampler3D(out_c, compress_time)])
            if add_upsample else None
        )

    def forward(self, x, z):
        for r in self.resnets:
            x = r(x, z)
        if self.upsamplers:
            for u in self.upsamplers:
                x = u(x)
        return x


class _DecoderMidBlock(nn.Module):
    def __init__(self, channels, num_resnet=2, z_channels=16):
        super().__init__()
        self.resnets = nn.ModuleList(
            [_DecoderResBlock(channels, channels, z_channels)
             for _ in range(num_resnet)]
        )

    def forward(self, x, z):
        for r in self.resnets:
            x = r(x, z)
        return x


class CogVideoX5BDecoder(nn.Module):
    """Decoder: latent [B,16,T',H',W'] → image [B,3,T,H,W]"""

    # (in_c, out_c, num_resnets, add_upsample, compress_time)
    _BLOCKS = [
        (512, 512, 4, True,  True),   # block 0: temporal+spatial 2×
        (512, 256, 4, True,  True),   # block 1: temporal+spatial 2×
        (256, 256, 4, True,  False),  # block 2: spatial only
        (256, 128, 4, False, False),  # block 3: no upsampling
    ]

    def __init__(self, z_channels=16):
        super().__init__()
        self.conv_in   = _CausalConvWrapper(z_channels, 512)
        self.mid_block = _DecoderMidBlock(512, z_channels=z_channels)
        self.up_blocks = nn.ModuleList([
            _DecoderUpBlock(ic, oc, nr, au, ct, z_channels)
            for ic, oc, nr, au, ct in self._BLOCKS
        ])
        self.norm_out  = SpatialNorm3D(128, z_channels)
        self.conv_out  = _CausalConvWrapper(128, 3)

    def forward(self, z):
        """z: [B, 16, T', H', W']"""
        x = self.conv_in(z)
        x = self.mid_block(x, z)
        for blk in self.up_blocks:
            x = blk(x, z)
        x = _silu(self.norm_out(x, z))
        x = self.conv_out(x)     # [B, 3, T, H, W] in raw range
        return x


# ─────────────────────────────────────────────────────────────
# Full VAE
# ─────────────────────────────────────────────────────────────

class CogVideoX5BVAE(nn.Module):
    """
    CogVideoX-5B 3D Causal VAE.

    Usage:
        vae = CogVideoX5BVAE()
        vae.load_state_dict(torch.load('cogvideo5bvae.safetensors'))

    encode(pixels):
        pixels: [B, 3, T, H, W]  (values in [-1, 1])
        returns mean: [B, 16, T', H', W']

    decode(z):
        z: [B, 16, T', H', W']
        returns pixels: [B, 3, T, H, W] (raw, NOT clamped)
    """

    SCALING_FACTOR = 1.15258426

    # Spatial / temporal compression factors
    SPATIAL_FACTOR  = 8
    TEMPORAL_FACTOR = 4

    def __init__(self):
        super().__init__()
        self.encoder = CogVideoX5BEncoder()
        self.decoder = CogVideoX5BDecoder()

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        pixels: [B, 3, T, H, W]  range [-1, 1]
        Returns **mean** latent [B, 16, T', H', W'] × SCALING_FACTOR
        (deterministic encode — no random sampling to avoid stochasticity).
        """
        h     = self.encoder(pixels)            # [B, 32, T', H', W']
        mean, _ = h.chunk(2, dim=1)             # clamp logvar, just keep mean
        return mean * self.SCALING_FACTOR

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, 16, T', H', W']  (already scaled by SCALING_FACTOR)
        Returns pixels [B, 3, T, H, W], range roughly [-1, 1]
        """
        z = z / self.SCALING_FACTOR
        return self.decoder(z)


# ─────────────────────────────────────────────────────────────
# Public helper
# ─────────────────────────────────────────────────────────────

def load_cogvideox_vae(path: str, device, dtype: torch.dtype) -> CogVideoX5BVAE:
    """Load cogvideo5bvae.safetensors → CogVideoX5BVAE."""
    from safetensors.torch import load_file
    sd    = load_file(path)
    model = CogVideoX5BVAE()
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        print(f"[VOID-VAE] Missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"[VOID-VAE] Unexpected keys ({len(unexpected)}): {unexpected[:4]}")
    return model.to(device=device, dtype=dtype).eval()
