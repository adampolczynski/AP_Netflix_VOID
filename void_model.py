"""
Netflix VOID Model Implementation for ComfyUI
----------------------------------------------
Architecture: CogVideoX-5B-based 3D video inpainting transformer

Deduced architecture parameters:
  - hidden_size      = 3072
  - num_heads        = 48   (head_dim = 64)
  - num_layers       = 42
  - time_embed_dim   = 512
  - text_embed_dim   = 4096  (T5-XXL)
  - in_channels      = 48   (16 noisy + 16 masked + 16 mask)
  - out_channels     = 16
  - temporal_patch   = 2
  - spatial_patch    = 2
  - ff_inner_dim     = 12288 (4×)
  - qk_norm          = LayerNorm (per head, dim=64)
  - modulation       = CogVideoX-style AdaLN-zero (6 params per block)
  - noise_pred_type  = v_prediction (CogVideoX-5B default)
  - positional_enc   = 3D RoPE  (no learned pos embed in weights)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_SIZE        = 3072
NUM_HEADS          = 48
HEAD_DIM           = 64          # HIDDEN_SIZE // NUM_HEADS
NUM_LAYERS         = 42
TIME_EMBED_DIM     = 512
TEXT_EMBED_DIM     = 4096
IN_CHANNELS        = 48          # 3 × 16 for inpainting
OUT_CHANNELS       = 16
TEMPORAL_PATCH     = 2
SPATIAL_PATCH      = 2
PATCH_IN_DIM       = IN_CHANNELS * TEMPORAL_PATCH * SPATIAL_PATCH * SPATIAL_PATCH  # 384
PATCH_OUT_DIM      = OUT_CHANNELS * TEMPORAL_PATCH * SPATIAL_PATCH * SPATIAL_PATCH  # 128
FF_INNER_DIM       = 12288       # 4 × HIDDEN_SIZE

# RoPE axis dims (must sum to HEAD_DIM=64)
ROPE_AXES_DIM      = [16, 24, 24]   # [temporal, height, width]
ROPE_THETA         = 10000


# ---------------------------------------------------------------------------
# 3D RoPE utilities (mirrors comfy/ldm/flux/math.py  'rope' function)
# ---------------------------------------------------------------------------

def _rope_freqs(pos: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Compute rotation matrices for one axis.

    pos:  [B, L]  integer positions
    return: [B, L, dim//2, 2, 2]  (2×2 rotation matrices)
    """
    assert dim % 2 == 0
    if pos.device.type in ("mps",):
        device = torch.device("cpu")
    else:
        device = pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2,
                           dtype=torch.float64, device=device)
    omega = 1.0 / (theta ** scale)                             # [dim//2]
    # outer product of positions and frequencies
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    # build 2×2 rotation matrices: [[cos, -sin], [sin, cos]]
    out = torch.stack([torch.cos(out), -torch.sin(out),
                       torch.sin(out),  torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def _apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation matrices to x.

    x:         [B, L, num_heads, head_dim]
    freqs_cis: [B, 1,  L, head_dim//2, 2, 2]  – stacked per-axis
    """
    # reshape x to pairs
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


def compute_3d_pe(T: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """Build 3D RoPE cosine/sine matrices for (T, H, W) grid.

    Returns: [1, 1, T*H*W, HEAD_DIM//2, 2, 2]  – used as freqs_cis.
    """
    t_ids = torch.arange(T, device=device, dtype=torch.float32)
    h_ids = torch.arange(H, device=device, dtype=torch.float32)
    w_ids = torch.arange(W, device=device, dtype=torch.float32)

    tt, hh, ww = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    t_pos = tt.flatten()  # [T*H*W]
    h_pos = hh.flatten()
    w_pos = ww.flatten()

    L = t_pos.shape[0]

    # Each axis contributes (axis_dim/2) rotation-matrix pairs
    # Shape per axis: [L, axis_dim//2, 2, 2]  → broadcast over batch+head
    rot_t = _rope_freqs(t_pos.unsqueeze(0), ROPE_AXES_DIM[0], ROPE_THETA)  # [1, L, d_t//2, 2, 2]
    rot_h = _rope_freqs(h_pos.unsqueeze(0), ROPE_AXES_DIM[1], ROPE_THETA)  # [1, L, d_h//2, 2, 2]
    rot_w = _rope_freqs(w_pos.unsqueeze(0), ROPE_AXES_DIM[2], ROPE_THETA)  # [1, L, d_w//2, 2, 2]

    # Concatenate along the frequency dimension
    freqs = torch.cat([rot_t, rot_h, rot_w], dim=2)  # [1, L, HEAD_DIM//2, 2, 2]
    freqs = freqs.unsqueeze(1)                         # [1, 1, L, HEAD_DIM//2, 2, 2]
    return freqs.to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

def timestep_sinusoidal(t: torch.Tensor, dim: int = HIDDEN_SIZE,
                        max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal timestep embedding, matching ComfyUI's convention.
    t:  [B]  float timesteps in [0, 1000)
    Returns [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Sinusoidal → MLP → time_embed_dim.
    Matches keys: time_embedding.linear_1, time_embedding.linear_2
    """

    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(HIDDEN_SIZE, TIME_EMBED_DIM)
        self.linear_2 = nn.Linear(TIME_EMBED_DIM, TIME_EMBED_DIM)
        self.act = nn.SiLU()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = timestep_sinusoidal(t, dim=HIDDEN_SIZE).to(dtype=self.linear_1.weight.dtype)
        x = self.act(self.linear_1(x))
        x = self.linear_2(x)
        return x


class VoidPatchEmbed(nn.Module):
    """3D patch embedding.
    Matches keys: patch_embed.proj, patch_embed.text_proj
    """

    def __init__(self):
        super().__init__()
        self.proj      = nn.Linear(PATCH_IN_DIM, HIDDEN_SIZE)          # [3072, 384]
        self.text_proj = nn.Linear(TEXT_EMBED_DIM, HIDDEN_SIZE)         # [3072, 4096]

    def forward(self, video: torch.Tensor, text: torch.Tensor):
        """
        video: [B, IN_CHANNELS=48, T, H, W]
               (noisy_latent + masked_latent + mask, all 16ch each)
        text:  [B, seq_len, TEXT_EMBED_DIM]
        Returns:
            vid_tokens:  [B, T_p*H_p*W_p, HIDDEN_SIZE]
            txt_tokens:  [B, seq_len,      HIDDEN_SIZE]
            T_p, H_p, W_p: grid sizes
        """
        B, C, T, H, W = video.shape
        tp, sp = TEMPORAL_PATCH, SPATIAL_PATCH

        # 3D patch extraction
        # step 1: reshape into patch groups
        vid = video.reshape(B, C, T // tp, tp, H // sp, sp, W // sp, sp)
        # step 2: permute → [B, T_p, H_p, W_p, C, tp, sp, sp]
        vid = vid.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        T_p, H_p, W_p = T // tp, H // sp, W // sp
        # step 3: flatten patch channels
        vid = vid.reshape(B, T_p * H_p * W_p, C * tp * sp * sp)   # [B, L, 384]

        vid_tokens = self.proj(vid)                                      # [B, L, HIDDEN_SIZE]
        txt_tokens = self.text_proj(text)                                # [B, S, HIDDEN_SIZE]
        return vid_tokens, txt_tokens, T_p, H_p, W_p


class VoidLayerNormZero(nn.Module):
    """CogVideoX-style AdaLN-zero for one block (norm1 or norm2).
    Matches keys: transformer_blocks.N.norm{1,2}.norm  and  .linear
    linear: Linear(TIME_EMBED_DIM=512, 6*HIDDEN_SIZE=18432)
    norm:   LayerNorm(HIDDEN_SIZE=3072)
    """

    def __init__(self):
        super().__init__()
        self.norm   = nn.LayerNorm(HIDDEN_SIZE, elementwise_affine=True, eps=1e-5)
        self.linear = nn.Linear(TIME_EMBED_DIM, 6 * HIDDEN_SIZE, bias=True)
        self.silu   = nn.SiLU()

    def forward(self, vid_tokens: torch.Tensor, txt_tokens: torch.Tensor,
                temb: torch.Tensor):
        """
        vid_tokens:  [B, L,  HIDDEN_SIZE]
        txt_tokens:  [B, S,  HIDDEN_SIZE]
        temb:        [B, TIME_EMBED_DIM]

        Returns: norm_vid, norm_txt, vid_gate, txt_gate
        Each gate is [B, 1, HIDDEN_SIZE] for broadcasting.
        """
        # 6 modulation params: vid_shift, vid_scale, vid_gate, txt_shift, txt_scale, txt_gate
        mods = self.linear(self.silu(temb))                              # [B, 18432]
        v_sh, v_sc, v_gt, t_sh, t_sc, t_gt = mods.chunk(6, dim=-1)     # each [B, H]

        norm_vid = self.norm(vid_tokens) * (1 + v_sc[:, None]) + v_sh[:, None]
        norm_txt = self.norm(txt_tokens) * (1 + t_sc[:, None]) + t_sh[:, None]
        return norm_vid, norm_txt, v_gt[:, None], t_gt[:, None]


class VoidAdaLayerNorm(nn.Module):
    """Final output AdaLayerNorm.
    Matches keys: norm_out.norm, norm_out.linear
    linear: Linear(TIME_EMBED_DIM=512, 2*HIDDEN_SIZE=6144)
    norm:   LayerNorm(HIDDEN_SIZE=3072)
    """

    def __init__(self):
        super().__init__()
        self.norm   = nn.LayerNorm(HIDDEN_SIZE, elementwise_affine=True, eps=1e-5)
        self.linear = nn.Linear(TIME_EMBED_DIM, 2 * HIDDEN_SIZE, bias=True)
        self.silu   = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        mods  = self.linear(self.silu(temb))    # [B, 6144]
        shift, scale = mods.chunk(2, dim=-1)    # each [B, 3072]
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class VoidAttention(nn.Module):
    """Multi-head self-attention with QK-norm and 3D RoPE (video tokens only).
    Matches keys under transformer_blocks.N.attn1.*
    """

    def __init__(self):
        super().__init__()
        self.to_q    = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)   # [3072, 3072]
        self.to_k    = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
        self.to_v    = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
        self.to_out  = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)])
        self.norm_q  = nn.LayerNorm(HEAD_DIM, elementwise_affine=True, eps=1e-5)  # [64]
        self.norm_k  = nn.LayerNorm(HEAD_DIM, elementwise_affine=True, eps=1e-5)

    def forward(self, combined: torch.Tensor, txt_len: int,
                freqs: torch.Tensor) -> tuple:
        """
        combined: [B, S+L, HIDDEN_SIZE]  (text prefix + video tokens)
        txt_len:  number of text tokens (prefix)
        freqs:    [1, 1, L, HEAD_DIM//2, 2, 2]  3D RoPE for video tokens

        Returns: vid_out [B, L, H], txt_out [B, S, H]
        """
        B, N, _ = combined.shape
        vid_len = N - txt_len

        Q = self.to_q(combined)   # [B, N, H]
        K = self.to_k(combined)
        V = self.to_v(combined)

        # Reshape to per-head
        Q = Q.view(B, N, NUM_HEADS, HEAD_DIM)  # [B, N, nh, hd]
        K = K.view(B, N, NUM_HEADS, HEAD_DIM)

        # QK normalisation (applied before RoPE, standard practice)
        Q = self.norm_q(Q)
        K = self.norm_k(K)

        # Apply 3D RoPE to video tokens only
        if freqs is not None and vid_len > 0:
            Q_vid = _apply_rope1(Q[:, txt_len:, :, :], freqs.to(Q.device))
            K_vid = _apply_rope1(K[:, txt_len:, :, :], freqs.to(K.device))
            Q = torch.cat([Q[:, :txt_len], Q_vid], dim=1)
            K = torch.cat([K[:, :txt_len], K_vid], dim=1)

        # Efficient attention (ComfyUI provides optimized_attention but we keep
        # self-contained here for portability)
        Q = Q.transpose(1, 2)   # [B, nh, N, hd]
        K = K.transpose(1, 2)
        V = V.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        scale = math.sqrt(HEAD_DIM) ** -1
        attn  = torch.nn.functional.scaled_dot_product_attention(
            Q.to(dtype=torch.float32),
            K.to(dtype=torch.float32),
            V.to(dtype=torch.float32),
            scale=scale,
        ).to(dtype=combined.dtype)

        # [B, nh, N, hd] → [B, N, H]
        attn = attn.transpose(1, 2).contiguous().view(B, N, HIDDEN_SIZE)
        attn = self.to_out[0](attn)

        txt_out = attn[:, :txt_len]
        vid_out = attn[:, txt_len:]
        return vid_out, txt_out


class VoidFeedForward(nn.Module):
    """GELU feedforward.
    Matches keys transformer_blocks.N.ff.net.0.proj  and  .net.2
    net.0.proj: Linear(3072, 12288)   – GELU activation class uses .proj
    net.2:      Linear(12288, 3072)
    """

    def __init__(self):
        super().__init__()
        # Mimic diffusers layout so key names serialise correctly
        self.net = nn.ModuleList([
            _GELUProj(),            # index 0 → ff.net.0.proj
            nn.Identity(),          # index 1 → dropout (no params)
            nn.Linear(FF_INNER_DIM, HIDDEN_SIZE, bias=True),   # index 2 → ff.net.2
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0](x)
        x = self.net[2](x)
        return x


class _GELUProj(nn.Module):
    """Holds the proj weight that maps hidden→ff_inner with GELU.
    Key: ff.net.0.proj.{weight,bias}
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN_SIZE, FF_INNER_DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class VoidBlock(nn.Module):
    """Single CogVideoX-style transformer block.
    Matches keys transformer_blocks.N.*
    """

    def __init__(self):
        super().__init__()
        self.norm1 = VoidLayerNormZero()
        self.attn1 = VoidAttention()
        self.norm2 = VoidLayerNormZero()
        self.ff    = VoidFeedForward()

    def forward(self, vid_tokens: torch.Tensor, txt_tokens: torch.Tensor,
                temb: torch.Tensor, freqs: torch.Tensor) -> tuple:
        """
        Returns updated (vid_tokens, txt_tokens)
        """
        S = txt_tokens.shape[1]
        combined = torch.cat([txt_tokens, vid_tokens], dim=1)   # text first

        # --- Attention ---
        norm_vid, norm_txt, v_gate, t_gate = self.norm1(vid_tokens, txt_tokens, temb)
        norm_combined = torch.cat([norm_txt, norm_vid], dim=1)
        vid_a, txt_a = self.attn1(norm_combined, S, freqs)
        vid_tokens = vid_tokens + v_gate * vid_a
        txt_tokens = txt_tokens + t_gate * txt_a

        # --- FFN ---
        norm_vid2, norm_txt2, v_gate2, t_gate2 = self.norm2(vid_tokens, txt_tokens, temb)
        norm_combined2 = torch.cat([norm_txt2, norm_vid2], dim=1)
        ff_out = self.ff(norm_combined2)
        txt_ff = ff_out[:, :S]
        vid_ff = ff_out[:, S:]
        vid_tokens = vid_tokens + v_gate2 * vid_ff
        txt_tokens = txt_tokens + t_gate2 * txt_ff

        return vid_tokens, txt_tokens


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class VoidTransformer(nn.Module):
    """
    Netflix VOID video inpainting transformer.
    Accepts float32/bfloat16 weights via standard state_dict loading.
    """

    def __init__(self):
        super().__init__()
        self.patch_embed      = VoidPatchEmbed()
        self.time_embedding   = TimestepEmbedder()
        self.transformer_blocks = nn.ModuleList([VoidBlock() for _ in range(NUM_LAYERS)])
        self.norm_final       = nn.LayerNorm(HIDDEN_SIZE, elementwise_affine=True, eps=1e-5)
        self.norm_out         = VoidAdaLayerNorm()
        self.proj_out         = nn.Linear(HIDDEN_SIZE, PATCH_OUT_DIM, bias=True)   # [128, 3072]

    # ----- Forward -----

    def forward(self, video: torch.Tensor, timesteps: torch.Tensor,
                text_embeds: torch.Tensor,
                transformer_options: dict = None) -> torch.Tensor:
        """
        video:       [B, 48, T, H, W]  – concatenated (noisy + masked + mask)
        timesteps:   [B]   float in [0, 999]
        text_embeds: [B, S, 4096]  – T5-XXL embeddings (padded/truncated)

        Returns: [B, 16, T, H, W]  – predicted output (velocity or eps)
        """
        B, _, T, H, W = video.shape

        # Temporal / spatial grid sizes after patching
        T_p = T // TEMPORAL_PATCH
        H_p = H // SPATIAL_PATCH
        W_p = W // SPATIAL_PATCH

        # 1. Timestep embedding
        temb = self.time_embedding(timesteps)                            # [B, 512]

        # 2. Patch embed
        vid_tokens, txt_tokens, T_p_, H_p_, W_p_ = self.patch_embed(video, text_embeds)

        # 3. Precompute 3D RoPE for video tokens
        freqs = compute_3d_pe(T_p, H_p, W_p, device=video.device,
                              dtype=video.dtype)                         # [1,1,L, head_dim//2, 2, 2]

        # 4. Transformer blocks
        for block in self.transformer_blocks:
            vid_tokens, txt_tokens = block(vid_tokens, txt_tokens, temb, freqs)

        # 5. Output head
        vid_tokens = self.norm_final(vid_tokens)                         # [B, L, H]
        vid_tokens = self.norm_out(vid_tokens, temb)                     # [B, L, H]
        vid_tokens = self.proj_out(vid_tokens)                           # [B, L, 128]

        # 6. Un-patch 3D
        out = _unpatch3d(vid_tokens, B, T_p, H_p, W_p)                  # [B, 16, T, H, W]
        return out


def _unpatch3d(tokens: torch.Tensor, B: int, T_p: int, H_p: int, W_p: int) -> torch.Tensor:
    """Inverse of 3D patching: [B, T_p*H_p*W_p, 128] → [B, 16, T, H, W]"""
    tp, sp = TEMPORAL_PATCH, SPATIAL_PATCH
    # tokens: [B, L, C*tp*sp*sp] where L = T_p*H_p*W_p
    tokens = tokens.view(B, T_p, H_p, W_p, OUT_CHANNELS, tp, sp, sp)
    # [B, T_p, H_p, W_p, 16, 2, 2, 2]
    tokens = tokens.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    # [B, 16, T_p, 2, H_p, 2, W_p, 2]
    tokens = tokens.reshape(B, OUT_CHANNELS, T_p * tp, H_p * sp, W_p * sp)
    return tokens


# ---------------------------------------------------------------------------
# Noise schedule helpers (CogVideoX-5B defaults)
# ---------------------------------------------------------------------------

def _betas_scaled_linear(num_train=1000, beta_start=0.00085, beta_end=0.012) -> torch.Tensor:
    """sqrt(linear) schedule (=scaled_linear in diffusers)."""
    return torch.linspace(beta_start**0.5, beta_end**0.5, num_train) ** 2


def build_alphas_cumprod(num_train: int = 1000) -> torch.Tensor:
    betas = _betas_scaled_linear(num_train)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


ALPHAS_CUMPROD = build_alphas_cumprod()   # precomputed, moved to device lazily


def get_alphas(timesteps: torch.Tensor, device) -> tuple:
    """Return (sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod) for given timesteps."""
    global ALPHAS_CUMPROD
    if ALPHAS_CUMPROD.device != device:
        ALPHAS_CUMPROD = ALPHAS_CUMPROD.to(device)
    t_int = timesteps.long().clamp(0, ALPHAS_CUMPROD.shape[0] - 1)
    acp   = ALPHAS_CUMPROD[t_int].float()           # [B]
    return acp.sqrt(), (1 - acp).sqrt()


def v_to_x0_eps(v_pred: torch.Tensor, x_t: torch.Tensor,
                sqrt_acp: torch.Tensor, sqrt_1m_acp: torch.Tensor):
    """
    CogVideoX v-prediction:
        v = sqrt(α) * ε - sqrt(1-α) * x0
    Solve for x0 and eps.
    """
    sac  = sqrt_acp.view(-1, 1, 1, 1, 1)
    s1ac = sqrt_1m_acp.view(-1, 1, 1, 1, 1)
    x0   =  sac * x_t  - s1ac * v_pred
    eps  = s1ac * x_t  + sac  * v_pred
    return x0, eps


# ---------------------------------------------------------------------------
# DDIM sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(
    model:        VoidTransformer,
    x_T:          torch.Tensor,       # [B, 16, T, H, W] initial noise
    masked_lat:   torch.Tensor,       # [B, 16, T, H, W] masked image latent
    mask:         torch.Tensor,       # [B,  1, T, H, W] binary float mask (1=inpaint)
    cond_embeds:  torch.Tensor,       # [B, S, 4096]
    uncond_embeds: torch.Tensor,      # [B, S, 4096]
    num_steps:    int   = 50,
    cfg_scale:    float = 7.5,
    eta:          float = 0.0,        # 0 = deterministic DDIM
    device:       str   = "cuda",
    dtype:        torch.dtype = torch.bfloat16,
    num_timesteps: int  = 1000,
    callback             = None,      # optional: fn(step, x_t, preview)
) -> torch.Tensor:
    """DDIM sampling loop for VOID.

    Returns denoised latent [B, 16, T, H, W].
    """
    model.eval()

    alphas_cp = build_alphas_cumprod(num_timesteps).to(device=device, dtype=torch.float32)

    # Build sub-sequence of timesteps (uniformly spaced, high → low)
    step_ratio   = num_timesteps // num_steps
    timesteps    = torch.arange(num_timesteps - 1, -1, -step_ratio)[:num_steps]
    timesteps    = timesteps.flip(0)              # ascending → we'll iterate reversed
    timesteps_t  = timesteps.flip(0)             # [T] from high to low

    x = x_T.to(device=device, dtype=dtype)
    mask_16 = mask.expand_as(x)                  # [B, 16, T, H, W]

    for i, t_val in enumerate(timesteps_t):
        B = x.shape[0]
        t = torch.full((B,), t_val.item(), device=device, dtype=torch.float32)

        # Build model input: noisy + masked + mask (48 channels)
        model_input = torch.cat([x, masked_lat, mask_16], dim=1).to(dtype=dtype)

        # CFG forward pass
        if cfg_scale != 1.0:
            cond_input   = torch.cat([model_input, model_input], dim=0)
            text_input   = torch.cat([cond_embeds, uncond_embeds], dim=0).to(dtype=dtype)
            t_doubled    = torch.cat([t, t], dim=0)
            v_both = model(cond_input, t_doubled, text_input)
            v_cond, v_uncond = v_both.chunk(2, dim=0)
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            text_input = cond_embeds.to(dtype=dtype)
            v_pred = model(model_input, t, text_input)

        # v-prediction → x0 & eps
        acp   = alphas_cp[t.long().clamp(0, num_timesteps - 1)]
        sqrt_acp  = acp.sqrt()
        sqrt_1mac = (1 - acp).sqrt()
        x0, eps = v_to_x0_eps(v_pred.float(), x.float(), sqrt_acp, sqrt_1mac)

        # DDIM step
        if i < len(timesteps_t) - 1:
            t_prev     = timesteps_t[i + 1].item()
            acp_prev   = alphas_cp[int(t_prev)]
        else:
            acp_prev   = torch.ones(1, device=device)
        sqrt_acp_prev  = acp_prev.sqrt()
        sqrt_1mac_prev = (1 - acp_prev).sqrt()

        if eta > 0:
            sigma = (eta * torch.sqrt((1 - acp_prev) / (1 - acp)) *
                     torch.sqrt(1 - acp / acp_prev))
            noise = torch.randn_like(x)
        else:
            sigma = 0.0
            noise = 0.0

        x_prev = (sqrt_acp_prev * x0 +
                  torch.sqrt(torch.clamp(1 - acp_prev - sigma**2, min=0)) * eps +
                  sigma * noise)

        x = x_prev.to(dtype=dtype)

        if callback is not None:
            callback(i, x, x0)

    # Keep original pixels outside the mask
    x = x * mask_16 + masked_lat * (1 - mask_16)
    return x
