"""
ComfyUI nodes for Netflix VOID video inpainting model.

Nodes:
  VoidModelLoader     – load void_pass2.safetensors → VOID_MODEL
  VoidSampler         – VOID_MODEL + image + mask + conditioning → LATENT
  VoidLatentDecode    – LATENT + VAE → IMAGE  (convenience wrapper)
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
import comfy.model_management as mm

from .void_model import (
    VoidTransformer,
    ddim_sample,
    TEMPORAL_PATCH, SPATIAL_PATCH,
    OUT_CHANNELS, IN_CHANNELS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_void_model(path: str, dtype: torch.dtype, device) -> VoidTransformer:
    """Load safetensors weights into VoidTransformer."""
    from safetensors.torch import load_file
    sd   = load_file(path)
    model = VoidTransformer()
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        print(f"[VOID] Missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"[VOID] Unexpected keys ({len(unexpected)}): {unexpected[:4]}")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _encode_mask_vae(vae, mask_bhw: torch.Tensor, T: int, device) -> torch.Tensor:
    """Encode the mask video through the VAE → [B, 16, T', Lh, Lw].

    VOID uses use_vae_mask=True: the 3rd block of 16 input channels is the
    VAE-encoded mask video, not a broadcast binary channel. Encoding the mask
    through the same VAE that processes the image gives the model the proper
    16-dimensional latent representation it was trained to expect.

    Mask convention: 0.0 = remove (inpaint), 1.0 = keep (background).
    Values between 0 and 1 (overlap=63/255, affected=127/255) are preserved.
    """
    if mask_bhw.ndim == 2:
        mask_bhw = mask_bhw.unsqueeze(0)            # [1, H, W]
    B, H, W = mask_bhw.shape
    vae_dtype = next(vae.parameters()).dtype
    # Mask values stay in [0, 1] range — official VOID does NOT normalize to [-1, +1].
    # The caller is responsible for inverting if needed (official: encodes 1-mask_condition).
    mask_vid = mask_bhw.float().to(device)           # [B, H, W], range [0, 1]
    mask_vid = mask_vid.unsqueeze(1).unsqueeze(2)    # [B, 1, 1, H, W]
    mask_vid = mask_vid.expand(B, 3, T, H, W).to(dtype=vae_dtype)  # [B, 3, T, H, W]
    with torch.no_grad():
        lat = vae.encode(mask_vid)                   # [B, 16, T', Lh, Lw]
    return lat.float()


def _vae3d_encode(vae, image_bhwc: torch.Tensor, T: int, device) -> torch.Tensor:
    """Encode ComfyUI IMAGE frames [B*T, H, W, C] with CogVideoX 3D VAE.
    Returns latent [B, 16, T', Lh, Lw] scaled by scaling_factor.
    """
    BT, H, W, C = image_bhwc.shape
    B = max(1, BT // T)
    pixels = image_bhwc[:B * T].view(B, T, H, W, min(C, 3))
    vae_dtype = next(vae.parameters()).dtype
    pixels = pixels.permute(0, 4, 1, 2, 3).float()   # [B, 3, T, H, W]
    pixels = (pixels * 2.0 - 1.0).to(device=device, dtype=vae_dtype)  # normalize [-1, 1]
    with torch.no_grad():
        lat = vae.encode(pixels)                      # [B, 16, T', Lh, Lw]
    return lat


def _vae3d_decode(vae, lat_bcthw: torch.Tensor, device) -> torch.Tensor:
    """Decode 3D latent [B, 16, T', Lh, Lw] to ComfyUI IMAGE [B*T, H, W, C]."""
    vae_dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        out = vae.decode(lat_bcthw.to(device=device, dtype=vae_dtype))  # [B, 3, T, H, W]
    out = (out.float() * 0.5 + 0.5).clamp(0, 1)
    B, _, T, H, W = out.shape
    return out.permute(0, 2, 3, 4, 1).reshape(B * T, H, W, 3).cpu()


def _prepare_mask_3d(mask_bhw: torch.Tensor, target_T: int, target_H: int,
                     target_W: int, device) -> torch.Tensor:
    """Resize mask to latent resolution [B, 1, T', Lh, Lw], preserving quadmask values.

    Expects VOID quadmask convention (output of VoidQuadMask node):
      0.0        = primary object to remove
      63/255     = overlap of primary + affected
      127/255    = affected region (falling objects, displaced items)
      1.0        = background (keep)

    NOTE: always use the VoidQuadMask node to prepare the mask before passing it here.
    If you connect a raw ComfyUI mask (1=painted region), put an InvertMask node first.
    """
    if mask_bhw.ndim == 2:
        mask_bhw = mask_bhw.unsqueeze(0)                 # [1, H, W]
    mask_4d = mask_bhw.float().to(device).unsqueeze(1)   # [B, 1, H, W]

    # Debug: print mask statistics to help diagnose convention issues
    uniq = mask_4d.unique()
    print(f"[VOID] mask values (unique): {uniq.tolist()[:8]}  "
          f"min={mask_4d.min():.3f}  max={mask_4d.max():.3f}  "
          f"mean={mask_4d.mean():.3f}")
    if mask_4d.max() < 0.5:
        print("[VOID] WARNING: mask has no background (keep) region! "
              "All pixels are 'remove'. Did you accidentally feed an inverted mask? "
              "Check VoidQuadMask connections — overlap_mask should NOT be set to the "
              "full background (InvertMask of remove_mask).")

    mask_ds = F.interpolate(mask_4d, size=(target_H, target_W), mode="nearest")
    mask_5d = mask_ds.unsqueeze(2).expand(-1, -1, target_T, -1, -1).contiguous()
    return mask_5d


def _extract_cond_embeds(conditioning) -> torch.Tensor:
    """Extract text embedding tensor from ComfyUI CONDITIONING.
    Returns [1, S, 4096] on cpu.
    """
    if conditioning is None or len(conditioning) == 0:
        return torch.zeros(1, 1, 4096)
    cond_tensor = conditioning[0][0]              # [1, S, D]
    return cond_tensor.float().cpu()


def _pad_or_trim_text(embeds: torch.Tensor, target_len: int = 226) -> torch.Tensor:
    """Pad or trim text embeddings to (1, target_len, 4096)."""
    B, S, D = embeds.shape
    if S >= target_len:
        return embeds[:, :target_len, :]
    pad = torch.zeros(B, target_len - S, D, dtype=embeds.dtype, device=embeds.device)
    return torch.cat([embeds, pad], dim=1)


def _ensure_even_temporal(lat: torch.Tensor) -> torch.Tensor:
    """Zero-pad temporal dim to be divisible by TEMPORAL_PATCH."""
    T = lat.shape[2]
    rem = T % TEMPORAL_PATCH
    if rem != 0:
        pad_t = TEMPORAL_PATCH - rem
        lat = F.pad(lat, (0, 0, 0, 0, 0, pad_t))
    return lat


def _ensure_spatial_div(lat: torch.Tensor) -> torch.Tensor:
    """Zero-pad spatial dims to be divisible by SPATIAL_PATCH."""
    H, W = lat.shape[-2], lat.shape[-1]
    ph = (SPATIAL_PATCH - H % SPATIAL_PATCH) % SPATIAL_PATCH
    pw = (SPATIAL_PATCH - W % SPATIAL_PATCH) % SPATIAL_PATCH
    if ph > 0 or pw > 0:
        lat = F.pad(lat, (0, pw, 0, ph))
    return lat


# ──────────────────────────────────────────────────────────────────────────────
# Node: VoidModelLoader
# ──────────────────────────────────────────────────────────────────────────────

class VoidModelLoader:
    """Load a VOID .safetensors weight file from diffusion_models folder."""

    RETURN_TYPES    = ("VOID_MODEL",)
    RETURN_NAMES    = ("void_model",)
    FUNCTION        = "load_model"
    CATEGORY        = "AP/VOID"

    @classmethod
    def INPUT_TYPES(cls):
        model_files = folder_paths.get_filename_list("diffusion_models")
        return {
            "required": {
                "model_name": (model_files,),
                "dtype":      (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    def load_model(self, model_name: str, dtype: str):
        dtype_map = {"bfloat16": torch.bfloat16,
                     "float16":  torch.float16,
                     "float32":  torch.float32}
        path   = folder_paths.get_full_path("diffusion_models", model_name)
        device = mm.get_torch_device()
        print(f"[VOID] Loading model from {path} as {dtype}")
        model  = _load_void_model(path, dtype_map[dtype], device)
        print(f"[VOID] Model loaded. Parameters: "
              f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        return (model,)


# ──────────────────────────────────────────────────────────────────────────────
# Node: VoidVAELoader
# ──────────────────────────────────────────────────────────────────────────────

class VoidVAELoader:
    """Load a CogVideoX-5B 3D VAE (e.g. cogvideo5bvae.safetensors) from the vae folder."""

    RETURN_TYPES    = ("VOID_VAE",)
    RETURN_NAMES    = ("void_vae",)
    FUNCTION        = "load_vae"
    CATEGORY        = "AP/VOID"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "dtype":    (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    def load_vae(self, vae_name: str, dtype: str):
        from .vae_3d import load_cogvideox_vae
        dtype_map = {"bfloat16": torch.bfloat16,
                     "float16":  torch.float16,
                     "float32":  torch.float32}
        path   = folder_paths.get_full_path("vae", vae_name)
        device = mm.get_torch_device()
        print(f"[VOID] Loading VAE from {path} as {dtype}")
        vae = load_cogvideox_vae(path, device, dtype_map[dtype])
        return (vae,)


# ──────────────────────────────────────────────────────────────────────────────
# Node: VoidSampler
# ──────────────────────────────────────────────────────────────────────────────

class VoidSampler:
    """
    Complete VOID video inpainting sampler.

    Pipeline:
      1. IMAGE + MASK + VAE  →  encode to latent space
      2. DDIM loop with CFG  →  denoised latent
      3. Optional VAE decode →  IMAGE output

    Text conditioning uses standard ComfyUI CONDITIONING (T5-XXL recommended).
    """

    RETURN_TYPES    = ("LATENT", "IMAGE")
    RETURN_NAMES    = ("latent", "image")
    FUNCTION        = "sample"
    CATEGORY        = "AP/VOID"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "void_model":   ("VOID_MODEL",),
                "vae":          ("VOID_VAE",),
                "image":        ("IMAGE",),           # [B, H, W, C]  or  [B*T, H, W, C]
                "mask":         ("MASK",),             # [B, H, W]
                "positive":     ("CONDITIONING",),
                "negative":     ("CONDITIONING",),
                "num_frames":   ("INT",  {"default": 1, "min": 1, "max": 32,
                                          "tooltip": "Video frames (1 = single image)"}),
                "steps":        ("INT",  {"default": 50, "min": 1, "max": 200}),
                "cfg":          ("FLOAT",{"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed":         ("INT",  {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "eta":          ("FLOAT",{"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "0=deterministic DDIM, 1=DDPM stochastic"}),
                "denoise":      ("FLOAT",{"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Start noise strength (1=full noise, 0=no denoising)"}),
            },
        }

    def sample(self, void_model, vae, image, mask, positive, negative,
               num_frames, steps, cfg, seed, eta, denoise):

        device = mm.get_torch_device()
        dtype  = next(void_model.parameters()).dtype

        # ── 1. Encode image to latent with 3D VAE ─────────────────────────
        B_img, H_img, W_img, _ = image.shape
        T = num_frames
        B = max(1, B_img // T) if B_img >= T else 1

        if B_img < T:
            # Single image → duplicate T times
            image = image[:1].expand(T, -1, -1, -1)
            B = 1

        lat = _vae3d_encode(vae, image, T, device).float()  # [B, 16, T', Lh, Lw]
        _, latC, latT, Lh, Lw = lat.shape

        # ── 2. Prepare mask ────────────────────────────────────────────────
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)           # [1, H, W]
        mask_b = mask[:B].to(device=device)    # [B, H_img, W_img]  (0=remove, 1=keep)

        # 1-channel mask at latent resolution — for final compositing only
        mask_lat = _prepare_mask_3d(mask_b, latT, Lh, Lw, device)

        # Block 2: VAE-encode the INVERTED mask.
        # Official VOID pipeline: encode (1 - mask_condition) in [0, 1] range.
        # Inverted: cup=1.0, background=0.0. Values stay in [0,1], NOT [-1,+1].
        mask_vae = _encode_mask_vae(vae, 1.0 - mask_b, T, device)  # [B, 16, T', Lh, Lw]

        # Block 3: full original video latent (zero_out_mask_region=False).
        # Official VOID: the model sees the cup as context and generates the background.
        # No cup-zeroing — just use lat as-is.

        # ── 3. Pad to model grid requirements ─────────────────────────────
        lat        = _ensure_even_temporal(_ensure_spatial_div(lat))
        mask_pad   = _ensure_even_temporal(_ensure_spatial_div(mask_lat))
        mask_vae   = _ensure_even_temporal(_ensure_spatial_div(mask_vae))

        T_eff, H_eff, W_eff = lat.shape[2], lat.shape[3], lat.shape[4]

        # ── 5. Build initial noisy latent ──────────────────────────────────
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) & 0xFFFFFFFF)
        noise = torch.randn(B, OUT_CHANNELS, T_eff, H_eff, W_eff,
                            generator=generator, device=device, dtype=torch.float32)

        # Partial noising based on denoise strength
        if denoise < 1.0:
            t_start = int(denoise * 999)
            from .void_model import ALPHAS_CUMPROD
            acp = ALPHAS_CUMPROD[t_start].to(device)
            x_T = acp.sqrt() * lat + (1 - acp).sqrt() * noise
        else:
            x_T = noise

        # ── 6. Text conditioning ───────────────────────────────────────────
        pos_embeds  = _extract_cond_embeds(positive).to(device=device)   # [1, S, 4096]
        neg_embeds  = _extract_cond_embeds(negative).to(device=device)

        # Expand to batch
        pos_embeds  = pos_embeds.expand(B, -1, -1)
        neg_embeds  = neg_embeds.expand(B, -1, -1)

        # ── 7. DDIM sampling ───────────────────────────────────────────────
        mm.soft_empty_cache()

        def _progress(step, x, x0):
            mm.throw_exception_if_processing_interrupted()

        out_lat = ddim_sample(
            model          = void_model,
            x_T            = x_T,
            mask_vae_lat   = mask_vae,
            video_ref_lat  = lat.to(dtype=torch.float32),
            compose_mask   = mask_pad,
            cond_embeds    = pos_embeds,
            uncond_embeds  = neg_embeds,
            num_steps      = steps,
            cfg_scale      = cfg,
            eta            = eta,
            device         = device,
            dtype          = dtype,
            callback       = _progress,
        )
        # out_lat: [B, 16, T_eff, H_eff, W_eff]

        # Trim padding back to original size
        out_lat = out_lat[:, :, :latT, :Lh, :Lw].contiguous()

        # ── 8. Build LATENT dict (ComfyUI format: [B, C, H, W] per frame → stack) ─
        # For simplicity, return the video latent reshaped for downstream use.
        # Most VAE decoders expect [B, C, H, W] so we return frame-by-frame.
        # ComfyUI uses {"samples": [B, C, H, W]} convention.
        # We store video as [B*T, C, H, W] → standard ComfyUI LATENT.
        B, latC, latT2, Lh2, Lw2 = out_lat.shape
        lat_frames = out_lat.permute(0, 2, 1, 3, 4).reshape(B * latT2, latC, Lh2, Lw2)
        latent_out = {"samples": lat_frames.cpu(), "void_T": latT2, "void_B": B}

        # ── 9. Optional VAE decode ─────────────────────────────────────────
        try:
            image_out = _vae3d_decode(vae, out_lat, device)   # [B*T, H, W, C]
        except Exception as e:
            print(f"[VOID] VAE decode failed: {e}. Returning blank image.")
            image_out = torch.zeros(B * latT2, H_img, W_img, 3, dtype=torch.float32)

        return (latent_out, image_out)


# ──────────────────────────────────────────────────────────────────────────────
# Node: VoidLatentToVideo
# ──────────────────────────────────────────────────────────────────────────────

class VoidLatentToVideo:
    """Decode a batch of latent frames (from VoidSampler) to IMAGE.
    Useful when you want to re-decode without re-running the sampler.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "decode"
    CATEGORY     = "AP/VOID"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae":    ("VOID_VAE",),
                "latent": ("LATENT",),
            }
        }

    def decode(self, vae, latent):
        device  = mm.get_torch_device()
        samples = latent["samples"]              # [B*T, 16, Lh, Lw]
        T       = latent.get("void_T", 1)
        B       = latent.get("void_B", max(1, samples.shape[0] // T))
        lat_5d  = samples.view(B, T, *samples.shape[1:]).permute(0, 2, 1, 3, 4)
        return (_vae3d_decode(vae, lat_5d, device),)


# ──────────────────────────────────────────────────────────────────────────────
# Node: VoidTextEncode (convenience – wraps ComfyUI CLIPTextEncode for T5)
# ──────────────────────────────────────────────────────────────────────────────

class VoidTextEncode:
    """
    Thin wrapper: accepts text + T5-capable CLIP model and returns CONDITIONING.
    Identical to CLIPTextEncode but grouped under AP/VOID for clarity.
    You can instead use a standard CLIPTextEncode node with a T5-XXL CLIP.
    """

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION     = "encode"
    CATEGORY     = "AP/VOID"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip": ("CLIP",),
            }
        }

    def encode(self, text: str, clip):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond   = output.pop("cond")
        return ([[cond, output]],)


# ──────────────────────────────────────────────────────────────────────────────
# Node: VoidQuadMask
# ──────────────────────────────────────────────────────────────────────────────

class VoidQuadMask:
    """
    Build a VOID quadmask from separate ComfyUI mask layers.

    VOID requires a 4-value mask where each pixel value encodes a semantic role:
      0.0       (  0/255) – primary object to remove
      63/255    ( 63/255) – overlap: primary object + affected region
      127/255   (127/255) – affected region (objects that fall/move as a result)
      1.0       (255/255) – background / keep

    Inputs:
      remove_mask   (required) – white = primary object to delete
      affected_mask (optional) – white = region affected by removal (e.g. falling objects)
      overlap_mask  (optional) – white = pixels that are both primary and affected

    Priority when masks overlap: overlap > remove > affected > background.

    If you only connect remove_mask, the output is a proper 2-level VOID mask
    (remove / keep), which is sufficient for most scenes without physics interactions.
    """

    RETURN_TYPES  = ("MASK",)
    RETURN_NAMES  = ("quadmask",)
    FUNCTION      = "build"
    CATEGORY      = "AP/VOID"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remove_mask":   ("MASK",),
            },
            "optional": {
                "affected_mask": ("MASK",),
                "overlap_mask":  ("MASK",),
            },
        }

    def build(self, remove_mask: torch.Tensor,
              affected_mask: torch.Tensor | None = None,
              overlap_mask:  torch.Tensor | None = None):

        # Normalise all inputs to [B, H, W] float32
        def _norm(m):
            if m is None:
                return None
            return m.float() if m.ndim == 3 else m.unsqueeze(0).float()

        rm = _norm(remove_mask)
        am = _norm(affected_mask)
        om = _norm(overlap_mask)

        # Reference shape from remove mask
        B, H, W = rm.shape

        def _resize(m):
            if m is None:
                return None
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m.unsqueeze(1), size=(H, W),
                                  mode="nearest").squeeze(1)
            return (m > 0.5)

        rm_b = _resize(rm)           # bool [B,H,W]
        am_b = _resize(am)
        om_b = _resize(om)

        # Start with background (1.0 = 255/255)
        quad = torch.ones(B, H, W, dtype=torch.float32)

        # Apply in ascending priority
        if am_b is not None:
            quad[am_b]  = 127.0 / 255.0
        quad[rm_b]      = 0.0           # primary object (remove)
        if om_b is not None:
            quad[om_b]  = 63.0 / 255.0

        return (quad,)


# ──────────────────────────────────────────────────────────────────────────────
# Node registry
# ──────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "VoidModelLoader":    VoidModelLoader,
    "VoidVAELoader":      VoidVAELoader,
    "VoidSampler":        VoidSampler,
    "VoidLatentToVideo":  VoidLatentToVideo,
    "VoidTextEncode":     VoidTextEncode,
    "VoidQuadMask":       VoidQuadMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoidModelLoader":    "VOID Model Loader (AP)",
    "VoidVAELoader":      "VOID VAE Loader (AP)",
    "VoidSampler":        "VOID Sampler (AP)",
    "VoidLatentToVideo":  "VOID Latent → Image (AP)",
    "VoidTextEncode":     "VOID Text Encode (AP)",
    "VoidQuadMask":       "VOID Quad Mask (AP)",
}
