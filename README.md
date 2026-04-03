# AP Netflix VOID – ComfyUI Custom Nodes

Custom nodes for the **Netflix VOID** video inpainting model (CogVideoX-5B architecture).

---

## Downloads

| File | Destination |
|------|-------------|
| `void_pass2.safetensors` (VOID model) | `ComfyUI/models/diffusion_models/` |
| `cogvideo5bvae.safetensors` (CogVideoX-5B VAE) | `ComfyUI/models/vae/` |
| A **T5-XXL** text encoder supported by ComfyUI (e.g. `t5xxl_fp16.safetensors`) | `ComfyUI/models/clip/` |

---

## Installation

Clone this repository into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/AP_Netflix_VOID
```

No extra Python packages are required beyond a standard ComfyUI installation.

---

## Nodes

### VOID Model Loader (AP)
Loads `void_pass2.safetensors` → **VOID_MODEL**

| Input | Description |
|-------|-------------|
| `model_name` | Select from `models/diffusion_models/` |
| `dtype` | `bfloat16` (recommended), `float16`, `float32` |

---

### VOID VAE Loader (AP)
Loads `cogvideo5bvae.safetensors` → **VOID_VAE**

| Input | Description |
|-------|-------------|
| `vae_name` | Select from `models/vae/` |
| `dtype` | `bfloat16` (recommended) |

---

### VOID Text Encode (AP)
Encodes a text prompt using a T5-XXL CLIP model → **CONDITIONING**

Connect a `CLIPLoader` (T5-XXL) to the `clip` input.

---

### VOID Sampler (AP)
Main inpainting node. Encodes image+mask, runs DDIM, decodes output.

| Input | Type | Description |
|-------|------|-------------|
| `void_model` | VOID_MODEL | From VOID Model Loader |
| `vae` | VOID_VAE | From VOID VAE Loader |
| `image` | IMAGE | Source frames (1 or T frames) |
| `mask` | MASK | White = inpaint region |
| `positive` | CONDITIONING | Positive text embed |
| `negative` | CONDITIONING | Negative text embed |
| `num_frames` | INT | Number of video frames (1 = single image) |
| `steps` | INT | DDIM steps (default 30) |
| `cfg` | FLOAT | Classifier-free guidance scale |
| `seed` | INT | RNG seed |
| `eta` | FLOAT | 0 = deterministic DDIM, 1 = DDPM |
| `denoise` | FLOAT | 1.0 = full noise, lower = partial denoising |

Outputs: **LATENT**, **IMAGE**

---

### VOID Latent → Image (AP)
Re-decodes a VOID LATENT using the 3D VAE. Useful when you want to decode without re-sampling.

---

## Recommended Workflow

```
[CLIPLoader T5-XXL] ──► [VOID Text Encode] ──► CONDITIONING
[Load Image]        ──► IMAGE ──────────────────────────────► [VOID Sampler] ──► IMAGE
[Load Mask]         ──► MASK  ──────────────────────────────►
[VOID Model Loader] ──► VOID_MODEL ─────────────────────────►
[VOID VAE Loader]   ──► VOID_VAE ───────────────────────────►
```

---

## Notes

- VOID uses the **CogVideoX-5B** inpainting transformer (42 blocks, hidden 3072, 48 input channels).
- The 3D VAE compresses **8× spatially** and **4× temporally** (causal). For perfect temporal reconstruction, use `num_frames = 4k + 1` (e.g. 1, 5, 9, 13, 17, 49).
- Text encoder must be **T5-XXL** (4096-dim embeddings). Using CLIP will produce silence conditioning.

---

## License

MIT — see [LICENSE](LICENSE)
