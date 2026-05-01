# Stable Diffusion — Text-to-Image Generation Assignment

This project implements **Stable Diffusion** (a deep-learning text-to-image generation model) in Python, following the GeeksforGeeks tutorial:
https://www.geeksforgeeks.org/deep-learning/generate-images-from-text-in-python-stable-diffusion/

It loads the pre-trained `CompVis/stable-diffusion-v1-4` model from Hugging Face — the original Stable Diffusion model, which is publicly available without an account or token (the GeeksforGeeks tutorial recommends it as the lightweight, faster option). The script then generates an image from the following prompt:

> *"A majestic dragon perched on a snowy mountain peak at sunset, breathtaking fantasy art, highly detailed scales, cinematic lighting, golden hour, ultra realistic, 8k"*

---

## Project files

| File | Purpose |
|------|---------|
| `stable_diffusion.py`        | Standalone Python script (run locally). |
| `stable_diffusion_colab.ipynb` | Jupyter/Colab notebook version (recommended). |
| `requirements.txt`           | All Python packages needed to run the project. |
| `README.md`                  | This file (also a video walkthrough script). |

---

## How to run

### Option A — Google Colab (recommended, no setup)

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. **File → Upload notebook → `stable_diffusion_colab.ipynb`**.
3. **Runtime → Change runtime type → T4 GPU**, then click **Save**.
4. **Runtime → Run all**.
5. The image will appear inline at the bottom of the notebook and be saved as `generated_image.png`.

### Option B — Local machine on Ubuntu (NVIDIA GPU recommended)

Tested on **Ubuntu 22.04 / 24.04 LTS** with Python 3.10+.

#### 1. Install the system prerequisites

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

If you have an NVIDIA GPU, also make sure the proprietary driver is installed (PyTorch ships its own CUDA runtime, so you do **not** need a separate CUDA toolkit):

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
nvidia-smi          # verify the GPU is visible after reboot
```

#### 2. Set up the project

```bash
cd ~/stable-diffusion-assignment

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch + CUDA:** `requirements.txt` already pins the PyTorch CUDA-12.4 wheel index (`--extra-index-url https://download.pytorch.org/whl/cu124`). This matches the CUDA version reported by `nvidia-smi` on most current NVIDIA drivers (550.xx / 551.xx series, including the WSL2 driver bundled with Windows 11). If you ever switch to a newer driver and want the default torch wheels, just remove that line.

#### 3. Run the script

```bash
python stable_diffusion.py
```

The first run downloads ~5 GB of model weights from Hugging Face into `~/.cache/huggingface/`. Later runs are instant.

The generated image is saved as `generated_image.png` in the project folder and opened in Ubuntu's default image viewer (`eog`/`xdg-open`).

> **CPU-only Ubuntu machine?** The script auto-detects the absence of a GPU and falls back to CPU with `float32`. Image generation will take several minutes per image — Google Colab (Option A) is strongly preferred in that case.

---

## Video walkthrough script

Use this as a prompt-by-prompt guide while recording your screen. Each section corresponds to a block of code in `stable_diffusion.py` / a cell in the notebook.

### 0. Intro (15–20 s)
> "Hi, in this video I'll demonstrate my implementation of Stable Diffusion — a deep-learning model that generates images from text prompts. I followed the GeeksforGeeks tutorial and used the Hugging Face `diffusers` library to load the pre-trained `CompVis/stable-diffusion-v1-4` model — the original publicly-available Stable Diffusion model. Let me walk you through the code and the output."

### 1. The libraries

| Library | Purpose |
|---------|---------|
| **`torch`** | PyTorch — the deep-learning framework. Stable Diffusion is a neural network, and PyTorch handles all the tensor math and GPU acceleration. We also use it to choose `float16` (half-precision) so the model fits in less GPU memory. |
| **`diffusers`** | A Hugging Face library that wraps the *entire* Stable Diffusion pipeline — text encoder, U-Net, VAE, and scheduler — behind one easy `StableDiffusionPipeline` class. |
| **`transformers`** | Provides the **CLIP text encoder** that is used internally by the pipeline to convert our prompt into 768-dim embedding vectors. |
| **`PIL` (Pillow)** | The Python Imaging Library. The pipeline returns a PIL `Image` object, which we then display and save to disk. |
| **`accelerate`, `safetensors`, `scipy`** | Helper packages that the pipeline uses internally for fast model loading and numerical routines. |
| **`os`** | Standard library — used to build a safe output file path. |

### 2. Detecting the device
> "Stable Diffusion is computationally heavy — billions of operations per image. So first I check whether a CUDA-enabled GPU is available. If yes, I run on the GPU with `float16` precision to halve the memory usage. Otherwise I fall back to CPU with `float32`."

### 3. Loading the pipeline
> "`StableDiffusionPipeline.from_pretrained` downloads the pre-trained weights from Hugging Face the first time it runs — about 4 GB — and caches them locally. I'm using `CompVis/stable-diffusion-v1-4`, the original Stable Diffusion model. The GeeksforGeeks tutorial recommends it as the lightweight option, and unlike the newer `stabilityai` repos it's publicly accessible and doesn't require a Hugging Face token. Then `pipeline.to(device)` moves every sub-network — the CLIP text encoder, the U-Net, and the VAE — onto the GPU. `enable_attention_slicing()` is a memory optimisation that lets the model fit on smaller GPUs."

### 4. Writing the prompt
> "The prompt is the most important input. The more descriptive it is, the more accurate the image. My prompt describes the **subject** (a dragon), the **setting** (a snowy mountain peak at sunset), the **style** (fantasy art), and extra **details** like cinematic lighting, golden hour, and 8k resolution. The negative prompt lists things I do NOT want — blurry, low quality, watermarks, text — which is a quick way to avoid common artefacts."

### 5. Generating the image — the diffusion loop
> "When I call `pipeline(...)`, four things happen under the hood:
> 1. **Text encoding** — CLIP turns my prompt into a sequence of 768-dim embeddings.
> 2. **Latent noise sampling** — a random tensor is created in the model's compressed latent space.
> 3. **The denoising loop** — the U-Net runs 50 times, each iteration removing a tiny bit of noise while staying faithful to the text embeddings.
> 4. **VAE decoding** — the variational autoencoder turns the cleaned latent tensor into a full-resolution RGB image.
>
> I'm using `num_inference_steps=50` on GPU (and 25 on CPU to keep generation reasonable) — 50 is the standard sweet spot. `guidance_scale=7.5` controls how strictly the model follows the prompt. `height=512, width=512` is SD 1.x's native training resolution. The seeded `generator` makes the result reproducible — running this script again with seed 42 will produce the exact same image."

### 6. Saving and displaying the output
> "`pipeline(...).images` is a list, because the pipeline can produce a batch of images at once. I only asked for one, so I take index 0. Then I save it as `generated_image.png` and call `image.show()` (or `display(image)` in Colab) to view it."

### 7. Walk through the output
> "And here is the result — a dragon on a snowy mountain peak at sunset, with detailed scales and cinematic lighting, exactly matching the prompt."

### 8. Wrap up (10 s)
> "That concludes my Stable Diffusion implementation. The complete pipeline — from text prompt to generated image — was implemented in around 30 lines of Python, thanks to the Hugging Face `diffusers` library."

---

## Tips for recording the video

1. Open both `stable_diffusion.py` (or the notebook) **and** this README side by side.
2. Run the code in **Google Colab** so viewers can clearly see the image being generated.
3. Walk through the script top-to-bottom — your video should be 3–5 minutes.
4. End by showing the saved `generated_image.png`.
