"""
====================================================================
 Stable Diffusion - Text-to-Image Generation in Python
====================================================================

 Assignment: Implement a text-to-image generation model (Stable Diffusion)
 Reference : https://www.geeksforgeeks.org/deep-learning/generate-images-from-text-in-python-stable-diffusion/

 What this script does
 ---------------------
 1. Loads the pre-trained Stable Diffusion v1.4 model from Hugging Face.
 2. Sends a text prompt of our choice to the model.
 3. Generates an image that visualizes the prompt.
 4. Displays the image and saves it to disk as "generated_image.png".

 How Stable Diffusion works (high-level)
 ---------------------------------------
 Stable Diffusion is a "latent diffusion" model with three core pieces:
   - Text Encoder (CLIP) : converts the prompt into 768-dim embeddings.
   - U-Net               : iteratively denoises a random latent tensor,
                            guided by the text embeddings.
   - Variational Autoencoder (VAE) : decodes the cleaned latent tensor
                            back into a full-resolution RGB image.
 Starting from pure noise, the U-Net removes a tiny amount of noise at
 every step, nudging the latent toward something that matches the text.
 After ~50 steps the VAE decodes the final latent into the output image.
====================================================================
"""

# --------------------------------------------------------------------
# 1. IMPORT THE LIBRARIES WE NEED
# --------------------------------------------------------------------
# torch        : PyTorch deep-learning framework. Stable Diffusion is a
#                neural network, and torch handles tensors, GPU compute,
#                and the half-precision (float16) we use to save memory.
# diffusers    : Hugging Face library that wraps the full Stable
#                Diffusion pipeline (text encoder + U-Net + VAE +
#                scheduler) behind a single easy-to-use class.
# PIL (Pillow) : Python Imaging Library. The pipeline returns PIL
#                Image objects, which we use to show / save the result.
# os           : Standard library used to build a safe output file path.
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os


# --------------------------------------------------------------------
# 2. PICK THE BEST AVAILABLE HARDWARE
# --------------------------------------------------------------------
# Stable Diffusion is heavy: a single image at 768x768 needs billions of
# floating-point operations. A GPU ("cuda") is dramatically faster than
# a CPU. We auto-detect what is available so the script runs anywhere.
#   - "cuda"  -> NVIDIA GPU (best, used on Google Colab T4 runtime).
#   - "mps"   -> Apple Silicon GPU (M1/M2/M3 Macs).
#   - "cpu"   -> fallback; works but can take several minutes per image.
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16          # half-precision = ~2x less VRAM
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32          # CPUs do not support float16 well

print(f"[info] Using device: {device}  (dtype={dtype})")


# --------------------------------------------------------------------
# 3. LOAD THE PRE-TRAINED STABLE DIFFUSION PIPELINE
# --------------------------------------------------------------------
# `from_pretrained` downloads the model weights from the Hugging Face
# Hub the first time it runs (~4 GB) and caches them locally so future
# runs are instant.
#
# We use "CompVis/stable-diffusion-v1-4" - the original, publicly
# accessible Stable Diffusion model. The GeeksforGeeks tutorial
# explicitly recommends it as the lightweight option for faster
# inference, and it does not require a Hugging Face account or token.
# (Newer Stability AI repos like "stabilityai/stable-diffusion-2-1"
# are now gated and need an authenticated HF token to download.)
MODEL_ID = "CompVis/stable-diffusion-v1-4"

print(f"[info] Loading model '{MODEL_ID}' (first run downloads ~4 GB)...")
pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,             # half precision when on GPU
)

# Move every sub-network (text encoder, U-Net, VAE) onto the chosen
# device so the matrix multiplications happen on the GPU.
pipeline = pipeline.to(device)

# Optional memory optimisation: lets the model swap layers in/out of
# VRAM so it fits on smaller GPUs (e.g. the free Colab T4 with 15 GB).
if device == "cuda":
    pipeline.enable_attention_slicing()


# --------------------------------------------------------------------
# 4. WRITE OUR TEXT PROMPT
# --------------------------------------------------------------------
# The prompt is the single most important input - the more descriptive
# it is, the more accurate and detailed the generated image will be.
# Good prompts usually include:
#   - SUBJECT  : what should appear (a dragon, a city, a person...).
#   - SETTING  : where it is (snowy mountain, neon street...).
#   - STYLE    : artistic style (oil painting, photorealistic, anime...).
#   - DETAILS  : lighting, mood, camera lens, colours, etc.
prompt = (
    "A majestic dragon perched on a snowy mountain peak at sunset, "
    "breathtaking fantasy art, highly detailed scales, cinematic "
    "lighting, golden hour, ultra realistic, 8k"
)

# A negative prompt tells the model what we DO NOT want. It is a
# simple but powerful way to remove common Stable Diffusion artefacts.
negative_prompt = "blurry, low quality, distorted, deformed, watermark, text"

print(f"[info] Prompt: {prompt}")


# --------------------------------------------------------------------
# 5. GENERATE THE IMAGE
# --------------------------------------------------------------------
# Calling the pipeline runs the full denoising loop:
#   a) Encode the prompt with CLIP into text embeddings.
#   b) Sample a random noise tensor in latent space.
#   c) Loop `num_inference_steps` times, each time asking the U-Net to
#      remove a little noise while staying faithful to the embeddings.
#   d) Decode the final latent through the VAE into a PIL image.
#
# Key arguments:
#   num_inference_steps : more steps -> better quality, slower (50 is the
#                          standard sweet spot).
#   guidance_scale      : how strictly the image follows the prompt.
#                          ~7.5 is the recommended default.
#   height / width      : output resolution. Must be multiples of 8.
#   generator           : a seeded random generator => reproducible runs.
generator = torch.Generator(device=device).manual_seed(42)

# SD 1.x was trained at 512x512 - using its native size produces the
# best results. On CPU we cut the step count to keep generation under
# a few minutes; on GPU we use the standard 50 steps for max quality.
inference_steps = 25 if device == "cpu" else 50
image_size = 512

print(f"[info] Generating image... "
      f"({inference_steps} steps at {image_size}x{image_size}, device={device})")
result = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=inference_steps,
    guidance_scale=7.5,
    height=image_size,
    width=image_size,
    generator=generator,
)

# `result.images` is a list because the pipeline can produce a batch of
# images at once. We only asked for one, so we pick index 0.
image: Image.Image = result.images[0]


# --------------------------------------------------------------------
# 6. SAVE AND DISPLAY THE IMAGE
# --------------------------------------------------------------------
# Build an output path next to this script so it is easy to find.
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "generated_image.png")
image.save(output_path)
print(f"[info] Image saved to: {output_path}")

# `image.show()` opens the system's default image viewer.
# In Jupyter / Colab, replace this with:
#     from IPython.display import display
#     display(image)
image.show()
