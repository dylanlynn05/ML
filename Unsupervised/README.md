# Unsupervised Learning

Projects in this folder train models on **unlabeled** data — no `y` is provided. Instead, the model learns the structure of the input distribution `p(x)` directly.

The two projects here are both **deep generative models**: they learn to *generate* new samples that look like the training data.

## Projects

| Folder | Model | Data | Framework |
|---|---|---|---|
| [`DCGAN-SVHN/`](DCGAN-SVHN/) | Deep Convolutional GAN — generator + discriminator trained adversarially | SVHN (Street-View House Numbers, 32×32 RGB) | TensorFlow / Keras |
| [`StableDiffusion/`](StableDiffusion/) | Pre-trained latent diffusion model (`CompVis/stable-diffusion-v1-4`) for text-to-image | Hugging Face checkpoint | PyTorch + 🤗 `diffusers` |

## Why are these unsupervised?

- **DCGAN** learns to map random noise `z ~ N(0, I)` to realistic images. The SVHN images are training samples, but their digit labels are *not* used by the generator — the goal is `p(x)`, not `p(y | x)`.
- **Stable Diffusion** is a denoising diffusion model. The training objective is to predict noise added to an image, with the image itself as the target — no human labels involved. (Text–image pairs are used for *conditioning*, not as supervised labels in the classical sense.)

Both are canonical examples of *unsupervised generative learning*.
