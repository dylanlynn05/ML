# DCGAN on SVHN

A **Deep Convolutional GAN** (Radford et al., 2015) trained from scratch on the **Street-View House Numbers (SVHN)** dataset to generate new 32×32 RGB digit images.

## Files

| File | Purpose |
|---|---|
| [`DCGAN_SVHN_Assignment.ipynb`](DCGAN_SVHN_Assignment.ipynb) | End-to-end training notebook (TensorFlow / Keras) |
| [`DCGAN_SVHN_Writeup.docx`](DCGAN_SVHN_Writeup.docx) | Written assignment summary |

## What the notebook does

1. **Download** SVHN cropped digits (`train_32x32.mat`) from <http://ufldl.stanford.edu/housenumbers/>.
2. **Preprocess** — scale pixel values to `[-1, 1]`, batch into `tf.data.Dataset` (`BATCH_SIZE = 256` on GPU, 128 on CPU).
3. **Define** generator and discriminator (`make_generator_model`, `make_discriminator_model`):
   - Generator: `Dense → BatchNorm → LeakyReLU → reshape → Conv2DTranspose stack → Tanh` → `(32, 32, 3)`
   - Discriminator: stack of `Conv2D + LeakyReLU + Dropout` → `Dense(1)` logits
4. **Train adversarially** with Binary Cross-Entropy losses and Adam optimisers for `EPOCHS = 10`:
   - Generator wants the discriminator to output 1 on its fakes.
   - Discriminator wants 1 on real SVHN images and 0 on fakes.
5. **Checkpoint** weights every epoch into `./training_checkpoints_svhn/`.
6. **Save** a 4×4 grid of generated samples per epoch (`image_at_epoch_XXXX.png`) and assemble them into `dcgan_svhn.gif`.

## How to run

```bash
# from the repo root
pip install tensorflow numpy scipy matplotlib pillow
jupyter notebook Unsupervised/DCGAN-SVHN/DCGAN_SVHN_Assignment.ipynb
```

A CUDA-capable GPU is *strongly* recommended — CPU training will work but is slow.

The notebook downloads SVHN (~180 MB) on first run; subsequent runs reuse the cached file.

## Hyperparameters at a glance

| Setting | Value |
|---|---|
| Image shape | `(32, 32, 3)` |
| Latent dim (`noise_dim`) | 100 |
| Batch size | 256 (GPU) / 128 (CPU) |
| Epochs | 10 |
| Loss | BCE (from logits) on real / fake |
| Optimiser | Adam (default DCGAN-style settings) |

## Why this is "unsupervised"

The generator never sees the digit labels. It only sees images and a noise vector and learns to produce samples from `p(x)` — the distribution of SVHN digits — without any supervised target.
