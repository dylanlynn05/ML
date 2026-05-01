# ML — Machine Learning Coursework

A collection of machine-learning and deep-learning assignments by **Dylan Lynn**, organized by learning paradigm.

## Repository layout

```
ML/
├── Supervised/                       # Labeled data — regression & classification
│   ├── Housing/                      # California / Boston housing regression
│   └── Assignment2-LossFunctions/    # Loss functions + CNN shape math (Assignment 2)
│
├── Unsupervised/                     # No labels — generative models, clustering, etc.
│   ├── DCGAN-SVHN/                   # Deep Convolutional GAN trained on SVHN
│   └── StableDiffusion/              # Pre-trained Stable Diffusion text-to-image
│
└── Semi-supervised/                  # Mix of a small labeled set + large unlabeled set
    └── assignment1_doe_john.ipynb    # Assignment 1 template
```

## Quick reference — which paradigm is which?

| Paradigm | What the model sees during training | Examples in this repo |
|---|---|---|
| **Supervised** | Inputs **and** their labels (`x`, `y`) | Housing price regression, Assignment 2 (losses for regression + classification) |
| **Unsupervised** | Inputs only (`x`) — the model discovers structure | DCGAN on SVHN (learns `p(x)`), Stable Diffusion (denoising diffusion) |
| **Semi-supervised** | A small labeled set + a large unlabeled set | Assignment 1 |

> **Why are GANs and Diffusion models in `Unsupervised/`?** Generative models such as DCGAN and Stable Diffusion are trained to model the data distribution `p(x)` directly without requiring class labels. They are the canonical example of *unsupervised generative learning*. (Diffusion models are sometimes called "self-supervised" because the training target is constructed from the input itself, but they don't use external labels.)

## Getting started

Most projects are Jupyter notebooks. The easiest path:

```bash
git clone https://github.com/dylanlynn05/ML.git
cd ML

python -m venv venv
# Windows:    venv\Scripts\activate
# Linux/Mac:  source venv/bin/activate

pip install --upgrade pip
pip install jupyter numpy pandas scikit-learn matplotlib

jupyter notebook
```

Some sub-projects have their own `requirements.txt` (e.g. `Unsupervised/StableDiffusion/`) — install those from inside the project folder.

## Per-project READMEs

Each project folder contains its own `README.md` with details on data, model architecture, how to run, and results:

- [`Supervised/Housing/`](Supervised/Housing/) — regression on housing prices
- [`Supervised/Assignment2-LossFunctions/`](Supervised/Assignment2-LossFunctions/) — loss functions and CNN shape math
- [`Unsupervised/DCGAN-SVHN/`](Unsupervised/DCGAN-SVHN/) — DCGAN trained on SVHN
- [`Unsupervised/StableDiffusion/`](Unsupervised/StableDiffusion/) — text-to-image with `CompVis/stable-diffusion-v1-4`
- [`Semi-supervised/`](Semi-supervised/) — assignment 1 template

## Notes on what's *not* committed

The `.gitignore` excludes:

- `venv/`, `.venv/`, `env/` — local Python environments
- `.idea/`, `.vscode/` — IDE settings
- `.ipynb_checkpoints/` — Jupyter autosaves
- `datasets/`, `data/`, `*.h5`, `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors` — large data and model files
- `huggingface/`, `.cache/` — Hugging Face / model caches

Datasets are downloaded automatically by the relevant notebooks (e.g. SVHN is fetched from Stanford's mirror).
