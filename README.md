# DFA-CON

Official repository for the paper **“DFA-CON: A Contrastive Learning Approach for Detecting Copyright Infringement in DeepFake Art”**, accepted at **IEEE MLSP 2025**.

## Overview

DFA-CON is a contrastive learning framework for detecting copyright infringement in AI-generated visual artworks. It learns an embedding space in which original artworks and their forged variants are pulled closer, while unrelated images are pushed apart.

This repository includes:
- contrastive training pipeline,
- similarity-based evaluation pipeline,
- wrappers for comparing against pretrained visual foundation models.

## Method

### DFA-CON Framework

<p align="center">
  <img src="assets/method.png" width="85%" alt="DFA-CON framework">
</p>

DFA-CON consists of three main components:
1. **Forgery-aware contrastive sampling**
2. **Representation learning** using an encoder and projection head
3. **Supervised contrastive loss** for training

### Inference Pipeline

<p align="center">
  <img src="assets/inference.png" width="85%" alt="Inference pipeline">
</p>

At inference time, a generated image is encoded into an embedding, compared against a reference bank of copyright-protected artwork embeddings using cosine similarity, and classified as potential infringement or non-infringement using a threshold.

## Features

- Supervised contrastive training with forgery-aware sampling
- Support for **ResNet-50**, **ViT**, **CLIP**, and **DINO-v2**
- Modular model wrapper for rapid comparison with new embedding models
- Batch inference for efficient evaluation
- Per-attack evaluation on:
  - Inpainting
  - Style Transfer
  - Adversarial Perturbation
  - CutMix

## Repository Structure

```text
DFA-CON/
├── configs/          # YAML configuration files
├── data/             # Dataset parsing and loading utilities
├── eval/             # Model wrappers and evaluation utilities
├── loss/             # Contrastive loss implementations
├── models/           # Backbone and projection head definitions
├── scripts/          # Training and evaluation entry points
├── train/            # Training loop and scheduler logic
├── assets/           # Figures for README / poster
└── requirements.txt
