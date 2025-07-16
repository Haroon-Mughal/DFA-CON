# DFA-CON
This is the official repository containing code for the paper "DFA-CON: A CONTRASTIVE LEARNING APPROACH FOR DETECTING COPYRIGHT
INFRINGEMENT IN DEEPFAKE ART" accepted (poster presentation) at IEEE MLSP 2025.
## DeepfakeArt Contrastive Benchmark

This repo implements contrastive learning for AI-generated art forgery detection using the DeepfakeArt dataset.

## Features
- Supports SupCon training with ResNet-50 or ViT-B/16
- Parses DeepfakeArt JSON structure into anchor–positive maps
- Evaluation via cosine similarity and binary metrics
