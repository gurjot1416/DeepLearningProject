***Replicated Astroformer***

This repository is part of a reproducibility challenge aimed at replicating the results of the original Astroformer paper, presented at the ICLR Workshop 2023. Our goal was to assess the reproducibility of Astroformer, a model designed for detection tasks in low-data regimes, and its performance on  Galaxy10 DECals Dataset.

*Original Paper*

Title: Astroformer: More Data Might Not Be All You Need for Classification
Authors: Rishit-dagli
Link to the Paper: [[arXiv link]](https://arxiv.org/abs/2304.05350)

**Implementation Overview**

Our implementation closely follows the original Astroformer model, with particular focus on the architecture and training procedure as outlined in the original repository. Key aspects of our replication include:

Utilizing the astroformer.py file as the backbone of our model implementation.
Training the model using the timm framework, with modifications as necessary to adapt to our computing resources and available datasets.
Modifications and additions to the pytorch-image-models library to incorporate the Astroformer model, specifically:
Adding timm/models/astroformer.py
Modifying timm/models/__init__.py

**Training**
Replication of the training procedure was conducted under similar conditions to the original paper, adjusted for our computational constraints.
