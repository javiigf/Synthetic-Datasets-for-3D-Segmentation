# Synthetic 3D Epithelial Segmentation – Master’s Thesis Project

**Author:** Javier García Flores  
**Institution:** University of Seville – CABD (Centro Andaluz de Biología del Desarrollo)  
**Supervisors:** Prof. Luis M. Escudero & Dr. Pedro J. Gómez Gálvez  
**Year:** 2025  

---

## 📖 Project Overview

This repository contains the core scripts developed for the Master’s Thesis *“Integration of Synthetic Data Generation and Deep Learning for 3D Epithelial Segmentation.”*  
The project explores how synthetic datasets, generated through mathematical modeling and generative networks, can reduce the dependency on manual annotations for training deep learning segmentation models in 3D epithelial tissues.

The pipeline combines:

1. **Synthetic dataset generation** using 3D Voronoi diagrams and a CycleGAN-based style transfer approach.  
2. **Training of deep learning models** (3D ResU-Net) using real, synthetic, and mixed datasets via [BiaPy](https://github.com/BiaPyX/BiaPy).  
3. **Quantitative performance evaluation** using custom scripts for 3D segmentation metrics (IoU, F1, PQ, etc.).

---

## 📁 Repository Structure


