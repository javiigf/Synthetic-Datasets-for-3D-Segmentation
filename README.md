# 🧬 Synthetic Datasets for 3D Segmentation – Master’s Thesis Project
This repository contains the scripts developed for the Master’s Thesis *“Integration of Synthetic Data Generation and Deep Learning for 3D Epithelial Segmentation”* by **Javier García Flores**, University of Seville (IBiS), under the supervision of **Prof. Luis M. Escudero** and **Dr. Pedro J. Gómez-Gálvez**.

The project focuses on the automatic generation and evaluation of synthetic 3D epithelial datasets using Voronoi-based modeling and deep learning segmentation pipelines.

## tissueMaker_gui.py

This Python script provides a **graphical user interface (GUI)** for generating in silico epithelial tissues based on **3D Voronoi tessellations**.
Main functionalities include:

- Interactive control of cell geometry parameters (number of seeds, cell height, ellipsoid axes).

- Real-time visualization of 2D/3D Voronoi diagrams representing epithelial arrangements.

- Export of generated volumes as TIFF images for downstream analysis.

- Optional elastic deformation and anisotropy adjustment to simulate embryonic curvature.

This GUI serves as an intuitive tool for building synthetic embryo-like epithelia for model training.

## voronoiGenerator_0.py

This script contains the **core computational functions** used by the GUI to create 3D Voronoi-based epithelial structures.
It performs the following tasks:

- Initializes random seed distributions to define cell centroids.

- Generates 3D Voronoi diagrams using scipy.spatial.Voronoi.

- Assigns voxel-wise labels to each cell domain, creating volumetric masks.

- Supports iterative relaxation for improved homogeneity.

The generated volumes can be used directly for deep-learning segmentation or as input for further texture synthesis using generative models.

## generateVoronoi_image.py

This standalone script automates **batch generation of synthetic embryo images** using the Voronoi model.
Key features:

- Runs multiple embryo simulations with variable geometric parameters.

- Outputs binary masks or multi-channel TIFF volumes (membranes, contours, regions).

- Enables reproducible generation of large-scale training datasets without manual annotation.

The resulting files replicate epithelial topology observed in confocal images of Drosophila embryos.

## Run_Allmetrics_3D.ipynb

This **Jupyter notebook** computes **quantitative metrics** to evaluate segmentation performance in 3D.
It compares model predictions (e.g., ResUNet, CellPose-SAM) with manually annotated ground truth.
Metrics include:

- Intersection over Union (IoU) at multiple thresholds

- Precision, Recall, and F1-score

- Panoptic Quality (PQ) for combined detection and segmentation accuracy

To execute this notebook, the folders *EM_Image_Segmentation* and *mAP_3Dvolume* must be present in the working directory. These directories contain auxiliary functions and reference data required for metric computation. The notebook provides reproducible evaluation for comparing real, synthetic, and hybrid-trained models.

## run_Cellpose_SAM.ipynb

This notebook provides an **adaptation of the official** CellPose-SAM (https://github.com/MouseLand/cellpose) implementation for direct comparison with models trained in this study.
It allows automatic segmentation of membrane-stained 3D confocal stacks without the need for nuclear channels, producing instance masks for benchmarking against ground truth data.

The notebook was customized to handle epithelial tissue morphology and to output results compatible with the metric evaluation pipeline (*Run_Allmetrics_3D.ipynb*).

## environment.yaml

This file defines the conda environment required to execute all scripts. It ensures reproducibility of dependencies across systems. 

Main packages include: *numpy*, *scipy*, *matplotlib*, *scikit-image*, *tifffile*, *SimpleITK*, *ttkbootstrap*, *opencv-python*, *elasticdeform*, and *notebook*.

Create the environment with:

*conda env create -f environment.yaml*

*conda activate voronoi-seg*

# log_GAN.txt

This file contains the **training log of the CycleGAN model** used for generating synthetic microscopy-like images from Voronoi-based embryo geometries. The experiment was conducted as a **proof-of-concept test** with a **limited number of iterations and only six epochs**, designed to verify the correct implementation of the generative framework rather than to optimize performance. Content overview:

- Training progress per epoch, including generator and discriminator loss values.

- Cycle-consistency and identity losses.

- Basic validation metrics for monitoring convergence trends.

## Data and Integration

- Synthetic and segmentation models were trained and evaluated using the **BiaPy** framework.
- The full BiaPy configuration files and trained models are available upon request from: **Prof. Luis M. Escudero (IBiS – University of Seville)**.
- The remaining code and training data are hosted in a **private institutional repository** and can be accessed under request for review or collaboration.
