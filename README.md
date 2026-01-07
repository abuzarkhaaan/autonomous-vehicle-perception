# Autonomous Vehicle Perception System

This repository provides a clean and modular research-oriented scaffold for **multi-camera 3D object detection** in autonomous driving, inspired by **BEVFormer-style bird’s-eye-view (BEV) encoding** and sensor fusion pipelines. The focus is on clarity, reproducibility, and extensibility rather than a monolithic implementation.

The codebase is structured to support **multi-camera perception**, **temporal modeling**, and **BEV-based reasoning**, making it suitable for experimentation, coursework, and future research extensions.

---

## Overview

The system follows a standard autonomous perception pipeline: synchronized multi-camera inputs are encoded using convolutional backbones, projected into a BEV representation, and processed with transformer-based attention for 3D object detection. Sensor fusion and temporal aggregation modules are included as explicit components to encourage modular experimentation.

---

## Repository Structure

- `configs/` – experiment and model configuration files  
- `data/` – dataset layout placeholders (e.g., nuScenes)  
- `models/` – BEVFormer, backbones, and fusion modules  
- `datasets/` – dataset wrappers and dataloaders  
- `scripts/` – training, evaluation, inference entrypoints  
- `utils/` – geometry, BEV utilities, metrics, logging  
- `notebooks/` – exploratory and qualitative analysis  
- `outputs/` – logs, predictions, visualizations  
- `checkpoints/` – saved model weights  

---

## Dataset

The repository is designed with **nuScenes** in mind, but the dataset layer is abstracted to allow easy adaptation to other multi-camera autonomous driving datasets. Dataset files are intentionally excluded from version control.

---

## Usage

Typical workflow:
1. Configure the experiment in `configs/`
2. Place dataset files under `data/`
3. Train, evaluate, or run inference using scripts in `scripts/`
4. Inspect outputs and visualizations under `outputs/`

---

## Notes

This project is a **structured scaffold**, not a drop-in production system. Core components are intentionally lightweight to make architectural choices explicit and easy to modify.

---

## License

This project is released under the MIT License.
