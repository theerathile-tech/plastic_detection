# Plastic Detection

Fine-tuned YOLOv8 model for detecting plastic waste in underwater environments. This repository contains the training code and the trained model weights used by the Navan autonomous water cleanup boat.

## Overview

This model is specifically trained to identify various types of plastic waste in aquatic environments, enabling the Navan boat to autonomously locate and navigate towards plastic debris for collection. The model is optimized for real-time inference on Raspberry Pi hardware.

## Model Details

- **Base Model**: YOLOv8 (You Only Look Once v8)
- **Training**: Fine-tuned on underwater plastic waste dataset
- **Detection Classes**:
  - Plastic bottles (pbottle)
  - Plastic waste (pwaste)
  - General plastic debris
  - Other plastic objects
