# Solar Flare Data Quality and Prediction Pipeline

This repository contains the code and data processing pipeline for our study on 
the impact of data consistency on solar flare prediction using machine learning models.

---

## Overview

This project investigates how defects and inconsistencies among different solar flare catalogs
affect machine learning-based flare forecasting performance. We construct a unified
data processing and modeling pipeline based on multiple flare event lists and related space weather data products.

The repository includes:
- Data preprocessing and matching scripts
- Logistic regression and LSTM models training scripts
- Model performance evaluation and visualization tools

---

## Related Paper

This repository accompanies the following paper:


> *Defects and Inconsistencies in Solar Flare Data Sources: Implications for Machine Learning Forecasting*  
> (2026)


---

## Repository Structure


├── data/ # Processed datasets\\
├── figures/ # Figures used in the paper\\
├── Logreg_models/ # Logistic regression models\\
├── LSTM_models/ # LSTM models\\
├── results/ # Evaluation results\\
├── scripts_model_training/ # Training scripts\\
├── scripts_updating_list/ # Flare list processing\\
├── Sci-Quality data augmentation pipline description # science-quality flare list augmentation logic\\
└── README.md



## Contact

Ke Hu\\
uhek@umich.edu

Yang Chen\\
ychenang@umich.edu


