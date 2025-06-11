# EEG Classification using LSTM

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Datasets](#datasets)
* [Preprocessing](#preprocessing)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Evaluation](#evaluation)
* [Limitations](#limitations)
* [Future Work](#future-work)
* [Contributing](#contributing)

---

## Project Overview

This repository contains code and documentation for building a patient-specific EEG seizure classifier. We leverage two public EEG datasets (CHB-MIT and Bonn EEG) to train a Long Short-Term Memory (LSTM) neural network that discriminates between seizure and non-seizure EEG segments in a supervised framework.

## Features

* **Data Preprocessing:** Automatic reading of EDF files, resampling to a common frequency, and normalization.
* **Feature Extraction:** Construction of time-domain feature vectors capturing spectral and spatial EEG characteristics.
* **Deep Learning Model:** A 2-layer LSTM network with dropout and fully connected layers for binary classification.
* **Evaluation Metrics:** Accuracy, precision, recall, F1-score via scikit-learn’s classification report.
* **Modular Code:** Implemented as a Jupyter notebook with clear sections for each pipeline step.

## Repository Structure

```
├── data/                            # Raw EDF datasets (not tracked)
│   ├── chb-mit-scalp-eeg-database-1.0.0/
│   └── bonn-eeg-dataset/
├── converted_artifacts/             # NumPy arrays of preprocessed signals
├── notebooks/                       # Experimental and preprocessing notebooks
│   └── FDA_Project_3_Group_10.ipynb
├── README.md                        # Project documentation (this file)
└── requirements.txt                 # Python dependencies
```

## Requirements

* Python 3.7+
* PyTorch
* NumPy
* Pandas
* SciPy
* pyEDFlib
* scikit-learn
* tqdm

Install with:

```bash
pip install -r requirements.txt
```

## Datasets

1. **CHB-MIT EEG Database**: Download from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/).
2. **Bonn EEG Dataset**: Download from the University of Bonn website.

Place the extracted directories under `data/` as shown in the repository structure.

## Preprocessing

All preprocessing steps are in the notebook:

1. **EDF Reading:** Uses `pyedflib.highlevel.read_edf` to load raw signals.
2. **Resampling:** Uniform sampling at 100,000 Hz via `scipy.signal.resample`.
3. **Segmentation & Labeling:** Convert continuous recordings into labeled segments and save as `.npy` in `converted_artifacts/`.

Run:

```bash
jupyter notebook notebooks/FDA_Project_3_Group_10.ipynb
```

## Model Architecture

* **LSTM Layers:** 2 layers, input size = 10,000 (samples per segment), hidden size = 100, `batch_first=True`.
* **Dropout:** 20% between LSTM and dense layers.
* **Dense Layers:** Linear(100 → 32) + ReLU → Linear(32 → 2) + Softmax.

## Training

Training parameters:

* **Loss:** Cross-Entropy Loss
* **Optimizer:** Adam (lr=3e-4)
* **Batch Size:** 4
* **Epochs:** 10

In the notebook, sections titled **Model Training** cover the full epoch loop and progress bars.

## Evaluation

* Split: 500 training samples / 186 test samples (patient-specific split).
* Metrics: Overall accuracy and class-wise precision, recall, F1-score printed at the end of training.


## Limitations

* **Dataset Size & Diversity:** Only two datasets; may not generalize across all patient populations.
* **Hardware Requirements:** GPU recommended for faster training on large EEG segments.
* **Interpretability:** LSTM black-box model; clinical applications may require explainable frameworks.

## Future Work

* Experiment with hybrid CNN–LSTM architectures.
* Apply data augmentation (time warping, noise injection).
* Develop real-time inference pipeline for seizure alert systems.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes and enhancements.

