# Sleep-Stage-Classification

A machine learning project for classifying sleep stages from ear-EEG recordings using deep learning, feature engineering, and ensemble modeling. This repository implements an end-to-end pipeline for preprocessing physiological time-series data, extracting spectral and statistical features, training convolutional neural networks (CNNs), and evaluating multi-class sleep stage classification performance using leakage-aware validation.

Overview

This project explores how machine learning and deep learning can be applied to ear-EEG signals for automated sleep stage classification. The workflow integrates raw signal processing, feature extraction, context-aware modeling, and ensemble techniques to improve classification performance across imbalanced sleep stages.

The project was built to strengthen practical skills in:

biomedical signal processing
time-series feature engineering
deep learning for physiological data
multi-class classification
class imbalance handling
leakage-aware validation
ensemble modeling (CNN + Random Forest)
performance evaluation on biosignals
Project Goals

The main goals of this project are to:

build a reproducible sleep stage classification pipeline
classify sleep into Wake, REM, and NREM stages
combine raw signal learning with engineered features
incorporate temporal context into predictions
improve performance using ensemble modeling
reduce data leakage through group-aware validation
evaluate model robustness on imbalanced physiological data
Dataset

This project uses ear-EEG sleep recordings from OpenNeuro, consisting of multi-channel in-ear EEG signals collected during sleep sessions.

The dataset includes:

multiple subjects with one or more recording sessions
ear-EEG channels from left and right ears
annotated sleep stages

For this project, labels were grouped into a 3-class classification task:

Wake
REM
NREM (combining N1, N2, N3)
Pipeline

The overall workflow is shown below:

load raw ear-EEG recordings
preprocess and filter signals
segment data into fixed-length epochs
extract signal-based and statistical features
generate temporal context windows
split data using group-aware cross-validation
train CNN and Random Forest models
combine models using ensemble weighting
apply prediction smoothing
evaluate performance across folds
Methods
1. Data Organization

Sleep recordings were organized by subject and session to ensure proper grouping during validation. This enabled consistent preprocessing and prevented subject-level data leakage.

2. Preprocessing

The preprocessing workflow included:

bandpass filtering of EEG signals
notch filtering to remove line noise
resampling to a consistent frequency
handling missing or noisy channels
epoch segmentation (30-second windows)

These steps ensured clean and standardized inputs for downstream modeling.

3. Feature Engineering

In addition to raw EEG signals, engineered features were extracted, including:

bandpower (delta, theta, alpha, beta, gamma)
relative bandpower
bandpower ratios
Hjorth parameters (activity, mobility, complexity)
spectral entropy
aperiodic components using spectral modeling

These features captured both physiological and statistical characteristics of sleep stages.

4. Context Windowing

To incorporate temporal dependencies, context windows were created by grouping neighboring epochs (e.g., 5–7 epochs). This allowed the model to learn transitions between sleep stages rather than treating each epoch independently.

5. Modeling

The project uses a hybrid modeling approach:

CNN for learning patterns directly from raw EEG signals
Random Forest for leveraging engineered features
Ensemble model (CNN + RF) using weighted averaging

Additional techniques included:

temporal smoothing of predictions
class weighting for imbalance handling
6. Validation Strategy

To prevent subject-level data leakage, the project used:

Stratified Group K-Fold Cross-Validation

This ensured:

balanced class distribution across folds
no overlap of subjects between training and validation sets
7. Class Imbalance Handling

Sleep stage data is inherently imbalanced. The following strategies were applied:

class weighting during training
balanced sampling in classical models
evaluation using balanced metrics (e.g., balanced accuracy, macro F1)
Model Architecture

The deep learning component consists of a 1D CNN applied to multi-channel ear-EEG signals, optionally extended with temporal context windows.

The architecture includes:

stacked convolutional layers for temporal feature extraction
batch normalization and nonlinear activations
pooling layers for dimensionality reduction
dense layers for classification

The CNN is combined with a Random Forest model trained on engineered features, forming an ensemble that leverages both raw signal learning and domain-informed features.

Results

The model achieved strong performance on multi-class sleep stage classification using group-aware cross-validation.

Mean cross-validation performance:

Accuracy: 0.7886

Balanced Accuracy: 0.7958

Precision (Macro): 0.7324

Recall (Macro): 0.7958

F1-score (Macro): 0.7443

Precision (Weighted): 0.8408

Recall (Weighted): 0.7886

F1-score (Weighted): 0.7994

ROC-AUC (OVR, Weighted): 0.9216

Key Insights

Combining CNN + Random Forest improved robustness compared to standalone models

Balanced accuracy exceeded overall accuracy, indicating strong performance on minority classes

Temporal context windows and smoothing significantly improved predictions

Feature engineering complemented deep learning by capturing domain-specific signal properties

Group-aware validation was critical for realistic performance estimation

Future Work

Potential improvements include:

expanding to full 5-stage sleep classification (Wake, REM, N1, N2, N3)

exploring transformer-based sequence models

optimizing CNN architecture and hyperparameters

incorporating additional biosignals (e.g., ECG, respiration)

deploying real-time sleep staging for wearable devices
