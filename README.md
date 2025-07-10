# Network-Intrusion-detection-IDS-

The dataset were taken from - https://www.unb.ca/cic/datasets/ids-2017.html

Developed a machine learning–based intrusion detection system (IDS) to classify network traffic as benign or malicious using the CIC-IDS 2017 dataset. The project aimed to automate detection of diverse cyber attacks such as DDoS, brute-force, port scans, botnets, and web attacks by building robust, production-ready classification pipelines.

Key Steps
Data Preprocessing

Parsed large raw CSV logs from CIC-IDS 2017.

Cleaned data, handled missing values and infinities.

Performed real-time feature extraction simulation and standard scaling.

Feature Engineering

Used ExtraTreesClassifier for feature importance ranking.

Selected the top 30 features to reduce dimensionality while retaining signal.

Addressed skewed feature distributions using PowerTransformer.

Handling Class Imbalance

Downsampled the majority (Benign) class for tree-based models.

Applied SMOTE combined with under-sampling for linear models to balance classes effectively.

Model Building & Evaluation

Trained multiple models:

Decision Tree

Random Forest

XGBoost

LightGBM

Logistic Regression

Support Vector Machine

K-Nearest Neighbors

Evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

Compared results across data variants: raw scaled, skewness-transformed, balanced datasets.

Deployment Preparation

Prepared the final balanced dataset and trained models for deployment.

Designed architecture for a potential real-time detection pipeline.
