# Gaussian vs. Gaussian Mixture Models for MNIST Classification

This repository contains a Jupyter Notebook that compares the performance of a **Single Gaussian Model (SGM)** and a **Gaussian Mixture Model (GMM)** built from scratch to classify the MNIST handwritten digits dataset.

The goal of this project is to analyze how each probabilistic model represents the distributions of the digits and to evaluate their effectiveness in classifying unseen samples.

##  Features & Project Breakdown

1. **Data Preprocessing**
   - Loads the MNIST dataset (60,000 training images and 10,000 testing images).
   - Normalizes pixel values to the range [0, 1].
   - Applies **Principal Component Analysis (PCA)** for dimensionality reduction, retaining the top 60 components to optimize computational efficiency while preserving variance.

2. **Single Gaussian Model (SGM)**
   - Builds a classifier representing each class (digits 0-9) as a single multivariate Gaussian distribution.
   - Calculates the Mean and Covariance Matrix for each class from the training data.
   - Evaluates the model based on Maximum Likelihood estimation.

3. **Gaussian Mixture Model (GMM)**
   - Improves classification by modeling each class as a mixture of multiple Gaussian components, effectively capturing complex and multimodal distributions.
   - Implements the **Expectation-Maximization (EM)** algorithm from scratch to iteratively estimate the parameters (weights, means, and covariance matrices).

4. **Empirical Accuracy**
   - Compares the performance of both models:
     - The Single Gaussian Model (SGM) achieves an accuracy of approximately **96.16%**.
     - The Gaussian Mixture Model (GMM) achieves slightly higher and more stable accuracy, reaching around **96.38%**.

5. **Performance Evaluation & ROC Curves**
   - Plots the Receiver Operating Characteristic (ROC) curves for all digits (0-9).
   - Evaluates the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR), highlighting where GMM outperforms SGM (and vice versa, such as specific edge cases like digit 8 at low FPRs).

## Requirements

To run this notebook successfully, make sure you have the following libraries installed:
- `numpy`
- `scikit-learn` (for PCA and data splitting)
- `tensorflow` (only for `keras.datasets` to easily load MNIST)
- `scipy` (for statistical functions like `multivariate_normal` and `logsumexp`)
- `matplotlib` (for plotting ROC curves and graphs)

You can install the dependencies using:
```bash
pip install numpy scikit-learn tensorflow scipy matplotlib
