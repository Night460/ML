from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Standardize the features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
Z = (X - X_mean) / X_std

# Compute covariance matrix and perform eigen decomposition
cov_matrix = np.cov(Z, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Compute explained variance
explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
n_components = np.argmax(explained_var >= 0.50) + 1

# Select top eigenvectors
top_eigenvectors = eigenvectors[:, :n_components]

# Transform data using selected eigenvectors
Z_pca = Z @ top_eigenvectors
pca_df = pd.DataFrame(Z_pca, columns=[f'PC{i+1}' for i in range(n_components)])

# Plot heatmap of PCA components
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(top_eigenvectors, columns=[f'PC{i+1}' for i in range(n_components)]), annot=True, cmap='coolwarm')
plt.title('PCA Components')
plt.show()

# Plot PCA-transformed data
plt.figure(figsize=(10, 6))
plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=y, cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Transformed Data')
plt.colorbar(label='Wine Class')
plt.show()
