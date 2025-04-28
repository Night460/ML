# 1. Use non-GUI backend to avoid Tkinter error
import matplotlib
matplotlib.use('Agg')  # Use backend that doesn't require GUI

# 2. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# 3. Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# 4. Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 5. Apply PCA (reduce to 3 components)
pca = PCA(n_components=3)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

# 6. Shape after PCA
print("Shape after PCA:", x_pca.shape)

# 7. Eigenvectors (principal components)
print("\nPrincipal Components (Eigenvectors):")
print(pca.components_)

# 8. 2D Visualization and save to file
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=data.target, cmap='plasma')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - 2D Plot')
plt.colorbar()
plt.savefig("pca_2d_plot.png")
plt.close()

# 9. 3D Visualization and save to file
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_pca[:,0], x_pca[:,1], x_pca[:,2], c=data.target, cmap='plasma')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("PCA - 3D Plot")
fig.savefig("pca_3d_plot.png")
plt.close()

# 10. Explained Variance Ratio
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
