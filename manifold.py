import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import umap
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA, KernelPCA

# Hyperparameters
epsilon = 0.1
pca_dim = 10

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform  = transforms.ToTensor()
dataset    = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

images, _  = next(iter(dataloader))  
images     = images.view(len(dataset), -1).numpy()  

### PCA ###
# pca = PCA(n_components=pca_dim)
# pca.fit(images)
# vec = pca.transform(images)

### Kernel PCA ###
# kernel_pca = KernelPCA(n_components=pca_dim, kernel='rbf', gamma=0.1)  # RBFカーネルを使用
# vec = kernel_pca.fit_transform(images)

### UMAP ###
reducer = umap.UMAP(n_components=pca_dim, random_state=42)
vec = reducer.fit_transform(images)
# 保存
np.save("umap_results.npy", vec)
# ロード
loaded_data = np.load("umap_results.npy")

sample_vec0 = loaded_data[0]
sample_vec1 = loaded_data[1]

print(sample_vec0)
print(sample_vec1)

# ノイズ付与しない場合
# reconstructed_image = pca.inverse_transform(sample_vec.reshape(1, -1))
# reconstructed_image = reconstructed_image.reshape(28, 28)

# # ノイズ付与する場合
# noise = np.random.normal(0, 1, size=sample_vec.shape)
# sample_vec_aug = sample_vec + epsilon * noise
# reconstructed_image_aug = pca.inverse_transform(sample_vec_aug.reshape(1, -1))
# reconstructed_image_aug = reconstructed_image_aug.reshape(28, 28)


original_image0 = images[0].reshape(28, 28)
original_image1 = images[1].reshape(28, 28)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(original_image0, cmap='gray')
ax[0].set_title("Image #1")
ax[1].imshow(original_image1, cmap='gray')
ax[1].set_title("Image #2")
# ax[2].imshow(reconstructed_image_aug, cmap='gray')
# ax[2].set_title("Image (noised)")
plt.savefig('./result/sample.png')