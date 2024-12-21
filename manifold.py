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
reducer = 'umap'
epsilon = 0.1
pca_dim = 20

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform  = transforms.ToTensor()
dataset    = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

images, _  = next(iter(dataloader))  
images     = images.view(len(dataset), -1).numpy()  

if reducer == 'pca':
    pca = PCA(n_components=pca_dim)
    pca.fit(images)
    images = images[dataset.targets == 0]
    vec = pca.transform(images)
elif reducer == 'kernel_pca':
    kernel_pca = KernelPCA(n_components=pca_dim, kernel='rbf', gamma=0.1)  # RBFカーネル
    vec = kernel_pca.fit_transform(images)
elif reducer == 'umap':
    images = images[dataset.targets == 0]
    reducer = umap.UMAP(n_components=pca_dim, random_state=42)
    # reducer = umap.UMAP()
    reduce_vec = reducer.fit_transform(images)
    plt.scatter(reduce_vec[:, 0], reduce_vec[:, 1])
    plt.title("UMAP projection") 
    plt.show()
    np.save("umap_results.npy", reduce_vec)
    vec = np.load("umap_results.npy")

sample_vec0 = vec[0]
sample_vec1 = vec[1]

print(sample_vec0)
print(sample_vec1)

original_image0 = images[0].reshape(28, 28)
original_image1 = images[1].reshape(28, 28)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(original_image0, cmap='gray')
ax[0].set_title("Image #1")
ax[1].imshow(original_image1, cmap='gray')
ax[1].set_title("Image #2")
plt.savefig('./result/sample.png')