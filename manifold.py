import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

### MNIST: 28x28 = 784 pixels ###

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform  = transforms.ToTensor()
dataset    = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

images, _  = next(iter(dataloader))  
images     = images.view(len(dataset), -1).numpy()  

pca = PCA(n_components=300)
pca.fit(images)

vec = pca.transform(images)

sample_vec = vec[0]

reconstructed_image = pca.inverse_transform(sample_vec.reshape(1, -1))
reconstructed_image = reconstructed_image.reshape(28, 28)

original_image = images[0].reshape(28, 28)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(original_image, cmap='gray')
ax[0].set_title("Original Image")
ax[1].imshow(reconstructed_image, cmap='gray')
ax[1].set_title("Image #1")
plt.savefig('./result/sample.png')


