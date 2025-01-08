import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torchvision
import matplotlib.cm as cm
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from src.reducer import *
from src.regressor import *
from src.sampling import *
from src.plot_data import *
from src.utils import *

N = 300

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
# dataset   = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
dataset   = torchvision.datasets.MNIST(root='./data', download=True, transform=transform)
dataset   = organize_by_class(dataset)

for l in range(10):
    print(f'MNIST Label: {l}')
    pca = PCA(n_components=N)
    pca.fit(dataset[l])

    # 主成分を画像化 (最初の50主成分を表示)
    rows, cols = 5, 10  # 表示する行数と列数
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
    for i in range(rows):
        for j in range(cols):
            component_index = i * cols + j
            if component_index < 50:  # 最初の50成分のみ表示
                component = pca.components_[component_index].reshape(28, 28)
                axes[i, j].imshow(component, cmap='gray')
                axes[i, j].set_title(f'C{component_index + 1}', fontsize=8)
                axes[i, j].axis('off')

    # cols = 10
    # rows = int(np.ceil(N/float(cols)))
    # fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20, 10))
    # for i in range(N):
    #     r = i // cols
    #     c = i % cols
    #     axes[r, c].imshow(pca.components_[i].reshape(28, 28),vmin=-0.5,vmax=0.5, cmap = cm.Greys_r)
    #     axes[r, c].set_title('component %d' % i)
    #     axes[r, c].get_xaxis().set_visible(False)
    #     axes[r, c].get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig(f'mnist_{l}.png')