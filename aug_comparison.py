import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from torchvision.transforms import v2

# # STL-10 データセットのダウンロードと読み込み
# dataset = torchvision.datasets.STL10(root="./data", split="train", download=True)
# image, label = dataset[0]  # 最初の画像を取得
# image = np.array(image)  # NumPy 配列に変換（形状 (96, 96, 3)）

# print(f"Image shape: {image.shape}")  # 確認 (96, 96, 3)

# # 画像を PIL 形式に変換
# image_pil = Image.fromarray(image)

# # 画像を表示する関数
# def show_images(images, titles, cols=3):
#     rows = (len(images) + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
#     axes = axes.flatten()
    
#     for img, title, ax in zip(images, titles, axes):
#         ax.imshow(img)
#         ax.set_title(title)
#         ax.axis("off")

#     plt.tight_layout()
#     plt.savefig("comparison.png")
#     # plt.show()

# # 反転（Flip）
# def apply_flip(img):
#     transform = transforms.RandomHorizontalFlip(p=1.0)
#     return transform(img)

# # クロップ（Crop）
# def apply_crop(img):
#     transform = transforms.RandomResizedCrop(size=(96, 96), scale=(0.5, 1.0))
#     return transform(img)

# # 回転（Rotation）
# def apply_rotation(img):
#     transform = transforms.RandomRotation(degrees=30)  # ±30度回転
#     return transform(img)

# # 平行移動（Translation）
# def apply_translation(img):
#     transform = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))  # 最大 20% 移動
#     return transform(img)

# # ノイズ注入（Noise Injection）
# def apply_noise(img):
#     img_np = np.array(img)
#     noise = np.random.normal(0, 20, img_np.shape)  # ガウシアンノイズ
#     noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
#     return Image.fromarray(noisy_img)

# # 色空間変換（Color Space Transformations）
# def apply_color_transform(img):
#     transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
#     return transform(img)

# # カーネルフィルタ（ぼかし & シャープ化）
# def apply_blur(img):
#     transform = transforms.GaussianBlur(kernel_size=5)
#     return transform(img)

# # Random Erasing
# def apply_random_erasing(img):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.RandomErasing(p=1.0, scale=(0.1, 0.3), ratio=(0.3, 3.3), value="random"),
#         transforms.ToPILImage()
#     ])
#     return transform(img)

# # 各種データ拡張を適用
# flipped_img = apply_flip(image_pil)
# cropped_img = apply_crop(image_pil)
# rotated_img = apply_rotation(image_pil)
# translated_img = apply_translation(image_pil)
# noisy_img = apply_noise(image_pil)
# color_transformed_img = apply_color_transform(image_pil)
# blurred_img = apply_blur(image_pil)
# random_erased_img = apply_random_erasing(image_pil)

# # 画像を表示
# show_images(
#     [image, flipped_img, cropped_img, rotated_img, translated_img, noisy_img, color_transformed_img, blurred_img, random_erased_img],
#     ["Original", "Flipped", "Cropped", "Rotated", "Translated", "Noisy", "Color Transformed", "Blurred", "Random Erased"]
# )


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# **デバイス設定**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **STL-10 データセットの取得**
dataset = torchvision.datasets.STL10(root="./data", split="train", download=True)
image, label = dataset[123]  # 最初の画像を取得

# **共通の基本変換**
base_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # グレースケールに変換
    transforms.ToTensor()  # Tensor に変換
])

# **データ拡張のリスト**
augmentations = {
        "Original": transforms.Compose([base_transform]),
        "Flipping": transforms.Compose([
            base_transform,
            transforms.RandomApply([transforms.RandomHorizontalFlip(p=1.0)], p=1.0),
        ]),
        "Cropping": transforms.Compose([
            base_transform,
            transforms.RandomApply([transforms.RandomResizedCrop(size=96, scale=(0.7, 1.0))], p=1.0),
        ]),
        "Rotation": transforms.Compose([
            base_transform,
            transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=1.0),
        ]),
        "Translation": transforms.Compose([
            base_transform,
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))], p=1.0),
        ]),
        "Noisy": transforms.Compose([
            base_transform,
            transforms.RandomApply([transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))], p=1.0),
        ]),
        "Blurring": transforms.Compose([
            base_transform, 
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=1.0),
        ]),
        "Random-Erasing": transforms.Compose([
            base_transform, 
            transforms.RandomApply([transforms.RandomErasing(p=1.0, scale=(0.1, 0.3), ratio=(0.3, 3.3), value=0)], p=1.0),
        ])
    }
augmentations2 = {
    "CutMix": transforms.Compose([
        base_transform,
        v2.CutMix(alpha=1.0, num_classes=10)
    ])
}

# **画像を表示する関数**
def show_images(images, titles, cols=3):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        img = img.squeeze(0).numpy()  # グレースケール画像（C=1）を 2D に変換
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):  # 余ったプロットを非表示にする
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(".png")

# **拡張後の画像を取得**
images = []
titles = []

for name, transform in augmentations.items():
    transformed_image = transform(image)
    images.append(transformed_image)
    titles.append(name)

# **画像を表示**
show_images(images, titles)
