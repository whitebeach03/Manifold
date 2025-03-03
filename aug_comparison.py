import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

# STL-10 データセットのダウンロードと読み込み
dataset = torchvision.datasets.STL10(root="./data", split="train", download=True)
image, label = dataset[0]  # 最初の画像を取得
image = np.array(image)  # NumPy 配列に変換（形状 (96, 96, 3)）

print(f"Image shape: {image.shape}")  # 確認 (96, 96, 3)

# 画像を PIL 形式に変換
image_pil = Image.fromarray(image)

# 画像を表示する関数
def show_images(images, titles, cols=3):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()
    
    for img, title, ax in zip(images, titles, axes):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("comparison.png")
    # plt.show()

# 反転（Flip）
def apply_flip(img):
    transform = transforms.RandomHorizontalFlip(p=1.0)
    return transform(img)

# クロップ（Crop）
def apply_crop(img):
    transform = transforms.RandomResizedCrop(size=(96, 96), scale=(0.5, 1.0))
    return transform(img)

# 回転（Rotation）
def apply_rotation(img):
    transform = transforms.RandomRotation(degrees=30)  # ±30度回転
    return transform(img)

# 平行移動（Translation）
def apply_translation(img):
    transform = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))  # 最大 20% 移動
    return transform(img)

# ノイズ注入（Noise Injection）
def apply_noise(img):
    img_np = np.array(img)
    noise = np.random.normal(0, 20, img_np.shape)  # ガウシアンノイズ
    noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# 色空間変換（Color Space Transformations）
def apply_color_transform(img):
    transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
    return transform(img)

# カーネルフィルタ（ぼかし & シャープ化）
def apply_blur(img):
    transform = transforms.GaussianBlur(kernel_size=5)
    return transform(img)

# Random Erasing
def apply_random_erasing(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.1, 0.3), ratio=(0.3, 3.3), value="random"),
        transforms.ToPILImage()
    ])
    return transform(img)

# 各種データ拡張を適用
flipped_img = apply_flip(image_pil)
cropped_img = apply_crop(image_pil)
rotated_img = apply_rotation(image_pil)
translated_img = apply_translation(image_pil)
noisy_img = apply_noise(image_pil)
color_transformed_img = apply_color_transform(image_pil)
blurred_img = apply_blur(image_pil)
random_erased_img = apply_random_erasing(image_pil)

# 画像を表示
show_images(
    [image, flipped_img, cropped_img, rotated_img, translated_img, noisy_img, color_transformed_img, blurred_img, random_erased_img],
    ["Original", "Flipped", "Cropped", "Rotated", "Translated", "Noisy", "Color Transformed", "Blurred", "Random Erased"]
)
