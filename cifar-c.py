# import matplotlib.pyplot as plt
# import numpy as np
# from torchvision.datasets import CIFAR100
# import torchvision.transforms as T

# image_idx = 2

# # --- 1. CIFAR-100 Original ---
# transform = T.ToTensor()
# cifar100 = CIFAR100(root='./data', train=False, download=True, transform=transform)
# orig_img, label = cifar100[image_idx]  # 1枚取得

# # --- 2. CIFAR-100-C Corruption Names ---
# corruptions = [
#     "gaussian_noise", "shot_noise", "impulse_noise",
#     "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
#     "snow", "frost", "fog", "brightness",
#     "contrast", "elastic_transform", "pixelate", "jpeg_compression"
# ]

# # --- 3. Severity Level（1〜5を指定）---
# severity = 3
# start_idx = (severity - 1) * 10000
# end_idx = severity * 10000

# # --- 4. Plot ---
# fig, axes = plt.subplots(1, len(corruptions) + 1, figsize=(20, 3))

# # Original image
# axes[0].imshow(np.transpose(orig_img, (1, 2, 0)))
# axes[0].set_title("Original", fontsize=10)
# axes[0].axis("off")

# # Corrupted versions
# for i, cname in enumerate(corruptions):
#     c_data = np.load(f"./data/CIFAR-100-C/{cname}.npy")
#     c_img = c_data[image_idx] / 255.0  # severityに対応する画像を取得
#     axes[i + 1].imshow(c_img)
#     axes[i + 1].set_title(f"{cname.replace('_', ' ').title()}\n(severity={severity})", fontsize=8)
#     axes[i + 1].axis("off")

# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import os

image_idx = 14

# --- 1. CIFAR-10 Original ---
transform = T.ToTensor()
cifar100 = CIFAR10(root='./data', train=False, download=True, transform=transform)
orig_img, label = cifar100[image_idx]

# --- 2. CIFAR-100-C corruption list ---
corruptions = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression"
]

# --- 3. severity の設定（1〜5） ---
severity = 5

# --- 4. 出力フォルダ ---
out_dir = f"./visualized_corruptions/severity_{severity}"
os.makedirs(out_dir, exist_ok=True)

# --- 5. オリジナル画像の保存 ---
plt.imshow(np.transpose(orig_img, (1, 2, 0)))
plt.title("Original", fontsize=12)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "original.png"), bbox_inches="tight")
plt.close()
print("Saved: Original")

# --- 6. 各 corruption タイプごとに出力 ---
for cname in corruptions:
    c_data = np.load(f"./data/CIFAR-10-C/{cname}.npy")
    c_img = c_data[image_idx] / 255.0  # severity に対応する最初の画像を使用
    
    plt.imshow(c_img)
    plt.title(f"{cname.replace('_', ' ').title()} (sev={severity})", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f"{cname}_sev{severity}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cname}")
