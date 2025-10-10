# augment_and_save.py
import os, random
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms.functional as F

# ───────────────────────────────────────────────
# 画像を読み込む
# ───────────────────────────────────────────────
SRC_IMG = "six.jpeg"              # ここを書き換えてください
OUT_DIR  = Path("output")
OUT_DIR.mkdir(exist_ok=True)

img = Image.open(SRC_IMG).convert("RGB")

# ───────────────────────────────────────────────
# Cutout と Sample Pairing 用のユーティリティ
# ───────────────────────────────────────────────
def cutout(pil_img, size_frac=0.3):
    """画像中央付近に正方形マスクを貼る（Cutout）"""
    w, h = pil_img.size
    cut = int(min(w, h) * size_frac)
    x0 = random.randint(0, w - cut)
    y0 = random.randint(0, h - cut)
    mask = Image.new("RGB", (cut, cut), (0, 0, 0))
    pil_img = pil_img.copy()
    pil_img.paste(mask, (x0, y0))
    return pil_img

def sample_pairing(pil_img, other_pil_img, alpha=0.5):
    """2枚の画像を平均合成（Sample Pairing）"""
    other_resized = other_pil_img.resize(pil_img.size)
    return Image.blend(pil_img, other_resized, alpha)

# ───────────────────────────────────────────────
# 拡張ごとの関数
# ───────────────────────────────────────────────
AUGS = {
    # 幾何変換
    "ShearX"      : lambda x: F.affine(x, angle=0, translate=(0,0), scale=1, shear=(20, 0)),
    "ShearY"      : lambda x: F.affine(x, angle=0, translate=(0,0), scale=1, shear=(0, 20)),
    "TranslateX"  : lambda x: F.affine(x, angle=0, translate=(int(0.2*x.size[0]),0), scale=1, shear=0),
    "TranslateY"  : lambda x: F.affine(x, angle=0, translate=(0,int(0.2*x.size[1])), scale=1, shear=0),
    "Rotate"      : lambda x: F.rotate(x, angle=160),

    # ピクセル変換
    "AutoContrast": lambda x: ImageOps.autocontrast(x),
    "Invert"      : lambda x: ImageOps.invert(x),
    "Equalize"    : lambda x: ImageOps.equalize(x),
    "Solarize"    : lambda x: ImageOps.solarize(x, threshold=128),
    "Posterize"   : lambda x: ImageOps.posterize(x, bits=4),
    "Contrast"    : lambda x: ImageEnhance.Contrast(x).enhance(1.8),
    "Color"       : lambda x: ImageEnhance.Color(x).enhance(1.8),
    "Brightness"  : lambda x: ImageEnhance.Brightness(x).enhance(1.5),
    "Sharpness"   : lambda x: ImageEnhance.Sharpness(x).enhance(2.0),

    # 領域／サンプル系
    "Cutout"      : cutout,
    # Sample Pairing は original と 180° 回転版を合成して例示
    "SamplePair"  : lambda x: sample_pairing(x, F.rotate(x, 180)),
}

# ───────────────────────────────────────────────
# 適用して保存
# ───────────────────────────────────────────────
for name, fn in AUGS.items():
    out_img = fn(img)
    out_img.save(OUT_DIR / f"{name}.jpg")
    print(f"[✓] {name} → {name}.jpg")
