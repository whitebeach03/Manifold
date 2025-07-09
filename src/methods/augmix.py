import numpy as np
import random
import torch
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as T

__all__ = ['AugMixTransform']

def _int_parameter(level, maxval):
    return int(level * maxval / 10)

def _float_parameter(level, maxval):
    return float(level) * maxval / 10.0

def autocontrast(img, severity):
    return ImageOps.autocontrast(img)

def equalize(img, severity):
    return ImageOps.equalize(img)

def rotate(img, severity):
    degrees = _int_parameter(severity, 30)
    if random.random() > 0.5:
        degrees = -degrees
    return img.rotate(degrees)

def solarize(img, severity):
    threshold = _int_parameter(severity, 256)
    return ImageOps.solarize(img, 256 - threshold)

def color(img, severity):
    factor = _float_parameter(severity, 1.8) + 0.1
    return ImageEnhance.Color(img).enhance(factor)

def posterize(img, severity):
    bits = 4 - _int_parameter(severity, 4)
    return ImageOps.posterize(img, bits)

def contrast(img, severity):
    factor = _float_parameter(severity, 1.8) + 0.1
    return ImageEnhance.Contrast(img).enhance(factor)

def brightness(img, severity):
    factor = _float_parameter(severity, 1.8) + 0.1
    return ImageEnhance.Brightness(img).enhance(factor)

def sharpness(img, severity):
    factor = _float_parameter(severity, 1.8) + 0.1
    return ImageEnhance.Sharpness(img).enhance(factor)

def shear_x(img, severity):
    level = _float_parameter(severity, 0.3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))

def shear_y(img, severity):
    level = _float_parameter(severity, 0.3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))

def translate_x(img, severity):
    level = _float_parameter(severity, img.size[0] / 3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))

def translate_y(img, severity):
    level = _float_parameter(severity, img.size[1] / 3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))

_AUGMENTATIONS = [
    autocontrast, equalize, rotate, solarize, color,
    posterize, contrast, brightness, sharpness,
    shear_x, shear_y, translate_x, translate_y
]

def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """
    Perform AugMix augmentations and compute mixture.
    Args:
      image: PIL Image
      severity: strength of augmentation operators (1 to 10)
      width: number of augmentation chains
      depth: depth of augmentation chain (-1 for random depth)
      alpha: Dirichlet/Beta distribution parameter
    Returns:
      PIL Image
    """
    ws = np.random.dirichlet([alpha] * width).astype('float32')
    m = np.random.beta(alpha, alpha)
    mix = np.zeros_like(np.asarray(image), dtype=np.float32)
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = random.choice(_AUGMENTATIONS)
            image_aug = op(image_aug, severity)
        mix += ws[i] * np.asarray(image_aug, dtype=np.float32)
    mixed = (1 - m) * np.asarray(image, dtype=np.float32) + m * mix
    mixed = mixed.astype(np.uint8)
    return Image.fromarray(mixed)

class AugMixTransform:
    """
    Combined transform: default pre-augmentation, AugMix, and post-augmentation.

    Applies:
      1) pre_transforms: RandomHorizontalFlip, Pad(4), RandomCrop(32)
      2) AugMix augment_and_mix
      3) post_transforms: ToTensor, Normalize
    """
    def __init__(self, severity=3, width=3, depth=-1, alpha=1.,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        # Pre and post transforms
        self.pre_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.Pad(4),
            T.RandomCrop(32)
        ])
        self.post_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        self.to_pil = T.ToPILImage()

    def __call__(self, img):
        # img: Tensor or PIL
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)
        # Pre-augmentation (flip, pad, crop)
        img = self.pre_transforms(img)
        # AugMix
        img = augment_and_mix(img, self.severity, self.width, self.depth, self.alpha)
        # Post-augmentation (ToTensor, Normalize)
        return self.post_transforms(img)
