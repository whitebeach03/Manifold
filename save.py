#!/usr/bin/env python3
import os
import argparse
from torchvision.datasets import CIFAR100
from torchvision import transforms, utils
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',   type=int,   default=10,
                        help='何エポック分保存するか (エポック0～N)')
    parser.add_argument('--n_images', type=int,   default=8,
                        help='一度に並べる画像枚数')
    parser.add_argument('--out_dir',  type=str,   default='epoch_images',
                        help='保存先ディレクトリ')
    parser.add_argument('--padding',  type=int,   default=8,
                        help='画像間のピクセル余白')
    parser.add_argument('--bg_white', action='store_true',
                        help='余白を白(255)にする場合はフラグをつける')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) CIFAR-100 の生データを PIL のままロード
    cifar = CIFAR100(root='./data', train=True, download=True, transform=None)

    # 2) 可視化用に最初の n_images 枚だけ PIL.Image のリストとして取得
    orig_imgs = [cifar[i][0] for i in range(args.n_images)]

    # 3) epoch0用の Tensor 化だけ transform
    to_tensor = transforms.ToTensor()

    # 4) 質問のデフォルト変換
    default_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(4),
        transforms.RandomCrop(32),
    ])

    for epoch in range(args.epochs + 1):
        if epoch == 0:
            # 拡張なし
            imgs = [to_tensor(img) for img in orig_imgs]
        else:
            # 毎回ランダム変換を適用
            imgs = [to_tensor(default_transform(img)) for img in orig_imgs]

        # 横一列に並べ，padding ピクセルだけ余白を入れる
        # pad_value=0: 黒，=255: 白
        pad_val = 255 if args.bg_white else 0
        grid = utils.make_grid(
            imgs,
            nrow=args.n_images,     # 全枚数を一行に
            padding=args.padding,    # 画像間の余白
            pad_value=255,
        )

        # 保存
        fname = f'epoch_{epoch:03d}.png'
        save_path = os.path.join(args.out_dir, fname)
        utils.save_image(grid, save_path)
        print(f'[Epoch {epoch:3d}] saved {save_path}')

if __name__ == '__main__':
    main()
