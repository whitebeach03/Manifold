import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

def foma_inputspace_per_class(x_batch, y_batch, num_classes, lam=0.5, k=20):
    B, C, H, W = x_batch.shape
    D = C * H * W
    x_flat = x_batch.view(B, -1)
    y_onehot = F.one_hot(y_batch, num_classes).float()
    x_aug_list, y_aug_list = [], []

    for cls in range(num_classes):
        mask = (y_batch == cls)
        if mask.sum() < 2:
            continue
        X_cls = x_flat[mask]
        Y_cls = y_onehot[mask]
        A = torch.cat([X_cls, Y_cls], dim=1)
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)
        n = S.shape[0]
        k = min(k, n)
        scale = torch.cat([
            torch.ones(k, device=A.device),
            torch.full((n - k,), lam, device=A.device)
        ])
        S_scaled = S * scale
        A_aug = U @ torch.diag(S_scaled) @ Vt
        X_aug = A_aug[:, :D].view(-1, C, H, W)
        Y_aug = F.softmax(A_aug[:, D:], dim=1)
        x_aug_list.append(X_aug)
        y_aug_list.append(Y_aug)

    if len(x_aug_list) == 0:
        return None, None
    x_aug_total = torch.cat(x_aug_list, dim=0)
    y_aug_total = torch.cat(y_aug_list, dim=0)
    return x_aug_total, y_aug_total

# def show_augmented_images_per_class(x_batch, y_batch, num_classes=10, lam=0.5, k=20, max_per_class=5):
#     x_aug, y_aug = foma_inputspace_per_class(x_batch, y_batch, num_classes, lam=lam, k=k)
#     if x_aug is None:
#         print("拡張できるクラスがバッチ内に存在しませんでした。")
#         return

#     pred_labels = y_aug.argmax(dim=1)

#     # クラスごとに画像を表示
#     fig, axs = plt.subplots(num_classes, max_per_class, figsize=(max_per_class * 2, num_classes * 2))
#     axs = axs if num_classes > 1 else [axs]

#     for cls in range(num_classes):
#         # クラスに属するインデックスを取得
#         indices = (pred_labels == cls).nonzero(as_tuple=True)[0][:max_per_class]
#         for j, idx in enumerate(indices):
#             img = x_aug[idx].cpu()
#             img = img.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
#             img = (img - img.min()) / (img.max() - img.min())  # 正規化（表示のため）
#             axs[cls][j].imshow(img)
#             axs[cls][j].axis('off')
#             axs[cls][j].set_title(f"Class {cls}")
#         # 空欄を埋める
#         for j in range(len(indices), max_per_class):
#             axs[cls][j].axis('off')

#     plt.tight_layout()
#     plt.show()

# # 🔧 テスト実行
# if __name__ == "__main__":
#     transform = transforms.Compose([transforms.ToTensor()])
#     dataset = datasets.STL10(root="./data", split="train", download=True, transform=transform)
#     loader = DataLoader(dataset, batch_size=256, shuffle=True)

#     x_batch, y_batch = next(iter(loader))
#     x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

#     show_augmented_images_per_class(x_batch, y_batch, num_classes=10, lam=0.5, k=20)


def compare_original_and_foma(x_batch, y_batch, num_classes=10, lam=0.5, k=20, max_per_class=5):
    x_aug, y_aug = foma_inputspace_per_class(x_batch, y_batch, num_classes, lam=lam, k=k)
    if x_aug is None:
        print("拡張可能なクラスがバッチに含まれていません。")
        return

    pred_labels = y_aug.argmax(dim=1)

    fig, axs = plt.subplots(num_classes, max_per_class * 2, figsize=(max_per_class * 4, num_classes * 2))

    for cls in range(num_classes):
        # クラスに属する拡張画像のインデックスを取得
        indices = (pred_labels == cls).nonzero(as_tuple=True)[0][:max_per_class]
        for j, idx in enumerate(indices):
            # 拡張画像
            img_aug = x_aug[idx].cpu()
            img_aug = img_aug.permute(1, 2, 0).numpy()
            img_aug = (img_aug - img_aug.min()) / (img_aug.max() - img_aug.min())

            # 対応する元画像（元バッチのうち、同じクラスの画像を j 番目に選ぶ）
            original_indices = (y_batch == cls).nonzero(as_tuple=True)[0]
            if len(original_indices) <= j:
                continue
            img_orig = x_batch[original_indices[j]].cpu()
            img_orig = img_orig.permute(1, 2, 0).numpy()
            img_orig = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min())

            # 表示：左に元画像、右にFOMA画像
            axs[cls][2 * j].imshow(img_orig)
            axs[cls][2 * j].axis('off')
            axs[cls][2 * j].set_title(f"Orig C{cls}")

            axs[cls][2 * j + 1].imshow(img_aug)
            axs[cls][2 * j + 1].axis('off')
            axs[cls][2 * j + 1].set_title(f"FOMA C{cls}")

        # 空欄埋め
        for j in range(len(indices), max_per_class):
            axs[cls][2 * j].axis('off')
            axs[cls][2 * j + 1].axis('off')

    plt.tight_layout()
    plt.savefig("FOMA-image.png")

# 🔧 テスト実行
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.STL10(root="./data", split="train", download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    x_batch, y_batch = next(iter(loader))
    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

    compare_original_and_foma(x_batch, y_batch, num_classes=10, lam=0.5, k=10)