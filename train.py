import argparse
import numpy as np
import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.utils import *
from src.models.resnet import ResNet18
from src.models.wide_resnet import Wide_ResNet
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
from foma import foma
from batch_sampler import extract_wrn_features, FeatureKNNBatchSampler, HybridFOMABatchSampler

def main():
    for i in range(1):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs",     type=int, default=400)
        parser.add_argument("--data_type",  type=str, default="cifar100",          choices=["stl10", "cifar100", "cifar10"])
        parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "wide_resnet_28_10"])
        args = parser.parse_args() 

        epochs     = args.epochs
        data_type  = args.data_type
        model_type = args.model_type
        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Number of Classes & Batch Size
        if data_type == "stl10":
            num_classes = 10
            batch_size  = 64
        elif data_type == "cifar100":
            num_classes = 100
            batch_size  = 128
        elif data_type == "cifar10":
            num_classes = 10
            batch_size  = 128
        
        default_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Cutout(n_holes=1, length=16),
        ])
                
        # Loading Dataset
        if data_type == "stl10":
            transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_dataset = STL10(root="./data", split="test",  download=True, transform=default_transform)
            test_dataset  = STL10(root="./data", split="train", download=True, transform=transform)
        elif data_type == "cifar100":
            transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_dataset = CIFAR100(root="./data", train=True,  transform=default_transform, download=True)
            test_dataset  = CIFAR100(root="./data", train=False, transform=transform, download=True)
        elif data_type == "cifar10":
            transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_dataset = CIFAR10(root="./data", train=True,  transform=default_transform, download=True)
            test_dataset  = CIFAR10(root="./data", train=False, transform=transform, download=True)
        
        n_samples = len(train_dataset)
        n_train   = int(n_samples * 0.8)
        n_val     = n_samples - n_train
        train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])

        val_loader   = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        # Augmentation List
        augmentations = [
            "Default",
            "Mixup",
            "Manifold-Mixup",
            # "FOMA",
            "FOMA_latent_random",
            
            # "FOMA_hard",
            # "FOMA_curriculum"
            # "FOMA_samebatch"
            # "FOMA_knn"

            # "Mixup-Original",
            # "Mixup-PCA",
            # "Mixup-Original&PCA",
            # "PCA",
        ]

        for augment in augmentations:
            print(f"\n==> Training with {augment} ...")

            # Select Model
            if model_type == "resnet18":
                model = ResNet18().to(device)
            elif model_type == "wide_resnet_28_10":
                model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
                
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())
            score     = 0.0
            history   = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

            os.makedirs(f"./logs/{model_type}/{augment}",    exist_ok=True)
            os.makedirs(f"./history/{model_type}/{augment}", exist_ok=True)

            ### TRAINING ###
            for epoch in range(epochs):
                train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, augment, num_classes, aug_ok=False, epochs=epoch)
                val_loss, val_acc     = val(model, val_loader, criterion, device, augment, aug_ok=False)

                if score <= val_acc:
                    print("Save model parameters...")
                    score = val_acc
                    model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}.pth"
                    torch.save(model.state_dict(), model_save_path)
                
                history["loss"].append(train_loss)
                history["accuracy"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                print(f"| {epoch+1} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f} |")

            with open(f"./history/{model_type}/{augment}/{data_type}_{epochs}_{i}.pickle", "wb") as f:
                pickle.dump(history, f)
            
            ### TEST ###
            model.load_state_dict(torch.load(model_save_path, weights_only=True))
            test_loss, test_acc = test(model, test_loader, criterion, device, augment, aug_ok=False)
            print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")

            test_history = {"acc": test_acc, "loss": test_loss}
            with open(f"./history/{model_type}/{augment}/{data_type}_{epochs}_{i}_test.pickle", "wb") as f:
                pickle.dump(test_history, f)

if __name__ == "__main__":
    main()
































# train.py

# import os
# import torch
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random
# import pickle
# import argparse
# from collections import defaultdict

# from sklearn.metrics import accuracy_score
# from tqdm import tqdm
# from torchvision.datasets import STL10, CIFAR10, CIFAR100
# from torch.utils.data import DataLoader, random_split, Subset

# from src.models.resnet import ResNet18
# from src.models.wide_resnet import Wide_ResNet
# from foma import foma, foma_hard
# from src.utils import train, val, test, mixup_data, mixup_criterion

# from batch_sampler import HybridFOMABatchSampler
# from batch_sampler import extract_wrn_features  # assuming this function exists as before

# def main():
#     for i in range(1):
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--epochs", type=int, default=250)
#         parser.add_argument("--data_type", type=str, default="cifar100", choices=["stl10", "cifar100", "cifar10"])
#         parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "wide_resnet_28_10"])
#         parser.add_argument("--alpha", type=float, default=0.6, help="Ratio of local-KNN in batch (default: 0.6)")
#         args = parser.parse_args()

#         epochs     = args.epochs
#         data_type  = args.data_type
#         model_type = args.model_type
#         alpha      = args.alpha
#         device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Number of Classes & Batch Size
#         if data_type == "stl10":
#             num_classes = 10
#             batch_size  = 64
#         elif data_type == "cifar100":
#             num_classes = 100
#             batch_size  = 128
#         elif data_type == "cifar10":
#             num_classes = 10
#             batch_size  = 128

#         # Loading Dataset
#         if data_type == "stl10":
#             transform     = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#             full_train_dataset = STL10(root="./data", split="test",  download=True, transform=transform)
#             test_dataset       = STL10(root="./data", split="train", download=True, transform=transform)
#         elif data_type == "cifar100":
#             transform     = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#             full_train_dataset = CIFAR100(root="./data", train=True,
#                                           transform=transform, download=True)
#             test_dataset       = CIFAR100(root="./data", train=False,
#                                           transform=transform, download=True)
#         elif data_type == "cifar10":
#             transform     = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#             full_train_dataset = CIFAR10(root="./data", train=True,
#                                          transform=transform, download=True)
#             test_dataset       = CIFAR10(root="./data", train=False,
#                                          transform=transform, download=True)

#         # Split into train / val
#         n_samples = len(full_train_dataset)
#         n_train   = int(n_samples * 0.8)
#         n_val     = n_samples - n_train
#         train_subset, val_subset = random_split(full_train_dataset, [n_train, n_val])

#         # DataLoaders for val and test
#         val_loader   = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)
#         test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#         ###  FOMA_knn  ############################################################################################################
#         # Load a pretrained Wide_ResNet to extract static features
#         wrn_path = f"./logs/wide_resnet_28_10/Original/{data_type}_{epochs}_{i}.pth"
#         wrn_model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
#         wrn_model.load_state_dict(torch.load(wrn_path,
#                                              map_location=device,
#                                              weights_only=True),
#                                   strict=True)
#         wrn_model.eval()
#         for param in wrn_model.parameters():
#             param.requires_grad = False

#         # Extract features for all samples in train_subset
#         all_features = extract_wrn_features(
#             model=wrn_model,
#             dataset=train_subset,
#             device=device,
#             batch_size=256,
#             num_workers=4
#         )
#         assert all_features.shape[0] == n_train

#         # Gather labels aligned with train_subset
#         # train_subset is a Subset, so train_subset.indices gives original dataset indices
#         if hasattr(full_train_dataset, 'targets'):
#             orig_targets = full_train_dataset.targets
#         else:
#             # For STL10, labels are in train_subset.dataset.labels
#             orig_targets = full_train_dataset.labels

#         labels = [orig_targets[idx] for idx in train_subset.indices]
#         ###########################################################################################################################

#         # Augmentation List
#         augmentations = {
#             "FOMA_knn"
#         }

#         for augment in augmentations:
#             if augment == "FOMA_knn":
#                 sampler = HybridFOMABatchSampler(
#                     feature_matrix=all_features,
#                     labels=labels,
#                     batch_size=batch_size,
#                     alpha=alpha,
#                     k_neighbors=None,   # デフォルトで ceil(alpha * B) を使う
#                     drop_last=False
#                 )
#                 train_loader = DataLoader(
#                     dataset=train_subset,
#                     batch_sampler=sampler,
#                     num_workers=4
#                 )
#             else:
#                 train_loader = DataLoader(
#                     dataset=train_subset,
#                     batch_size=batch_size,
#                     shuffle=True
#                 )

#             print(f"\n==> Training with {augment} ...")

#             # Select Model for training from scratch
#             if model_type == "resnet18":
#                 model = ResNet18().to(device)
#             elif model_type == "wide_resnet_28_10":
#                 model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)

#             criterion = nn.CrossEntropyLoss()
#             optimizer = optim.Adam(model.parameters())
#             best_score = 0.0
#             history   = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

#             os.makedirs(f"./logs/{model_type}/{augment}",    exist_ok=True)
#             os.makedirs(f"./history/{model_type}/{augment}", exist_ok=True)

#             # TRAINING #
#             for epoch in range(epochs):
#                 train_loss, train_acc = train(
#                     model, train_loader,
#                     criterion, optimizer,
#                     device, augment,
#                     num_classes,
#                     aug_ok=False,
#                     epochs=epoch
#                 )
#                 val_loss, val_acc = val(
#                     model, val_loader,
#                     criterion, device,
#                     augment, aug_ok=False
#                 )

#                 if best_score <= val_acc:
#                     print("Save model parameters...")
#                     best_score = val_acc
#                     model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}.pth"
#                     torch.save(model.state_dict(), model_save_path)

#                 history["loss"].append(train_loss)
#                 history["accuracy"].append(train_acc)
#                 history["val_loss"].append(val_loss)
#                 history["val_accuracy"].append(val_acc)
#                 print(f"| {epoch+1} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} "
#                       f"| Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f} |")

#             # Save history
#             with open(f"./history/{model_type}/{augment}/{data_type}_{epochs}_{i}.pickle", "wb") as f:
#                 pickle.dump(history, f)

#             # TEST #
#             model.load_state_dict(torch.load(model_save_path, weights_only=True))
#             model.eval()
#             test_loss, test_acc = test(
#                 model, test_loader,
#                 criterion, device,
#                 augment, aug_ok=False
#             )
#             print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")

#             test_history = {"acc": test_acc, "loss": test_loss}
#             with open(f"./history/{model_type}/{augment}/{data_type}_{epochs}_{i}_test.pickle", "wb") as f:
#                 pickle.dump(test_history, f)


# if __name__ == "__main__":
#     main()
