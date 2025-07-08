import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from src.models.resnet import ResNet18

# — 1) デバイス & クラス名 —
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']

# — 2) STL-10 の Test セット準備 —
transform_stl = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4467,0.4398,0.4066],
        std =[0.2241,0.2210,0.2239]
    ),
])
test_stl = STL10(
    root="./data",
    split="train",
    download=True,
    transform=transform_stl
)

# — 3) “bird” と “truck” のサンプルをひとつずつ選ぶ —
#    最初に見つかったものを使います
def find_first(dataset, target_label):
    for img, label in dataset:
        if label == target_label:
            return img
    raise ValueError(f"No sample with label={target_label}")

airplane_img  = find_first(test_stl, class_names.index("cat")).to(device)   # (C,H,W)
monkey_img    = find_first(test_stl, class_names.index("deer")).to(device)  # (C,H,W)

# — 4) 学習済み STL-10 モデルをロード —
#    すでに訓練した Wide-ResNet / ResNet18 などを model に代入済みとします
model = ResNet18().to(device)
model.load_state_dict(torch.load("./logs/resnet18/Mixup/stl10_200.pth"))
model.eval()

# — 5) λ を変えつつ混合→予測→確率取得 →
lams = np.linspace(0, 1, 100)
p_plane, p_bird, p_car, p_cat, p_deer, p_dog, p_horse, p_monkey, p_ship, p_truck = [], [], [], [], [], [], [], [], [], []

with torch.no_grad():
    for lam in lams:
        mixed = lam * airplane_img + (1 - lam) * monkey_img       # (C,H,W)
        out   = model(mixed.unsqueeze(0), labels=None, device=device, augment="Mixup")                    # (1,10)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()   # (10,)

        p_plane.append (probs[class_names.index("airplane")])
        p_bird.append(probs[class_names.index("bird")])
        p_car.append(probs[class_names.index("car")])
        p_cat.append (probs[class_names.index("cat")])
        p_deer.append(probs[class_names.index("deer")])
        p_dog.append(probs[class_names.index("dog")])
        p_horse.append (probs[class_names.index("horse")])
        p_monkey.append(probs[class_names.index("monkey")])
        p_ship.append(probs[class_names.index("ship")])
        p_truck.append(probs[class_names.index("truck")])

# — 6) プロット —
plt.figure(figsize=(6,4))
plt.plot(lams, p_plane, label='airplane')
plt.plot(lams, p_bird, label='bird')
plt.plot(lams, p_car, label='car')
plt.plot(lams, p_cat, label='cat')
plt.plot(lams, p_deer, label='deer')
plt.plot(lams, p_dog, label='dog')
plt.plot(lams, p_horse, label='horse')
plt.plot(lams, p_monkey, label='monkey')
plt.plot(lams, p_ship, label='ship')
plt.plot(lams, p_truck, label='truck')

plt.xlabel("λ")
plt.ylabel("Prediction probability")
plt.title("STL10 deer-cat mixup")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lambda4.png")
plt.show()
