import pickle

dic = {}

pickle_file_path = f'./history/wide_resnet_28_10/Manifold-Mixup/cifar100_400_0.pickle'
with open(pickle_file_path, 'rb') as f:
    dic = pickle.load(f)

for i in range(400):
    val_acc = dic["val_accuracy"][i]
    if val_acc >= 0.66:
        print(i+1, ": ", val_acc)