import pickle

dic = {}

pickle_file_path = f'./history/wide_resnet_28_10/FOMA-Mixup/cifar100_400_0.pickle'
with open(pickle_file_path, 'rb') as f:
    dic = pickle.load(f)

for epoch in range(400):
    train_acc  = dic["accuracy"][epoch]
    train_loss = dic["loss"][epoch]
    val_acc    = dic["val_accuracy"][epoch]
    val_loss   = dic["val_loss"][epoch]
    if val_acc >= 0.78:
        print(f"| {epoch+1} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f} |")
