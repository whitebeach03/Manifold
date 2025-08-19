import matplotlib.pyplot as plt

k = [10, 16, 20, 32, 64]
cifar100_acc = [77.96, 78.59, 78.61, 78.93, 78.51]
cifar100_C_acc = [37.37, 38.39, 37.49, 38.77, 38.62]

fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# CIFAR-100
ax[0].plot(k, cifar100_acc, marker='o', linestyle='-', color='tab:blue')
ax[0].set_ylim(77.5, 79.0)
ax[0].set_ylabel("CIFAR-100 Acc (%)")
ax[0].grid(True)

# CIFAR-100-C
ax[1].plot(k, cifar100_C_acc, marker='s', linestyle='--', color='tab:orange')
ax[1].set_ylim(37.0, 39.0)
ax[1].set_ylabel("CIFAR-100-C Acc (%)")
ax[1].set_xlabel("k")
ax[1].grid(True)
ax[1].set_xticks(k)

plt.suptitle("Accuracy vs k (Zoomed per dataset)")
plt.savefig("k_plot_sub.png")
