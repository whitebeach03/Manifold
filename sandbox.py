augment = "Manifold-Mixup(alpha=0.5)"

alpha = float(augment[len("Manifold-Mixup(alpha="):-1])
print(alpha + 3.0)