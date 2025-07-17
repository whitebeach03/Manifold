import numpy as np
import matplotlib.pyplot as plt

# 1. Toy manifold: 半円 (θ ∈ [0, π])
theta = np.linspace(0, np.pi, 200)
manifold = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # shape (200, 2)

# 2. 代表点の選択
theta1 = np.pi * 0.2
theta2 = np.pi * 0.8
p1 = np.array([np.cos(theta1), np.sin(theta1)])
p2 = np.array([np.cos(theta2), np.sin(theta2)])

# 3. 直線補間
t = np.linspace(0, 1, 100)
line = np.outer(1 - t, p1) + np.outer(t, p2)  # shape (100, 2)

# 4. プロット
plt.figure()
plt.plot(manifold[:, 0], manifold[:, 1], label='Manifold (半円)')
plt.plot(line[:, 0], line[:, 1], label='Linear Interpolation')
plt.scatter(p1[0], p1[1], marker='o', label='Point A')
plt.scatter(p2[0], p2[1], marker='o', label='Point B')
mid = (p1 + p2) / 2
plt.scatter(mid[0], mid[1], marker='x', label='Midpoint')

plt.title('曲がったマニフォールド vs 直線補間')
plt.axis('equal')
plt.legend()
plt.savefig("manifold.png")
