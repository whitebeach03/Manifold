import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# x軸のデータ
x = np.linspace(0, 1, 1000)

# プロットするalphaのリスト
alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
# alphas = [0.5, 1, 2, 5, 10]

plt.figure()
for a in alphas:
    y = beta.pdf(x, a, a)  # Beta(α, α)分布のPDF
    plt.plot(x, y, label=f'α={a}')

plt.title('Beta(α, α)')
plt.xlabel('λ')
plt.ylabel('Probability Density Function')
plt.legend()
plt.grid(True)
plt.savefig("beta_distribution.png")