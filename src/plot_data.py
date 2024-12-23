import matplotlib.pyplot as plt

### 高次元データのプロット ###
def plot_high_dim_comparison(original_data, original_color, generated_data):
    """
    元の高次元データと生成された高次元データの比較をプロット
    """
    fig = plt.figure(figsize=(12, 6))

    # 元のデータ
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], c=original_color, cmap=plt.cm.viridis, s=10)
    ax1.set_title("Original High-Dimensional Data")

    # 生成されたデータ
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], c='black', s=10, alpha=0.5)
    ax2.set_title("Generated High-Dimensional Data")

    plt.savefig("result/high_dim_comparison.png")
    plt.show()

def plot_high_dim_comparison_with_overlay(original_data, original_color, generated_data):
    """
    高次元データを元データと生成データで重ねてプロット
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 元の高次元データをプロット
    ax.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], 
               c=original_color, cmap=plt.cm.viridis, s=10, label="Original Data", alpha=0.5)

    # 生成された高次元データをプロット
    ax.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], 
               c='black', s=10, label="Generated Data", alpha=0.7)

    ax.set_title("Overlay of Original and Generated High-Dimensional Data")
    ax.legend()
    plt.savefig("result/high_dim_overlay_comparison.png")
    plt.show()

### 低次元データのプロット ###
def plot_low_dim_comparison(original_low_dim_data, generated_low_dim_data):
    """
    元の低次元データと生成された低次元データの分布を比較
    """
    plt.figure(figsize=(8, 6))

    # 元の低次元データ
    plt.scatter(original_low_dim_data[:, 0], original_low_dim_data[:, 1], label='Original Low-Dim Data', alpha=0.5)

    # 生成された低次元データ
    plt.scatter(generated_low_dim_data[:, 0], generated_low_dim_data[:, 1], label='Generated Low-Dim Data (KDE)', alpha=0.5, color='black')

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Low-Dimensional Data Comparison")
    plt.legend()
    plt.savefig("result/low_dim_comparison.png")
    plt.show()