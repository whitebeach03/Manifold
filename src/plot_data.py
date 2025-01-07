import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### 高次元データのプロット ###
def plot_high_dim_comparison(original_data, original_color, generated_data, red, reg, sam, data_type):
    fig = plt.figure(figsize=(12, 6))

    # 元のデータ
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], 
                c=original_color, s=10, cmap=plt.cm.viridis)
    ax1.set_title("Original High-Dimensional Data")

    # 生成されたデータ
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], 
                c='black', s=10, alpha=0.5)
    ax2.set_title("Generated High-Dimensional Data")

    filename = f"result/{data_type}/high_dim/comparison/{red}_{reg}_{sam}.png"
    plt.savefig(filename)
    plt.show()

def plot_high_dim_comparison_with_overlay(original_data, original_color, generated_data, red, reg, sam, data_type):
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
    filename = f"result/{data_type}/high_dim/overlay/{red}_{reg}_{sam}.png"
    plt.savefig(filename)
    plt.show()

def plot_high_dim(original_data, original_color, generated_data, red, reg, sam, data_type):
    fig = plt.figure(figsize=(18, 12))

    # Overlayプロット
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], 
                c=original_color, cmap=plt.cm.viridis, s=10, alpha=0.5, label="Original Data")
    ax1.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], 
                c='black', s=10, alpha=0.7, label="Generated Data")
    ax1.set_title("Overlay of Original and Generated High-Dimensional Data")
    ax1.legend()

    # 元のデータプロット
    ax2 = fig.add_subplot(223, projection='3d')
    ax2.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], 
                c=original_color, cmap=plt.cm.viridis, s=10)
    ax2.set_title("Original High-Dimensional Data")

    # 生成されたデータプロット
    ax3 = fig.add_subplot(224, projection='3d')
    ax3.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], 
                c='black', s=10, alpha=0.5)
    ax3.set_title("Generated High-Dimensional Data")

    plt.tight_layout()
    filename = f"result/{data_type}/high_dim/all/{red}_{reg}_{sam}.png"
    plt.savefig(filename)


### 低次元データのプロット ###
def plot_low_dim(original_low_dim_data, generated_low_dim_data, red, reg, sam, data_type):
    """
    低次元データの比較プロットを表示
    - Overlayと2つのサブプロットを1つの図で同時に表示
    """
    fig = plt.figure(figsize=(18, 12))

    # Overlayプロット
    ax1 = fig.add_subplot(221)
    ax1.scatter(original_low_dim_data[:, 0], original_low_dim_data[:, 1], 
                c='blue', alpha=0.5, label='Original Low-Dim Data')
    ax1.scatter(generated_low_dim_data[:, 0], generated_low_dim_data[:, 1], 
                c='black', alpha=0.5, label='Generated Low-Dim Data')
    ax1.set_title("Overlay of Original and Generated Low-Dimensional Data")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    ax1.legend()

    # 元のデータプロット
    ax2 = fig.add_subplot(223)
    ax2.scatter(original_low_dim_data[:, 0], original_low_dim_data[:, 1], 
                c='blue', alpha=0.5)
    ax2.set_title("Original Low-Dimensional Data")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")

    # 生成されたデータプロット
    ax3 = fig.add_subplot(224)
    ax3.scatter(generated_low_dim_data[:, 0], generated_low_dim_data[:, 1], 
                c='black', alpha=0.5)
    ax3.set_title("Generated Low-Dimensional Data")
    ax3.set_xlabel("Component 1")
    ax3.set_ylabel("Component 2")

    plt.tight_layout()
    filename = f"result/{data_type}/low_dim/{red}_{reg}_{sam}.png"
    plt.savefig(filename)
    plt.show()

def plot_3d_data(data, color, title="3D Data Visualization", cmap="viridis"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        data[:, 0], data[:, 1], data[:, 2],
        c=color, cmap=cmap, s=10
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()



def plot_low_dim_comparison(original_low_dim_data, generated_low_dim_data):
    plt.figure(figsize=(12, 6))

    # 元のデータ
    ax1 = plt.subplot(121)
    ax1.scatter(original_low_dim_data[:, 0], original_low_dim_data[:, 1], 
                c='blue', alpha=0.5)
    ax1.set_title("Original Low-Dimensional Data")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")

    # 生成されたデータ
    ax2 = plt.subplot(122)
    ax2.scatter(generated_low_dim_data[:, 0], generated_low_dim_data[:, 1], 
                c='black', alpha=0.5)
    ax2.set_title("Generated Low-Dimensional Data")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")

    filename = "result/low_dim/comparison.png"
    plt.savefig("result/low_dim/comparison.png")
    plt.show()

def plot_low_dim_comparison_with_overlay(original_low_dim_data, generated_low_dim_data):
    plt.figure(figsize=(8, 6))

    # 元の低次元データ
    plt.scatter(original_low_dim_data[:, 0], original_low_dim_data[:, 1], 
                c='blue', alpha=0.5, label='Original Low-Dim Data')

    # 生成された低次元データ
    plt.scatter(generated_low_dim_data[:, 0], generated_low_dim_data[:, 1], 
                c='black', alpha=0.5, label='Generated Low-Dim Data')

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Low-Dimensional Data Comparison")
    plt.legend()
    plt.savefig("result/low_dim/overlay.png")
    plt.show()



