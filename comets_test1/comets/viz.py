# comets/viz.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def _to_np(x):
    return x if isinstance(x, np.ndarray) else (x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x))

# Z1, Z2 are [T, D]; color can be angle or scalar per timepoint
def plotemb2(Z1, Z2, color, titles=("View 0", "View 1"), mode="direct", dims=(0,1,2)):
    Z1 = _to_np(Z1); Z2 = _to_np(Z2); c = _to_np(color)
    if mode == "pca":
        Z1 = PCA(n_components=3).fit_transform(Z1)
        Z2 = PCA(n_components=3).fit_transform(Z2)
    else:
        Z1 = Z1[:, dims]; Z2 = Z2[:, dims]

    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.scatter(Z1[:,0], Z1[:,1], Z1[:,2], c=c, s=6)
    ax1.set_title(titles[0])
    ax2.scatter(Z2[:,0], Z2[:,1], Z2[:,2], c=c, s=6)
    ax2.set_title(titles[1])
    plt.show()
