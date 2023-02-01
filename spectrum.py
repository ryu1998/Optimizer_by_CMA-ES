import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.patches as patches
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from scipy.interpolate import RectBivariateSpline
import rcparams


def plot_contour(Ex, Ey, x_dim, y_dim):
    """
    円孔のシフトデータとLumerical FDTDで出力した電界データを元に，電界強度分布を描画します．

    Parameters
    ----------
    Ex : Ex電界・磁界強度分布を記録したテキストデータの名前
    Ey : Ey電界・磁界強度分布を記録したテキストデータの名前
    x_dim : FDTDのx方向メッシュ数
    y_dim : FDTDのy方向メッシュ数
    """

    # Lumerical FDTDからエクスポートしたデータを読み込む（先頭のNanは取り除く）
    X = pd.read_csv(Ex, header = None, delimiter = " ", skiprows = 3, nrows = x_dim, na_filter = False).iloc[:, 1:]
    Y = pd.read_csv(Ey, header = None, delimiter = " ", skiprows = (x_dim + 5), nrows = y_dim, na_filter = False).iloc[:, 1:]
    Ex = pd.read_csv(Ex, header = None, delimiter = " ", skiprows = (x_dim + y_dim + 7), na_filter = False).iloc[:, 1:]
    Ey = pd.read_csv(Ey, header = None, delimiter = " ", skiprows = (x_dim + y_dim + 7), na_filter = False).iloc[:, 1:]

    # ndarrayに変換
    X = np.array(X).flatten() * 1e9
    Y = np.array(Y).flatten()  * 1e9
    Z = np.abs(np.array(Ex)) ** 2 + np.abs(np.array(Ey)) ** 2

    print("x:{}, y:{}, z:{}".format(X.shape, Y.shape, Z.shape))

    # xy空間をspline関数により補完
    print("強度データを補完中・・")
    f = RectBivariateSpline(Y, X, Z.T)

    # xy空間を更に細分化
    x = np.linspace(X.min(), X.max(), 1000)
    y = np.linspace(Y.min(), Y.max(), 1000)
    z = f(x, y)

    # zを0 ~ 1で正規化
    z = (z - np.min(z)) / (np.max(z) - np.min(z))

    # 電界強度分布を描画
    lv = np.linspace(0, 1, 100)
    pb = plt.contourf(x, y, z, levels = lv, cmap = "jet")

    # 描画範囲と体裁を指定
    plt.xlim(-750, 750)
    plt.ylim(-5000, 5000)
    plt.xlabel("$\it{x}$ [nm]")
    plt.ylabel("$\it{y}$ [nm]")
    plt.gca().set_xticks(np.linspace(-750, 750, 3))
    plt.gca().set_xticks(np.linspace(-750, 750, 21), minor = True)
    plt.gca().set_yticks(np.linspace(-5000, 5000, 5))
    plt.gca().set_yticks(np.linspace(-5000, 5000, 21), minor = True)
    plt.gca().set_aspect('equal')

    # 電界強度分布を画像に保存
    plt.savefig(os.path.splitext(os.path.basename(Ey))[0] + "_DFT.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.savefig(os.path.splitext(os.path.basename(Ey))[0] + "_DFT.png", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.show()

    fig, ax = plt.subplots()
    bar = plt.colorbar(pb, ax = ax, ticks = [0.0,0.5,1.0], shrink = 0.8)
    bar.set_label("Normalized intensity")
    ax.remove()
    plt.savefig(os.path.splitext(Ey)[0] + "_DFT_colorbar.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.close()


if __name__ == "__main__":

    #rcparams.report(3,3)

    plot_contour("Ex.txt", "Ey.txt", 21, 517)
