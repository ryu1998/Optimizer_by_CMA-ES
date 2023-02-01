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



def shift_to_coord(shift):
    """
    円孔のシフトを座標に変換します．

    Parameters
    ----------
    shift : 円孔のシフト

    Returns
    ----------
    coord : 円孔の座標
    """

    if shift.shape[0] != 10:
        raise ValueError("Input is not n=10.")

    coord = np.zeros((21,2))
    a = 200

    for i in range(21):
        coord[i,0] = (i - 10) * a
        coord[i,1] = 0

    coord = np.delete(coord, 10, axis = 0)

    coord[0,0] -= shift[9]
    coord[1,0] -= shift[8]
    coord[2,0] -= shift[7]
    coord[3,0] -= shift[6]
    coord[4,0] -= shift[5]
    coord[5,0] -= shift[4]
    coord[6,0] -= shift[3]
    coord[7,0] -= shift[2]
    coord[8,0] -= shift[1]
    coord[9,0] -= shift[0]
    coord[10,0] += shift[0]
    coord[11,0] += shift[1]
    coord[12,0] += shift[2]
    coord[13,0] += shift[3]
    coord[14,0] += shift[4]
    coord[15,0] += shift[5]
    coord[16,0] += shift[6]
    coord[17,0] += shift[7]
    coord[18,0] += shift[8]
    coord[19,0] += shift[9]

    coord = np.ravel(coord) * 1e-9

    return coord


def plot_holes(before, after):
    """
    円孔のシフトデータをmatplotlibで描画します．

    Parameters
    ----------
    before : 最適化前の円孔のシフトデータ
    after : 最適化後の円孔のシフトデータ
    """

    x_min, x_max = -2.5, 2.5
    y_min, y_max = -0.2, 0.2

    # 最適化前後のシフトデータを座標として読み込む
    data1 = shift_to_coord(before) * 1e9 * 1e-3
    data2 = shift_to_coord(after) * 1e9 * 1e-3

    r = 50 * 1e-3
    circle_list = []

    # 最適化前の構造をリストに登録
    for m in range(20):
        #if data1[2*m] < x_max*1.1 and data1[2*m] > x_min*1.1 and data1[2*m+1] < x_max*1.1 and data1[2*m+1] > x_min*1.1:
        circle_list.append(patches.Circle(xy = (data1[2*m], data1[2*m+1]), radius = r, color = "black", lw = 2, linestyle = (0, (1.5, 1)), fill = False))

    # 最適化後の構造をリストに登録
    for m in range(20):
        #if data2[2*m] < x_max*1.1 and data2[2*m] > x_min*1.1 and data2[2*m+1] < y_max*1.1 and data2[2*m+1] > y_min*1.1:
        if abs(data2[2*m]-data1[2*m]) >= 1 * 1e-3 or abs(data2[2*m+1]-data1[2*m+1]) >= 1 * 1e-3:
            circle_list.append(patches.Circle(xy = (data2[2*m], data2[2*m+1]), radius = r, color = "red", lw = 2, fill = False))
        else:
            circle_list.append(patches.Circle(xy = (data2[2*m], data2[2*m+1]), radius = r, color = "black", lw = 2, fill = False))

    # 円孔をグラフに描画
    [plt.gca().add_patch(c) for c in circle_list]

    plt.gca().add_patch(patches.Rectangle(xy = (-2.5,-0.1), width = 5, height = 0.2, color = "black", lw = 2, fill = False))

    # 矢印の追加
    for m in range(20):
        #if data2[2*m] < x_max*1.1 and data2[2*m] > x_min*1.1 and data2[2*m+1] < x_max*1.1 and data2[2*m+1] > x_min*1.1:
        if abs(data2[2*m]-data1[2*m]) >= 1 * 1e-3 or abs(data2[2*m+1]-data1[2*m+1]) >= 1 * 1e-3:
            plt.quiver(data1[2*m], data1[2*m+1], (data2[2*m]-data1[2*m])*2, (data2[2*m+1]-data1[2*m+1])*2, angles="xy", scale_units="xy", scale = 1, zorder=2, headwidth = 3, headlength = 3, headaxislength = 3)

    # 描画範囲・体裁を指定
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("$\it{x}$ " + u"[\u03bcm]")
    plt.ylabel("$\it{y}$ " + u"[\u03bcm]")
    #plt.gca().set_xticks(np.linspace(x_min, x_max, 5))
    plt.xticks([-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0])
    plt.gca().set_xticks(np.linspace(x_min, x_max, 26), minor = True)
    plt.gca().set_yticks(np.linspace(y_min, y_max, 3))
    plt.gca().set_yticks(np.linspace(y_min, y_max, 5), minor = True)
    plt.gca().set_aspect('equal')

    # 画像を保存
    plt.savefig("hole.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.savefig("hole.png", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.show()


if __name__ == "__main__":

    rcparams.report(7,1)

    x0 = np.array([0,0,0,0,0,0,0,0,0,0])
    #plot_contour(path1 = "DFT_data\\Ex_300.txt", path2 = "DFT_data\\Ey_300.txt", shift = x0)

    x = np.loadtxt("data\\pop_0.csv", delimiter = ",")
    x = x[np.argsort(x[:,-2])[::-1], :][0,:-3]

    #x = np.loadtxt("2021-12-31 H1 QV\\pop_111.csv", delimiter = ",")
    #x = x[np.argsort(x[:,-2])[::-1], :]
    #print(x[0,-3:])
    #x = x[0,:-3]

    #x = np.loadtxt("2022-01-06 H0 QV\\pop_49.csv", delimiter = ",")
    #x = x[np.argsort(x[:,-2] / x[:,-1])[::-1], :][0,:-3]

    plot_holes(x0, x)
