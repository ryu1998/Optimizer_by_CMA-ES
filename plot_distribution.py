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

    x_min, x_max = -2, 2
    y_min, y_max = -0.5, 0.5

    # 最適化前後のシフトデータを座標として読み込む
    data1 = shift_to_coord(before) * 1e9 * 1e-3
    data2 = shift_to_coord(after) * 1e9 * 1e-3

    r = 50 * 1e-3
    circle_list = []

    # 最適化前の構造をリストに登録
    for m in range(20):
        if data1[2*m] < x_max*1.1 and data1[2*m] > x_min*1.1 and data1[2*m+1] < x_max*1.1 and data1[2*m+1] > x_min*1.1:
            circle_list.append(patches.Circle(xy = (data1[2*m], data1[2*m+1]), radius = r, color = "black", lw = 1, linestyle = "--", fill = False))

    # 最適化後の構造をリストに登録
    for m in range(20):
        if data2[2*m] < x_max*1.1 and data2[2*m] > x_min*1.1 and data2[2*m+1] < y_max*1.1 and data2[2*m+1] > y_min*1.1:
            if abs(data2[2*m]-data1[2*m]) >= 1 * 1e-3 or abs(data2[2*m+1]-data1[2*m+1]) >= 1 * 1e-3:
                circle_list.append(patches.Circle(xy = (data2[2*m], data2[2*m+1]), radius = r, color = "red", lw = 1, fill = False))
            else:
                circle_list.append(patches.Circle(xy = (data2[2*m], data2[2*m+1]), radius = r, color = "black", lw = 1, fill = False))

    # 円孔をグラフに描画
    [plt.gca().add_patch(c) for c in circle_list]

    plt.gca().add_patch(patches.Rectangle(xy = (-2,-0.1), width = 4, height = 0.2, color = "black", lw = 2, fill = False))

    # 矢印の追加
    for m in range(20):
        if data2[2*m] < x_max*1.1 and data2[2*m] > x_min*1.1 and data2[2*m+1] < x_max*1.1 and data2[2*m+1] > x_min*1.1:
            if abs(data2[2*m]-data1[2*m]) >= 1 * 1e-3 or abs(data2[2*m+1]-data1[2*m+1]) >= 1 * 1e-3:
                plt.quiver(data1[2*m], data1[2*m+1], (data2[2*m]-data1[2*m])*10, (data2[2*m+1]-data1[2*m+1])*10, angles="xy", scale_units="xy", scale = 1, zorder=2, linewidth = 2)

    # 描画範囲・体裁を指定
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("$\it{x}$ " + u"[\u03bcm]")
    plt.ylabel("$\it{y}$ " + u"[\u03bcm]")
    plt.gca().set_xticks(np.linspace(x_min, x_max, 9))
    plt.gca().set_xticks(np.linspace(x_min, x_max, 21), minor = True)
    plt.gca().set_yticks(np.linspace(y_min, y_max, 3))
    plt.gca().set_yticks(np.linspace(y_min, y_max, 11), minor = True)
    plt.gca().set_aspect('equal')

    # 画像を保存
    plt.savefig("hole.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.savefig("hole.png", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.show()


def plot_contour(path1, path2, shift):
    """
    Lumerical FDTDでエクスポートした電界強度分布.txtと円孔のシフトを元に，Splineで補完した電界強度分布を描画します．

    Parameters
    ----------
    Ex : エクスポートしたEx.txtのパス
    Ey : エクスポートしたEy.txtのパス
    shift : 描画する円孔のシフト
    """

    # データの次元数を取得（三桁を想定）
    with open(path1, "r") as f:
        t = f.read()
        x_dim = int(t[t.find("x(m)") + 5 : t.find("x(m)") + 8])
        y_dim = int(t[t.find("y(m)") + 5 : t.find("y(m)") + 7])

    # Lumerical FDTDからエクスポートした.txtを読み込む
    x = pd.read_csv(path1, header = None, delimiter = " ", skiprows = 3, nrows = x_dim).iloc[:, 1:]
    y = pd.read_csv(path1, header = None, delimiter = " ", skiprows = (x_dim + 5), nrows = y_dim).iloc[:, 1:]
    Ex = pd.read_csv(path1, header = None, delimiter = " ", skiprows = (x_dim + y_dim + 7)).iloc[:, 1:]
    Ey = pd.read_csv(path2, header = None, delimiter = " ", skiprows = (x_dim + y_dim + 7)).iloc[:, 1:]

    # DataframeをNdarrayに変換
    x = np.array(x).flatten() * 1e9 * 1e-3
    y = np.array(y).flatten()  * 1e9 * 1e-3
    Ex = np.array(Ex)
    Ey = np.array(Ey)
    z = np.sqrt(Ex**2 + Ey**2)

    print("x:{}, y:{}, z:{}".format(x.shape, y.shape, z.shape))

    # zを0 ~ 1で正規化
    z = (z - np.min(z)) / (np.max(z) - np.min(z))
    pb = plt.pcolormesh(x, y, z.T, cmap = "jet", shading = "auto", rasterized = True, alpha = 1.0)

    # グラフの範囲を指定
    #min, max = -1000, 1000
    x_min, x_max = -2, 2
    y_min, y_max = -0.5, 0.5
    # シフトデータを座標データに変換
    data = shift_to_coord(shift) * 1e9 * 1e-3

    # 円孔を描画
    circle_list = []
    for m in range(20):
        if data[2*m] < x_max*1.1 and data[2*m] > x_min*1.1 and data[2*m+1] < y_max*1.1 and data[2*m+1] > y_min*1.1:
            circle_list.append(patches.Circle(xy = (data[2*m], data[2*m+1]), radius = 50 * 1e-3, color = "white", lw = 1, fill = False))
    [plt.gca().add_patch(c) for c in circle_list]

    plt.gca().add_patch(patches.Rectangle(xy = (-2,-0.1), width = 4, height = 0.2, color = "white", lw = 1, fill = False))

    # 描画範囲と体裁を指定
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("$\it{x}$ " + u"[\u03bcm]")
    plt.ylabel("$\it{y}$ " + u"[\u03bcm]")
    plt.gca().set_xticks(np.linspace(x_min, x_max, 5))
    plt.gca().set_xticks(np.linspace(x_min, x_max, 9), minor = True)
    plt.gca().set_yticks(np.linspace(y_min, y_max, 3))
    plt.gca().set_yticks(np.linspace(y_min, y_max, 11), minor = True)
    plt.gca().set_aspect('equal')

    # 画像を保存
    plt.savefig(os.path.splitext(path2)[0] + "_DFT.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.savefig(os.path.splitext(path2)[0] + "_DFT.png", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.show()

    # カラーバーを保存
    fig, ax = plt.subplots()
    bar = plt.colorbar(pb, ax = ax, ticks = [0.0,0.5,1.0], shrink = 0.8)
    bar.set_label("Normalized intensity")
    ax.remove()
    plt.savefig(os.path.splitext(path2)[0] + "_DFT_colorbar.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.close()


def plot_vector(path1, path2, shift, slot):
    """
    Lumerical FDTDでエクスポートした電界強度分布.txtと円孔のシフトを元に，電界強度分布をベクトルプロットで描画します．

    Parameters
    ----------
    path1 : エクスポートしたEx.txtのパス
    path2 : エクスポートしたEy.txtのパス
    shift : 描画する円孔のシフト
    """

    # データの次元数を取得（三桁を想定）
    with open(path1, "r") as f:
        t = f.read()
        x_dim = int(t[t.find("x(m)") + 5 : t.find("x(m)") + 8])
        y_dim = int(t[t.find("y(m)") + 5 : t.find("y(m)") + 7])

    # Lumerical FDTDからエクスポートしたデータを読み込む
    x = pd.read_csv(path1, header = None, delimiter = " ", skiprows = 3, nrows = x_dim).iloc[:, 1:]
    y = pd.read_csv(path1, header = None, delimiter = " ", skiprows = (x_dim + 5), nrows = y_dim).iloc[:, 1:]
    Ex = pd.read_csv(path1, header = None, delimiter = " ", skiprows = (x_dim + y_dim + 7)).iloc[:, 1:]
    Ey = pd.read_csv(path2, header = None, delimiter = " ", skiprows = (x_dim + y_dim + 7)).iloc[:, 1:]

    # ndarrayに変換 (強度だけ転置の必要あり)
    x = np.array(x).flatten() * 1e9 * 1e-3
    y = np.array(y).flatten()  * 1e9 * 1e-3
    Ex = np.array(Ex)
    Ey = np.array(Ey)
    z= np.sqrt(Ex**2 + Ey**2)

    print("x:{}, y:{}, z:{}".format(x.shape, y.shape, z.shape))

    # Zを0 ~ 1で正規化
    z = (z - np.min(z)) / (np.max(z) - np.min(z))

    # 電界強度分布を描画
    pb = plt.pcolormesh(x, y, z.T, cmap = "jet", shading = "nearest", rasterized = True)

    # グラフの範囲を指定
    min, max = -1, 1

    # シフトデータを座標データに変換
    data = shift_to_coord(shift) * 1e9 * 1e-3

    # 円孔をグラフに描画
    circle_list = []
    for m in range(24+12+4*12*12):
        if data[2*m] < max*1.1 and data[2*m] > min*1.1 and data[2*m+1] < max*1.1 and data[2*m+1] > min*1.1:
            circle_list.append(patches.Circle(xy = (data[2*m], data[2*m+1]), radius = 130 * 1e-3, color = "black", lw = 1, fill = False))
    [plt.gca().add_patch(c) for c in circle_list]

    # ベクトルを描画
    arrow_list = []
    for i in range(len(x)):
        for j in range(len(y)):
            if i > 250 and i < 350 and j > 150 and j < 350 and i % 5 == 0 and j % 5 == 0:
                print("quiver {}-{}".format(i,j))
                plt.quiver(x[i], y[j], (Ex[i,j]*3e9), (Ey[i,j]*3e9), angles="xy", scale_units="xy", color = "white", lw = 0.5, scale = 1.0, zorder=2)

    # 描画範囲と体裁を指定
    plt.xlim(min, max)
    plt.ylim(min, max)
    plt.xlabel("$\it{x}$ " + u"[\u03bcm]")
    plt.ylabel("$\it{y}$ " + u"[\u03bcm]")
    plt.gca().set_xticks(np.linspace(min, max, 5))
    plt.gca().set_xticks(np.linspace(min, max, 21), minor = True)
    plt.gca().set_yticks(np.linspace(min, max, 5))
    plt.gca().set_yticks(np.linspace(min, max, 21), minor = True)
    plt.gca().set_aspect('equal')

    # 画像を保存
    plt.savefig(os.path.splitext(path2)[0] + "_vector.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.savefig(os.path.splitext(path2)[0] + "_vector.png", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.show()

    # カラーバーを保存
    fig, ax = plt.subplots()
    bar = plt.colorbar(pb, ax = ax, ticks = [0,0.5,1], shrink = 0.8)
    bar.set_label("Intensity")
    ax.remove()
    plt.savefig(os.path.splitext(path2)[0] + "_vector_colorbar.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.close()


def plot_fourier(Ex, Ey):
    """
    Lumerical FDTDでエクスポートした電界強度分布.txtをFFT・IFFTし，周波数成分を描画します．

    Parameters
    ----------
    Ex : エクスポートしたEx.txtのパス
    Ey : エクスポートしたEy.txtのパス
    """

    # LumericalにおけるFDTD計算の条件
    field_size_x = 15000 * 1e-9         # x方向のモニターサイズ
    field_size_y = 13000 * 1e-9         # y方向のモニターサイズ
    lattice_constant = 500 * 1e-9       # 格子定数
    NA = 0.55                           # 測定系の開口数
    dk = 2*np.pi / lattice_constant     # 単位波数？

    #if slot == True:
       # n_clad = 1.321                      # 媒質の屈折率（水中）
   #else:
    n_clad = 1.000                      # 媒質の屈折率（空気中）

    # データの次元数と波長を取得（次元数は三桁，波長は四桁を想定）
    with open(Ex, "r") as f:
        t = f.read()
        x_dim = int(t[t.find("x(m)") + 5 : t.find("x(m)") + 8])
        y_dim = int(t[t.find("y(m)") + 5 : t.find("y(m)") + 8])
        wavelength = int(t[t.find("lambda=") + 7 : t.find("lambda=") + 11]) * 1e-6

    # データを読み込む
    x = pd.read_csv(Ex, header = None, delimiter = " ", skiprows = 3, nrows = x_dim).iloc[:, 1:]
    y = pd.read_csv(Ex, header = None, delimiter = " ", skiprows = (x_dim + 5), nrows = y_dim).iloc[:, 1:]
    Ex = pd.read_csv(Ex, header = None, delimiter = " ", skiprows = (x_dim + y_dim + 7)).iloc[:, 1:]
    Ey = pd.read_csv(Ey, header = None, delimiter = " ", skiprows = (x_dim + y_dim + 7)).iloc[:, 1:]

    # データをndarrayに変換
    x = np.array(x).ravel() * 1e9
    y = np.array(y).ravel() * 1e9
    Ex = np.array(Ex)
    Ey = np.array(Ey)

    # データを二次元FFTで周波数成分に変換
    E_FFT = np.fft.fft2(Ex + Ey)

    # 低周波成分を内側にシフト
    spectrum = np.fft.fftshift(E_FFT)

    # 周波数空間のパラメータ
    x_shape = spectrum.shape[0]
    y_shape = spectrum.shape[1]
    center_x = int(x_shape / 2)
    center_y = int(y_shape / 2)
    cell_size_x = field_size_x / x_shape
    cell_size_y = field_size_y / y_shape
    max_wavenum = 2*np.pi / cell_size_x
    resolution = max_wavenum / x_shape

    # ライトライン，検出ゾーン，ブリユアンゾーンの半径を計算
    theta = (np.pi / 2) - np.arcsin(NA / n_clad)
    lightline_radius = n_clad * (lattice_constant / wavelength) * dk / resolution
    detection_radius = n_clad * (lattice_constant / wavelength) * np.cos(theta) * dk / resolution
    brillouin_radius = dk / np.sqrt(3) / resolution

    # ライトライン内の成分を計算
    lightline_power = np.zeros((x_shape, y_shape),  dtype = np.complex)
    for i in range(x_shape):
        for j in range(y_shape):
            r = np.sqrt(np.square(i - center_x) + np.square(j - center_y))
            if r < lightline_radius:
                lightline_power[i,j] = spectrum[i,j]

    lightline_power_total = np.sum(np.abs(lightline_power))
    print("lightline_power_total:", lightline_power_total)

    # 検出ゾーン内の成分を計算
    detection_power = np.zeros((x_shape, y_shape), dtype = np.complex)
    for i in range(x_shape):
        for j in range(y_shape):
            r = np.sqrt(np.square(i-center_x) + np.square(j - center_y))
            if r < detection_radius:
                detection_power[i,j] = spectrum[i,j]
    detection_power_total = np.sum(np.abs(detection_power))
    print("detection_power_total:", detection_power_total)

    # 検出効率を計算
    detection_efficency = detection_power_total / lightline_power_total * 100
    print("detection_efficency:", detection_efficency)

    # 周波数成分の絶対値のlogをとって-1 ~ 1に正規化
    z = np.log10(np.abs(spectrum))
    z = (z - np.min(z)) / (np.max(z) - np.min(z)) * 2 - 1

    # 周波数成分を描画
    lv = np.linspace(-1, 1, 100)
    pb = plt.contourf(z.T, levels = lv, cmap = "YlOrRd_r")

    # ライトライン，検出ゾーン，ブリユアンゾーンを描画
    lightline = plt.gca().add_patch(patches.Circle(xy = (center_x, center_y), radius = lightline_radius, fc = "None", ec = "w", linewidth = 1))
    detection_line = plt.gca().add_patch(patches.Circle(xy = (center_x, center_y), radius = detection_radius, fc = "None", ec = "w", linewidth = 1))
    brillouin_zone = plt.gca().add_patch(patches.RegularPolygon(xy = (center_x, center_y), numVertices = 6, radius = brillouin_radius, orientation = np.pi / 2, fc = "None", ec = "w", linewidth = 1))

    # 描画範囲・体裁を指定
    plt.xlim(center_x - 20, center_x + 20)
    plt.ylim(center_y - 20, center_y + 20)
    plt.xlabel("$\it{kx}$")
    plt.ylabel("$\it{ky}$")
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    plt.gca().set_aspect('equal')

    # 画像を保存
    plt.savefig(os.path.splitext(Ey)[0] + "_FFT.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.savefig(os.path.splitext(Ey)[0] + "_FFT.png", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.show()

    # カラーバーを保存
    fig, ax = plt.subplots()
    bar = plt.colorbar(pb, ax = ax, ticks = [-1,0,1], shrink = 0.8)
    bar.set_label("Intensity [a.u.]")
    ax.remove()
    plt.savefig(os.path.splitext(Ey)[0] + "_FFT_colorbar.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.close()


def plot_leaky(path1, path2, shift, max_intensity):
    """
    Lumerical FDTDで出力した電界強度分布のデータをFFT・IFFTし，漏れ成分を描画します．

    Parameters
    ----------
    Ex : エクスポートしたEx.txtのパス
    Ey : エクスポートしたEy.txtのパス
    shift : 円孔のシフト
    max_intensity : 描画の基準とする電界強度分布の最大値
    """

    # LumericalにおけるFDTD計算の条件
    field_size_x = 15000 * 1e-9         # x方向のモニターサイズ
    field_size_y = 13000 * 1e-9         # y方向のモニターサイズ
    lattice_constant = 500 * 1e-9       # 格子定数
    NA = 0.55                           # 測定系の開口数
    dk = 2*np.pi / lattice_constant     # 単位波数？
    n_clad = 1.000                  # 媒質の屈折率（空気中）

    # データの次元数と波長を取得（次元数は三桁，波長は四桁を想定）
    with open(path1, "r") as f:
        t = f.read()
        x_dim = int(t[t.find("x(m)") + 5 : t.find("x(m)") + 8])
        y_dim = int(t[t.find("y(m)") + 5 : t.find("y(m)") + 7])
        try:
            wavelength = float(t[t.find("lambda=") + 7 : t.find("lambda=") + 12]) * 1e-6
        except ValueError:
            try:
                wavelength = float(t[t.find("lambda=") + 7 : t.find("lambda=") + 11]) * 1e-6
            except ValueError:
                    wavelength = float(t[t.find("lambda=") + 7 : t.find("lambda=") + 10]) * 1e-6

    # データを読み込む
    x = pd.read_csv(path1, header = None, delimiter = " ", skiprows = 3, nrows = x_dim).iloc[:, 1:]
    y = pd.read_csv(path1, header = None, delimiter = " ", skiprows = (x_dim + 5), nrows = y_dim).iloc[:, 1:]
    Ex = pd.read_csv(path1, header = None, delimiter = " ", skiprows = (x_dim + y_dim + 7)).iloc[:, 1:]
    Ey = pd.read_csv(path2, header = None, delimiter = " ", skiprows = (x_dim + y_dim + 7)).iloc[:, 1:]

    # データをndarrayに変換
    x = np.array(x).ravel() * 1e9
    y = np.array(y).ravel() * 1e9
    Ex = np.array(Ex)
    Ey = np.array(Ey)

    # データを二次元FFTで周波数成分に変換
    Ex_FFT = np.fft.fft2(Ex)
    Ey_FFT = np.fft.fft2(Ey)

    # 低周波成分を内側にシフト
    spectrum_Ex = np.fft.fftshift(Ex_FFT)
    spectrum_Ey = np.fft.fftshift(Ey_FFT)

    # 周波数空間のパラメータ
    x_shape = spectrum_Ex.shape[0]
    y_shape = spectrum_Ex.shape[1]
    center_x = int(x_shape / 2)
    center_y = int(y_shape / 2)
    cell_size_x = field_size_x / x_shape
    cell_size_y = field_size_y / y_shape
    max_wavenum = 2*np.pi / cell_size_x
    resolution = max_wavenum / x_shape

    # ライトラインの半径を計算
    theta = (np.pi / 2) - np.arcsin(NA / n_clad)
    lightline_radius = n_clad * (lattice_constant / wavelength) * dk / resolution

    # ライトライン内の成分を計算
    lightline_power_Ex = np.zeros((x_shape, y_shape),  dtype = np.complex)
    lightline_power_Ey = np.zeros((x_shape, y_shape),  dtype = np.complex)
    for i in range(x_shape):
        for j in range(y_shape):
            r = np.sqrt(np.square(i - center_x) + np.square(j - center_y))
            if r < lightline_radius:
                lightline_power_Ex[i,j] = spectrum_Ex[i,j]
                lightline_power_Ey[i,j] = spectrum_Ey[i,j]

    # ライトライン内成分の低周波成分を外側にシフト
    spectrum_Ex = np.fft.ifftshift(lightline_power_Ex)
    spectrum_Ey = np.fft.ifftshift(lightline_power_Ey)

    # ライトライン内の周波数成分をIFFT
    Ex_IFFT = np.fft.ifft2(spectrum_Ex)
    Ey_IFFT = np.fft.ifft2(spectrum_Ey)

    # 漏れ成分の絶対値をとる
    z = np.abs(Ex_IFFT) ** 2 + np.abs(Ey_IFFT) ** 2

    # 漏れ成分を正規化
    print("z_max before = {}".format(np.max(z)))
    z = 10 * np.log10(z)
    z = z - 10 * np.log10(max_intensity)
    print("z_max after = {}".format(np.max(z)))

    # カラーバーをの色を生成
    cmap = pl.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    # 漏れ成分を描画
    lv = np.linspace(-30, 0, 100)
    pb = plt.contourf(x * 1e-3, y * 1e-3, z.T, levels = lv, cmap = my_cmap)

    # 描画範囲・体裁を指定
    x_min, x_max = -2, 2
    y_min, y_max = -0.5, 0.5
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("$\it{x}$ " + u"[\u03bcm]")
    plt.ylabel("$\it{y}$ " + u"[\u03bcm]")
    plt.gca().set_xticks(np.linspace(x_min, x_max, 5))
    plt.gca().set_xticks(np.linspace(x_min, x_max, 9), minor = True)
    plt.gca().set_yticks(np.linspace(y_min, y_max, 3))
    plt.gca().set_yticks(np.linspace(y_min, y_max, 11), minor = True)
    plt.gca().set_aspect('equal')

    # シフトデータを座標に変換
    data1 = shift_to_coord(shift) * 1e9 * 1e-3

    # 円孔を描画
    circle_list = []
    for m in range(20):
        #if data1[2*m] < max*1.1 and data1[2*m] > min*1.1 and data1[2*m+1] < max*1.1 and data1[2*m+1] > min*1.1:
        circle_list.append(patches.Circle(xy = (data1[2*m], data1[2*m+1]), radius = 50 * 1e-3, color = "black", lw = 1, fill = False))
    [plt.gca().add_patch(c) for c in circle_list]

    plt.gca().add_patch(patches.Rectangle(xy = (-2,-0.1), width = 4, height = 0.2, color = "black", lw = 1, fill = False))

    # 画像を保存
    plt.savefig(os.path.splitext(path2)[0] + "_leaky.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.savefig(os.path.splitext(path2)[0] + "_leaky.png", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.show()

    # カラーバーを保存
    fig, ax = plt.subplots()
    bar = plt.colorbar(pb, ax = ax, ticks = [-30, -20, -10, 0], shrink = 0.8)
    bar.set_label("Intensity [dB]")
    ax.remove()
    plt.savefig(os.path.splitext(path2)[0] + "_leaky_colorbar.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.close()


if __name__ == "__main__":

    rcparams.report(5,2.5)

    x0 = np.array([50,0,0,0,0,0,0,0,0,0])
    plot_contour("DFT_data\\Ex_gen0.txt", "DFT_data\\Ey_gen0.txt", x0)
    plot_leaky("DFT_data\\Ex_gen0.txt", "DFT_data\\Ey_gen0.txt", x0, 9.171e-30)

    x = np.loadtxt("data\\pop_200.csv", delimiter = ",")
    x = x[np.argsort(x[:,-2])[::-1], :][0,:-3]
    plot_contour("DFT_data\\Ex_gen200.txt", "DFT_data\\Ey_gen200.txt", x)
    plot_leaky("DFT_data\\Ex_gen200.txt", "DFT_data\\Ey_gen200.txt", x, 9.171e-30)
    """
    x0 = np.array([0,0,0,0,0,0,0,0,0,0])
    plot_contour("DFT_data\\Ex_406.txt", "DFT_data\\Ey_406.txt", x0)
    #plot_leaky("DFT_data\\Ex_406.txt", "DFT_data\\Ey_406.txt", x0, 9.6e-37)

    x = np.loadtxt("2022-01-06 H0 QV\\pop_49.csv", delimiter = ",")
    x = x[np.argsort(x[:,-2] / x[:,-1])[::-1], :][0,:-3]
    plot_contour("DFT_data\\Ex_35200.txt", "DFT_data\\Ey_35200.txt", x)
    #plot_leaky("DFT_data\\Ex_35200.txt", "DFT_data\\Ey_35200.txt", x, 9.6e-37)
    """
