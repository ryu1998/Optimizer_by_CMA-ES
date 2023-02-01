import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import rcparams


def plot_iterations(path1, path2):
    """
    CMAESによる探索過程を，世代を横軸にプロットします．

    Parameters
    ----------
    path1 : pop_*.csvが格納されているフォルダのパス
    """

    # 一番目の世代数・個体数を指定
    NGEN = len(glob.glob(path1 + "\\pop_*.csv"))
    LAMBDA = 20

    # 世代ごとに読み込み最大値を格納
    best = np.zeros(NGEN)
    for i in range(NGEN):
        best[i] = np.max(np.loadtxt(path1 + "\\pop_{}.csv".format(i), delimiter = ",")[:,-2])

    # 世代ごとの最大値をプロット
    plt.plot(np.arange(0, NGEN), best, color = "C0")
    plt.plot((NGEN-1), best[-1], marker = "o", color = "C0")

    # 二番目の世代数・個体数を指定
    NGEN = len(glob.glob(path2 + "\\pop_*.csv"))
    LAMBDA = 20

    # 世代ごとの最大値をプロット
    best = np.zeros(NGEN)
    for i in range(NGEN):
        best[i] = np.max(np.loadtxt(path2 + "\\pop_{}.csv".format(i), delimiter = ",")[:,-2])

    # 世代ごとの最大値をプロット
    plt.plot(np.arange(0, NGEN), best, color = "C1")
    plt.plot((NGEN-1), best[-1], marker = "o", color = "C1")

    # 描画範囲・体裁を指定
    plt.xlim(0, 200)
    plt.ylim(0, 6000)
    plt.xlabel("Generations")
    plt.ylabel("$\it{Q}$")
    plt.gca().xaxis.set_major_locator(tick.MultipleLocator(20))
    plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(5))
    plt.gca().yaxis.set_major_locator(tick.MultipleLocator(1000))
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(200))

    # 画像を保存
    plt.savefig("Iterations.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.savefig("Iterations.png", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.show()


def plot_iterations_QV(path1, path2):
    """
    CMAESによる探索過程を，計算時間を横軸にプロットします．また，探索過程のσも別の軸でプロットします．

    Parameters
    ----------
    path1 : pop_*.csvが格納されている一番目のフォルダのパス
    path2 : pop_*.csvが格納されている二番目のフォルダのパス
    """

    # 一番目の世代数・個体数を指定
    NGEN = len(glob.glob(path1 + "\\pop_*.csv"))
    LAMBDA = 20

    # 世代ごとに読み込み最大値を格納
    Q = np.zeros(NGEN)
    V = np.zeros(NGEN)
    for i in range(NGEN):
        data = np.loadtxt(path1 + "\\pop_{}.csv".format(i), delimiter = ",")
        best_index = np.argsort(data[:,-2] / data[:,-1])[::-1][0]
        Q[i] = data[best_index,-2]
        V[i] = data[best_index,-1]

    # 世代ごとのQをプロット
    plt.scatter(Q,V, color = "C0", marker = "o")

    # 2番目の世代数・個体数を指定
    NGEN = len(glob.glob(path2 + "\\pop_*.csv"))
    LAMBDA = 20

    # 世代ごとに読み込み最大値を格納
    Q = np.zeros(NGEN)
    V = np.zeros(NGEN)
    for i in range(NGEN):
        data = np.loadtxt(path2 + "\\pop_{}.csv".format(i), delimiter = ",")
        best_index = np.argsort(data[:,-2] / data[:,-1])[::-1][0]
        Q[i] = data[best_index,-2]
        V[i] = data[best_index,-1]

    # 世代ごとのQをプロット
    plt.scatter(Q, V, color = "C1", marker = "o")

    # 描画範囲・体裁を指定
    plt.xlim(-5000, 25000)
    plt.ylim(-1, 3)
    plt.xlabel("$\it{Q}$")
    plt.ylabel("$\it{V}$")
    plt.gca().xaxis.set_major_locator(tick.MultipleLocator(5000))
    plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(1000))
    plt.gca().yaxis.set_major_locator(tick.MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(0.2))

    # 画像を保存
    plt.savefig("iterations_QV.pdf", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.savefig("iterations_QV.png", transparent = True, bbox_inches = "tight", pad_inches = 0.1)
    plt.show()


if __name__ == "__main__":

    rcparams.report(3,2.5)

    plot_iterations(path1 = "data", path2 = "data")
    #plot_iterations(path1 = "2022-01-06 H0 QV", path2 = "2021-12-31 H1 QV")
