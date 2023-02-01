import matplotlib.pyplot as plt


def report(x,y):
    """
    matplotlibのデフォルト体裁を論文用に設定します．

    Parameters
    ----------
    x : 画像サイズの幅（インチ）
    y : 画像サイズの高さ（インチ）
    """

    plt.rcParams["ps.fonttype"] = 42                        # 画像で使用するフォントタイプ
    plt.rcParams["pdf.fonttype"] = 42                       # PDFで使用するフォントタイプ
    plt.rcParams["font.family"] ="Times New Roman"          # 使用するフォント
    plt.rcParams["mathtext.fontset"] = "custom"             # 数式用フォントセット
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"         # 数式用フォントセット
    plt.rcParams["font.size"] = 10                          # フォントサイズ
    plt.rcParams["axes.linewidth"] = 1                      # 軸の線幅
    plt.rcParams["xtick.direction"] = "out"                  # x軸の目盛線の向き
    plt.rcParams["ytick.direction"] = "out"                  # y軸の目盛線の向き
    plt.rcParams["xtick.minor.visible"] = True              # x軸の副目盛線の有無
    plt.rcParams["ytick.minor.visible"] = True              # y軸副目盛り線の有無
    plt.rcParams["xtick.major.size"] = 6                    # x軸の目盛線の長さ
    plt.rcParams["ytick.major.size"] = 6                    # y軸の目盛線の長さ
    plt.rcParams["xtick.minor.size"] = 3                    # x軸の副目盛線の長さ
    plt.rcParams["ytick.minor.size"] = 3                    # y軸の副目盛線の長さ
    plt.rcParams["xtick.major.width"] = 1                   # x軸の目盛線の線幅
    plt.rcParams["ytick.major.width"] = 1                   # y軸の目盛線の線幅
    plt.rcParams["xtick.minor.width"] = 1                   # x軸の副目盛線の線幅
    plt.rcParams["ytick.minor.width"] = 1                   # y軸の副目盛線の線幅
    plt.rcParams["figure.figsize"] = x, y                   # 図のサイズ(x, y)


def slide(x,y):
    """
    matplotlibのデフォルト体裁をスライド用に設定します．

    Parameters
    ----------
    x : 画像サイズの幅（インチ）
    y : 画像サイズの高さ（インチ）
    """

    plt.rcParams["ps.fonttype"] = 42                        # 画像で使用するフォントタイプ
    plt.rcParams["pdf.fonttype"] = 42                       # PDFで使用するフォントタイプ
    plt.rcParams["font.family"] ="Arial"                    # 使用するフォント
    plt.rcParams["mathtext.fontset"] = "custom"             # 数式用フォントセット
    plt.rcParams["mathtext.it"] = "Arial:italic"            # 数式用フォントセット
    plt.rcParams["font.size"] = 16                          # フォントサイズ
    plt.rcParams["axes.linewidth"] = 1                      # 軸の線幅
    plt.rcParams["xtick.direction"] = "out"                  # x軸の目盛線の向き
    plt.rcParams["ytick.direction"] = "out"                  # y軸の目盛線の向き
    plt.rcParams["xtick.minor.visible"] = True              # x軸の副目盛線の有無
    plt.rcParams["ytick.minor.visible"] = True              # y軸副目盛り線の有無
    plt.rcParams["xtick.major.size"] = 6                    # x軸の目盛線の長さ
    plt.rcParams["ytick.major.size"] = 6                    # y軸の目盛線の長さ
    plt.rcParams["xtick.minor.size"] = 3                    # x軸の副目盛線の長さ
    plt.rcParams["ytick.minor.size"] = 3                    # y軸の副目盛線の長さ
    plt.rcParams["xtick.major.width"] = 1                   # x軸の目盛線の線幅
    plt.rcParams["ytick.major.width"] = 1                   # y軸の目盛線の線幅
    plt.rcParams["xtick.minor.width"] = 1                  # x軸の副目盛線の線幅
    plt.rcParams["ytick.minor.width"] = 1                   # y軸の副目盛線の線幅
    plt.rcParams["figure.figsize"] = x, y                   # 図のサイズ(x, y)


def conference(x,y):
    """
    matplotlibのデフォルト体裁を学会発表用に設定します．

    Parameters
    ----------
    x : 画像のサイズの幅（インチ）
    y : 画像のサイズの高さ（インチ）
    """

    plt.rcParams["ps.fonttype"] = 42                        # 画像で使用するフォントタイプ
    plt.rcParams["pdf.fonttype"] = 42                       # PDFで使用するフォントタイプ
    plt.rcParams["font.family"] ="Times New Roman"          # 使用するフォント
    plt.rcParams["mathtext.fontset"] = "custom"             # 数式用フォントセット
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"         # 数式用フォントセット
    plt.rcParams["font.size"] = 8                          # フォントサイズ
    plt.rcParams["axes.linewidth"] = 1                      # 軸の線幅
    plt.rcParams["xtick.direction"] = "out"                  # x軸の目盛線の向き
    plt.rcParams["ytick.direction"] = "out"                  # y軸の目盛線の向き
    plt.rcParams["xtick.minor.visible"] = True              # x軸の副目盛線の有無
    plt.rcParams["ytick.minor.visible"] = True              # y軸副目盛り線の有無
    plt.rcParams["xtick.major.size"] = 6                    # x軸の目盛線の長さ
    plt.rcParams["ytick.major.size"] = 6                    # y軸の目盛線の長さ
    plt.rcParams["xtick.minor.size"] = 3                    # x軸の副目盛線の長さ
    plt.rcParams["ytick.minor.size"] = 3                    # y軸の副目盛線の長さ
    plt.rcParams["xtick.major.width"] = 1                   # x軸の目盛線の線幅
    plt.rcParams["ytick.major.width"] = 1                   # y軸の目盛線の線幅
    plt.rcParams["xtick.minor.width"] = 1                   # x軸の副目盛線の線幅
    plt.rcParams["ytick.minor.width"] = 1                   # y軸の副目盛線の線幅
    plt.rcParams["figure.figsize"] = x, y                   # 図のサイズ(x, y)


def poster(x,y):
    """
    matplotlibのデフォルト体裁をポスター用に設定します．

    Parameters
    ----------
    x : 画像のサイズの幅（インチ）
    y : 画像のサイズの高さ（インチ）
    """

    plt.rcParams["ps.fonttype"] = 42                        # 画像で使用するフォントタイプ
    plt.rcParams["pdf.fonttype"] = 42                       # PDFで使用するフォントタイプ
    plt.rcParams["font.family"] ="Arial"                    # 使用するフォント
    plt.rcParams["mathtext.fontset"] = "custom"             # 数式用フォントセット
    plt.rcParams["mathtext.default"] = "regular"         # 数式用フォントセット
    plt.rcParams["mathtext.it"] = "Arial:italic"            # 数式用フォントセット
    plt.rcParams["font.size"] = 10                          # フォントサイズ
    plt.rcParams["axes.linewidth"] = 1                      # 軸の線幅
    plt.rcParams["xtick.direction"] = "out"                  # x軸の目盛線の向き
    plt.rcParams["ytick.direction"] = "out"                  # y軸の目盛線の向き
    plt.rcParams["xtick.minor.visible"] = True              # x軸の副目盛線の有無
    plt.rcParams["ytick.minor.visible"] = True              # y軸副目盛り線の有無
    plt.rcParams["xtick.major.size"] = 6                    # x軸の目盛線の長さ
    plt.rcParams["ytick.major.size"] = 6                    # y軸の目盛線の長さ
    plt.rcParams["xtick.minor.size"] = 3                    # x軸の副目盛線の長さ
    plt.rcParams["ytick.minor.size"] = 3                    # y軸の副目盛線の長さ
    plt.rcParams["xtick.major.width"] = 1                   # x軸の目盛線の線幅
    plt.rcParams["ytick.major.width"] = 1                   # y軸の目盛線の線幅
    plt.rcParams["xtick.minor.width"] = 1                   # x軸の副目盛線の線幅
    plt.rcParams["ytick.minor.width"] = 1                   # y軸の副目盛線の線幅
    plt.rcParams["figure.figsize"] = x, y                   # 図のサイズ(x, y)
