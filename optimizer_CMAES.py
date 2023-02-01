import os
import numpy as np
import glob
import shutil
import time
import datetime
from socket import gethostname
import getpass
from multiprocessing import Pool
from subprocess import Popen, STDOUT
import traceback


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


def makelsf_addpc(path_work, model_name):
    """
    円孔をMaster.fspに追加し，新しいfspファイルを作成するlsfスクリプトを生成します．

    Parameters
    ----------
    path_work : クライアント側の一時作業フォルダへのパス
    model_name : タスク名
    """

    with open(path_work + "\\" + "addPC" + model_name + ".lsf" ,"w") as f:

        f.write("groupscope('::model');\n")
        f.write("groupscope('air holes');\n")
        f.write("importmaterialdb('material.mdf');\n\n")

        f.write("r = 50e-09;\n")

        f.write("data = readdata('Model" + model_name + ".csv');\n\n")

        f.write("count = 0;\n")
        f.write("for(i=1:20){\n")
        f.write("    addcircle;\n")
        f.write("    set('x', data(2*i-1,1));\n")
        f.write("    set('y', data(2*i,1));\n")
        f.write("    set('z min', 0);\n")
        f.write("    set('z max', 500e-09);\n")
        f.write("    set('radius', r);\n\n")
        f.write("    set('material', 'etch');\n\n")
        f.write("    set('name', 'x' + num2str(count));\n")
        f.write("    count = count + 1;\n\n")
        f.write("}\n\n")

        f.write("unselectall;\n")

        f.write("save('Model" + model_name + ".fsp');\n")
        f.write("exit(2);\n")


def makelsf_runQanalysis(path_work, model_name):
    """
    FDTD計算したfspファイルから，Q値を計算して保存するlsfスクリプトを生成します．

    Parameters
    ----------
    path_work : クライアント側の一時作業フォルダへのパス
    model_name : タスク名
    """

    with open(path_work + "\\" + "runQanalysis" + model_name + ".lsf" ,"w") as f:

        f.write("# run Qanalysis\n")
        f.write("runanalysis('Qanalysis');\n")
        f.write("result = getresult('Qanalysis', 'Q');\n\n")

        f.write("# get wavelength, Q and dQ\n")
        f.write("w = result.lambda;\n")
        f.write("Q = result.Q;\n")
        f.write("dQ = result.dQ;\n")

        f.write("# run mode_volume analysis\n")
        f.write("runanalysis('mode_volume');\n")
        f.write("result2 = getresult('mode_volume', 'Volume');\n\n")

        f.write("# calculate modal volume as l^3/n^3\n")
        f.write("lambda = result2.lambda;\n")
        f.write("V = result2.V(find(lambda, w));\n")
        f.write("n = real(getindex('Diamond', w));\n")
        f.write("V = V / (w^3) * (n^3);\n\n")

        f.write("# write result to file\n")
        f.write("str = num2str(w) + ',' + num2str(Q) + ',' + num2str(dQ) + ',' + num2str(V);\n")
        f.write("write('Q' + '{}' + '.csv', str);\n".format(model_name))
        f.write("exit(2);\n")


def automation(shift, server, gen, index, password, path_work):
    """
    プロセスで並列計算させる部分のみを分離したモジュールです．

    Parameters
    ----------
    shift : シフトデータ
    server : サーバー名
    gen : 世代番号
    index : データ番号
    password : サーバーのパスワード
    path_work : クライアント側の一時作業用フォルダへのパス
    """

    # ログを展開
    logfile = open(server + ".log" ,"w")

    try:
        # タイマーを開始
        start = time.time()

        # サーバーの一時作業フォルダを作製
        path_server = "\\\\" + server + "\\user\\takahashi\\temp"
        if os.path.exists(path_server) == True:
            shutil.rmtree(path_server)
            os.mkdir(path_server)
        else:
            os.mkdir(path_server)

        # シフトデータをcsvに保存
        model_name = "_{}_{}".format(gen, index)
        np.savetxt(path_work + "\\Model" + model_name + ".csv", shift, delimiter = ",")

        # シフトデータを座標データに変換してサーバーに保存
        coord = shift_to_coord(shift)
        np.savetxt(path_server + "\\Model" + model_name + ".csv", coord, delimiter = ",")

        while os.path.exists(path_server + "\\BASE.fsp") == False:
            try:
                # Master.fspをサーバーに移動
                shutil.copy("BASE.fsp", path_server)
                shutil.copy("material.mdf", path_server)
            except OSError:
                pass

        # fsp作成スクリプト・Q値解析スクリプトを作成
        makelsf_addpc(path_work, model_name)
        makelsf_runQanalysis(path_work, model_name)

        while os.path.exists(path_server + "\\addPC" + model_name + ".lsf") == False or os.path.exists(path_server + "\\runQanalysis" + model_name + ".lsf") == False:
            try:
                # 構造データとスクリプトをサーバーに移動
                shutil.move(path_work + "\\" + "addPC" + model_name + ".lsf", path_server)
                shutil.move(path_work + "\\" + "runQanalysis" + model_name + ".lsf", path_server)
            except OSError:
                pass

        # AddPC_X_Y.lsfを実行，fspを生成
        proc_runaddpc = Popen("PsExec.exe -h -u babalab -p " + password + " \\\\" + server + " \"C:\\Program Files\\Lumerical\\v211\\bin\\fdtd-solutions.exe\" -nw -trust-script -run C:\\user\\takahashi\\temp\\addPC" + model_name + ".lsf C:\\user\\takahashi\\temp\\BASE.fsp", stdout = logfile, stderr = STDOUT)
        proc_runaddpc.wait(120)
        time.sleep(2)

        # FDTD計算を開始
        proc_solver = Popen("\"C:\\Program Files (x86)\\IntelSWToolsMPI\\mpi\\2018.4.274\\intel64\\bin\\mpiexec.exe\" -hosts 1 " + server + " 4 -env APPDATA \"C:\\Users\\babalab\\AppData\\Roaming\" \"C:\\Program Files\\Lumerical\\v211\\bin\\fdtd-engine-impi.exe\" -log-stdout C:\\user\\takahashi\\temp\\Model" + model_name + ".fsp", stdout = logfile, stderr = STDOUT)
        proc_solver.wait(1000)

        # Q値を計算
        proc_runQanalysis = Popen("PsExec.exe -h -u babalab -p " + password + " \\\\" + server + " \"C:\\Program Files\\Lumerical\\v211\\bin\\fdtd-solutions.exe\" -nw -trust-script -run C:\\user\\takahashi\\temp\\runQanalysis" + model_name + ".lsf  C:\\user\\takahashi\\temp\\Model" + model_name + ".fsp", stdout = logfile, stderr = STDOUT)
        proc_runQanalysis.wait(120)
        time.sleep(2)

        # Q_X_Y.csvをクライアントに送信
        shutil.move(path_server + "\\" + "Q" + model_name + ".csv", path_work)

        # 経過時間と計算Q値を表示
        elapsed = time.time() - start
        df = np.loadtxt(path_work + "\\" + "Q" + model_name + ".csv", delimiter = ",")
        logfile.write("\033[33m" + "Q" + model_name + ".csv" + "\033[0m" + "を取得．経過時間:{:.0f}時間{:.0f}分{:.0f}秒 Q = {}, V = {}".format(elapsed//3600, elapsed%3600//60, elapsed%60, "\033[33m" + "{:.0f}".format(df[1]) + "\033[0m", "\033[33m" + "{:.4f}".format(df[3]) + "\033[0m"))

        time.sleep(5)

        # サーバに残ったデータを削除
        shutil.rmtree(path_server)

    except TimeoutError:
        print("\033[36m" + "[{}] ".format(server) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "タイムアウトが発生しました．")

    except:
        traceback.print_exc(file = logfile)
        print("\033[36m" + "[{}] ".format(server) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "不明なエラーが発生しました．")

    finally:
        logfile.close()


def calculate_Q(x):
    """
    CMAESで生成した個体のシフトを基に並列計算を行い，Q値を求めます．

    Parameters
    ----------
    x : CMAESで生成した個体

    Returns
    ----------
    Q : 各個体のQ値のリスト
    """

    # 前回のログを削除
    [os.remove(f) for f in glob.glob("*.log") if len(glob.glob("*.log")) != 0]

    # クライアント側の一時作業用フォルダを作成／初期化
    path_work = os.getcwd() + "\\" + "temp"
    if os.path.exists(path_work) == True:
        shutil.rmtree(path_work)
        os.mkdir(path_work)
    else:
        os.mkdir(path_work)

    while len(glob.glob(path_work + "\\" + "Q_{}_*.csv".format(gen))) != LAMBDA:

        print("\033[32m" + "[{}] ".format(gethostname()) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "===============================================================")
        print("\033[32m" + "[{}] ".format(gethostname()) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "Gen {} の計算を開始します．".format(gen))

        # 実行中のプロセスとその数を格納
        process = []
        current_server = []

        # 並列計算用のプロセスを生成
        pool = Pool(LAMBDA)

        # 個体ごとに以下を実行
        for i, shift in enumerate(x):

            if os.path.exists(path_work + "\\" + "Q_{}_{}.csv".format(gen, i)) == False:

                # 計算を新プロセスで開始．使う変数は全てタプルで与えること
                process.append(pool.apply_async(automation, (shift, serverlist[i], gen, i, password, path_work,)))
                current_server.append(serverlist[i])

                print("\033[36m" + "[{}] ".format(serverlist[i]) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "他のサーバーを待っています．．")
                time.sleep(2)

        # 全プロセスの計算が終了するまでログの内容を更新して表示
        while np.all([p.ready() for p in process]) == False:
            time.sleep(2)
            text = "\033[{}A".format(len(process)) + "\033[0J"
            for j in range(len(current_server)):
                with open(current_server[j] + ".log", "r") as f:
                    try:
                        text += "\033[36m" + "[{}] ".format(current_server[j]) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + f.readlines()[-1].strip() + "\n"
                    except IndexError:
                        text += "\033[36m" + "[{}] ".format(current_server[j]) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "\n"
            print(text, end = "")

        if len(glob.glob(path_work + "\\" + "Q_{}_*.csv".format(gen))) != LAMBDA:
            print("\033[32m" + "[{}] ".format(gethostname()) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "Gen {} の全てのデータが揃っていません．再び Gen {} の計算を開始します．".format(gen, gen))

        # 生成した全プロセスを停止
        pool.close()

    # 取得したQ_X_Y.csvを元に個体のQ値を評価
    f = np.array([np.loadtxt(path_work + "\\Q_{}_{}.csv".format(gen, i), delimiter = ",")[0] for i in range(LAMBDA)])
    Q = np.array([np.loadtxt(path_work + "\\Q_{}_{}.csv".format(gen, i), delimiter = ",")[1] for i in range(LAMBDA)])
    V = np.array([np.loadtxt(path_work + "\\Q_{}_{}.csv".format(gen, i), delimiter = ",")[3] for i in range(LAMBDA)])

    # 構造とQ値のセットをテキストに保存
    np.savetxt(path_data + "\\pop_{}.csv".format(gen), np.concatenate((x, np.reshape(f, (-1, 1)), np.reshape(Q, (-1, 1)), np.reshape(V, (-1, 1))), axis = 1), delimiter = ",")

    # クライアント側の作業フォルダを削除
    shutil.rmtree(path_work)

    # ログを削除
    [os.remove(f) for f in glob.glob("*.log")]

    return Q, V


if __name__ == "__main__":

    # 探索の初期基準となる構造のシフト
    x0 = np.array([0,0,0,0,0,0,0,0,0,0])

    # 計算可能なサーバーの一覧を読み込む
    serverlist = [
        "dell-i7-05",
        "dell-i7-06",
        "dell-i7-07",
        "dell-i7-08",
        "dell-i7-09",
        "dell-i7-10",
        "dell-i7-11",
        "dell-i7-12",
        "dell-i7-13",
        "dell-i7-14",
        "dell-i7-15",
        "dell-i7-16",
        "dell-i7-17",
        "dell-i7-18",
        "dell-i7-19",
        "dell-i7-20",
    ]

    # CMAESのパラメータ設定
    m = x0                                                                              # 正規分布の中心点
    dim = np.shape(m)[0]                                                                # 探索の次元数
    LAMBDA = len(serverlist)                                                                         # 生成する個体数
    mu = int(LAMBDA / 2)                                                                # 選別する個体数
    c_m = 1.0                                                                           # 学習率

    weights = [np.log((LAMBDA+1)/ 2) - np.log(i) for i in range(1, mu+1)]               # 各特徴量の重み（式49）
    weights = weights / np.sum(weights)                                                 # 重みを1以下に
    mu_eff = 1.0 / np.sum(weights**2)                                                   # 有効選択偏差（？）

    sigma = 5.0                                                                         # 探索範囲の初期ステップサイズ
    path_sigma = np.zeros(dim)                                                          # ステップサイズの遷移ベクトル
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 3)                                         # ステップサイズ更新重み（式55）Deapの値を使用
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff-1) / (dim + 1)) - 1) + c_sigma             # ステップサイズダンピング重み（式55）

    C = np.identity(dim)                                                                # 共分散行列C
    path_C = np.zeros(dim)                                                              # Cの遷移ベクトル
    ccov_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)                          # Cの遷移ベクトル更新の重み（式56）
    ccov_1 = 2.0 / ((dim + 1.3)**2 + mu_eff)                                            # Cのランク1更新の重み（式57）
    ccov_mu = min(1 - ccov_1, 2.0 * (mu_eff - 2 + 1/mu_eff) / ((dim+2) ** 2 + mu_eff))  # Cのランクμ更新の重み（式58）

    # サーバーのパスワードを入力
    password = getpass.getpass("\033[32m" + "[{}] ".format(gethostname()) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "Enter password:")

    # クライアント側のデータ保存用フォルダを作成
    path_data = os.getcwd() + "\\" + "data"
    if os.path.exists(path_data) == False:
        os.mkdir(path_data)

    NGEN = 301

    for gen in range(NGEN):

        if os.path.exists(path_data + "\\gen_{}.npy".format(gen)) == False:

            # 平均0，分散Iの多変量正規分布N(0,I)を生成（式38）
            z = np.random.normal(0, 1, (LAMBDA, dim))

            # 共分散行列Cの固有値B・固有ベクトルD^2を計算．CはB*D^2*B^Tで表せる
            diagD, B = np.linalg.eigh(C)
            D = np.sqrt(diagD)

            # N(0,I)にB*Dを掛けてN(0,C)に（式39）
            y = B @ np.diag(D) @ z.T

            # 中心mにN(0,σC)を足して，N(m,σC)に従う新しい個体群xを生成（式40）
            x = m + sigma * y.T

            # 各個体のQ値を計算開始
            Q, V = calculate_Q(x)

            # 個体を評価値が大きい順にソート
            x = x[np.argsort(Q/V), :][::-1, :]

            # 個体を選別（x:多変量正規分布，y:線形変換した分布）
            x_elite = x[:mu, :]
            y_elite = (x_elite - m) / sigma

            # 評価値を基にyを重み付け（式41）
            y_w = weights @ y_elite

            # 正規分布の中心を更新（式42）
            old_m = m
            m = m + c_m * sigma * y_w

            # C^-0.5を計算
            C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T

            # ステップサイズの遷移ベクトルを更新（式43）
            path_sigma = (1 - c_sigma) * path_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w)

            # ステップサイズを更新（式44）
            E_normal = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(path_sigma) / E_normal - 1))

            # Cの遷移ベクトルを更新（式45）
            h_sigma = 1 if (np.linalg.norm(path_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen+1)))) < ((1.4 + 2 / (dim+1)) * E_normal) else 0
            path_C = (1 - ccov_c) * path_C + h_sigma * np.sqrt(ccov_c * (2 - ccov_c) * mu_eff) * y_w

            # 共分散行列Cを更新（式47）
            C = (1 + ccov_1 * (1 - h_sigma) * ccov_c * (2 - ccov_c) - ccov_1 - ccov_mu) * C                 # 共分散行列Cの更新
            C += ccov_1 * np.outer(path_C, path_C)                                                          # ランク1更新
            C += ccov_mu * np.dot((weights * (x[0:mu] - old_m).T), (x[0:mu] - old_m)) / sigma ** 2          # ランクμ更新

            # 更新された各種パラメータを.npy形式で保存
            dict = {"m" : m, "sigma" : sigma, "path_sigma" : path_sigma, "C" : C, "path_C" : path_C}
            np.save(path_data + "\\gen_{}.npy".format(gen), dict)

            # 計算結果を画面に表示
            print("\033[32m" + "[{}] ".format(gethostname()) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "Gen {} 現在の最適構造:{}".format(gen, "\033[33m" + "{}".format(np.round(x[np.argsort(Q), :][0, :])) + "\033[0m"))
            print("\033[32m" + "[{}] ".format(gethostname()) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "Gen {} 現在の最適構造のQ = {}".format(gen, "\033[33m" + "{:.0f}".format(Q[np.argsort(Q/V)][-1]) + "\033[0m"))
            print("\033[32m" + "[{}] ".format(gethostname()) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "Gen {} 現在の最適構造のV = {}".format(gen, "\033[33m" + "{:.4f}".format(V[np.argsort(Q/V)][-1]) + "\033[0m"))
            print("\033[32m" + "[{}] ".format(gethostname()) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "Gen {} 現在のσ = {}".format(gen, "\033[33m" + "{:.2f}".format(sigma) + "\033[0m"))
            time.sleep(2)

        else:
            # 保存した各種パラメータを展開
            dict = np.load(path_data + "\\gen_{}.npy".format(gen), allow_pickle =  True).item()
            m, sigma, path_sigma, C, path_C = dict["m"], dict["sigma"], dict["path_sigma"], dict["C"], dict["path_C"]

    print("\033[32m" + "[{}] ".format(gethostname()) + "\033[0m" + "[{}] ".format(datetime.datetime.now()) + "全世代の計算が終了しました．")
