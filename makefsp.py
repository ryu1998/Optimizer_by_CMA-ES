import numpy as np
import sys
sys.path.append("C:\\Program Files\\Lumerical\\v211\\api\\python")
import lumapi as lm



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


def make_fsp(shift):
    """
    円孔のシフトデータからfspファイルを作成します．

    Parameters
    ----------
    shift : 円孔のシフトデータ
    """

    with lm.FDTD("BASE.fsp", hide = True) as fdtd:

        print("Starting lumerical solutions FDTD...")

        data = shift_to_coord(shift)

        fdtd.groupscope("::model")
        fdtd.groupscope("air holes")
        fdtd.importmaterialdb("material.mdf")

        radius = 50e-9
        count = 0

        # 円孔をシミュレーションに追加
        for m in range(20):
            #if m != 0:
                fdtd.addcircle(x = data[2*m], y = data[2*m+1], z_min = 0, z_max = 500e-9, radius = radius, material = "etch", name = str(count))
                count += 1

        # シミュレーションを保存
        fdtd.save("Model.fsp")

        print("Model.fsp saved.")


if __name__ == "__main__":

    x = np.array([0,0,0,0,0,0,0,0,0,0])

    x = np.loadtxt("data\\pop_200.csv", delimiter = ",")
    x = x[np.argsort(x[:,-2])[::-1], :][0,:-3]

    #x = np.loadtxt("2022-01-06 H0 QV\\pop_49.csv", delimiter = ",")
    #x = x[np.argsort(x[:,-2] / x[:,-1])[::-1], :][0,:-3]

    make_fsp(shift = x)
