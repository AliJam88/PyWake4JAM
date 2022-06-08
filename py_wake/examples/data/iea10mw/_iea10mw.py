from py_wake import np
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine
power_curve = np.array([[0.00000000e+00, 0.00000000e+00],
                        [2.90000000e+00, 0.00000000e+00],
                        [3.00000000e+00, 3.76809496e+01],
                        [4.00000000e+00, 3.92394850e+02],
                        [4.51470000e+00, 6.52877703e+02],
                        [5.00080000e+00, 9.49787484e+02],
                        [5.45740000e+00, 1.27397015e+03],
                        [5.88330000e+00, 1.62453737e+03],
                        [6.27770000e+00, 1.99417169e+03],
                        [6.63970000e+00, 2.36991416e+03],
                        [6.96840000e+00, 2.74278637e+03],
                        [7.26320000e+00, 3.10582353e+03],
                        [7.52340000e+00, 3.45171734e+03],
                        [7.74840000e+00, 3.77075976e+03],
                        [7.93770000e+00, 4.05393526e+03],
                        [8.09090000e+00, 4.29322121e+03],
                        [8.20770000e+00, 4.48184867e+03],
                        [8.28770000e+00, 4.61418318e+03],
                        [8.33080000e+00, 4.68654608e+03],
                        [8.33700000e+00, 4.69701742e+03],
                        [8.36780000e+00, 4.74926760e+03],
                        [8.43560000e+00, 4.86564815e+03],
                        [8.54010000e+00, 5.04872405e+03],
                        [8.68120000e+00, 5.30312729e+03],
                        [8.85850000e+00, 5.63473290e+03],
                        [9.07170000e+00, 6.05144103e+03],
                        [9.32020000e+00, 6.56248708e+03],
                        [9.60350000e+00, 7.17928821e+03],
                        [9.92100000e+00, 7.91514937e+03],
                        [1.02720000e+01, 8.79963266e+03],
                        [1.06557000e+01, 1.00000041e+04],
                        [1.07577000e+01, 1.00000101e+04],
                        [1.15177000e+01, 9.99998670e+03],
                        [1.19941000e+01, 1.00000090e+04],
                        [1.24994000e+01, 1.00000110e+04],
                        [1.30324000e+01, 9.99998525e+03],
                        [1.35920000e+01, 1.00000103e+04],
                        [1.41769000e+01, 1.00000051e+04],
                        [1.47859000e+01, 1.00000202e+04],
                        [1.54175000e+01, 1.00000170e+04],
                        [1.60704000e+01, 1.00000304e+04],
                        [1.67432000e+01, 1.00000238e+04],
                        [1.74342000e+01, 1.00000370e+04],
                        [1.81421000e+01, 1.00000458e+04],
                        [1.88652000e+01, 1.00000053e+04],
                        [1.96019000e+01, 9.99999288e+03],
                        [2.03506000e+01, 9.99996326e+03],
                        [2.11096000e+01, 9.99997681e+03],
                        [2.18773000e+01, 1.00000281e+04],
                        [2.26519000e+01, 9.99989737e+03],
                        [2.34317000e+01, 1.00000827e+04],
                        [2.42150000e+01, 1.00000140e+04],
                        [2.50100000e+01, 1.00118719e+04],
                        [2.50200000e+01, 0.00000000e+00],
                        [5.00000000e+01, 0.00000000e+00]])

ct_curve = np.array([[0.00000e+00, 0.00000e+00],
                     [2.90000e+00, 0.00000e+00],
                     [3.00000e+00, 7.70100e-01],
                     [4.00000e+00, 7.70100e-01],
                     [4.51470e+00, 7.76300e-01],
                     [5.00080e+00, 7.82400e-01],
                     [5.45740e+00, 7.82000e-01],
                     [5.88330e+00, 7.80200e-01],
                     [6.27770e+00, 7.77200e-01],
                     [6.63970e+00, 7.71900e-01],
                     [6.96840e+00, 7.76800e-01],
                     [7.26320e+00, 7.76800e-01],
                     [7.52340e+00, 7.76800e-01],
                     [7.74840e+00, 7.76800e-01],
                     [7.93770e+00, 7.76800e-01],
                     [8.09090e+00, 7.76800e-01],
                     [8.20770e+00, 7.76800e-01],
                     [8.28770e+00, 7.76800e-01],
                     [8.33080e+00, 7.76800e-01],
                     [8.33700e+00, 7.76800e-01],
                     [8.36780e+00, 7.76800e-01],
                     [8.43560e+00, 7.76800e-01],
                     [8.54010e+00, 7.76800e-01],
                     [8.68120e+00, 7.76800e-01],
                     [8.85850e+00, 7.76800e-01],
                     [9.07170e+00, 7.76800e-01],
                     [9.32020e+00, 7.76800e-01],
                     [9.60350e+00, 7.76800e-01],
                     [9.92100e+00, 7.76800e-01],
                     [1.02720e+01, 7.67500e-01],
                     [1.06557e+01, 7.65100e-01],
                     [1.07577e+01, 7.58700e-01],
                     [1.15177e+01, 5.05600e-01],
                     [1.19941e+01, 4.31000e-01],
                     [1.24994e+01, 3.70800e-01],
                     [1.30324e+01, 3.20900e-01],
                     [1.35920e+01, 2.78800e-01],
                     [1.41769e+01, 2.43200e-01],
                     [1.47859e+01, 2.12800e-01],
                     [1.54175e+01, 1.86800e-01],
                     [1.60704e+01, 1.64500e-01],
                     [1.67432e+01, 1.45400e-01],
                     [1.74342e+01, 1.28900e-01],
                     [1.81421e+01, 1.14700e-01],
                     [1.88652e+01, 1.02400e-01],
                     [1.96019e+01, 9.18000e-02],
                     [2.03506e+01, 8.25000e-02],
                     [2.11096e+01, 7.45000e-02],
                     [2.18773e+01, 6.75000e-02],
                     [2.26519e+01, 6.13000e-02],
                     [2.34317e+01, 5.59000e-02],
                     [2.42150e+01, 5.12000e-02],
                     [2.50100e+01, 4.70000e-02],
                     [2.50200e+01, 0.00000e+00],
                     [5.00000e+01, 0.00000e+00]])


class IEA10MW(WindTurbine):
    '''
    Data from:
    https://www.nrel.gov/docs/fy19osti/73492.pdf
    Pietro Bortolotti, Helena Canet Tarr ́es, Katherine Dykes, Karl Merz,
    Latha Sethuraman, David Verelst, and Frederik Zahle
    "IEA Wind Task 37 on Systems Engineering in Wind Energy WP2.1 Reference
    Wind Turbines", May 23 2019
    '''

    def __init__(self, method='linear'):
        u, p = power_curve.T
        WindTurbine.__init__(
                             self,
                             'IEA10MW',
                             diameter=198.0,
                             hub_height=119,
                             powerCtFunction=PowerCtTabular(u, p * 1000, 'w',
                                                            ct_curve[:, 1],
                                                            ws_cutin=4,
                                                            ws_cutout=25,
                                                            method=method))


IEA10WM_RWT = IEA10MW


def main():
    wt = IEA10MW()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.arange(3, 25)
    import matplotlib.pyplot as plt
    plt.plot(ws, wt.power(ws), '.-', label='power [W]')
    c = plt.plot([], label='ct')[0].get_color()
    plt.legend()
    ax = plt.twinx()
    ax.plot(ws, wt.ct(ws), '.-', color=c)
    plt.show()


if __name__ == '__main__':
    main()
