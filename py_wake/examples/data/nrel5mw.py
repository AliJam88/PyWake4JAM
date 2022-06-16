from py_wake import np
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine
rotor_diameter = 126.0
wind_speeds = np.array([0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
                        6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
                        10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0,
                        14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0,
                        18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0,
                        22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.01, 25.02,
                        50.0])

cp_vals = np.array([0.0, 0.0, 0.0, 0.178085, 0.289075, 0.349022, 0.384728,
                    0.406059, 0.420228, 0.428823, 0.433873, 0.436223,
                    0.436845, 0.436575, 0.436511, 0.436561, 0.436517,
                    0.435903, 0.434673, 0.43323, 0.430466, 0.378869,
                    0.335199, 0.297991, 0.266092, 0.238588, 0.214748,
                    0.193981, 0.175808, 0.159835, 0.145741, 0.133256,
                    0.122157, 0.112257, 0.103399, 0.095449, 0.088294,
                    0.081836, 0.075993, 0.070692, 0.065875, 0.061484,
                    0.057476, 0.053809, 0.050447, 0.047358, 0.044518,
                    0.0419, 0.039483, 0.0, 0.0])

ct_vals = np.array([0.0, 0.0, 0.0, 0.99, 0.99, 0.97373036, 0.92826162,
                    0.89210543, 0.86100905, 0.835423, 0.81237673, 0.79225789,
                    0.77584769, 0.7629228, 0.76156073, 0.76261984, 0.76169723,
                    0.75232027, 0.74026851, 0.72987175, 0.70701647, 0.54054532,
                    0.45509459, 0.39343381, 0.34250785, 0.30487242, 0.27164979,
                    0.24361964, 0.21973831, 0.19918151, 0.18131868, 0.16537679,
                    0.15103727, 0.13998636, 0.1289037, 0.11970413, 0.11087113,
                    0.10339901, 0.09617888, 0.09009926, 0.08395078, 0.0791188,
                    0.07448356, 0.07050731, 0.06684119, 0.06345518, 0.06032267,
                    0.05741999, 0.05472609, 0.0, 0.0])


power_curve = np.column_stack([wind_speeds,
                               cp_vals * 0.5 * 1.225 *
                               np.pi * (rotor_diameter // 2) ** 2 *
                               wind_speeds ** 3 / 1e3])

ct_curve = np.column_stack([wind_speeds, ct_vals])


class NREL5MW(WindTurbine):
    '''
    Data from:
    https://www.nrel.gov/docs/fy09osti/38060.pdf
    J. Jonkman, S. Butterfield, W. Musial, and G. Scott
    "Definition of a 5-MW Reference Wind Turbine for Offshore
    System Development", February 2009
    '''

    def __init__(self, method='linear'):
        u, p = power_curve.T
        WindTurbine.__init__(
            self,
            'NREL5MW',
            diameter=126.0,
            hub_height=90.0,
            powerCtFunction=PowerCtTabular(u, p * 1000, 'w',
                                           ct_curve[:, 1],
                                           ws_cutin=3, ws_cutout=25,
                                           method=method))


NREL5MW_RWT = NREL5MW


def main():
    wt = NREL5MW()
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
