import pytest

import matplotlib.pyplot as plt
from py_wake import np
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.deficit_models.fuga import FugaDeficit, Fuga
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit, IEA37SimpleBastankhahGaussianDeficit,\
    ZongGaussianDeficit, NiayifarGaussianDeficit, BastankhahGaussian, ZongGaussian,\
    NiayifarGaussian, CarbajofuertesGaussianDeficit, TurboGaussianDeficit, IEA37SimpleBastankhahGaussian
from py_wake.deficit_models.gcl import GCLDeficit, GCL, GCLLocal
from py_wake.deficit_models.noj import NOJDeficit, NOJ, NOJLocalDeficit, NOJLocal, TurboNOJDeficit
from py_wake.deficit_models import VortexDipole
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm
from py_wake.flow_map import HorizontalGrid, XYGrid
from py_wake.superposition_models import SquaredSum, WeightedSum
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.turbulence_models.gcl_turb import GCLTurbulence
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.utils.model_utils import get_models
from numpy import newaxis as na
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.rotor_avg_models.rotor_avg_model import CGIRotorAvg
import warnings
from py_wake.deficit_models.utils import mom1d_a0


class GCLLocalDeficit(GCLDeficit):
    def __init__(self):
        GCLDeficit.__init__(self, use_effective_ws=True, use_effective_ti=True)


@pytest.mark.parametrize(
    'deficitModel,aep_ref',
    # test that the result is equal to last run (no evidens that these number are correct)
    [
        (NOJDeficit(), (368764.7199209298, [9863.98445,  8458.99574, 10863.93795, 14022.65446, 22279.85026,
                                            25318.68166, 37461.855  , 42999.89503, 24857.24081, 13602.39867,
                                            14348.77375, 31867.18218, 75509.51435, 17661.32988, 11773.35282,
                                            7875.07291])),
        (NOJLocalDeficit(a0=mom1d_a0), (335151.6404628441, [8355.71335, 7605.92379, 10654.172, 13047.6971, 19181.46408,
                                                 23558.34198, 36738.52415, 38663.44595, 21056.39764, 12042.79324,
                                                 13813.46269, 30999.42279, 63947.61202, 17180.40299, 11334.12323,
                                                 6972.14345])),
        (TurboNOJDeficit(a0=mom1d_a0), (354154.2962989713, [9320.85263, 8138.29496, 10753.75662, 13398.00865, 21189.29438,
                                                 24190.84895, 37081.91938, 41369.66605, 23488.54863, 12938.48451,
                                                 14065.00719, 30469.75602, 71831.78532, 16886.85274, 11540.51872,
                                                 7490.70156])),

        (BastankhahGaussianDeficit(), (355597.1277349052,
                                       [9147.1739 ,  8136.64045, 11296.20887, 13952.43453, 19784.35422,
                                        25191.89568, 38952.44439, 41361.25562, 23050.87822, 12946.16957,
                                        14879.10238, 32312.09672, 66974.91909, 17907.90902, 12208.49426,
                                        7495.1508])),
        (IEA37SimpleBastankhahGaussianDeficit(), read_iea37_windfarm(iea37_path + 'iea37-ex16.yaml')[2]),
        (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc'),
         (404441.6306021485, [9912.33731, 9762.05717, 12510.14066, 15396.76584, 23017.66483,
                              27799.7161, 43138.41606, 49623.79059, 24979.09001, 15460.45923,
                              16723.02619, 35694.35526, 77969.14805, 19782.41376, 13721.45739,
                              8950.79218])),
        (GCLDeficit(), (370863.6246093183,
                        [9385.75387, 8768.52105, 11450.13309, 14262.42186, 21178.74926,
                         25751.59502, 39483.21753, 44573.31533, 23652.09976, 13924.58752,
                         15106.11692, 32840.02909, 71830.22035, 18200.49805, 12394.7626,
                         8061.6033])),
        (GCLLocalDeficit(), (381187.36105425097,
                             [9678.85358, 9003.65526, 11775.06899, 14632.42259, 21915.85495,
                              26419.65189, 40603.68618, 45768.58091, 24390.71103, 14567.43106,
                              15197.82861, 32985.67922, 75062.92788, 18281.21981, 12470.01322,
                              8433.77587])),
        (ZongGaussianDeficit(eps_coeff=0.35),
         (353802.41756446497, [8960.79647,  8071.467  , 11308.13887, 13919.16109, 19923.64896,
                               25131.81863, 38993.58231, 41029.95727, 22581.2071 , 12906.39753,
                               14753.38477, 32283.50946, 66469.81638, 17892.06549, 12105.34135,
                               7472.12489])),
        (NiayifarGaussianDeficit(),
         (362094.6051760222, [9216.49947,  8317.79564, 11380.71714, 14085.18651, 20633.99691,
                              25431.58675, 39243.85219, 42282.12782, 23225.57866, 13375.79217,
                              14794.64817, 32543.64467, 69643.8641 , 18036.2368 , 12139.1985 ,
                              7743.87967])),
        (CarbajofuertesGaussianDeficit(),
         (362181.7902397156, [9147.75141,  8391.35733, 11374.77771, 14056.87594, 20620.92221,
                              25380.47044, 39223.37142, 42656.06645, 23052.33356, 13470.08994,
                              14782.42464, 32454.40658, 69656.52099, 17986.77955, 12129.16893,
                              7798.47313])),
        (TurboGaussianDeficit(rotorAvgModel=None),
         (332442.50441823964, [7669.15984,  7870.54675, 11070.85926, 13466.62205, 17127.5009 ,
                               24314.73426, 38175.37675, 40008.61265, 19326.28279, 12468.08061,
                               14565.28281, 31605.97384, 58087.54436, 17516.56382, 11951.00128,
                               7218.36246])),
    ])
def test_IEA37_ex16(deficitModel, aep_ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel,
                                 superpositionModel=SquaredSum(), turbulenceModel=GCLTurbulence())

    aep_ilk = wf_model(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep_ilk(normalize_probabilities=True)
    aep_MW_l = aep_ilk.sum((0, 2)) * 1000

    # check if ref is reasonable
    aep_est = 16 * 3.35 * 24 * 365 * .8  # n_wt * P_rated * hours_pr_year - 20% wake loss = 375628.8
    npt.assert_allclose(aep_ref[0], aep_est, rtol=.12)
    npt.assert_allclose(aep_ref[1], [9500, 8700, 11500, 14300, 21300, 25900, 39600, 44300, 23900,
                                     13900, 15200, 33000, 72100, 18300, 12500, 8000], rtol=.2)

    npt.assert_almost_equal(aep_MW_l.sum(), aep_ref[0], 5)
    npt.assert_array_almost_equal(aep_MW_l, aep_ref[1], 5)


@pytest.mark.parametrize('deficitModel', get_models(WakeDeficitModel))
def test_huge_distance(deficitModel):
    ref = {"NOJDeficit": 9.799733,
           "FugaDeficit": 9.8,
           "FugaYawDeficit": 9.8,
           "FugaMultiLUTDeficit": 9.8,
           "BastankhahGaussianDeficit": 9.79916,
           "CarbajofuertesGaussianDeficit": 9.798729,
           "IEA37SimpleBastankhahGaussianDeficit": 9.799151,
           "NiayifarGaussianDeficit": 9.799162,
           "TurboGaussianDeficit": 9.67046,
           "ZongGaussianDeficit": 9.799159,
           "GCLDeficit": 9.728704,
           "NoWakeDeficit": 9.8,
           "NOJLocalDeficit": 9.797536,
           "TurboNOJDeficit": 9.795322, }
    site = IEA37Site(16)

    windTurbines = IEA37_WindTurbines()
    wfm = All2AllIterative(site, windTurbines, wake_deficitModel=deficitModel(), turbulenceModel=GCLTurbulence())
    sim_res = wfm([0, 100000], [0, 0], wd=[0, 90, 180, 270], yaw=0)
    # print(f'"{deficitModel.__name__}": {np.round(sim_res.WS_eff.sel(wt=1, ws=9.8, wd=270).item(),6)},')

    npt.assert_array_almost_equal([9.8, 9.8, 9.8, ref[deficitModel.__name__]], sim_res.WS_eff.sel(wt=1).squeeze())


@pytest.mark.parametrize('deficitModel', get_models(BlockageDeficitModel))
def test_huge_distance_blockage(deficitModel):
    if deficitModel is None:
        return
    ref = {"FugaDeficit": 9.8,
           "FugaYawDeficit": 9.8,
           "FugaMultiLUTDeficit": 9.8,
           "HybridInduction": 9.799999,
           "SelfSimilarityDeficit2020": 9.799999,
           "VortexDipole": 9.799999,
           "RankineHalfBody": 9.799999,
           "Rathmann": 9.799999,
           "RathmannScaled": 9.799999,
           "SelfSimilarityDeficit": 9.799999,
           "VortexCylinder": 9.799999, }
    site = IEA37Site(16)

    windTurbines = IEA37_WindTurbines()
    wfm = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                           blockage_deficitModel=deficitModel(),
                           turbulenceModel=GCLTurbulence())
    sim_res = wfm([0, 100000], [0, 0], wd=[0, 90, 180, 270], yaw=0)
    # print(f'"{deficitModel.__name__}": {np.round(sim_res.WS_eff.sel(wt=0, ws=9.8, wd=270).item(),6)},')

    npt.assert_array_almost_equal([9.8, 9.8, 9.8, ref[deficitModel.__name__]], sim_res.WS_eff.sel(wt=1).squeeze())


@pytest.mark.parametrize(
    'deficitModel,ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(NOJDeficit(),
      [3.38, 3.38, 9.  , 7.49, 7.49, 7.49, 7.49, 7.34, 7.34, 7.34, 7.34, 8.31, 8.31, 8.31, 8.31, 8.31, 8.31]),
     (NOJLocalDeficit(a0=mom1d_a0),
      [3.09, 3.09, 9., 9., 5.54, 5.54, 5.54, 5.54, 5.54, 5.54, 6.73, 6.73, 6.73, 6.73, 6.73, 6.73, 6.73]),
     (TurboNOJDeficit(a0=mom1d_a0),
      [3.51, 3.51, 3.51, 7.45, 7.45, 7.45, 7.45, 7.45, 7.13, 7.13, 7.13, 7.96, 7.96, 7.96, 7.96, 7.96, 7.96]),
     (BastankhahGaussianDeficit(),
      [-2.61,  1.89,  6.73,  8.26,  7.58,  6.59,  5.9 ,  5.98,  6.75, 7.67,  8.08,  7.87,  7.59,  7.46,  7.56,  7.84,  8.2]),
     (IEA37SimpleBastankhahGaussianDeficit(),
      [3.32, 4.86, 7.0, 8.1, 7.8, 7.23, 6.86, 6.9, 7.3, 7.82, 8.11, 8.04, 7.87, 7.79, 7.85, 8.04, 8.28]),
     (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc'),
      [7.06, 7.87, 8.77, 8.85, 8.52, 7.96, 7.49, 7.55, 8.06, 8.58, 8.69, 8.45, 8.18, 8.05, 8.15, 8.41, 8.68]),
     (GCLDeficit(),
      [2.39, 5.01, 7.74, 8.34, 7.95, 7.58, 7.29, 7.32, 7.61, 7.92, 8.11, 8.09, 7.95, 7.83, 7.92, 8.1, 8.3]),
     (GCLLocalDeficit(),
      [3.05, 5.24, 7.61, 8.36, 8.08, 7.81, 7.61, 7.63, 7.82, 8.01, 8.11, 8.07, 7.94, 7.83, 7.92, 8.1, 8.3]),
     (ZongGaussianDeficit(eps_coeff=0.35),
      [6.3 , 7.05, 8.08, 8.35, 7.47, 6.14, 5.14, 5.25, 6.37, 7.61, 8.07, 7.75, 7.35, 7.18, 7.31, 7.69, 8.14]),
     (NiayifarGaussianDeficit(),
      [-2.61,  1.9 ,  6.73,  8.26,  7.58,  6.6 ,  5.91,  5.98,  6.75, 7.67,  8.08,  7.88,  7.59,  7.46,  7.56,  7.84,  8.2 ]),
     (CarbajofuertesGaussianDeficit(),
      [6.45, 7.13, 8.06, 8.29, 7.66, 6.86, 6.33, 6.38, 6.99, 7.72, 8.06, 7.86, 7.57, 7.44, 7.54, 7.83, 8.19]),
     (TurboGaussianDeficit(),
      [3.34, 4.88, 7.03, 8.14, 7.5 , 6.31, 5.44, 5.53, 6.52, 7.66, 8.01, 7.28, 6.38, 5.94, 6.27, 7.14, 8.03]),
     ])
def test_deficitModel_wake_map(deficitModel, ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel, superpositionModel=SquaredSum(),
                                 turbulenceModel=GCLTurbulence())

    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)

    flow_map = wf_model(x, y, wd=0, ws=9).flow_map(HorizontalGrid(x_j, y_j))
    X, Y = flow_map.X, flow_map.Y
    Z = flow_map.WS_eff_xylk[:, :, 0, 0]

    mean_ref = [3.2, 4.9, 8., 8.2, 7.9, 7.4, 7., 7., 7.4, 7.9, 8.1, 8.1, 8., 7.8, 7.9, 8.1, 8.4]

    if 0:
        flow_map.plot_wake_map()
        plt.plot(X[49, 100:133:2], Y[49, 100:133:2], '.-')
        windTurbines.plot(x, y)
        plt.figure()
        plt.plot(Z[49, 100:133:2])
        plt.plot(ref, label='ref')
        plt.plot(mean_ref, label='Mean ref')
        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(ref, Z[49, 100:133:2], 2)

    # check that ref is reasonable
    npt.assert_allclose(ref[3:], mean_ref[3:], atol=2.6)


@pytest.mark.parametrize(
    'deficitModel,wake_radius_ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [
        (NOJDeficit(), [100., 75., 150., 100., 100.]),
        (NOJLocalDeficit(), [71., 46., 92., 71., 61.5]),
        (TurboNOJDeficit(), [99.024477, 61.553917, 123.107833, 92.439673, 97.034049]),
        (BastankhahGaussianDeficit(), [83.336286, 57.895893, 115.791786, 75.266662, 83.336286]),
        (IEA37SimpleBastankhahGaussianDeficit(), [103.166178, 67.810839, 135.621678, 103.166178, 103.166178]),
        (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc'), [100, 50, 100, 100, 100]),
        (GCLDeficit(), [156.949964, 97.763333, 195.526667, 113.225695, 111.340236]),
        (GCLLocalDeficit(), [156.949964, 97.763333, 195.526667, 113.225695, 111.340236]),
        (ZongGaussianDeficit(eps_coeff=0.35), [91.15734, 66.228381, 132.456762, 94.90156, 79.198215]),
        (NiayifarGaussianDeficit(), [92.880786, 67.440393, 134.880786, 84.811162, 73.880786]),
        (CarbajofuertesGaussianDeficit(), [102.914211, 68.866465, 137.866624, 102.914211, 85.457105]),
        (TurboGaussianDeficit(), [76.674176, 41.548202, 83.096405, 64.831198, 76.143396]),
    ])
def test_wake_radius(deficitModel, wake_radius_ref):

    mean_ref = [105, 68, 135, 93, 123]
    # check that ref is reasonable
    npt.assert_allclose(wake_radius_ref, mean_ref, rtol=.5)

    npt.assert_array_almost_equal(deficitModel.wake_radius(
        D_src_il=np.reshape([100, 50, 100, 100, 100], (5, 1)),
        dw_ijlk=np.reshape([500, 500, 1000, 500, 500], (5, 1, 1, 1)),
        WS_ilk=np.reshape([10, 10, 10, 10, 10], (5, 1, 1)),
        ct_ilk=np.reshape([.8, .8, .8, .4, .8], (5, 1, 1)),
        TI_ilk=np.reshape([.1, .1, .1, .1, .05], (5, 1, 1)),
        TI_eff_ilk=np.reshape([.1, .1, .1, .1, .05], (5, 1, 1)))[:, 0, 0, 0],
        wake_radius_ref)

    # Check that it works when called from WindFarmModel
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel, turbulenceModel=GCLTurbulence())
    wfm(x=[0, 500], y=[0, 0], wd=[30], ws=[10])

    if 0:
        ax1, ax2 = plt.subplots(2, 1)[1]
        sim_res = wfm([0], [0], wd=[270], ws=10)
        fm = sim_res.flow_map(HorizontalGrid(x=np.arange(-100, 1500, 10)))
        fm.WS_eff.plot(ax=ax1)
        x = np.arange(0, 1500, 10)
        wr = deficitModel.wake_radius(
            WS_ilk=np.reshape([10], (1, 1, 1)),
            D_src_il=np.reshape([130], (1, 1)),
            dw_ijlk=np.reshape(x, (1, len(x), 1, 1)),
            ct_ilk=sim_res.CT.values,
            TI_ilk=np.reshape(sim_res.TI.values, (1, 1, 1)),
            TI_eff_ilk=sim_res.TI_eff.values)[0, :, 0, 0]
        ax1.set_title(deficitModel.__class__.__name__)
        ax1.plot(x, wr)
        ax1.plot(x, -wr)
        ax1.axvline(500, linestyle='--', color='k')
        ax1.axis('equal')
        fm.WS_eff.sel(x=500).plot(ax=ax2)
        ax2.axvline(-wr[x == 500], color='k')
        ax2.axvline(wr[x == 500], color='k')
        ax2.grid()
        plt.show()


def test_wake_radius_not_implemented():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    class MyDeficitModel(WakeDeficitModel):

        def calc_deficit(self, WS_ilk, dw_ijlk, cw_ijlk, **_):
            # 10% deficit in downstream triangle
            ws_10pct_ijlk = 0.1 * WS_ilk[:, na]
            triangle_ijlk = ((.2 * dw_ijlk) > cw_ijlk)
            return ws_10pct_ijlk * triangle_ijlk

    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=MyDeficitModel(),
                            turbulenceModel=GCLTurbulence())
    with pytest.raises(NotImplementedError, match="wake_radius not implemented for MyDeficitModel"):
        wfm(x, y)


@pytest.mark.parametrize(
    'deficitModel,aep_ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(BastankhahGaussianDeficit(), (345563.56158101023,
                                    [8840.49603,  7861.52773, 11065.6338 , 13561.93359, 18886.67187,
                                     24486.82453, 38157.35794, 39962.76595, 22278.05001, 12638.95639,
                                     14630.37761, 31286.00381, 65246.02835, 17339.23102, 12004.4124 ,
                                     7317.29054])),
     (ZongGaussianDeficit(eps_coeff=0.35),
      (342537.2931214204, [8677.52913,  7784.55762, 11095.77536, 13542.73642, 18884.86515,
                           24452.16298, 38261.29436, 39571.50123, 21867.37342, 12558.19339,
                           14505.28555, 31149.32656, 63750.90399, 17263.48219, 11901.77276,
                           7270.53301])),
     (NiayifarGaussianDeficit(),
      (348957.0089966384, [8897.27267,  8015.35488, 11124.24696, 13644.89638, 19497.63969,
                           24636.61846, 38359.47228, 40744.72065, 22421.12712, 12962.14355,
                           14493.47879, 31339.84373, 66054.63976, 17369.07002, 11892.08516,
                           7504.3989])),

     ])
def test_IEA37_ex16_convection(deficitModel, aep_ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel,
                                 superpositionModel=WeightedSum(), turbulenceModel=GCLTurbulence())

    aep_ilk = wf_model(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep_ilk(normalize_probabilities=True)
    aep_MW_l = aep_ilk.sum((0, 2)) * 1000

    # check if ref is reasonable
    aep_est = 16 * 3.35 * 24 * 365 * .8  # n_wt * P_rated * hours_pr_year - 20% wake loss = 375628.8
    npt.assert_allclose(aep_ref[0], aep_est, rtol=.11)
    npt.assert_allclose(aep_ref[1], [9500, 8700, 11500, 14300, 21300, 25900, 39600, 44300, 23900,
                                     13900, 15200, 33000, 72100, 18300, 12500, 8000], rtol=.15)

    npt.assert_almost_equal(aep_MW_l.sum(), aep_ref[0], 5)
    npt.assert_array_almost_equal(aep_MW_l, aep_ref[1], 5)


@pytest.mark.parametrize(
    'deficitModel,ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(BastankhahGaussianDeficit(),
      [-2.62,  1.84,  7.76,  8.15,  7.55,  6.58,  5.87,  5.99,  6.61, 7.32,  7.68,  7.65,  7.55,  7.45,  7.55,  7.84,  8.19]),
     (ZongGaussianDeficit(eps_coeff=0.35),
      [6.96, 7.5 , 8.1 , 8.21, 7.41, 6.12, 5.12, 5.18, 6.33, 7.3 , 7.69, 7.53, 7.33, 7.18, 7.31, 7.69, 8.14]),
     (NiayifarGaussianDeficit(),
      [-2.62,  1.84,  7.76,  8.15,  7.55,  6.58,  5.87,  5.99,  6.61, 7.32,  7.68,  7.65,  7.55,  7.45,  7.56,  7.84,  8.19]),
     ])
def test_deficitModel_wake_map_convection(deficitModel, ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel, superpositionModel=WeightedSum(),
                                 turbulenceModel=GCLTurbulence())

    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)

    flow_map = wf_model(x, y, wd=0, ws=9).flow_map(HorizontalGrid(x_j, y_j))
    X, Y = flow_map.X, flow_map.Y
    Z = flow_map.WS_eff_xylk[:, :, 0, 0]

    mean_ref = [3.2, 4.9, 8., 8.2, 7.9, 7.4, 7., 7., 7.4, 7.9, 8.1, 8.1, 8., 7.8, 7.9, 8.1, 8.4]

    if 0:
        flow_map.plot_wake_map()
        plt.plot(X[49, 100:133:2], Y[49, 100:133:2], '.-')
        windTurbines.plot(x, y)
        plt.figure()
        plt.plot(Z[49, 100:133:2], label='Actual')
        plt.plot(ref, label='Reference')
        plt.plot(mean_ref, label='Mean ref')
        plt.legend()
        plt.show()

    # check that ref is reasonable
    npt.assert_allclose(ref[2:], mean_ref[2:], atol=2.6)

    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)


@pytest.mark.parametrize(
    'deficitModel,ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(ZongGaussianDeficit(eps_coeff=0.35),
      [6.96, 7.5 , 8.38, 8.3 , 7.46, 6.13, 5.12, 5.17, 6.32, 7.3 , 7.69, 7.53, 7.33, 7.18, 7.32, 7.69, 8.16])
     ])
def test_deficitModel_wake_map_convection_all2all(deficitModel, ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = All2AllIterative(site, windTurbines, wake_deficitModel=deficitModel, superpositionModel=WeightedSum(),
                                blockage_deficitModel=VortexDipole(), turbulenceModel=STF2017TurbulenceModel())

    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)

    flow_map = wf_model(x, y, wd=0, ws=9).flow_map(HorizontalGrid(x_j, y_j))
    X, Y = flow_map.X, flow_map.Y
    Z = flow_map.WS_eff_xylk[:, :, 0, 0]

    mean_ref = [3.2, 4.9, 8., 8.2, 7.9, 7.4, 7., 7., 7.4, 7.9, 8.1, 8.1, 8., 7.8, 7.9, 8.1, 8.4]

    if 0:
        flow_map.plot_wake_map()
        plt.plot(X[49, 100:133:2], Y[49, 100:133:2], '.-')
        windTurbines.plot(x, y)
        plt.figure()
        plt.plot(Z[49, 100:133:2], label='Actual')
        print(np.round(Z[49, 100:133:2], 2).values.tolist())
        plt.plot(ref, label='Reference')
        plt.plot(mean_ref, label='Mean ref')
        plt.legend()
        plt.show()

    # check that ref is reasonable
    npt.assert_allclose(ref[2:], mean_ref[2:], atol=2.6)

    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)


@pytest.mark.parametrize(
    'windFarmModel,aep_ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(NOJ,
      (368764.7199209298, [9863.98445,  8458.99574, 10863.93795, 14022.65446, 22279.85026,
                           25318.68166, 37461.855  , 42999.89503, 24857.24081, 13602.39867,
                           14348.77375, 31867.18218, 75509.51435, 17661.32988, 11773.35282,
                           7875.07291])),
     (NOJLocal,
      (336587.0633777636, [8393.09366,  7644.39035, 10690.37193, 13095.25925, 19271.97188,
                           23644.21809, 36863.35147, 38858.98426, 21150.59603, 12106.08548,
                           13868.64937, 31092.31296, 64287.70342, 17231.88429, 11379.40461,
                           7008.78633])),
     (BastankhahGaussian, (355597.1277349052,
                           [9147.1739 ,  8136.64045, 11296.20887, 13952.43453, 19784.35422,
                            25191.89568, 38952.44439, 41361.25562, 23050.87822, 12946.16957,
                            14879.10238, 32312.09672, 66974.91909, 17907.90902, 12208.49426,
                            7495.1508])),
     (IEA37SimpleBastankhahGaussian, read_iea37_windfarm(iea37_path + 'iea37-ex16.yaml')[2]),
     (lambda *args, **kwargs: Fuga(tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc', *args, **kwargs),
      (404441.6306021485, [9912.33731, 9762.05717, 12510.14066, 15396.76584, 23017.66483,
                           27799.7161, 43138.41606, 49623.79059, 24979.09001, 15460.45923,
                           16723.02619, 35694.35526, 77969.14805, 19782.41376, 13721.45739,
                           8950.79218])),
     (GCL, (370863.6246093183,
            [9385.75387, 8768.52105, 11450.13309, 14262.42186, 21178.74926,
             25751.59502, 39483.21753, 44573.31533, 23652.09976, 13924.58752,
             15106.11692, 32840.02909, 71830.22035, 18200.49805, 12394.7626,
             8061.6033])),
     (GCLLocal, (381187.36105425097,
                 [9678.85358, 9003.65526, 11775.06899, 14632.42259, 21915.85495,
                  26419.65189, 40603.68618, 45768.58091, 24390.71103, 14567.43106,
                  15197.82861, 32985.67922, 75062.92788, 18281.21981, 12470.01322,
                  8433.77587])),
     (ZongGaussian, (354394.892004361,
                     [8980.72989,  8091.71749, 11310.53734, 13932.45776, 19973.7833 ,
                      25155.82651, 39001.85288, 41132.89722, 22631.43932, 12939.46983,
                      14755.07905, 32302.53737, 66685.94946, 17902.61107, 12106.73153,
                      7491.272])),
     (NiayifarGaussian, (362094.6051760222,
                         [9216.49947,  8317.79564, 11380.71714, 14085.18651, 20633.99691,
                          25431.58675, 39243.85219, 42282.12782, 23225.57866, 13375.79217,
                          14794.64817, 32543.64467, 69643.8641 , 18036.2368 , 12139.1985 ,
                          7743.87967])),
     ])
def test_IEA37_ex16_windFarmModel(windFarmModel, aep_ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        wf_model = windFarmModel(site, windTurbines, turbulenceModel=GCLTurbulence())
    wf_model.superpositionModel = SquaredSum()

    aep_ilk = wf_model(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep_ilk(normalize_probabilities=True)
    aep_MW_l = aep_ilk.sum((0, 2)) * 1000

    # check if ref is reasonable
    aep_est = 16 * 3.35 * 24 * 365 * .8  # n_wt * P_rated * hours_pr_year - 20% wake loss = 375628.8
    npt.assert_allclose(aep_ref[0], aep_est, rtol=.11)
    npt.assert_allclose(aep_ref[1], [9500, 8700, 11500, 14300, 21300, 25900, 39600, 44300, 23900,
                                     13900, 15200, 33000, 72100, 18300, 12500, 8000], rtol=.15)

    npt.assert_almost_equal(aep_MW_l.sum(), aep_ref[0], 5)
    npt.assert_array_almost_equal(aep_MW_l, aep_ref[1], 5)


def test_own_deficit_is_zero():
    site = Hornsrev1Site()
    windTurbines = IEA37_WindTurbines()
    for deficitModel in get_models(WakeDeficitModel):
        wf_model = All2AllIterative(site, windTurbines, wake_deficitModel=deficitModel(),
                                    turbulenceModel=STF2017TurbulenceModel())
        sim_res = wf_model([0], [0], yaw=0)
        npt.assert_array_equal(sim_res.WS_eff, sim_res.WS.broadcast_like(sim_res.WS_eff))


@pytest.mark.parametrize('upstream_only,ref', [(False, [[9, 9, 9],  # -1 upstream
                                                        [7, 7, 7]]),  # - (1+2) downstream
                                               (True, [[9, 9, 9],  # -1 upstream
                                                       [8, 8, 8]])])  # -2 downstream
def test_wake_blockage_split(upstream_only, ref):
    class MyWakeModel(WakeDeficitModel):
        def calc_deficit(self, dw_ijlk, **kwargs):
            return np.ones_like(dw_ijlk) * 2

    class MyBlockageModel(BlockageDeficitModel):

        def __init__(self, upstream_only):
            BlockageDeficitModel.__init__(self, upstream_only=upstream_only)

        def calc_deficit(self, dw_ijlk, **kwargs):
            return np.ones_like(dw_ijlk)

    site = Hornsrev1Site()
    windTurbines = IEA37_WindTurbines()
    wf_model = All2AllIterative(site, windTurbines, wake_deficitModel=MyWakeModel(),
                                blockage_deficitModel=MyBlockageModel(upstream_only=upstream_only))
    sim_res = wf_model([0], [0], ws=10, wd=270)
    fm = sim_res.flow_map(XYGrid(x=[-100, 100], y=[-100, 0, 100]))
    if 0:
        sim_res.flow_map().plot_wake_map()
        print(fm.WS_eff.values.squeeze().T)
        plt.show()

    npt.assert_array_equal(fm.WS_eff.values.squeeze().T, ref)


@pytest.mark.parametrize('deficitModel', get_models(WakeDeficitModel))
def test_All2AllIterative_WakeDeficit_RotorAvg(deficitModel):
    if deficitModel == NOJLocalDeficit:
        site = IEA37Site(16)
        windTurbines = IEA37_WindTurbines()
        wf_model = All2AllIterative(site, windTurbines,
                                    wake_deficitModel=deficitModel(rotorAvgModel=CGIRotorAvg(4)),
                                    turbulenceModel=STF2017TurbulenceModel())
        sim_res = wf_model([0, 500, 1000, 1500], [0, 0, 0, 0], wd=270, ws=10)

        if 0:
            sim_res.flow_map(XYGrid(x=np.linspace(-200, 2000, 100))).plot_wake_map()
            plt.show()
