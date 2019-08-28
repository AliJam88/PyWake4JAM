from py_wake.site._site import UniformWeibullSite, UniformSite
import numpy as np
from py_wake.tests import npt
import pytest
from py_wake.examples.data.ParqueFicticio import ParqueFicticio_path
from py_wake.examples.data.ParqueFicticio.parque_ficticio import ParqueFicticioSite
from py_wake.site.wasp_grid_site import WaspGridSite
import os
import time
from py_wake.tests.test_files.wasp_grid_site import one_layer


@pytest.fixture
def site():
    return ParqueFicticioSite()


def test_local_wind(site):

    x_i, y_i = site.initial_position.T
    h_i = x_i * 0 + 70
    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    WD_ilk, WS_ilk, TI_ilk, P_ilk = site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i, wd=wdir_lst, ws=wsp_lst)
    npt.assert_array_equal(WS_ilk.shape, (8, 4, 3))

    WD_ilk, WS_ilk, TI_ilk, P_ilk = site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i)
    npt.assert_array_equal(WS_ilk.shape, (8, 360, 23))

    # check probability local_wind()[-1]
    npt.assert_almost_equal(site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i, wd=[0], ws=[10])[-1],
                            site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i, wd=[0], ws=[10], wd_bin_size=2)[-1] * 180, 6)

    z = np.arange(30, 100)
    zero = [0] * len(z)

    ws = site.local_wind(x_i=x_i[:1], y_i=y_i[:1], h_i=z, wd=[0], ws=[10])[1][:, 0, 0]
    ws35 = site.local_wind(x_i=x_i[:1], y_i=y_i[:1], h_i=[35], wd=[0], ws=[10])[1][:, 0, 0]
    z0 = 10

    if 0:
        import matplotlib.pyplot as plt
        plt.plot(ws, z)
        plt.plot(ws35 * np.log(z / z0) / np.log(35 / z0), z)
        plt.show()
    # npt.assert_array_equal(10 * (z / 50)**.3, ws)


def test_distances(site):
    x, y = site.initial_position.T
    dw_ijl, cw_ijl, dh_ijl, dwo = site.distances(src_x_i=x, src_y_i=y, src_h_i=np.array([70]),
                                                 dst_x_j=x, dst_y_j=y, dst_h_j=np.array([70]),
                                                 wd_il=np.array([[0]]))
    npt.assert_almost_equal(dw_ijl[0, :, 0], np.array([0., 207.7973259, 484.8129285, 727.1261764, 1039.5612311, 1263.5467003, 1490.7972623, 1841.0639107]))
    npt.assert_almost_equal(cw_ijl[:, 1, 0], np.array([236.1, 0., -131.1, -167.8, -204.5, -131.1, -131.1, -45.4]))
    npt.assert_almost_equal(dh_ijl, np.zeros_like(dh_ijl))


def test_distances_different_points(site):
    x, y = site.initial_position.T
    with pytest.raises(NotImplementedError):
        dw_ijl, cw_ijl, dh_ijl, dwo = site.distances(src_x_i=x, src_y_i=y, src_h_i=np.array([70]),
                                                     dst_x_j=x[1:], dst_y_j=y[1:], dst_h_j=np.array([70]),
                                                     wd_il=np.array([[0]]))


def test_distances_no_turning(site):
    x, y = site.initial_position.T
    site.turning = False
    site.local_wind(x, y, np.array([70]))
    dw_ijl, cw_ijl, dh_ijl, dwo = site.distances(src_x_i=x, src_y_i=y, src_h_i=np.array([70]),
                                                 dst_x_j=x, dst_y_j=y, dst_h_j=np.array([70]),
                                                 wd_il=np.array([[180]]))
    npt.assert_almost_equal(dw_ijl[0, :, 0], np.array([-0., 207.7973259, 484.8129285, 727.1261764, 1039.5612311, 1263.5467003, 1490.7972623, 1841.0639107]))
    npt.assert_almost_equal(cw_ijl[:, 1, 0], np.array([236.1, 0., -131.1, -167.8, -204.5, -131.1, -131.1, -45.4]))
    npt.assert_almost_equal(dh_ijl, np.zeros_like(dh_ijl))


def test_distances_ri(site):
    x, y = site.initial_position.T
    site.calc_all = False
    site.r_i = np.ones(len(x)) * 90
    dw_ijl, cw_ijl, dh_ijl, dwo = site.distances(src_x_i=x, src_y_i=y, src_h_i=np.array([70]),
                                                 dst_x_j=x, dst_y_j=y, dst_h_j=np.array([70]),
                                                 wd_il=np.array([[180]]))
    npt.assert_almost_equal(dw_ijl[0, :, 0], np.array([0., -207., -477., -710., -1016., -1236., -1456., -1799.]))
    npt.assert_almost_equal(cw_ijl[:, 1, 0], np.array([-236.1, 0., 131.1, 167.8, 204.5, 131.1, 131.1, 45.4]))
    npt.assert_almost_equal(dh_ijl, np.zeros_like(dh_ijl))


def test_distances_uniform(site):
    x, y = site.initial_position.T
    site.distance_type = None
    dw_ijl, cw_ijl, dh_ijl, dwo = site.distances(src_x_i=x, src_y_i=y, src_h_i=np.array([70]),
                                                 dst_x_j=x, dst_y_j=y, dst_h_j=np.array([70]),
                                                 wd_il=np.array([[180]]))
    npt.assert_almost_equal(dw_ijl[0, :, 0], np.array([0., -207., -477., -710., -1016., -1236., -1456., -1799.]))
    npt.assert_almost_equal(np.abs(cw_ijl[:, 1, 0]), np.array([236.1, -0., 131.1, 167.8, 204.5, 131.1, 131.1, 45.4]))
    npt.assert_almost_equal(dh_ijl, np.zeros_like(dh_ijl))


def test_distances_wd_shape(site):
    x, y = site.initial_position.T
    dw_ijl, cw_ijl, dh_ijl, dwo = site.distances(src_x_i=x, src_y_i=y, src_h_i=np.array([70]),
                                                 dst_x_j=x, dst_y_j=y, dst_h_j=np.array([70]),
                                                 wd_il=np.ones((len(x), 1)) * 180)
    npt.assert_almost_equal(dw_ijl[0, :, 0], np.array([0., -207., -477., -710., -1016., -1236., -1456., -1799.]))
    npt.assert_almost_equal(cw_ijl[:, 1, 0], np.array([-236.1, 0., 131.1, 167.8, 204.5, 131.1, 131.1, 45.4]))
    npt.assert_almost_equal(dh_ijl, np.zeros_like(dh_ijl))


def test_speed_up_using_pickle():
    pkl_fn = ParqueFicticio_path + "ParqueFicticio.pkl"
    if os.path.exists(pkl_fn):
        os.remove(pkl_fn)
    start = time.time()
    site = WaspGridSite.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=False)
    time_wo_pkl = time.time() - start
    site = WaspGridSite.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=True)
    assert os.path.exists(pkl_fn)
    start = time.time()
    site = WaspGridSite.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=True)
    time_w_pkl = time.time() - start
    npt.assert_array_less(time_w_pkl * 10, time_wo_pkl)


def test_interp_funcs_initialization_missing_key(site):
    site.interp_funcs_initialization(['missing'])


def test_one_layer():
    site = WaspGridSite.from_wasp_grd(os.path.dirname(one_layer.__file__) + "/", speedup_using_pickle=False)


def test_missing_path():
    with pytest.raises(NotImplementedError):
        WaspGridSite.from_wasp_grd("missing_path/", speedup_using_pickle=True)

    with pytest.raises(Exception, match='Path was not a directory'):
        WaspGridSite.from_wasp_grd("missing_path/", speedup_using_pickle=False)


def test_elevation(site):
    x_i, y_i = site.initial_position.T
    npt.assert_array_less(np.abs(site.elevation(x_i=x_i, y_i=y_i) -
                                 [519.4, 567.7, 583.6, 600, 574.8, 559.9, 517.7, 474.5]  # ref from wasp
                                 ), 5)


def test_plot_ws_distribution(site):
    x, y = site.initial_position[0]
    p1 = site.plot_ws_distribution(x=x, y=y, h=70, wd=[0, 90, 180, 270])
    p2 = site.plot_ws_distribution(x=x, y=y, h=70, wd=[0, 90, 180, 270], include_wd_distribution=True)
    if 0:
        import matplotlib.pyplot as plt
        plt.show()

    # print(np.round(p1[::30], 4).tolist())
    npt.assert_array_almost_equal(p1[::30], [0.0001, 0.0042, 0.0072, 0.0077, 0.0063,
                                             0.0041, 0.0021, 0.0009, 0.0003, 0.0001], 4)
    # print(np.round(p2[::30], 4).tolist())
    npt.assert_array_almost_equal(p2[::30], [0.0001, 0.0021, 0.0033, 0.0033, 0.0025,
                                             0.0014, 0.0007, 0.0002, 0.0001, 0.0], 4)


def test_plot_wd_distribution(site):
    x, y = site.initial_position[0]
    import matplotlib.pyplot as plt
    p = site.plot_wd_distribution(x=x, y=y, h=70, n_wd=12, ax=plt)
    # print(np.round(p, 3).tolist())
    npt.assert_array_almost_equal(p, [0.052, 0.043, 0.058, 0.085, 0.089, 0.061,
                                      0.047, 0.083, 0.153, 0.152, 0.108, 0.068], 3)

    if 0:
        plt.show()


def test_plot_wd_distribution_with_ws_levels(site):
    x, y = site.initial_position[0]
    p = site.plot_wd_distribution(x=x, y=y, n_wd=12, ws_bins=[0, 5, 10, 15, 20, 25])
    # print(np.round(p, 3).tolist())
    npt.assert_array_almost_equal(p, [[0.031, 0.019, 0.003, 0.0, 0.0],
                                      [0.023, 0.018, 0.002, 0.0, 0.0],
                                      [0.018, 0.032, 0.007, 0.0, 0.0],
                                      [0.018, 0.048, 0.018, 0.001, 0.0],
                                      [0.024, 0.052, 0.014, 0.0, 0.0],
                                      [0.029, 0.031, 0.003, 0.0, 0.0],
                                      [0.023, 0.022, 0.002, 0.0, 0.0],
                                      [0.022, 0.041, 0.016, 0.002, 0.0],
                                      [0.023, 0.062, 0.048, 0.016, 0.003],
                                      [0.027, 0.058, 0.044, 0.018, 0.004],
                                      [0.034, 0.044, 0.023, 0.007, 0.001],
                                      [0.036, 0.026, 0.006, 0.001, 0.0]], 3)

    if 0:
        import matplotlib.pyplot as plt
        plt.show()
