import matplotlib.pyplot as plt
import numpy as np
from py_wake.deficit_models.gcl import GCLDeficit, GCLLocalDeficit, get_dU, get_Rw
from py_wake.superposition_models import LinearSum
from py_wake.tests import npt
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.tests.check_speed import timeit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.turbulence_models.stf import STF2017TurbulenceModel


def test_GCL_ex80():
    site = Hornsrev1Site()

    x, y = site.initial_position.T
    windTurbines = V80()
    wfm = PropagateDownwind(
        site,
        windTurbines,
        wake_deficitModel=GCLDeficit(),
        superpositionModel=LinearSum())
    if 0:
        windTurbines.plot(x, y)
        plt.show()

    if 0:
        sim_res = timeit(wfm)(x, y, ws=np.arange(10, 15))[0]

        def run():
            wfm(x, y, ws=np.arange(10, 15))

        from line_profiler import LineProfiler
        lp = LineProfiler()
        lp.timer_unit = 1e-6
        lp.add_function(GCLDeficit.calc_deficit)
        lp.add_function(get_dU)
        lp.add_function(get_Rw)
        lp_wrapper = lp(run)
        res = lp_wrapper()
        lp.print_stats()
    else:
        sim_res = wfm(x, y, ws=np.arange(10, 15))

    # test that the result is equal to previuos runs (no evidens that  these number are correct)
    aep_ref = 1055.956615887197
    npt.assert_almost_equal(sim_res.aep_ilk(normalize_probabilities=True).sum(), aep_ref, 5)

    sim_res = wfm(x, y, ws=np.arange(3, 10))
    npt.assert_array_almost_equal(sim_res.aep_ilk(normalize_probabilities=True).sum(), 261.6143039016946, 5)