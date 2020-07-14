import logging
from typing import Dict

import pytest
from pathlib import Path
import numpy as np

import porepy as pp
from GTS.isc_modelling.isc_model import ISCBiotContactMechanics
from GTS.isc_modelling.parameter import (
    BiotParameters,
    stress_tensor,
    shearzone_injection_cell,
    GrimselGranodiorite,
    nd_sides_shearzone_injection_cell,
    nd_and_shearzone_injection_cell,
    center_of_shearzone_injection_cell,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def biot_params() -> BiotParameters:
    """ Initialize BiotParameters"""
    here = Path(__file__).parent / "simulations"
    _sz = 6
    params = BiotParameters(
        folder_name=here,
        stress=stress_tensor(),
        injection_rate=1 / 6,
        frac_permeability=1e-9,
        intact_permeability=1e-12,
        well_cells=shearzone_injection_cell,
        rock=GrimselGranodiorite(),
        mesh_args={
            "mesh_size_frac": _sz,
            "mesh_size_min": 0.2 * _sz,
            "mesh_size_bound": 3 * _sz,
        },
    )
    return params


@pytest.fixture
def biot_params_small() -> Dict:
    """ Initialize Biot parameters with 'easy' values on small grid"""
    here = Path(__file__).parent / "simulations"
    _sz = 6

    values = {
        "folder_name": here,
        "stress": stress_tensor(),
        "injection_rate": 1 / 6,
        "frac_permeability": 3,  # -> frac_aperture = 6.0
        "intact_permeability": 1e-12,
        "well_cells": shearzone_injection_cell,
        "rock": GrimselGranodiorite(),
        "mesh_args": {
            "mesh_size_frac": 10,
            "mesh_size_min": 10,
            "mesh_size_bound": 30,
        },
        "length_scale": 7,
        "scalar_scale": 11,
    }
    return values


class TestISCBiotContactMechanics:
    def test_create_grid(self, biot_params):
        setup = ISCBiotContactMechanics(biot_params)
        setup.create_grid()

    def test_well_cells(self):
        assert False

    def test_compute_initial_aperture(self, biot_params_small):
        """ Test computation of initial aperture in intact rock and fractured rock."""
        # Setup model
        params = BiotParameters(**biot_params_small)
        setup = ISCBiotContactMechanics(params)
        setup.create_grid()

        g3: pp.Grid = setup._nd_grid()
        a = setup.compute_initial_aperture(g3, scaled=False)
        assert np.isclose(np.sum(a), g3.num_cells)

        g2: pp.Grid = setup.gb.grids_of_dimension(2)[0]
        a = setup.compute_initial_aperture(g2, scaled=False)
        assert np.isclose(np.sum(a), 6.0 * g2.num_cells)

    def test_mechanical_aperture(self, biot_params_small):
        """ Test computation of mechanical aperture in fractured rock"""
        # Prepare the model
        biot_params_small["shearzone_names"] = ["F1"]
        biot_params_small["length_scale"] = 1
        params = BiotParameters(**biot_params_small)
        setup = ISCBiotContactMechanics(params)

        # --- Structured Grid with one fracture
        nx = np.array([2, 2, 2])
        physdims = np.array([10, 10, 10])

        # fmt: off
        frac_pts = np.array(
            [[0, 10, 10, 0],
             [5, 5, 5, 5],
             [0, 0, 5, 5]]) / params.length_scale
        # fmt: on
        gb = pp.meshing.cart_grid(
            [frac_pts], nx=nx, physdims=physdims / params.length_scale,
        )
        # --- ---
        setup.gb = gb
        setup.assign_biot_variables()

        # Assign dummy displacement values to the fracture edge
        nd_grid = setup.grids_by_name(params.intact_name)[0]
        frac = setup.grids_by_name(params.shearzone_names[0])[0]
        edge = (frac, nd_grid)
        data_edge = setup.gb.edge_props(edge)
        mg: pp.MortarGrid = data_edge["mortar_grid"]
        nd = setup.Nd
        var_mortar = setup.mortar_displacement_variable

        if pp.STATE not in data_edge:
            data_edge[pp.STATE] = {}
        nc = frac.num_cells

        # Set mechanical mortar displacement to STATE.
        # Set u_n = 1 and ||u_t|| = 0 on side 1
        # Set all zero on side 1.
        s1 = np.vstack((np.zeros(nc), np.ones(nc), np.zeros(nc),))  # mortar side 1
        mortar_u = np.hstack(
            (s1, np.zeros((nd, nc)),)  # mortar side 1  # mortar side 2
        ).ravel("F")

        # Set mechanical mortar displacement to previous iterate.
        # Set u = (1, 1, 1) on side 1
        # Set u = (3, 3, 3) on side 2.
        # This should give an aperture of 2.
        s2 = np.vstack((3 * np.ones((nd, nc)),))
        mortar_u_prev_iter = np.hstack(
            (np.ones((nd, nc)), s2,)  # mortar side 1  # mortar side 2
        ).ravel("F")

        data_edge[pp.STATE].update(
            {
                var_mortar: mortar_u,
                "previous_iterate": {var_mortar: mortar_u_prev_iter},
            }
        )

        # Get mechanical aperture
        # Nd
        aperture_nd = setup.mechanical_aperture(nd_grid, from_iterate=False)
        assert np.allclose(aperture_nd, 0)
        aperture_nd_iter = setup.mechanical_aperture(nd_grid, from_iterate=False)
        assert np.allclose(aperture_nd_iter, 0)
        # fracture
        aperture_frac = setup.mechanical_aperture(frac, from_iterate=False)
        assert np.allclose(aperture_frac, np.ones(nc))
        aperture_frac_iter = setup.mechanical_aperture(frac, from_iterate=True)
        assert np.allclose(aperture_frac_iter, 2 * np.ones(nc))

    def test_aperture(self):
        pass

    def test_permeability(self):
        assert False

    def test_source_flow_rate(self):
        assert False

    def test_source_scalar(self):
        assert False

    def test_after_simulation(self):
        assert False

    def test_faces_to_fix(self):
        assert False

    def test_bc_type_mechanics(self):
        assert False

    def test_bc_values_mechanics(self):
        assert False

    def test_source(self):
        assert False

    def test_prepare_simulation(self, biot_params):
        setup = ISCBiotContactMechanics(biot_params)
        setup.prepare_simulation()

    def test_run_simulation(self, biot_params):
        setup = ISCBiotContactMechanics(biot_params)
        setup.prepare_simulation()
        pp.run_time_dependent_model(setup, {})

    def test_realistic_setup(self):
        """ For a 50 000 cell setup, test Contact mechanics Biot model on 5 shear zones.
        Parameters are close to the ISC setup. Model is run for 10 minutes,
        with time steps of 1 minute. Injection to a shear zone.
        """
        # _sz=6 => ~50k cells,
        # _sz=5.5 => ~66k cells
        _sz = 6
        params = BiotParameters(
            # Base parameters
            length_scale=12.8,
            scalar_scale=1e10,
            base="/home/haakonervik/mastersproject-data",
            head="2frac/steady-stateinj-1_lminMPa-frac-center-4MPa",
            time_step=pp.MINUTE,
            end_time=3 * pp.MINUTE,
            rock=GrimselGranodiorite(),
            # Geometry parameters
            shearzone_names=["S1_2", "S3_1"],
            mesh_args={
                "mesh_size_frac": _sz,  # 0.3 * _sz
                "mesh_size_min": 0.2 * _sz,
                "mesh_size_bound": 3 * _sz,  # 3.2 * _sz
            },
            # Mechanical parameters
            stress=stress_tensor(),
            dilation_angle=(np.pi / 180) * 5,  # 5 degrees dilation angle.
            newton_options={
                "max_iterations": 60,
                "nl_convergence_tol": 1e-8,
                "nl_divergence_tol": 1e5,
            },
            # Flow parameters
            source_scalar_borehole_shearzone={"shearzone": "S1_2", "borehole": "INJ1",},
            well_cells=center_of_shearzone_injection_cell,
            injection_rate=(1/60) * 4,# steady-state injection. For intvl T=1e-7 m2/s => 1 (l/min)/MPa  ||  (10 / 60) * 2,  # (10/60)*2 = 20l per 60s = 20 l/min
            frac_transmissivity=[3.7e-7,3.7e-7]  # (original): [1e-9, 3.7e-7],
        )

        setup = NeverFailtBiotCM(params)
        newton_params = params.newton_options
        pp.run_time_dependent_model(setup, newton_params)

    def test_realistic_only_fracture_zone(self):
        # See changelist: reset changes for only fracture zone setup of 2020/06/25.
        # _sz = 2, bounding_box=None -> ~53k 3d, ~6k 2d, 80 1d cells
        _sz = 3.5
        params = BiotParameters(
            # Base parameters
            length_scale=0.05,
            scalar_scale=1e6,
            head="60k-5frac/medium-sized-intact-grid/t_end-10min",
            time_step=pp.MINUTE,
            end_time=10 * pp.MINUTE,
            rock=GrimselGranodiorite(),
            # Geometry parameters
            mesh_args={
                "mesh_size_frac": 0.9 * _sz,
                "mesh_size_min": 0.2 * _sz,
                "mesh_size_bound": 3 * _sz,
            },
            bounding_box={
                "xmin": -1 - 50,
                "ymin": 80 - 50,
                "zmin": -4 - 50,
                "xmax": 86 + 50,
                "ymax": 151 + 50,
                "zmax": 41 + 50,
            },
            shearzone_names=["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"],
            # Mechanical parameters
            stress=stress_tensor(),
            newton_options={
                "max_iterations": 30,
                "nl_convergence_tol": 1e-10,
                "nl_divergence_tol": 1e5,
            },
            # Flow parameters
            source_scalar_borehole_shearzone={"shearzone": "S1_2", "borehole": "INJ1",},
            well_cells=nd_and_shearzone_injection_cell,
            injection_rate=(1 / 6)
            / 3,  # = 10 l/min  - Divide by 3 due to 3 injection cells.
            frac_permeability=[2.3e-15, 4.5e-17, 2.3e-17, 6.4e-14, 1e-16],
            intact_permeability=2e-20,
        )
        setup = NeverFailtBiotCM(params)
        newton_params = params.newton_options
        pp.run_time_dependent_model(setup, newton_params)


class NeverFailtBiotCM(ISCBiotContactMechanics):
    def after_newton_failure(self, solution, errors, iteration_counter):
        """ Instead of raising error on failure, simply continue.
        """
        logger.error("Newton iterations did not converge")
        self.after_newton_convergence(solution, errors, iteration_counter)
