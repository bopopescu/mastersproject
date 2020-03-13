import logging
from typing import (  # noqa
    Any,
    Coroutine,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import os
from pathlib import Path

import porepy as pp
import numpy as np
from porepy.models.contact_mechanics_biot_model import ContactMechanicsBiot
from GTS.isc_modelling.contact_mechanics_biot import ContactMechanicsBiotISC
import pendulum

import GTS as gts
from src.mastersproject.util.logging_util import (
    __setup_logging,
    timer,
    trace,
)
from refinement import gb_coarse_fine_cell_mapping
from refinement.convergence import grid_error
import GTS.test.util as test_util

logger = logging.getLogger(__name__)


@trace(logger=logger)
def test_unit_convergence_study():
    """ Unit test for convergence study

    Simple prototype-like setup.
    """

    # 1. Prepare parameters
    stress = gts.isc_modelling.stress_tensor()
    hydrostatic = np.mean(np.diag(stress)) * np.ones(stress.shape[0])
    stress = np.diag(hydrostatic)

    params = {
        "mesh_args":
            {
                'mesh_size_frac': 10,
                'mesh_size_min': .1 * 10,
                'mesh_size_bound': 6 * 10,
            },
        "stress":
            stress,
        "shearzone_names":
            None,  # ['S1_1'],
    }

    n_refinements = 2

    # Run ContactMechanicsISC model
    gb_list, errors = gts.isc_modelling.setup.run_models_for_convergence_study(
        model=gts.ContactMechanicsISC,
        run_model_method=pp.run_stationary_model,
        params=params,
        n_refinements=n_refinements,
        variable=['u'],
        variable_dof=[3],
    )

    return gb_list, errors







