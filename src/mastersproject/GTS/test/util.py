""" Common util methods for tests"""
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

from porepy.models.contact_mechanics_model import ContactMechanics
from GTS.isc_modelling.contact_mechanics_biot import ContactMechanicsBiotISC
from GTS.isc_modelling.mechanics import ContactMechanicsISC
import pendulum

import GTS as gts
from util.logging_util import (
    __setup_logging,
    timer,
    trace,
)
logger = logging.getLogger(__name__)


def prepare_setup(
        model: Type[ContactMechanicsISC],
        path_head: str,
        params: dict = None,
        prepare_simulation: bool = True,
        setup_loggers: bool = True,
) -> ContactMechanicsISC:
    """ Helper method to create grids, etc. for test methods

    Parameters
    ----------
    model : Type[ContactMechanicsISC]
        Which model to setup
        Subclass of porepy.models.abstract_model.AbstractModel
    path_head : str
        folder structure to store results in.
        Computed relative to:
            f".../GTS/test/{path_head}"
    params : dict (Default: None)
        Update or pass additional model parameters
    prepare_simulation : bool (Default: True)
        Whether to run setup.prepare_simulation()
    setup_loggers : bool (Default: True)
        Whether to setup loggers

    Returns
    -------
    setup : ContactMechanicsISC
        Instance of ContactMechanicsISC or subclass
    """

    in_params = prepare_params(
        path_head=path_head,
        params=params,
        setup_loggers=setup_loggers,
    )

    setup = model(in_params)
    if prepare_simulation:
        setup.prepare_simulation()
    return setup


@trace(logger, timeit=False, level='DEBUG')
def prepare_params(
        path_head: str,
        params: dict,
        setup_loggers: bool,
) -> dict:
    """
    Method to prepare default set of parameters for model runs.

    Parameters
    ----------
    path_head : str
        folder structure to store results in.
        Computed relative to:
            '...GTS/test/{test_method_name}/{path_head}'
    params : dict
        Update or pass additional parameters to params
    setup_loggers: bool
        Whether to setup loggers
        (usually false for subsequent calls to this method in one test run)

    Returns
    -------
    default_params : dict
        model parameters
    """

    if params is None:
        params = {}

    _this_file = Path(os.path.abspath(__file__)).parent
    _results_path = _this_file / f"results/{path_head}"
    _results_path.mkdir(parents=True, exist_ok=True)  # Create path if not exists
    if setup_loggers:
        __setup_logging(_results_path)
        logger.info(f"Set up logger at {pendulum.now().to_atom_string()}")
    logger.info(f"Path to results: {_results_path}")

    # --- DOMAIN ARGUMENTS ---
    default_params = {
        'mesh_args':
            {'mesh_size_frac': 10, 'mesh_size_min': .1 * 10, 'mesh_size_bound': 6 * 10},
        'bounding_box':
            {'xmin': -20, 'xmax': 80, 'ymin': 50, 'ymax': 150, 'zmin': -25, 'zmax': 75},
        'shearzone_names':
            ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"],
        'folder_name':
            _results_path,
        'solver':
            'direct',
        'source_scalar_borehole_shearzone':  # Only relevant for biot
            {"shearzone": "S1_2", "borehole": "INJ1"},
        'stress':
            gts.isc_modelling.stress_tensor(),
        'length_scale':
            1,
        'scalar_scale':
            1,
    }
    default_params.update(params)

    return default_params
