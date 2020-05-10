from pathlib import Path
from typing import Dict, List

import porepy as pp
import numpy as np

from GTS.ISC_data.fracture import fracture_network


def create_grid(
        mesh_args: Dict[str, float],
        length_scale: float,
        box: Dict[str, float],
        shearzone_names: List[str],
        viz_folder_name: str,
):
    """ Create a GridBucket of a 3D domain with fractures defined by the ISC data set.

    shearzone_names are used to give names to each fracture grid. We assume that the
    order of names appearing in shearzone_names is preserved as fracture grids are constructed.


    Parameters
    ----------
    mesh_args : Dict[float]
        Mesh arguments (unscaled)
    length_scale : float
        length scale coefficient
    box : Dict[str, float]
        bounding box of domain (unscaled)
    shearzone_names : List[str]
        names of ISC shearzones to include or None
    viz_folder_name : str
        Path to store grid files

    Returns
    -------
        gb : pp.GridBucket
            The produced grid bucket
        box : dict
            The SCALED bounding box of the domain, defined through
            minimum and maximum values in each dimension.
        network : pp.FractureNetwork3d
            fracture network

    """
    # Scale mesh args by length_scale:
    mesh_args = {k: v / length_scale for k, v in mesh_args.items()}
    # Scale bounding box by length_scale:
    box = {k: v / length_scale for k, v in box.items()}

    network = fracture_network(
        shearzone_names=shearzone_names,
        export_vtk=True,
        domain=box,
        length_scale=length_scale,
        network_path=f"{viz_folder_name}/fracture_network.vtu",
    )
    path = f"{viz_folder_name}/gmsh_frac_file"
    gb = network.mesh(mesh_args=mesh_args, file_name=path)

    pp.contact_conditions.set_projections(gb)

    # --- Set fracture grid names: ---
    # The 3D grid is tagged by 'None'
    # 2D fractures are tagged by their shearzone name (S1_1, S1_2, etc.)
    # 1D (and 0D) fracture intersections are tagged by 'None'.
    gb.add_node_props(keys=["name"])  # Add 'name' as node prop to all grids. (value is 'None' by default)
    fracture_grids = gb.get_grids(lambda _g: _g.dim == gb.dim_max() - 1)

    # Set node property 'name' to each fracture with value being name of the shear zone.
    if fracture_grids.size > 0:
        for i, sz_name in enumerate(shearzone_names):
            gb.set_node_prop(fracture_grids[i], key="name", val=sz_name)
            # Note: Use self.gb.node_props(g, 'name') to get value.

    return gb, box, network


def create_structured_grid(
        nx: np.ndarray,
        physdims: np.ndarray,
        length_scale: float,
):
    """ Create a structured 3d grid

    nx : np.ndarray
        Number of cells in (x,y,z)
    physdims : np.ndarray
        Physical dimensions of (x,y,z).
    length_scale : float
        Length scale of physical dimension.
    """
    gb = pp.meshing.cart_grid(
        [],
        nx=nx,
        physdims=physdims/length_scale,
    )
    return gb


def optimize_grid(in_file, out_file=None, method='', force=False, dim_tags=[]):
    """ Optimize a grid using an optimizer

    See: https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/api/gmsh.py#L1444

    Parameters
    ----------
    in_file : str
        path to .msh file to be optimized
    out_file : str
        output file. By default, in_file+"optimized"
    method : str
        name of optimizer.
        Default: '' (gmsh default tetrahedral optimizer)
        Other options:
            'Netgen': Netgen optimizer
            'HighOrder': direct high-order mesh optimizer
            'HighOrderElastic': high-order elastic smoother
            'HighOrderFastCurving': fast curving algorithm
            'Laplace2D': Laplace smoothing
            'Relocate2D': Node relocation, 2d
            'Relocate3D': Node relocation, 3d
    force : bool
        If set, apply the optimization also to discrete entities
    dim_tags : List
        If supplied, only apply the optimizer to the given entities

    """
    assert Path(in_file).is_file()

    import gmsh
    gmsh.initialize()
    gmsh.open(in_file)

    gmsh.model.mesh.optimize(method=method, force=force, dimTags=dim_tags)




