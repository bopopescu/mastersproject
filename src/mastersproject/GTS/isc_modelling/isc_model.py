import logging
import time
from typing import Dict

import numpy as np

import porepy as pp
from GTS.isc_modelling.ISCGrid import create_grid
from GTS.isc_modelling.contact_mechanics_biot import ContactMechanicsBiotBase
from GTS.isc_modelling.parameter import BiotParameters

from mastersproject.util.logging_util import trace, timer

logger = logging.getLogger(__name__)


class ISCBiotContactMechanics(ContactMechanicsBiotBase):
    def __init__(self, params: BiotParameters):
        super().__init__(params)

        # -- GRAVITY OPTIONS

        # TODO: Add these options to the MechanicsParameters
        # Turn on/off mechanical gravity term
        self._gravity_src = params.dict().get("_gravity_src", False)

        # Turn on/off gravitational effects on (Neumann) mechanical boundary conditions
        self._gravity_bc = params.dict().get("_gravity_bc", False)

    # --- Grid methods ---

    def create_grid(self):
        """
        Method that creates a GridBucket of a 2D domain with one fracture and sets
        projections to local coordinates for all fractures.

        The method requires the following attribute:
            mesh_args (dict): Containing the mesh sizes.

        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            bounding_box (dict): The SCALED bounding box of the domain,
                defined through minimum and maximum values in each dimension.
            Nd (int): The dimension of the matrix, i.e., the highest dimension in the
                grid bucket.
            network (pp.FractureNetwork3d): The fracture network associated to
                the domain.

        After self.gb is set, the method should also call

            pp.contact_conditions.set_projections(self.gb)

        """

        # Create grid
        gb, network = create_grid(
            **self.params.dict(
                include={
                    "mesh_args",
                    "length_scale",
                    "bounding_box",
                    "shearzone_names",
                    "folder_name",
                }
            )
        )
        self.gb = gb
        self.bounding_box = gb.bounding_box(as_dict=True)
        self.network = network

    @property
    def gb(self) -> pp.GridBucket:
        return self._gb

    @gb.setter
    def gb(self, gb: pp.GridBucket):
        """ Set a grid bucket to the class
        """
        self._gb = gb
        if gb is None:
            return
        pp.contact_conditions.set_projections(self.gb)
        self.gb.add_node_props(keys=["name"])  # Add 'name' as node prop to all grids.

        # Set the bounding box
        self.bounding_box = gb.bounding_box(as_dict=True)

        # Set Nd grid name
        self.gb.set_node_prop(self._nd_grid(), key="name", val=self.params.intact_name)

        # Set fracture grid names
        if self.params.n_frac > 0:
            fracture_grids = self.gb.get_grids(lambda g: g.dim == self.Nd - 1)
            assert (
                len(fracture_grids) == self.params.n_frac
            ), "There should be equal number of Nd-1 fractures as shearzone names"
            # We assume that order of fractures on grid creation (self.create_grid)
            # is preserved.
            for i, sz_name in enumerate(self.params.shearzone_names):
                self.gb.set_node_prop(fracture_grids[i], key="name", val=sz_name)

    def grids_by_name(self, name, key="name") -> np.ndarray:
        """ Get grid by grid bucket node property 'name'

        """
        gb = self.gb
        grids = gb.get_grids(lambda g: gb.node_props(g, key) == name)

        return grids

    @trace(logger, timeit=False, level="INFO")
    def well_cells(self) -> None:
        """
        Tag well cells with unity values, positive for injection cells and
        negative for production cells.
        """
        # Initiate all tags to zero
        for g, d in self.gb:
            tags = np.zeros(g.num_cells)
            g.tags["well_cells"] = tags
            pp.set_state(d, {"well": tags.copy()})

        # Set injection cells
        if self.params.well_cells:
            self.params.well_cells(self.params, self.gb)

    # ---- FLOW -----

    def biot_alpha(self, g: pp.Grid) -> float:  # noqa
        # if g.dim == self.Nd:
        #     return self.params.alpha
        # else:
        #     # Set to zero to turn off effects of DivUCoupling in 2d fractures.
        #     return 0.0
        return self.params.alpha

    # --- Aperture related methods ---

    def specific_volume(
        self, g: pp.Grid, scaled=False, from_iterate=True
    ) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in codimensions 2 and 3.
        """
        a = self.aperture(g, scaled, from_iterate)
        return np.power(a, self.Nd - g.dim)

    def aperture(
        self, g: pp.Grid, scaled: bool, from_iterate: bool = True
    ) -> np.ndarray:
        """ Compute the total aperture of each cell on a grid

        Parameters
        ----------
        g : pp.Grid
            grid
        scaled : bool
            whether to scale the aperture, which has units [m]
        from_iterate : bool
            whether to compute mechanical aperture from iterate.


        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        if g.dim == self.Nd - 2:
            fracs = self.gb.grids_of_dimension(self.Nd-1)
            aps = [self.aperture(f, scaled, from_iterate) for f in fracs]
            max_ap = np.max(np.hstack(aps))
            return np.ones(g.num_cells) * max_ap

        a_init = self.compute_initial_aperture(g, scaled=scaled)
        a_mech = self.mechanical_aperture(g, from_iterate=from_iterate)
        if not scaled:
            # Mechanical aperture is scaled by default, whereas initial aperture (above)
            # is not. This "upscaling" ensures they are consistent.
            a_mech *= self.params.length_scale / pp.METER

        # TODO: This is commented out to test effects of aperture variations in injection cell.
        # # Find the cells that are not injection (or sink) cells
        # injection_tags = g.tags["well_cells"]
        # not_injection_cells = 1 - np.abs(injection_tags)

        # Only add the mechanical contribution to non-injection cells:
        a_total = a_init + a_mech  # * not_injection_cells
        return a_total

    def compute_initial_aperture(self, g: pp.Grid, scaled) -> np.ndarray:
        """ Fetch the initial aperture. See __init__ for details """
        aperture = np.ones(g.num_cells)
        nd = self.Nd
        gb = self.gb

        # Get the aperture in the corresponding shearzone (is 1 for 3D matrix)
        if g.dim == nd:
            return aperture
        elif g.dim == nd - 1:
            shearzone = gb.node_props(g, "name")
            aperture *= self.params.initial_fracture_aperture[shearzone]
        elif g.dim == nd - 2:
            master_grids = gb.node_neighbors(g, only_higher=True)
            shearzones = (gb.node_props(g_h, "name") for g_h in master_grids)
            apertures = np.fromiter(
                (self.params.initial_fracture_aperture[sz] for sz in shearzones),
                dtype=float,
                count=len(master_grids),
            )
            aperture *= np.mean(apertures)
        else:
            raise ValueError("Not implemented 1d intersection points")

        if scaled:
            aperture *= pp.METER / self.params.length_scale

        return aperture

    def mechanical_aperture(self, g: pp.Grid, from_iterate: bool) -> np.ndarray:
        """ Compute aperture contribution from mechanical opening of fractures

        Parameters
        ----------
        g : pp.Grid
            a grid
        from_iterate : bool
            whether to fetch displacement jump from from iterate or from pp.STATE.

        Returns
        -------
        aperture : np.ndarray
            Apertures for each cell in the grid.

        Notes
        -----
        This method relies on a current (pp.STATE) or previous ('previous_iterate')
        being set for the mortar displacement variable.
        In case pp.STATE doesn't exist, uniformly 0 is returned.
        Otherwise, we require values for the mortar displacement variable.
        One way to ensure this in practice, is to set initial conditions
        before discretizations
        """
        nd = self.Nd
        gb = self.gb
        nd_grid = self._nd_grid()

        def _aperture_from_edge(data: Dict):
            """ Compute the mechanical contribution to the aperture
            for a given fracture. data is the edge data.
            """
            # Get normal displacement component from solution
            u_mortar_local = self.reconstruct_local_displacement_jump(
                data, from_iterate=from_iterate,
            )
            # Jump distances in each cell
            normal_jump = np.abs(u_mortar_local[-1, :])
            # tangential_jump = np.linalg.norm(u_mortar_local[:-1, :], axis=0)

            aperture_contribution = normal_jump
            return aperture_contribution

        # -- Computations --

        # No contribution in 3d
        if g.dim == nd:
            return np.zeros(g.num_cells)

        master_grids = gb.node_neighbors(g, only_higher=True)
        n_edges = len(master_grids)
        de = [gb.edge_props((g, e)) for e in master_grids]
        initialized = np.alltrue(
            np.fromiter((pp.STATE in d for d in de), dtype=bool, count=n_edges)
        )
        if not initialized:
            return np.zeros(g.num_cells)

        # In fractures
        elif g.dim == nd - 1:
            data_edge = gb.edge_props((g, nd_grid))
            aperture = _aperture_from_edge(data_edge)
            return aperture

        # TODO: Think about how to accurately do aperture computation (and specific volume)
        #  for fracture intersections where either side of fracture has different aperture.
        # In fracture intersections
        elif g.dim == nd - 2:
            # (g is the slave grid.)
            # Fetch edges of g that points to a higher-dimensional grid
            intx_edges = ((g, g_h) for g_h in master_grids)
            nd_edges = ((g_h, nd_grid) for g_h in master_grids)
            cell_faces = (g_h.cell_faces for g_h in master_grids)

            data_edges = (gb.edge_props(edge) for edge in nd_edges)
            master_apertures = (
                _aperture_from_edge(data_edge) for data_edge in data_edges
            )

            # Map parent apertures to faces
            master_face_apertures = (
                np.abs(cell_face) * parent_aperture
                for cell_face, parent_aperture in zip(cell_faces, master_apertures)
            )

            # .. and project face apertures to slave grid
            mortar_grids = (gb.edge_props(edge)["mortar_grid"] for edge in intx_edges)

            # Use _int() here to sum apertures, then we average at the end.
            master_to_slave_apertures = (
                mg.mortar_to_slave_int()
                * mg.master_to_mortar_int()
                * master_face_aperture
                for mg, master_face_aperture in zip(mortar_grids, master_face_apertures)
            )

            # expand the iterator
            apertures = np.zeros((n_edges, g.num_cells))
            for i, ap in enumerate(master_to_slave_apertures):
                apertures[i, :] = ap

            # average the apertures from master to determine the slave aperture
            avg_aperture = np.mean(
                apertures, axis=0
            )  # / 2  <-- divide by 2 for each mortar side???
            return avg_aperture

        else:
            raise ValueError("Not implemented 1d intersection points")

    # --- Flow parameter related methods ---

    def permeability(self, g, scaled) -> np.ndarray:
        """ Set (uniform) permeability in a subdomain"""
        # intact rock gets permeability from rock
        if g.dim == self.Nd:
            k = self.params.rock.PERMEABILITY * np.ones(g.num_cells)
            if scaled:
                k *= (pp.METER / self.params.length_scale) ** 2
        # fractures get permeability from cubic law
        else:

            def cubic_law(a):
                return np.power(a, 2) / 12

            aperture = self.aperture(g, scaled=scaled, from_iterate=True)
            k = cubic_law(aperture)
        return k

    @property
    def source_flow_rate(self) -> float:
        """ Scaled source flow rate """
        injection_rate = (
            self.params.injection_rate
        )  # 10 / 60  # 10 l/min  # injection rate [l / s], unscaled
        return (
            injection_rate * pp.MILLI * (pp.METER / self.params.length_scale) ** self.Nd
        )

    def source_scalar(self, g: pp.Grid) -> np.ndarray:
        """ Well-bore source (scaled)"""
        flow_rate = self.source_flow_rate  # scaled
        values = flow_rate * g.tags["well_cells"] * self.time_step
        return values

    def source_scalar_pressure(self, pressure):
        """ Pressure-controlled injection"""
        for g, d in self.gb:
            wells = g.tags["well_cells"]
            # Assume for now that only one grid injection cell
            if np.sum(np.abs(wells)) > 0:
                dof_ind = self.assembler.dof_ind(g, self.scalar_variable)
                loc = np.where(wells != 0)[0]
                glob_ind = dof_ind[loc]
                break

        A, b = self.assembler.assemble_matrix_rhs(matrix_format="csr")
        self.csr_zero_rows(A, glob_ind)
        A[glob_ind, glob_ind] = 1
        b[glob_ind] = pressure
        return A, b

    def csr_zero_rows(self, csr, rows_to_zero):
        """ Efficient way to set csr sparse matrix row to zero.
        https://stackoverflow.com/questions/19784868/
        what-is-most-efficient-way-of-setting-row-to-zeros-for-a-sparse-scipy-matrix/
        19800305#19800305"""
        rows, cols = csr.shape
        mask = np.ones((rows,), dtype=np.bool)
        mask[rows_to_zero] = False
        nnz_per_row = np.diff(csr.indptr)

        mask = np.repeat(mask, nnz_per_row)
        nnz_per_row[rows_to_zero] = 0
        csr.data = csr.data[mask]
        csr.indices = csr.indices[mask]
        csr.indptr[1:] = np.cumsum(nnz_per_row)

    # --- Simulation and solvers ---

    def _prepare_grid(self):
        """ Tag well cells right after creation.
        Called by self.prepare_simulation()
        """
        if self.gb is None:
            super()._prepare_grid()
        self.well_cells()  # tag well cells

    @timer(logger, level="INFO")
    def before_newton_iteration(self) -> None:
        # Re-discretize the nonlinear term
        super().before_newton_iteration()
        self.assembler.discretize(term_filter=["!grad_p", "!div_u", "!stabilization", "!mpsa"])
        # Pressure-controlled injection. See ISC Experiment Description (~avg. pressure used).

        # for g, _ in self.gb:
        #     if g.dim < self.Nd:
        #         self.assembler.discretize(grid=g)

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        super().after_newton_iteration(solution_vector)
        # Update Biot parameters using aperture from iterate
        # (i.e. displacements from iterate)
        self.set_biot_parameters()

    # -- For testing --

    def after_simulation(self):
        super().after_simulation()

        # Intro:
        summary_intro = f"Time of simulation: {time.asctime()}\n"

        # Get negative values
        g: pp.Grid = self._nd_grid()
        d = self.gb.node_props(g)
        p: np.ndarray = d[pp.STATE][self.p_exp]
        neg_ind = np.where(p < 0)[0]
        negneg_ind = np.where(p < -1e-10)[0]

        p_neg = p[neg_ind]

        summary_p_common = (
            f"\nInformation on negative values:\n"
            f"pressure values. "
            f"max: {np.max(p):.2e}. "
            f"Mean: {np.mean(p):.2e}. "
            f"Min: {np.min(p):.2e}\n"
        )
        if neg_ind.size > 0:
            summary_p = (
                f"{summary_p_common}"
                f"all negative indices: p<0: count:{neg_ind.size}, indices: {neg_ind}\n"
                f"very negative indices: p<-1e-10: count: {negneg_ind.size}, "
                f"indices: {negneg_ind}\n"
                f"neg pressure range: [{p_neg.min():.2e}, {p_neg.max():.2e}]\n"
            )
        else:
            summary_p = (
                f"{summary_p_common}"
                f"No negative pressure values. count:{neg_ind.size}\n"
            )

        self.neg_ind = neg_ind  # noqa
        self.negneg_ind = negneg_ind  # noqa

        # Condition number
        A, _ = self.assembler.assemble_matrix_rhs()  # noqa
        row_sum = np.sum(np.abs(A), axis=1)
        pp_cond = np.max(row_sum) / np.min(row_sum)
        diag = np.abs(A.diagonal())
        umfpack_cond = np.max(diag) / np.min(diag)

        summary_param = (
            f"\nSummary of relevant parameters:\n"
            f"length scale: {self.params.length_scale:.2e}\n"
            f"scalar scale: {self.params.scalar_scale:.2e}\n"
            f"3d permeability: "
            f"{np.mean(self.permeability(self._nd_grid(), scaled=False)):.2e}\n"
            f"time step: {self.time_step / pp.HOUR:.4f} hours\n"
            f"3d cells: {g.num_cells}\n"
            f"pp condition number: {pp_cond:.2e}\n"
            f"umfpack condition number: {umfpack_cond:.2e}\n"
        )

        scalar_parameters = d[pp.PARAMETERS][self.scalar_parameter_key]
        diffusive_term = scalar_parameters["second_order_tensor"].values[0, 0, 0]
        mass_term = scalar_parameters["mass_weight"][0]
        source_term = scalar_parameters["source"]
        nnz_source = np.where(source_term != 0)[0].size
        cv = g.cell_volumes
        summary_terms = (
            f"\nEstimates on term sizes, 3d grid:\n"
            f"diffusive term: {diffusive_term:.2e}\n"
            f"mass term: {mass_term:.2e}\n"
            f"source; max: {source_term.max():.2e}; "
            f"number of non-zero sources: {nnz_source}\n"
            f"cell volumes. "
            f"max:{cv.max():.2e}, "
            f"min:{cv.min():.2e}, "
            f"mean:{cv.mean():.2e}\n"
        )

        # Write summary to file
        summary_path = self.params.folder_name / "summary.txt"
        summary_text = summary_intro + summary_p + summary_param + summary_terms
        logger.info(summary_text)
        with summary_path.open(mode="w") as f:
            f.write(summary_text)

    # --- MECHANICS ---

    def faces_to_fix(self, g: pp.Grid):
        """ Fix some boundary faces to dirichlet to ensure unique solution to problem.

        Identify three boundary faces to fix (u=0). This should allow us to assign
        Neumann "background stress" conditions on the rest of the boundary faces.

        Credits: Keilegavlen et al (2019) - Source code.
        """

        all_bf, *_ = self.domain_boundary_sides(g)
        point = np.array(
            [
                [(self.bounding_box["xmin"] + self.bounding_box["xmax"]) / 2],
                [(self.bounding_box["ymin"] + self.bounding_box["ymax"]) / 2],
                [self.bounding_box["zmin"]],
            ]
        )
        distances = pp.distances.point_pointset(point, g.face_centers[:, all_bf])
        indexes = np.argpartition(distances, self.Nd)[: self.Nd]
        old_indexes = np.argsort(distances)
        assert np.allclose(
            np.sort(indexes), np.sort(old_indexes[: self.Nd])
        )  # Temporary: test new argpartition method
        faces = all_bf[indexes[: self.Nd]]
        return faces

    def bc_type_mechanics(self, g: pp.Grid) -> pp.BoundaryConditionVectorial:
        """ We set Neumann values on all but a few boundary faces.
        Fracture faces are set to Dirichlet.

        Three boundary faces (see self.faces_to_fix())
        are set to 0 displacement (Dirichlet).
        This ensures a unique solution to the problem.
        Furthermore, the fracture faces are set to 0 displacement (Dirichlet).
        """

        all_bf, *_ = self.domain_boundary_sides(g)
        faces = self.faces_to_fix(g)
        bc = pp.BoundaryConditionVectorial(g, faces, ["dir"] * len(faces))
        fracture_faces = g.tags["fracture_faces"]
        bc.is_neu[:, fracture_faces] = False
        bc.is_dir[:, fracture_faces] = True
        return bc

    def bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
        """ Mechanical stress values as ISC

        All faces are Neumann, except 3 faces fixed
        by self.faces_to_fix(g), which are Dirichlet.
        """
        # Retrieve the domain boundary
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)

        # Boundary values
        bc_values = np.zeros((g.dim, g.num_faces))

        # --- mechanical state ---
        # Get outward facing normal vectors for domain boundary, weighted for face area

        # 1. Get normal vectors on the faces. These are already weighed by face area.
        bf_normals = g.face_normals
        # 2. Adjust direction so they face outwards
        flip_normal_to_outwards = np.where(g.cell_face_as_dense()[0, :] >= 0, 1, -1)
        outward_normals = bf_normals * flip_normal_to_outwards
        bf_stress = np.dot(self.params.stress, outward_normals[:, all_bf])
        bc_values[:, all_bf] += (
            bf_stress / self.params.scalar_scale
        )  # Mechanical stress

        # --- gravitational forces ---
        # See MechanicsParameters to turn on/off gravity effects (Default: OFF)
        if self.params.gravity_bc:
            lithostatic_bc = self._adjust_stress_for_depth(g, outward_normals)

            # NEUMANN
            bc_values[:, all_bf] += lithostatic_bc[:, all_bf] / self.params.scalar_scale

        # DIRICHLET
        faces = self.faces_to_fix(g)
        bc_values[:, faces] = 0  # / self.length scale

        return bc_values.ravel("F")

    def _adjust_stress_for_depth(self, g: pp.Grid, outward_normals):
        """ Compute a stress tensor purely accounting for depth.

        The true_stress_depth determines at what depth we consider
        the given stress tensor (self.stress) to be equal to
        the given value. This can in principle be any number,
        but will usually be zmin <= true_stress_depth <= zmax

        Need the grid g, and outward_normals (see method above).

        Returns the marginal **traction** for the given face (g.face_centers)

        # TODO This could perhaps be integrated directly in the above method.
        """
        # TODO: Only do computations over 'all_bf'.
        # TODO: Test this method
        true_stress_depth = self.bounding_box["zmax"] * self.params.length_scale

        # We assume the relative sizes of all stress components scale with sigma_zz.
        # Except if sigma_zz = 0, then we don't scale.
        if np.abs(self.params.stress[2, 2]) < 1e-12:
            logger.critical("The stress scaler is set to 0 since stress[2, 2] = 0")
            stress_scaler = np.zeros(self.params.stress.shape)
        else:
            stress_scaler = self.params.stress / self.params.stress[2, 2]

        # All depths are translated in terms of the assumed depth
        # of the given stress tensor.
        relative_depths = (
            g.face_centers[2] * self.params.length_scale - true_stress_depth
        )
        rho_g_h = self.params.rock.lithostatic_pressure(relative_depths)
        lithostatic_stress = stress_scaler.dot(np.multiply(outward_normals, rho_g_h))
        return lithostatic_stress

    def source_mechanics(self, g: pp.Grid) -> np.ndarray:
        """ Gravity term.

        Gravity points downward, but we give the term
        on the RHS of the equation, thus we take the
        negative (i.e. the vector given will be
        pointing upwards)
        """
        # See MechanicsParameters to turn on/off gravity effects (Default: OFF)
        if self.params.gravity_src:
            # Gravity term
            values = np.zeros((self.Nd, g.num_cells))
            scaling = self.params.length_scale / self.params.scalar_scale
            values[2] = self.params.rock.lithostatic_pressure(g.cell_volumes) * scaling
            return values.ravel("F")
        else:
            # No gravity term
            return np.zeros(self.Nd * g.num_cells)

    # --- Set mechanics parameters ---

    def rock_friction_coefficient(self, g: pp.Grid) -> np.ndarray:  # noqa
        """ The friction coefficient is uniform, and equal to 1.

        Assumes self.set_rock() is called
        """
        return np.ones(g.num_cells) * self.params.rock.FRICTION_COEFFICIENT

    def set_mechanics_parameters(self) -> None:
        """ Set the dilation angle for slip in fractures"""
        super().set_mechanics_parameters()
        gb = self.gb

        for g, d in gb:
            if g.dim == self.Nd - 1:
                params: pp.Parameters = d[pp.PARAMETERS]
                mech_params = {"dilation_angle": self.params.dilation_angle}
                params.update_dictionaries(
                    [self.mechanics_parameter_key], [mech_params],
                )
                

    @timer(logger, level="INFO")
    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """ Assemble a solve the linear system"""
        from pypardiso import spsolve

        #p = 4 * pp.MEGA * (pp.PASCAL / self.params.scalar_scale)
        #A, b = self.source_scalar_pressure(p)
        A, b = self.assembler.assemble_matrix_rhs()

        # Estimate condition number
        logger.info(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.info(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and "
            f"min {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )

        # UMFPACK Estimate of condition number
        sum_diag_abs_A = np.abs(A.diagonal())
        logger.info(
            f"UMFPACK Condition number estimate: "
            f"{np.min(sum_diag_abs_A) / np.max(sum_diag_abs_A) :.2e}"
        )

        if self.params.linear_solver == "direct":
            tic = time.time()
            logger.info("Solve Ax=b using scipy")
            # sol = spla.spsolve(A, b)
            sol = spsolve(A, b)  # pypardiso
            logger.info(f"Done. Elapsed time {time.time() - tic}")
            norm = np.linalg.norm(b - A * sol)
            logger.info(f"||b-Ax|| = {norm}")

            rhs_norm = np.linalg.norm(b)
            identical_zero = np.isclose(rhs_norm, 0) and np.isclose(norm, 0)
            rel_norm = norm / rhs_norm if not identical_zero else norm
            logger.info(f"||b-Ax|| / ||b|| = {rel_norm}")
            return sol

        else:
            raise ValueError(f"Unknown linear solver {self.params.linear_solver}")
