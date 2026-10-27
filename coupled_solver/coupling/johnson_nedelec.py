import numpy as np
import dolfinx
import ufl
import bempp_cl.api

from bempp_cl.api.external import fenicsx
from dolfinx import fem, default_scalar_type
from scipy.sparse import coo_matrix

def _compute_trace_matrix(fem_space, bm_nodes):
    """
    Matriz T (Nv_bnd x Ndof_FEM) que extrae el valor nodal en vértices de frontera,
    con filas en el mismo orden que bm_nodes.
    """
    mesh = fem_space.mesh
    tdim = mesh.topology.dim

    bm_nodes = np.asarray(bm_nodes, dtype=np.int32)

    # Conectividades necesarias
    mesh.topology.create_connectivity(0, tdim)   # v -> cell
    mesh.topology.create_connectivity(tdim, 0)   # cell -> v

    v2c = mesh.topology.connectivity(0, tdim)
    c2v = mesh.topology.connectivity(tdim, 0)

    imap = fem_space.dofmap.index_map
    bs = fem_space.dofmap.index_map_bs
    num_fem_dofs = (imap.size_local + imap.num_ghosts) * bs

    dof_layout = fem_space.dofmap.dof_layout

    rows = np.empty(len(bm_nodes), dtype=np.int64)
    cols = np.empty(len(bm_nodes), dtype=np.int64)
    data = np.ones(len(bm_nodes), dtype=np.float64)

    for i, v in enumerate(bm_nodes):
        cells = v2c.links(int(v))
        if len(cells) == 0:
            raise RuntimeError(f"Vértice {v} no tiene celdas incidentes (¿id incorrecto?).")

        c = int(cells[0])
        cell_verts = c2v.links(c)

        # encontrar índice local del vértice dentro de la celda
        lv = int(np.where(cell_verts == v)[0][0])

        cell_dofs = fem_space.dofmap.cell_dofs(c)
        ldofs = dof_layout.entity_dofs(0, lv)  # dof(s) del vértice local lv
        if len(ldofs) != 1:
            raise RuntimeError("Esperaba 1 dof por vértice (Lagrange escalar).")

        dof = int(cell_dofs[ldofs[0]])

        rows[i] = i
        cols[i] = dof

    T = coo_matrix((data, (rows, cols)), shape=(len(bm_nodes), num_fem_dofs)).tocsc()
    return T

