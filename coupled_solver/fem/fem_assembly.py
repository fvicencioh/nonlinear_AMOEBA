import numpy as np
import dolfinx

from dolfinx import fem
from dolfinx.geometry import compute_colliding_cells, compute_collisions_points

def _get_fem_spaces(mesh):
    fenics_space = fem.functionspace(mesh, ('Lagrange', 2)) # Espacio de funciones para FEM (H1)
    P1_space = fem.functionspace(mesh, ('Lagrange', 1)) # Espacio para la condición de Neumann
    return fenics_space, P1_space

def _get_local_coordinates(vertices, point):
    """Get the local coordinates of the point in the cell."""
    origin = vertices[0]
    axes = [v - origin for v in vertices[1:]]
    tdim = 3
    if len(axes) == 2:
        axes.append(np.cross(axes[0], axes[1]))
        tdim = 2
    
    assert len(axes) == 3
    
    return np.linalg.solve(np.array(axes).T, point - origin)[:tdim]

def _locate_cell(point, mesh, tree):

    '''
    Encuentra que tetrahedro contiene al punto ingresado

    point = Arreglo 3x1 con el punto buscado.
    mesh = Malla volumétrica donde buscar.

    Retorna:
        cell = N° del tetrahedro que contiene el punto
    '''
    cell_candidates = compute_collisions_points(tree, point)
    if len(compute_colliding_cells(mesh, cell_candidates, point).array)==0:
        return None
    else:
        cell = compute_colliding_cells(mesh, cell_candidates, point).array[0]

        return cell
    
def _compute_reaction_potential(phi_rf, x_q, fem_space):
    """
    Compute the reaction potential in the multipoles location
    using shape functions
    
    """
    N = x_q.shape[0]
    phi = np.zeros(N)
    mesh = fem_space.mesh

    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)

    basix_el = fem_space.element.basix_element
    dofmap = fem_space.dofmap

    D000 = 0 # Indice de funciones de forma base

    for k in range(N):
        x = x_q[k]

        cell = _locate_cell(x, mesh, tree)

        if cell == None:
            continue

        verts_idx = mesh.geometry.dofmap[cell] # dof del tetrahedro que contiene el multipolo
        verts = np.array([mesh.geometry.x[i] for i in verts_idx], dtype=np.float64)  # Coordenadas de los vertices. (4,3)
        xi = _get_local_coordinates(verts, x)  # Coordenadas locales de los vertices (4,3)

        tab = basix_el.tabulate(0, np.array([xi], dtype=np.float64)) # Funciones de forma

        shape_fun = tab[D000, 0, :, 0]
        dofs = fem_space.dofmap.cell_dofs(cell)

        for d in range(len(dofs)):
            phi[k] += phi_rf[dofs[d]] * shape_fun[d]
            
    return phi

def _compute_solvent_derivatives(fem_space, x_q, phi_rf):
    """
    
    Compute the first and second derivative of the reaction potential in the multipoles location
    using shape functions derivatives
    
    """
    N     = x_q.shape[0]
    dphi  = np.zeros((N, 3))
    ddphi = np.zeros((N, 3, 3))
    mesh  = fem_space.mesh

    fem_ndof = fem_space.dofmap.index_map.size_global

    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)

    basix_el = fem_space.element.basix_element
    dofmap = fem_space.dofmap

    # Indices de derivadas
    # (1,0,0)=1, (0,1,0)=2, (0,0,1)=3
    # (2,0,0)=4, (1,1,0)=5, (1,0,1)=6, (0,2,0)=7, (0,1,1)=8, (0,0,2)=9

    D100 = 1; D010 = 2; D001 = 3 # Indices de primeras derivadas
    D200 = 4; D110 = 5; D101 = 6; D020 = 7; D011 = 8; D002 = 9 # Indices de segundas derivadas

    for k in range(N):
        x = x_q[k]
        cell = _locate_cell(x, mesh, tree)

        if cell == None:
            continue

        verts_idx = mesh.geometry.dofmap[cell] # dof del tetrahedro que contiene el multipolo       
        verts = np.array([mesh.geometry.x[i] for i in verts_idx], dtype=np.float64)  # Coordenadas de los vertices. (4,3)
        xi = _get_local_coordinates(verts, x)  # Coordenadas locales de los vertices (4,3)

        tab = basix_el.tabulate(2, np.array([xi], dtype=np.float64)) # Funciones de forma hasta la segunda derivada
        dofs = fem_space.dofmap.cell_dofs(cell)

        #--- Calculo de Jacobiano ---#

        X0 = verts[0]
        J = np.column_stack((verts[1]-X0, verts[2]-X0, verts[3]-X0))  # Jacobiano. (3,3)
        invJ = np.linalg.inv(J) # J^-1
        invJT = invJ.T # J^-T

        g_ref = np.vstack([         # Valor de la primera derivada de las funciones de forma. (ndofs, 3)
            tab[D100, 0, :, 0],
            tab[D010, 0, :, 0],
            tab[D001, 0, :, 0],
        ])

        H_ref = np.zeros((3, 3, len(dofs)), dtype=np.float64)   # Segunda derivada de las funciones de forma. (ndofs, 3, 3)
        H_ref[0, 0, :] = tab[D200, 0, :, 0]
        H_ref[0, 1, :] = tab[D110, 0, :, 0]
        H_ref[0, 2, :] = tab[D101, 0, :, 0]
        H_ref[1, 0, :] = H_ref[0, 1, :]
        H_ref[1, 1, :] = tab[D020, 0, :, 0]
        H_ref[1, 2, :] = tab[D011, 0, :, 0]
        H_ref[2, 0, :] = H_ref[0, 2, :]
        H_ref[2, 1, :] = H_ref[1, 2, :]
        H_ref[2, 2, :] = tab[D002, 0, :, 0]

        g_phys = invJT @ g_ref  # (3, ndofs)
        H_phys = np.zeros_like(H_ref)
        for a in range(len(dofs)):
            H_phys[:, :, a] = invJT @ H_ref[:, :, a] @ invJ

        H_phys = np.transpose(H_phys, (2, 0, 1))
        for d in range(len(dofs)):
            dphi[k]  += phi_rf[dofs[d]] * g_phys.T[d, :]
            ddphi[k] += phi_rf[dofs[d]] * H_phys[d, :]

    return dphi, ddphi
