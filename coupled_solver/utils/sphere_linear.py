import numpy as np
import gmsh
import dolfinx
import ufl
import bempp_cl.api
import petsc4py
import scipy.sparse
import os

from bempp_cl.api.external import fenicsx
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import io
from numba import jit

from dolfinx.geometry import compute_colliding_cells, compute_collisions_points
from scipy.sparse import coo_matrix, diags
from petsc4py import PETSc

from coupled_solver.utils.generate_mesh import load_dolfin_mesh
from coupled_solver.utils.generate_mesh import build_boundary_mesh
from coupled_solver.utils.check_charge_mesh_distance import check_charge_mesh_distance

from bempp_cl.api.operators.boundary import sparse, laplace, modified_helmholtz
from bempp_cl.api.assembly.blocked_operator import BlockedDiscreteOperator
from bempp_cl.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
from bempp_cl.api.operators.potential.laplace import single_layer, double_layer

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres

def coulomb_potential(q, p, Q, xq, E):
    """
    Computes the electrostatic potential due to a point monopole, dipole, 
    quadrupole distribution at the position of the multipoles.
    See equation 29 of amoeba bem document. 
    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        phi: electrostatic potential at the position of the multipoles
    """

    phi = np.zeros(len(xq))
    T2  = np.zeros((len(xq),3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))

        for j in np.where(Rnorm>1e-10)[0]: #remove singularity
            T0 = 1/Rnorm[j]
            T1 = Ri[j,:]/Rnorm[j]**3
            T2[j,:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])/Rnorm[j]**5

            phi[i] += q[j]*T0 + np.sum(T1[:]*p[j,:]) + 0.5*np.sum(np.sum(T2[j,:,:]*Q[j,:,:],axis=1),axis=0)

    phi /= (4*np.pi*E)

    return phi

def coulomb_potential_thole(p, alpha, xq, E):
    """
    Computes the electrostatic potential due to a point dipole 
    distribution at the position of the multipoles.
    See equation 29 of amoeba bem document. 
    Uses Thole damping
    Inputs:
    ------- 
        p: array size (Nx3) with polarizable dipoles
        alpha: array size (Nx3x3) with polarizabilities (tensor)
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        phi: electrostatic potential at the position of the multipoles
    """

    phi = np.zeros(len(xq))
    T2  = np.zeros((len(xq),3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))

        for j in np.where(Rnorm>1e-10)[0]: #remove singularity

            # Thole damping for polarizable dipoles (valid for thole factor=1)
            damp = (alpha[i,0,0]*alpha[j,0,0])**(0.16666667)
            damp += 1e-12
            damp = -(Rnorm[j]/damp)**3
            expdamp = np.exp(damp)
            scale3 = 1 - expdamp
            scale5 = 1 - expdamp*(1-damp)

            T1 = Ri[j,:]/Rnorm[j]**3 * scale3

            phi[i] += np.sum(T1[:]*p[j,:]) 

    phi /= (4*np.pi*E)

    return phi

def coulomb_field(q, p, Q, xq, E):
    """
    Computes the electric field due to a point monopole, dipole, quadrupole distribution
    at the position of the multipoles. The field is defined as E=-nabla*phi.
    See equation 52 of kirkwood multipole, and Equation 30 of amoeba bem document.
    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        Efield: electric field at the position of the multipoles
    """
    Efield = np.zeros((len(xq),3))
    T0 = np.zeros((len(xq),3))
    T1 = np.zeros((len(xq),3,3))
    T2 = np.zeros((len(xq),3,3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))


        for j in np.where(Rnorm>1e-10)[0]: #remove singularity

            T0[j,:]   = -Ri[j,:]/Rnorm[j]**3
            T1[j,:,:] = np.identity(3)/Rnorm[j]**3 - 3*np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])/Rnorm[j]**5

            # the ordering in aux will be k,j,i looking at Eq 52 of kirkwood multipole
            aux = np.zeros((3,3,3))
            for k in range(3):
                aux[k,:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])*Ri[j,k]
            aux *= -5/Rnorm[j]**7

            for k in range(3):
                aux[:,:,k] += np.identity(3)*Ri[j,k]/Rnorm[j]**5
            for k in range(3):
                aux[:,k,:] += np.identity(3)*Ri[j,k]/Rnorm[j]**5

            T2[j,:,:,:] = aux

            for k in range(3):
                Efield[i,k] += T0[j,k]*q[j] + np.sum(T1[j,k,:]*p[j,:]) + 0.5*np.sum(np.sum(T2[j,k,:,:]*Q[j,:,:],axis=1),axis=0)

    Efield /= -(4*np.pi*E)
    return Efield

def coulomb_field_thole(q, p, Q, alpha, xq, E):
    """
    Computes the electric field due to a point monpole, dipole and quadrupole
    at the position of the multipoles. The field is defined as E=-nabla*phi.
    Uses Thole damping.
    See equation 52 of kirkwood multipole, and Equation 30 of amoeba bem document.
    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        alpha: array size (Nx3x3) with polarizabilities
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        Efield: electric field at the position of the multipoles
    """
    Efield = np.zeros((len(xq),3))
    T0 = np.zeros((len(xq),3))
    T1 = np.zeros((len(xq),3,3))
    T2 = np.zeros((len(xq),3,3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))


        for j in np.where(Rnorm>1e-10)[0]: #remove singularity

            # Thole damping for polarizable dipoles (valid for thole factor=1)
            damp = (alpha[i,0,0]*alpha[j,0,0])**(0.16666667)
            damp += 1e-12
            damp = -(Rnorm[j]/damp)**3
            expdamp = np.exp(damp)
            scale3 = 1 - expdamp
            scale5 = 1 - expdamp*(1-damp)
            scale7 = 1 - expdamp*(1-damp+0.6*damp*damp) 

            T0[j,:]   = -Ri[j,:]/Rnorm[j]**3 * scale3
            T1[j,:,:] = np.identity(3)/Rnorm[j]**3*scale3 - 3*np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])/Rnorm[j]**5*scale5

            # the ordering in aux will be k,j,i looking at Eq 52 of kirkwood multipole
            aux = np.zeros((3,3,3))
            for k in range(3):
                aux[k,:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])*Ri[j,k]
            aux *= -5/Rnorm[j]**7*scale7

            for k in range(3):
                aux[:,:,k] += np.identity(3)*Ri[j,k]/Rnorm[j]**5*scale5
            for k in range(3):
                aux[:,k,:] += np.identity(3)*Ri[j,k]/Rnorm[j]**5*scale5

            T2[j,:,:,:] = aux

            for k in range(3):
                Efield[i,k] += T0[j,k]*q[j] + np.sum(T1[j,k,:]*p[j,:]) + 0.5*np.sum(np.sum(T2[j,k,:,:]*Q[j,:,:],axis=1),axis=0) 

    Efield /= -(4*np.pi*E)
    return Efield

def coulomb_ddpotential(q, p, Q, xq, E):
    """
    Computes the second derivative of the electrostatic potential due 
    to a point monopole, dipole, and quadrupole distribution at the 
    position of the multipoles. See equation 29 and 43 of amoeba bem document. 
    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        ddphi: second derivative of electrostatic potential at the 
                position of the multipoles
    """
    ddphi = np.zeros((len(xq),3,3))
    T0 = np.zeros((len(xq),3,3))
    T1 = np.zeros((len(xq),3,3,3))
    T2 = np.zeros((len(xq),3,3,3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))

        for j in np.where(Rnorm>1e-10)[0]: #remove singularity
            T0[j,:,:] = -np.identity(3)/Rnorm[j]**3 + 3*np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])/Rnorm[j]**5

            # the ordering in aux will be k,j,i looking at Eq 52 of kirkwood multipole
            aux = np.zeros((3,3,3))
            for k in range(3):
                aux[k,:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])*Ri[j,k]
            aux *= 15/Rnorm[j]**7

            for k in range(3):
                aux[:,:,k] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5
                aux[:,k,:] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5
                aux[k,:,:] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5

            T1[j,:,:,:] = aux

            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            dkl = (k==l)*1.0
                            dkm = (k==m)*1.0
                            dkn = (k==n)*1.0
                            dlm = (l==m)*1.0
                            dln = (l==n)*1.0

                            T2[j,k,l,m,n] = -7*Ri[j,k]*Ri[j,l]*Ri[j,m]*Ri[j,n]/Rnorm[j]**2  \
                                           + Ri[j,m]*Ri[j,n]*dkl + Ri[j,l]*Ri[j,n]*dkm      \
                                           + Ri[j,m]*Ri[j,l]*dkn + Ri[j,k]*Ri[j,n]*dlm      \
                                           + Ri[j,m]*Ri[j,k]*dln                            \
                                           - (dkm*dln + dlm*dkn)*Rnorm[j]**2/5
            T2 *= -5/Rnorm[j]**7

            for k in range(3):
                for l in range(3):
                    ddphi[i,k,l] += T0[j,k,l]*q[j] + np.sum(T1[j,k,l,:]*p[j,:]) + 0.5*np.sum(np.sum(T2[j,k,l,:,:]*Q[j,:,:],axis=1),axis=0)
    
    ddphi /= (4*np.pi*E)

    return ddphi

def coulomb_ddpotential_thole(p, alpha, xq, E):
    """
    Computes the second derivative of the electrostatic potential due 
    to a point dipole distribution at the 
    position of the multipoles. 
    Uses Thole damping.
    See equation 29 and 43 of amoeba bem document. 
    Inputs:
    ------- 
        p: array size (Nx3) with dipoles
        alpha: array size (Nx3x3) with polarizabilities
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        ddphi: second derivative of electrostatic potential at the 
                position of the multipoles
    """
    ddphi = np.zeros((len(xq),3,3))
    T1 = np.zeros((len(xq),3,3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))

        for j in np.where(Rnorm>1e-10)[0]: #remove singularity

            # Thole damping for polarizable dipoles (valid for thole factor=1)
            damp = (alpha[i,0,0]*alpha[j,0,0])**(0.16666667)
            damp += 1e-12
            damp = -(Rnorm[j]/damp)**3
            expdamp = np.exp(damp)
            scale3 = 1 - expdamp
            scale5 = 1 - expdamp*(1-damp)
            scale7 = 1 - expdamp*(1-damp+0.6*damp*damp) 

            # the ordering in aux will be k,j,i looking at Eq 52 of kirkwood multipole
            aux = np.zeros((3,3,3))
            for k in range(3):
                aux[k,:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])*Ri[j,k]
            aux *= 15/Rnorm[j]**7 * scale7

            for k in range(3):
                aux[:,:,k] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5 * scale5
                aux[:,k,:] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5 * scale5
                aux[k,:,:] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5 * scale5

            T1[j,:,:,:] = aux

            for k in range(3):
                for l in range(3):
                    ddphi[i,k,l] += np.sum(T1[j,k,l,:]*p[j,:]) 
    
    ddphi /= (4*np.pi*E)

    return ddphi

def solvation_energy(q, p, Q, phi, dphi, ddphi):
    solv_energy = 2*np.pi*332.064*(np.sum(q*phi) + np.sum(p*dphi) + np.sum(Q*ddphi)/6)
    return solv_energy

def coulomb_energy_multipole(q, p_per, p_pol, Q, alpha, xq, E):
    """
    Computes the Coulomb energy from a collection of point
    multipoles, according to equation 38 of amoeba bem document.
    
    Inputs:
    ------ 
        q : array size N with charges of multipoles
        p_per : array size (Nx3) with permanent dipoles of multipoles
        p_pol : array size (Nx3) with polarizable dipoles of multipoles
        Q : array size (Nx3x3) with quadrupoles of multipoles
        xq: array size Nx3 with positions of multipoles
    p_perm: array size (Nx3) with permanent dipoles of multipoles
        E : float, dielectric constant
    Outputs:
    -------
        E_coul: (float) free energy  
    """
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    phi   = coulomb_potential(q, p_per, Q, xq, E)
    dphi  = -1*coulomb_field(q, p_per, Q, xq, E)
    ddphi = coulomb_ddpotential(q, p_per, Q, xq, E)

    dummy1 = np.zeros((len(q)))        # dummy charges
    dummy2 = np.zeros((len(q),3,3))    # dummy quadrupoles

    phi += coulomb_potential_thole(p_pol, alpha, xq, E)
    dphi += -1*coulomb_field_thole(dummy1, p_pol, dummy2, alpha, xq, E)
    ddphi += coulomb_ddpotential_thole(p_pol, alpha, xq, E)

    cons = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    E_coul = 0.5*cons*(np.sum(q*phi) + np.sum(np.sum(p_per*dphi,axis=1)) + np.sum(np.sum(np.sum(Q*ddphi,axis=2),axis=1))/6)

    return E_coul

def get_local_coordinates(vertices, point):
    """Get the local coordinates of the point in the cell."""
    origin = vertices[0]
    axes = [v - origin for v in vertices[1:]]
    tdim = 3
    if len(axes) == 2:
        axes.append(np.cross(axes[0], axes[1]))
        tdim = 2
    
    assert len(axes) == 3
    
    return np.linalg.solve(np.array(axes).T, point - origin)[:tdim]

def locate_cell(point, mesh, tree):

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

def compute_reaction_potential(phi_rf, x_q, fem_space):
    """
    Compute the reaction potential in the multipoles location
    
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

        cell = locate_cell(x, mesh, tree)

        if cell == None:
            continue

        verts_idx = mesh.geometry.dofmap[cell] # dof del tetrahedro que contiene el multipolo
        verts = np.array([mesh.geometry.x[i] for i in verts_idx], dtype=np.float64)  # Coordenadas de los vertices. (4,3)
        xi = get_local_coordinates(verts, x)  # Coordenadas locales de los vertices (4,3)

        tab = basix_el.tabulate(0, np.array([xi], dtype=np.float64)) # Funciones de forma

        shape_fun = tab[D000, 0, :, 0]
        dofs = fem_space.dofmap.cell_dofs(cell)

        for d in range(len(dofs)):
            phi[k] += phi_rf[dofs[d]] * shape_fun[d]
            
    return phi

def compute_solvent_derivatives(fem_space, x_q, phi_rf):
    """
    
    Compute the first and second derivative of the reaction potential in the multipoles location
    
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
        cell = locate_cell(x, mesh, tree)

        if cell == None:
            continue

        verts_idx = mesh.geometry.dofmap[cell] # dof del tetrahedro que contiene el multipolo       
        verts = np.array([mesh.geometry.x[i] for i in verts_idx], dtype=np.float64)  # Coordenadas de los vertices. (4,3)
        xi = get_local_coordinates(verts, x)  # Coordenadas locales de los vertices (4,3)

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

def coulomb_polarizable_dipole(q, p_per, Q, alpha, xq, E, dtol):
    """
    Computes polarized dipole component of a collection of polarizabe multipoles
    Used in eq 56 of kirkwood multipole
    Inputs:
    ------
        q: array size N with charges of multipoles
        p_per: array size (Nx3) with permanent dipoles of multipoles
        Q: array size (Nx3x3) with quadrupoles of multipoles
        alpha: array size (Nqx3x3) with polarizability of dipoles (considered as a tensor)
        xq: array size Nx3 with positions of multipoles
        E : float, dielectric constant
    Returns:
    -------
        p_pol: array size (Nx3) with polarizable component of dipoles
        Efield: array size (Nx3) with electrostatic field that polarized the multipoles 
    """
    p_pol      = np.zeros((len(xq),3))
    dipole_diff= 1. 
    p_pol_prev = np.ones((len(xq),3))

    iteration = 0
    SOR = 0.7
    while dipole_diff>dtol:
        iteration += 1
        p_tot = p_per + p_pol
    
        Efield = coulomb_field_thole(q, p_tot, Q, alpha, xq, E)
        
        for k in range(len(q)):
            p_pol[k] = p_pol[k]*(1-SOR) + np.dot(alpha[k],4*np.pi*Efield[k])*SOR # 4*pi because alpha in Tinker
                                                                           # comes in atomic units that
                                                                           # that already include the 1/4pi

        dipole_diff = np.max(np.sqrt(np.sum((p_pol-p_pol_prev)**2, axis = 1)))
        p_pol_prev = p_pol.copy()

    print('Took %i iterations for vacuum induced dipole to converge'%iteration)
    return p_pol, Efield

def compute_trace_matrix(fem_space, bm_nodes):
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

def gmres_callback(residual_norm):
        global residuals
        residuals.append(residual_norm)
        print(f"GMRES iter {len(residuals):4d} | residual = {residual_norm:.3e}")

@jit(nopython=True)
def electric_field_coulomb(x, x_q, q, p, Q, ep_in):
    dphi_x = 0.0
    dphi_y = 0.0
    dphi_z = 0.0
    for j in range(len(x_q)):
        dx = x[0] - x_q[j, 0]
        dy = x[1] - x_q[j, 1]
        dz = x[2] - x_q[j, 2]
        norm = np.sqrt(dx*dx + dy*dy + dz*dz)
        if norm > 1e-10:
            r2 = norm**2
            r3 = norm**3
            r5 = r3 * r2
            r7 = r5 * r2

            T0_x = -dx/r3
            T0_y = -dy/r3
            T0_z = -dz/r3
            dphi_x += T0_x * q[j]
            dphi_y += T0_y * q[j]
            dphi_z += T0_z * q[j]

            dot_rd = dx*p[j,0] + dy*p[j,1] + dz*p[j,2]
            T1_x = p[j, 0]/r3 - 3.0 * dx * dot_rd / r5
            T1_y = p[j, 1]/r3 - 3.0 * dy * dot_rd / r5
            T1_z = p[j, 2]/r3 - 3.0 * dz * dot_rd / r5

            dphi_x += T1_x
            dphi_y += T1_y
            dphi_z += T1_z

            Q_rr = (dx*dx*Q[j,0,0] + dx*dy*Q[j,0,1] + dx*dz*Q[j,0,2] +
                    dy*dx*Q[j,1,0] + dy*dy*Q[j,1,1] + dy*dz*Q[j,1,2] +
                    dz*dx*Q[j,2,0] + dz*dy*Q[j,2,1] + dz*dz*Q[j,2,2])

            Q_rx = dx * Q[j,0,0] + dy * Q[j,0,1] + dz * Q[j,0,2]
            Q_ry = dx * Q[j,1,0] + dy * Q[j,1,1] + dz * Q[j,1,2]
            Q_rz = dx * Q[j,2,0] + dy * Q[j,2,1] + dz * Q[j,2,2]

            T2_x = Q_rx/r5 - 2.5 * dx * Q_rr / r7
            T2_y = Q_ry/r5 - 2.5 * dy * Q_rr / r7
            T2_z = Q_rz/r5 - 2.5 * dz * Q_rr / r7

            dphi_x += T2_x
            dphi_y += T2_y
            dphi_z += T2_z

    dphi_x /= (4.0 * np.pi * ep_in)
    dphi_y /= (4.0 * np.pi * ep_in)
    dphi_z /= (4.0 * np.pi * ep_in)

    return -1*np.array([dphi_x, dphi_y, dphi_z])

class Sphere:

    def __init__(self, x_q, q, d_scaled, Q_scaled, alpha, ep_in=1., ep_ex=80., k=0.125):
        self.ep_in = ep_in
        self.ep_ex = ep_ex
        self.k = k
        bohr = 0.52917721067
        self.x_q = x_q
        self.q = q
        self.d = d_scaled * bohr
        self.Q = Q_scaled * 2 * bohr**2
        self.alpha = alpha
        self.N = x_q.shape[0]
        self.mu = np.zeros((self.N, 3))
        self.p = self.d + self.mu
        self.gmres_rtol = 1e-5
        self.gmres_atol = 0.
        self.SOR = 0.7
        self.dtol = 1e-3
        self.maxiter = 100

    def upload_mesh(self, external_mesh, gradation=0.1, probe_radius=1.4, stern_thickness=1.0, algorithm=10, grid_scale=2.0):
        self.mesh, self.cell_tags, self.facet_tags = load_dolfin_mesh(external_mesh, gradation, probe_radius, stern_thickness, algorithm, grid_scale)
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1
        self.mesh.topology.create_connectivity(0, self.tdim)
        self.mesh.topology.create_connectivity(self.tdim, 0)
        self.mesh.topology.create_entities(self.fdim)
        self.mesh.topology.create_connectivity(self.tdim, self.fdim)  # cell -> facet
        self.mesh.topology.create_connectivity(self.fdim, self.tdim)  # facet -> cell
        self.mesh.topology.create_connectivity(self.fdim, 0)     # facet -> vertex 
        self.mesh.topology.create_entity_permutations()
        self.ses_facets = self.facet_tags.find(1)
        self.stern_facets = self.facet_tags.find(2)
        self.solute_cells = self.cell_tags.find(1)
        self.stern_cells = self.cell_tags.find(2)

    def extract_bmesh(self):
        self.ses_mesh, self.ses_nodes = build_boundary_mesh(self.mesh, self.ses_facets, cell_tags=self.cell_tags, inner_marker=1)
        self.stern_mesh, self.stern_nodes = build_boundary_mesh(self.mesh, self.stern_facets)

    def build_function_spaces(self):
        self.fenics_space = fem.functionspace(self.mesh, ('Lagrange', 2)) # Espacio de funciones para FEM (H1)
        self.P1_space = fem.functionspace(self.mesh, ('Lagrange', 1)) # Espacio para la condición de Neumann
        self.dirichlet_space_ses = bempp_cl.api.function_space(self.ses_mesh, 'P', 1)
        self.neumann_space_ses = bempp_cl.api.function_space(self.ses_mesh, 'P', 1)
        self.dirichlet_space_stern = bempp_cl.api.function_space(self.stern_mesh, 'P', 1) # Espacios de funciones para BEM (H1/2)
        self.neumann_space_stern = bempp_cl.api.function_space(self.stern_mesh, 'P', 1)
        self.trace_matrix = compute_trace_matrix(self.fenics_space, self.stern_nodes)
        self.trace_ses = compute_trace_matrix(self.P1_space, self.ses_nodes)
        self.trace_stern = compute_trace_matrix(self.P1_space, self.stern_nodes)
        self.fem_ndof = self.fenics_space.dofmap.index_map.size_global
        self.bem_ndof = self.dirichlet_space_stern.global_dof_count
        self.total_ndof = self.fem_ndof + self.bem_ndof
        
        self.sol_prev = np.zeros(self.total_ndof)
        
        print("FEM dofs: {0}".format(self.fem_ndof))
        print("BEM dofs: {0}".format(self.bem_ndof))
        print("Total dofs: {0}".format(self.total_ndof))

    def extract_solute_dofs(self):
        self.x_dofs = self.fenics_space.tabulate_dof_coordinates() # ndof x 3
        self.solute_dofs = set() # Para almacenar los dof del volumen 1 sin repetir.
        for cell_idx in self.solute_cells:
            cell_ndofs = self.fenics_space.dofmap.cell_dofs(cell_idx)
            for dof in cell_ndofs:
                self.solute_dofs.add(dof)
        self.solute_dofs = list(self.solute_dofs)
        self.x_dofs_solute = self.x_dofs[self.solute_dofs]

    def potential_operators(self):
        x_dofs_solute, x_q = self.x_dofs_solute, self.x_q
        self.phi0_dofs = np.zeros(self.fem_ndof)
        
        self.slpo_dofs = single_layer(self.neumann_space_ses, np.transpose(x_dofs_solute))
        self.dlpo_dofs = double_layer(self.dirichlet_space_ses, np.transpose(x_dofs_solute))
        
        self.slpo_mult = single_layer(self.neumann_space_ses, np.transpose(x_q))
        self.dlpo_mult = double_layer(self.dirichlet_space_ses, np.transpose(x_q))

    def fem_functions(self):
        self.u = ufl.TrialFunction(self.fenics_space) # función de aproximación
        self.lam = ufl.TrialFunction(self.P1_space) # función P1 para los datos de frontera
        self.v = ufl.TestFunction(self.fenics_space) # Función de prueba
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags) # cell_tags almacena ambos volumenes
        self.ds = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.facet_tags) # En caso de necesitarlo, dS es la frontera interna
        self.ds_ext = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags) # ds es la frontera exterior, pinche fenics
        self.A = fenicsx.FenicsOperator(self.ep_in*(ufl.inner(ufl.nabla_grad(self.u), ufl.nabla_grad(self.v)))*self.dx(1) + self.ep_ex*(ufl.inner(ufl.nabla_grad(self.u), ufl.nabla_grad(self.v)))*self.dx(2) + self.ep_ex*self.k*self.k*ufl.inner(self.u, self.v)*self.dx(2))

    def bem_operators(self):
        self.Is = sparse.identity(self.dirichlet_space_ses, self.neumann_space_ses, self.neumann_space_ses)
        self.I = sparse.identity(self.dirichlet_space_stern, self.neumann_space_stern, self.neumann_space_stern)
        self.M = sparse.identity(self.neumann_space_stern, self.neumann_space_stern, self.dirichlet_space_stern)
        self.VL = laplace.single_layer(self.neumann_space_ses, self.dirichlet_space_ses, self.neumann_space_ses)
        self.KL = laplace.double_layer(self.dirichlet_space_ses, self.dirichlet_space_ses, self.neumann_space_ses)
        self.VY = modified_helmholtz.single_layer(self.dirichlet_space_stern, self.dirichlet_space_stern, self.dirichlet_space_stern, self.k);
        self.KY = modified_helmholtz.double_layer(self.neumann_space_stern, self.dirichlet_space_stern, self.dirichlet_space_stern, self.k);
        self.trace_op = LinearOperator(self.trace_matrix.shape, lambda x:self.trace_matrix * x)
        
    def build_mass_matrix(self):
        self.mass = self.lam * self.v * self.ds_ext(2)
        self.mass_matrix = fem.petsc.assemble_matrix(fem.form(self.mass))
        self.mass_matrix.assemble()
        
        iv, jv, kv = self.mass_matrix.getValuesCSR()
        self.mass_matrix_sparse = scipy.sparse.csr_matrix((kv, jv, iv), shape=self.mass_matrix.getSize())

    def build_lhs(self):
        blocks = [[None,None],[None,None]]
        
        blocks[0][0] = self.A.weak_form()
        blocks[0][1] = -self.ep_ex * self.mass_matrix_sparse @ self.trace_stern.T
        blocks[1][0] = (.5 * self.I - self.KY).weak_form() * self.trace_op
        blocks[1][1] = self.VY.weak_form()
        
        self.cterm_lhs = BlockedDiscreteOperator(np.array(blocks)) # Lado izquierdo de la ecuación
        
        P1 = diags(1./self.cterm_lhs[0,0].to_sparse().diagonal())
        
        P2 = InverseSparseDiscreteBoundaryOperator(
                bempp_cl.api.operators.boundary.sparse.identity(
                self.neumann_space_stern, self.neumann_space_stern, self.neumann_space_stern).weak_form())
        
        def apply_prec(x):
        
            m1 = P1.shape[0]
            m2 = P2.shape[0]
            n1 = P1.shape[1]
            n2 = P2.shape[1]
        
            res1 = P1.dot(x[:n1])
            res2 = P2.dot(x[n1:])
        
            return np.concatenate([res1, res2])
        
        p_shape = (P1.shape[0] + P2.shape[0], P1.shape[1] + P2.shape[1])
        self.P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('float64'))

    def build_perm_rhs(self):
        x_q, q, d, Q, ep_in = self.x_q, self.q, self.d, self.Q, self.ep_in
        @bempp_cl.api.real_callable
        def charges_fun_perm(x, n, i, result):
            phi_c = 0.0
            for j in range(len(x_q)):
                dx = x[0] - x_q[j, 0]
                dy = x[1] - x_q[j, 1]
                dz = x[2] - x_q[j, 2]
                norm = np.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 1e-10:
                    T0 = 1/norm
                    phi_c += q[j] * T0
        
                    T1_x = dx/(norm**3)
                    T1_y = dy/(norm**3)
                    T1_z = dz/(norm**3)
                    phi_c += T1_x*d[j,0] + T1_y*d[j,1] + T1_z*d[j,2]
        
                    T2_xx = (dx * dx)/(2*norm**5)
                    T2_xy = (dx * dy)/(2*norm**5)
                    T2_xz = (dx * dz)/(2*norm**5)
                    T2_yx = (dy * dx)/(2*norm**5)
                    T2_yy = (dy * dy)/(2*norm**5)
                    T2_yz = (dy * dz)/(2*norm**5)
                    T2_zx = (dz * dx)/(2*norm**5)
                    T2_zy = (dz * dy)/(2*norm**5)
                    T2_zz = (dz * dz)/(2*norm**5)
                    phi_c += T2_xx*Q[j,0,0] + T2_xy*Q[j,0,1] + T2_xz*Q[j,0,2]
                    phi_c += T2_yx*Q[j,1,0] + T2_yy*Q[j,1,1] + T2_yz*Q[j,1,2]
                    phi_c += T2_zx*Q[j,2,0] + T2_zy*Q[j,2,1] + T2_zz*Q[j,2,2]
            result[0] = phi_c/(4*np.pi*ep_in)
        
        self.G_fun_perm = bempp_cl.api.GridFunction(self.dirichlet_space_ses, fun = charges_fun_perm)
        
        @bempp_cl.api.real_callable
        def lambda_fun_perm(x, n, i, result):
            dphi_x = 0.0
            dphi_y = 0.0
            dphi_z = 0.0
            for j in range(len(x_q)):
                dx = x[0] - x_q[j, 0]
                dy = x[1] - x_q[j, 1]
                dz = x[2] - x_q[j, 2]
                norm = np.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 1e-10:
                    r2 = norm**2
                    r3 = norm**3
                    r5 = r3 * r2
                    r7 = r5 * r2
        
                    T0_x = -dx/r3
                    T0_y = -dy/r3
                    T0_z = -dz/r3
                    dphi_x += T0_x * q[j]
                    dphi_y += T0_y * q[j]
                    dphi_z += T0_z * q[j]
        
                    dot_rd = dx*d[j,0] + dy*d[j,1] + dz*d[j,2]
                    T1_x = d[j, 0]/r3 - 3.0 * dx * dot_rd / r5
                    T1_y = d[j, 1]/r3 - 3.0 * dy * dot_rd / r5
                    T1_z = d[j, 2]/r3 - 3.0 * dz * dot_rd / r5
        
                    dphi_x += T1_x
                    dphi_y += T1_y
                    dphi_z += T1_z
        
                    Q_rr = (dx*dx*Q[j,0,0] + dx*dy*Q[j,0,1] + dx*dz*Q[j,0,2] +
                            dy*dx*Q[j,1,0] + dy*dy*Q[j,1,1] + dy*dz*Q[j,1,2] +
                            dz*dx*Q[j,2,0] + dz*dy*Q[j,2,1] + dz*dz*Q[j,2,2])
        
                    Q_rx = dx * Q[j,0,0] + dy * Q[j,0,1] + dz * Q[j,0,2]
                    Q_ry = dx * Q[j,1,0] + dy * Q[j,1,1] + dz * Q[j,1,2]
                    Q_rz = dx * Q[j,2,0] + dy * Q[j,2,1] + dz * Q[j,2,2]
        
                    T2_x = Q_rx/r5 - 2.5 * dx * Q_rr / r7
                    T2_y = Q_ry/r5 - 2.5 * dy * Q_rr / r7
                    T2_z = Q_rz/r5 - 2.5 * dz * Q_rr / r7
        
                    dphi_x += T2_x
                    dphi_y += T2_y
                    dphi_z += T2_z
        
            dphi_x /= (4.0 * np.pi * ep_in)
            dphi_y /= (4.0 * np.pi * ep_in)
            dphi_z /= (4.0 * np.pi * ep_in)
            result[0] = n[0]*dphi_x + n[1]*dphi_y + n[2]*dphi_z
        
        dGdn_fun_perm = bempp_cl.api.GridFunction(self.dirichlet_space_ses, fun = lambda_fun_perm)
        
        #--- RHS permanente componente armónica ---#
        
        self.hterm_rhs_perm = -(0.5*self.Is + self.KL) * self.G_fun_perm
        
        #--- RHS permanente componente de corrección ---#
        self.bem_rhs = np.zeros(self.bem_ndof)
        
        mass_ses = self.lam('+') * self.v('+') * self.ds(1)
        mass_ses_matrix = fem.petsc.assemble_matrix(fem.form(mass_ses))
        mass_ses_matrix.assemble()
        iv, jv, kv = mass_ses_matrix.getValuesCSR()
        self.mass_ses_sparse = scipy.sparse.csr_matrix((kv, jv, iv), shape=mass_ses_matrix.getSize())
        
        cterm_rhs_perm = -(self.ep_in) * dGdn_fun_perm  # elimine el /ep_ex # Lo volví a incluir porque como no va a ir po felipe # no va po, estaba bien al principio
        neumann_values_perm = cterm_rhs_perm.coefficients
        
        self.fem_rhs_perm = self.mass_ses_sparse @ (self.trace_ses.T @ neumann_values_perm)

    def harmonic_component(self):
        x_q, mu, ep_in = self.x_q, self.mu, self.ep_in
        @bempp_cl.api.real_callable
        def charges_fun_var(x, n, i, result):
            phi_c = 0.0
            for j in range(len(x_q)):
                dx = x[0] - x_q[j, 0]
                dy = x[1] - x_q[j, 1]
                dz = x[2] - x_q[j, 2]
                norm = np.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 1e-10:
        
                    T1_x = dx/(norm**3)
                    T1_y = dy/(norm**3)
                    T1_z = dz/(norm**3)
                    phi_c += T1_x*mu[j,0] + T1_y*mu[j,1] + T1_z*mu[j,2]
        
            result[0] = phi_c/(4*np.pi*ep_in)
        
        self.G_fun_var = bempp_cl.api.GridFunction(self.dirichlet_space_ses, fun = charges_fun_var)
        hterm_rhs_var = -(0.5*self.Is + self.KL) * self.G_fun_var
        hterm_rhs = self.hterm_rhs_perm + hterm_rhs_var
        self.dphi0dn, info = bempp_cl.api.linalg.cg(self.VL, hterm_rhs, tol=1e-5)

    def correction_component(self):
        x_q, mu, ep_in = self.x_q, self.mu, self.ep_in
        global residuals
        @bempp_cl.api.real_callable
        def lambda_fun_var(x, n, i, result):
            dphi_x = 0.0
            dphi_y = 0.0
            dphi_z = 0.0
            for j in range(len(x_q)):
                dx = x[0] - x_q[j, 0]
                dy = x[1] - x_q[j, 1]
                dz = x[2] - x_q[j, 2]
                norm = np.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 1e-10:
                    r3 = norm**3
                    r5 = r3 * norm**2
        
                    dot_rd = dx*mu[j,0] + dy*mu[j,1] + dz*mu[j,2]
                    T1_x = mu[j, 0]/r3 - 3.0 * dx * dot_rd / r5
                    T1_y = mu[j, 1]/r3 - 3.0 * dy * dot_rd / r5
                    T1_z = mu[j, 2]/r3 - 3.0 * dz * dot_rd / r5
        
                    dphi_x += T1_x
                    dphi_y += T1_y
                    dphi_z += T1_z
        
            dphi_x /= (4.0 * np.pi * ep_in)
            dphi_y /= (4.0 * np.pi * ep_in)
            dphi_z /= (4.0 * np.pi * ep_in)
            result[0] = n[0]*dphi_x + n[1]*dphi_y + n[2]*dphi_z
        
        dGdn_fun_var = bempp_cl.api.GridFunction(self.dirichlet_space_ses, fun = lambda_fun_var)
        
        cterm_rhs_var = -(self.ep_in)*(dGdn_fun_var + self.dphi0dn) # eliminé el /ep_ex # Lo volví a incluir porque como no va a ir po felipe # no va po, estaba bien al principio
        neumann_values_var = cterm_rhs_var.coefficients
        
        fem_rhs_var = self.mass_ses_sparse @ (self.trace_ses.T @ neumann_values_var)
        fem_rhs_array = self.fem_rhs_perm + fem_rhs_var
        cterm_rhs = np.concatenate([fem_rhs_array, self.bem_rhs])
        
        residuals = []
        
        sol, info = gmres(
                    self.cterm_lhs,
                    cterm_rhs,
                    x0 = self.sol_prev,
                    rtol=self.gmres_rtol,
                    atol=self.gmres_atol,
                    M=self.P,
                    callback=gmres_callback,
                    callback_type="pr_norm",
                    restart=500,
                    maxiter=2000
                    )
        
        residual_real = np.linalg.norm(self.cterm_lhs @ sol - cterm_rhs) / np.linalg.norm(cterm_rhs)
        print(f"Residual relativo real: {residual_real:.3e}")
        
        self.sol_prev = sol.copy()
        
        self.phi_hat  = sol[:self.fem_ndof] # Componente de corrección en la ubicación de los dofs

    def reaction_potential(self):
        #--- 3. Determinar phi0 en la ubicación de los dofs y multipolos ---#
        
        self.G_fun = self.G_fun_perm + self.G_fun_var
        
        phi0_dofs_solvent = self.slpo_dofs * self.dphi0dn + self.dlpo_dofs * self.G_fun # Componente armónica en los dofs
        self.phi0_dofs[self.solute_dofs] = phi0_dofs_solvent[0]
        
        #--- 5. Se calcula el potencial de reacción en los dofs y multipolos ---#
        
        self.phi_rf_dofs = self.phi_hat + self.phi0_dofs

    def compute_derivatives(self):
        self.dphi_rf, self.ddphi_rf = compute_solvent_derivatives(self.fenics_space, self.x_q, self.phi_rf_dofs)

    def compute_multipoles_efield(self):
        self.p = self.d + self.mu # Dipolo total
        
        E_mult      = coulomb_field_thole(self.q, self.p, self.Q, self.alpha, self.x_q, self.ep_in)
        self.E_total     = E_mult - self.dphi_rf # Signo (-) porque E = -nabla phi

    def compute_mu(self):
        mu_prev = self.mu.copy()
        for i in range(self.N):
            self.mu[i] = self.mu[i]*(1-self.SOR) + np.dot(self.alpha[i], self.E_total[i]*4*np.pi)*self.SOR
        self.dipole_diff = np.max(np.sqrt(np.sum((self.mu-mu_prev)**2, axis = 1)))

    def compute_energy(self):
        phi_hat_mult = compute_reaction_potential(self.phi_hat, self.x_q, self.fenics_space)
        phi0_mult = self.slpo_mult * self.dphi0dn + self.dlpo_mult * self.G_fun
        phi_rf_mult = phi_hat_mult + phi0_mult[0]
        
        G_diss_solv = solvation_energy(self.q, self.d, self.Q, phi_rf_mult, self.dphi_rf, self.ddphi_rf)
        print(F"Solvent contributión: {G_diss_solv}")
        G_diss_mult = coulomb_energy_multipole(self.q, self.d, self.mu, self.Q, self.alpha, self.x_q, self.ep_in)
        print(F"Multipoles contributión: {G_diss_mult}")
        
        #--- Cálculo de G_vacc ---#
        
        p_pol_vac, Epol_vac = coulomb_polarizable_dipole(self.q, self.d, self.Q, self.alpha, self.x_q, self.ep_in, self.dtol)
        G_vacc = coulomb_energy_multipole(self.q, self.d, p_pol_vac, self.Q, self.alpha, self.x_q, self.ep_in)
        print(F"Coulomb vacuum energy: {G_vacc}")
        
        self.total_energy = G_diss_solv + G_diss_mult - G_vacc
        print(F"Total solvation energy: {self.total_energy} [kcal/Mol]")

    def compute_efield(self, x):
        Efield = np.zeros((1,3))
        E_solv = np.zeros((1,3))
        E_coul = np.zeros((1,3))
        tree = dolfinx.geometry.bb_tree(self.mesh, self.mesh.geometry.dim)
        basix_el = self.fenics_space.element.basix_element
        # Indices de derivadas
        # (1,0,0)=1, (0,1,0)=2, (0,0,1)=3
        # (2,0,0)=4, (1,1,0)=5, (1,0,1)=6, (0,2,0)=7, (0,1,1)=8, (0,0,2)=9
        D100 = 1; D010 = 2; D001 = 3 # Indices de primeras derivadas

        # Componente debido al solvente
        cell = locate_cell(x, self.mesh, tree)
        if cell == None:
            print("The coordinate is outside the volumetric mesh")
        else:
            verts_idx = self.mesh.geometry.dofmap[cell] # dof del tetrahedro que contiene el punto 
            verts = np.array([self.mesh.geometry.x[i] for i in verts_idx], dtype=np.float64)  # Coordenadas de los vertices. (4,3)
            xi = get_local_coordinates(verts, x)  # Coordenadas locales de los vertices (4,3)
            
            tab = basix_el.tabulate(1, np.array([xi], dtype=np.float64)) # Funciones de forma hasta la primera derivada
            dofs = self.fenics_space.dofmap.cell_dofs(cell)
            
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
            
            g_phys = invJT @ g_ref  # (3, ndofs)
            for coord in range(len(dofs)):
                E_solv[:]  -= self.phi_rf_dofs[dofs[coord]] * g_phys.T[coord, :] # - porque E=-nabla phi

        # Componente de coulomb

        E_coul = electric_field_coulomb(x, self.x_q, self.q, self.p, self.Q, self.ep_in)

        Efield = E_coul + E_solv
        print(f'Electric field in the location: {Efield}')

    def solve_system(self):
        self.extract_bmesh()
        self.build_function_spaces()
        self.extract_solute_dofs()
        self.potential_operators()
        self.fem_functions()
        self.bem_operators()
        self.build_mass_matrix()
        self.build_lhs()
        self.build_perm_rhs()

        for iter_number in range(self.maxiter):

            self.harmonic_component()
            self.correction_component()
            self.reaction_potential()
            self.compute_derivatives()
            self.compute_multipoles_efield()
            self.compute_mu()
            if self.dipole_diff < self.dtol:
                print(F"The result has converged in {iter_number+1} iterations")
                break
            print(F"Induced dipole residual: {self.dipole_diff}")

        self.compute_energy()