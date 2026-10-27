import numpy as np
import gmsh
import dolfinx
import ufl
import bempp_cl.api
import time
import scipy.sparse
import os
import glob
import math

from bempp_cl.api.external import fenicsx
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI

from dolfinx.geometry import compute_colliding_cells, compute_collisions_points
from scipy.sparse import coo_matrix, diags
from petsc4py import PETSc

from coupled_solver.utils.generate_mesh import load_dolfin_mesh, build_boundary_mesh
from coupled_solver.utils.get_data import read_tinker

from coupled_solver.bem.direct import solvation_energy_solvent, get_coulomb_energy, compute_induced_dipole
from coupled_solver.bem.direct import _get_bem_spaces, _get_potential_operators
from coupled_solver.bem.direct import electric_field_coulomb

from coupled_solver.coupling.johnson_nedelec import _compute_trace_matrix

from coupled_solver.fem.fem_assembly import _compute_reaction_potential, _compute_solvent_derivatives, _get_fem_spaces
from coupled_solver.fem.fem_assembly import _locate_cell, _get_local_coordinates

from bempp_cl.api.operators.boundary import sparse, laplace, modified_helmholtz
from bempp_cl.api.assembly.blocked_operator import BlockedDiscreteOperator
from bempp_cl.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
from bempp_cl.api.operators.potential.laplace import single_layer, double_layer

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres

def precon(lhs, neumann_space):

    P1 = diags(1./lhs[0,0].to_sparse().diagonal())

    P2 = InverseSparseDiscreteBoundaryOperator(
            bempp_cl.api.operators.boundary.sparse.identity(
            neumann_space, neumann_space, neumann_space).weak_form())

    def apply_prec(x):

        m1 = P1.shape[0]
        m2 = P2.shape[0]
        n1 = P1.shape[1]
        n2 = P2.shape[1]

        res1 = P1.dot(x[:n1])
        res2 = P2.dot(x[n1:])

        return np.concatenate([res1, res2])

    p_shape = (P1.shape[0] + P2.shape[0], P1.shape[1] + P2.shape[1])
    P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('float64'))
    return P

def gmres_callback(residual_norm):
    residuals.append(residual_norm)
    print(f"GMRES iter {len(residuals):4d} | residual = {residual_norm:.3e}")

class solver:

    def __init__(self, file, ep_in=1., ep_ex=80., k=0.125):
        self.ep_in = ep_in
        self.ep_ex = ep_ex
        self.k = k

        self.mol_name = file.split('/')[-1]

        self.x_q, self.q, self.d, self.Q, self.alpha, self.mass, self.polar_group, self.thole, \
        self.connections_12, self.connections_13, \
        self.pointer_connections_12, self.pointer_connections_13, \
        self.p12scale, self.p13scale, self.N = read_tinker('molecules/'+file, float)

        self.mu = np.zeros((self.N, 3))
        self.cg_tol = 1e-5
        self.gmres_rtol = 1e-5
        self.gmres_atol = 0.
        self.SOR = 0.7
        self.dtol = 1e-3
        self.maxiter = 100

        self.ntol = 1e-3 # Tolerancia de Newton-Rapson
        self.wtol = 1e-1 # Tolerancia del factor de relajación

        self.mu_flag = False
        self.mesh = None
        self.sol_prev = None

    def upload_mesh(self, external_mesh):
        self.mesh, self.cell_tags, self.facet_tags = load_dolfin_mesh(external_mesh)

    def create_mesh(self, new_mesh, gradation=0.1, probe_radius=1.4, stern_thickness=3.0, algorithm=10, grid_scale=2.0, grid_scale_ratio=2.0):
        self.mesh, self.cell_tags, self.facet_tags = load_dolfin_mesh(self.mol_name, gradation, probe_radius, stern_thickness, algorithm, grid_scale, grid_scale_ratio)

    def solve_vacuum(self):
        from coupled_solver import dir_name
        self.mu_flag = True
        record_info_path = os.path.join(dir_name, 'initial_mu', f'{self.mol_name}.npy')
        if os.path.exists(record_info_path):
            print('Saved information found. Loading initial dipole.')
            self.mu = np.load(record_info_path)

        else:
            p_pol_vacc = np.zeros((self.N,3))
            dipole_diff_vacc = 1.
            iteration = 0
            
            while dipole_diff_vacc>self.dtol:
            
                iteration += 1
            
                p_pol_prev = p_pol_vacc.copy()
            
                p_pol_vacc = compute_induced_dipole(self.x_q, self.q, self.d, p_pol_vacc, self.Q, self.alpha, self.thole, self.polar_group,
                                                    self.connections_12, self.pointer_connections_12, self.connections_13,
                                                    self.pointer_connections_13, np.zeros((self.N,3)), self.ep_in)
            
                dipole_diff_vacc = np.max(np.sqrt(np.sum((p_pol_vacc-p_pol_prev)**2, axis = 1)))
            
                print(F"Induced dipole residual in vacuum: {dipole_diff_vacc}")
            
            print(F"{iteration} iterations for vacuum induced dipole to converge")
            self.mu = p_pol_vacc.copy()
            os.makedirs(os.path.dirname(record_info_path), exist_ok=True)
            np.save(record_info_path, p_pol_vacc)
        
        self.G_vacc = get_coulomb_energy(self.x_q, self.q, self.d, self.mu, self.Q, self.alpha, self.ep_in, self.thole, self.polar_group, 
                                    self.connections_12, self.connections_13, self.pointer_connections_12, 
                                    self.pointer_connections_13, self.p12scale, self.p13scale)
        
    def extract_solute_dofs(self):
        self.x_dofs = self.fenics_space.tabulate_dof_coordinates() # ndof x 3
        self.solute_dofs = set() # Para almacenar los dof del volumen 1 sin repetir.
        for cell_idx in self.solute_cells:
            cell_ndofs = self.fenics_space.dofmap.cell_dofs(cell_idx)
            for dof in cell_ndofs:
                self.solute_dofs.add(dof)
        self.solute_dofs = list(self.solute_dofs)
        self.x_dofs_solute = self.x_dofs[self.solute_dofs]
    
    def fem_functions(self):
        self.u = ufl.TrialFunction(self.fenics_space) # función de aproximación
        self.lam = ufl.TrialFunction(self.P1_space) # función P1 para los datos de frontera
        self.v = ufl.TestFunction(self.fenics_space) # Función de prueba
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags) # cell_tags almacena ambos volumenes
        self.ds = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.facet_tags) # En caso de necesitarlo, dS es la frontera interna
        self.ds_ext = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags) # ds es la frontera exterior, pinche fenics
        self.A_l = fenicsx.FenicsOperator(self.ep_in*(ufl.inner(ufl.nabla_grad(self.u), ufl.nabla_grad(self.v)))*self.dx(1) + self.ep_ex*(ufl.inner(ufl.nabla_grad(self.u), ufl.nabla_grad(self.v)))*self.dx(2) + self.ep_ex*self.k*self.k*ufl.inner(self.u, self.v)*self.dx(2))
        self.A_nl = fenicsx.FenicsOperator(self.ep_in*(ufl.inner(ufl.nabla_grad(self.u), ufl.nabla_grad(self.v)))*self.dx(1) + self.ep_ex*(ufl.inner(ufl.nabla_grad(self.u), ufl.nabla_grad(self.v)))*self.dx(2))
    
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
        # Lineal matrix
        linear_lhs = [[None,None],[None,None]] 
            
        linear_lhs[0][0] = self.A_l.weak_form()
        linear_lhs[0][1] = -self.ep_ex * self.mass_matrix_sparse @ self.trace_stern.T
        linear_lhs[1][0] = (.5 * self.I - self.KY).weak_form() * self.trace_op
        linear_lhs[1][1] = self.VY.weak_form()
            
        self.cterm_lhs_l = BlockedDiscreteOperator(np.array(linear_lhs)) # Lado izquierdo de la ecuación lineal

        nonlinear_lhs = linear_lhs.copy()

        nonlinear_lhs[0][0] = self.A_nl.weak_form()

        self.cterm_lhs_nl = BlockedDiscreteOperator(np.array(nonlinear_lhs)) # Lado izquierdo de la ecuación no lineal

        self.dN = np.empty((2, 2), dtype=object) # Lado variable de NR, dN[0][0] se actualiza despues

        self.dN[0][1] = scipy.sparse.csr_matrix((self.fem_ndof, self.bem_ndof))
        self.dN[1][0] = scipy.sparse.csr_matrix((self.bem_ndof, self.fem_ndof))
        self.dN[1][1] = scipy.sparse.csr_matrix((self.bem_ndof, self.bem_ndof))
            
        self.P_l  = precon(self.cterm_lhs_l, self.neumann_space_stern)
        self.P_nl = precon(self.cterm_lhs_nl, self.neumann_space_stern)
    
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
        self.dphi0dn, info = bempp_cl.api.linalg.cg(self.VL, hterm_rhs, tol=self.cg_tol)
    
    def build_var_rhs(self):
        x_q, mu, ep_in = self.x_q, self.mu, self.ep_in
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
        self.cterm_rhs = np.concatenate([fem_rhs_array, self.bem_rhs])

    def solve_linear_system(self):
        global residuals
        residuals = []
        self.sol_l, info = gmres(
                    self.cterm_lhs_l,
                    self.cterm_rhs,
                    x0 = self.sol_prev,
                    rtol=self.gmres_rtol,
                    atol=self.gmres_atol,
                    M=self.P_l,
                    callback=gmres_callback,
                    callback_type="pr_norm",
                    restart=500,
                    maxiter=2000
                    )

        self.sol_prev = self.sol_l.copy()
        self.phi_hat_l  = self.sol_l[:self.fem_ndof] # Componente de corrección en la ubicación de los dofs

        self.u_l = dolfinx.fem.Function(self.fenics_space)
        self.u_l.x.array[:] = np.ascontiguousarray(self.phi_hat_l)
    
    def reaction_potential_dofs(self):
        self.G_fun = self.G_fun_perm + self.G_fun_var
        self.phi0_dofs_solvent = self.slpo_dofs * self.dphi0dn + self.dlpo_dofs * self.G_fun
        self.phi0_dofs[self.solute_dofs] = self.phi0_dofs_solvent[0]

        self.phi_rf_dofs = self.phi_hat_l + self.phi_hat_nl + self.phi0_dofs # Phi_rf = phi_h + phi_hat
    
    def compute_efield(self, x):
        E_solv = np.zeros((1,3))
        E_coul = np.zeros((1,3))
        tree = dolfinx.geometry.bb_tree(self.mesh, self.mesh.geometry.dim)
        basix_el = self.fenics_space.element.basix_element
        # Indices de derivadas
        # (1,0,0)=1, (0,1,0)=2, (0,0,1)=3
        # (2,0,0)=4, (1,1,0)=5, (1,0,1)=6, (0,2,0)=7, (0,1,1)=8, (0,0,2)=9
        D100 = 1; D010 = 2; D001 = 3 # Indices de primeras derivadas
        
        # Componente debido al solvente
        cell = _locate_cell(x, self.mesh, tree)
        if cell is None:
            print("The coordinate is outside the volumetric mesh")
        else:
            verts_idx = self.mesh.geometry.dofmap[cell] # dof del tetrahedro que contiene el punto 
            verts = np.array([self.mesh.geometry.x[i] for i in verts_idx], dtype=np.float64)  # Coordenadas de los vertices. (4,3)
            xi = _get_local_coordinates(verts, x)  # Coordenadas locales de los vertices (4,3)
                    
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
            self.p = self.mu + self.d
            E_coul = electric_field_coulomb(x, self.x_q, self.q, self.p, self.Q, self.ep_in)
        
            self.Efield = E_coul + E_solv
            print(f'Electric field in the location: {self.Efield}')
    
    def Evaluate_fd_dfd(self, Taylor_expansion, w0, derivate):
        C1 = 7042.938                       
        k2 = self.k**2
        V  = self.fenics_space
        zb = np.zeros(self.bem_ndof)        

        if not hasattr(self, '_nl_kits'):
            self._nl_kits = {}
        if Taylor_expansion not in self._nl_kits:
            u_w, dU = dolfinx.fem.Function(V), dolfinx.fem.Function(V)
            S_f, C_f, D_f = (dolfinx.fem.Function(V) for _ in range(3))
            US = C1*u_w
            if Taylor_expansion == 'T11':
                S_e = US + US**3/6 + US**5/120 + US**7/5040 + US**9/362880 + US**11/39916800
                C_e = 1 + US**2/2 + US**4/24 + US**6/720 + US**8/40320 + US**10/3628800
                D_e = US + US**3/6 + US**5/120 + US**7/5040 + US**9/362880   
            elif Taylor_expansion == 'T3':
                S_e, C_e, D_e = US + US**3/6, 1 + US**2/2, US
            else:  
                S_e, C_e, D_e = ufl.sinh(US), ufl.cosh(US), ufl.sinh(US)
            pts = self._ipts()
            self._nl_kits[Taylor_expansion] = dict(
                u_w=u_w, dU=dU, S=S_f, C=C_f, D=D_f,
                S_expr=dolfinx.fem.Expression(S_e, pts),
                C_expr=dolfinx.fem.Expression(C_e, pts),
                D_expr=dolfinx.fem.Expression(D_e, pts),
                f_N   =dolfinx.fem.form(self.ep_ex*k2/C1*S_f*self.v*self.dx(2)),         
                f_NpD =dolfinx.fem.form(self.ep_ex*k2*C_f*dU*self.v*self.dx(2)),         
                f_Npp =dolfinx.fem.form(self.ep_ex*k2*C1*D_f*dU**2*self.v*self.dx(2)),   
            )
        kit = self._nl_kits[Taylor_expansion]

        d     = self.d_soln_nl
        sol_k = self.sol_l + self.soln0_nl            
        c = getattr(self, '_w_cache', None)
        if (c is None or not np.array_equal(c['d'], d)
                or not np.array_equal(c['sol_k'], sol_k)
                or not np.array_equal(c['b'], self.cterm_rhs)):
            c = dict(d=d.copy(), sol_k=sol_k.copy(), b=self.cterm_rhs.copy(),
                    AD=self.cterm_lhs_nl*d,
                    R0=self.cterm_lhs_nl*sol_k - self.cterm_rhs)
            self._w_cache = c

        kit['u_w'].x.array[:] = sol_k[:self.fem_ndof] + w0*d[:self.fem_ndof]
        kit['dU'].x.array[:]  = d[:self.fem_ndof]
        kit['S'].interpolate(kit['S_expr'])
        kit['C'].interpolate(kit['C_expr'])

        N   = dolfinx.fem.assemble_vector(kit['f_N']).array
        NpD = dolfinx.fem.assemble_vector(kit['f_NpD']).array

        rhs_A  = -c['AD']
        rhs_B  = -(c['R0'] + np.concatenate([N, zb]))
        rhs_dB = -np.concatenate([NpD, zb])

        AA_AdB = np.dot(rhs_A + rhs_dB, rhs_A)
        AB_BdB = np.dot(rhs_A + rhs_dB, rhs_B)
        fD_w0  = -AB_BdB/AA_AdB - w0

        if not derivate:
            return fD_w0, None

        if Taylor_expansion == 'SINH':
            kit['D'].x.array[:] = kit['S'].x.array      
        else:
            kit['D'].interpolate(kit['D_expr'])
        Npp     = dolfinx.fem.assemble_vector(kit['f_Npp']).array
        rhs_ddB = -np.concatenate([Npp, zb])

        AddB     = np.dot(rhs_A, rhs_ddB)
        BddB     = np.dot(rhs_B, rhs_ddB)
        AdB_dBdB = np.dot(rhs_A + rhs_dB, rhs_dB)
        dfD_w0   = (-AA_AdB*(AdB_dBdB + BddB) + AB_BdB*AddB)/(AA_AdB**2) - 1
        return fD_w0, dfD_w0
    
    def w_optimal_by_Bisection(self, Taylor_expansion, wa, wb, Iter0, Secant_equation):  
        Iter=Iter0
        Tol_aditional=0.001
        eps=10
        w, d = (wa+wb)/2, (wb-wa)/2    
        while (eps > self.wtol): 
            w0, d = w, d/2
            Iter=Iter+1 
            fD,_ = self.Evaluate_fd_dfd(Taylor_expansion,w0,derivate=False)
            Sgn, eps = np.sign(fD), abs(fD)
            if Sgn>0:
                w=w0+d
            else:
                w=w0-d  
            print('Iter BI S=%d: norm=%g: w=%g' % (Iter, eps, w0))   
            if Secant_equation==True: 
                if Iter==(1+Iter0):
                    wb_sec, fD_wb_sec = w0, fD
                    wc_sec = 100
                else:
                    wa_sec, fD_wa_sec = wb_sec, fD_wb_sec
                    wb_sec, fD_wb_sec = w0, fD
                    wc0_sec = wc_sec
                    wc_sec = wa_sec-(wa_sec-wb_sec)*(fD_wa_sec)/(fD_wa_sec-fD_wb_sec) #Equation of the line
                    if Iter==(2+Iter0):
                        print('w calculate for Secant equation use point S=%d: and S=%d: wc_sec=%g:' % (Iter, (Iter-1), wc_sec))
                    else:
                        Diff = abs(wc_sec-wc0_sec)
                        print('w calculate for Secant equation use point S=%d: and S=%d: wc_sec=%g: Diff=%g' % (Iter, (Iter-1), wc_sec, Diff))
                        if Diff<0.05 and abs(fD_wb_sec)<50 and abs(fD_wa_sec)<50:
                            break            
            if Iter==50 or w0>=(wb-Tol_aditional) or w0<=(wa+Tol_aditional):  
                break
        return w0, Iter, wc_sec
    
    def w_optimal_by_Newton_Rapson(self, Taylor_expansion, w0, Iter0):  
        Iter=Iter0
        fD,dfD = self.Evaluate_fd_dfd(Taylor_expansion,w0,derivate=True)
        w, eps = w0-fD/dfD, abs(fD)
        print('Iter NR S=%d: norm=%g: w=%g' % (Iter, eps, w0))   
    
        while (eps > self.wtol):   
            w0=w
            Iter=Iter+1    
            fD,dfD = self.Evaluate_fd_dfd(Taylor_expansion,w0,derivate=True)
            w, eps = w0-fD/dfD, abs(fD)
            print('Iter NR S=%d: norm=%g: w=%g' % (Iter, eps, w0))    
        
            if Iter==30: 
                break         
        return w0, Iter 
    
    def Taylor_Expansion_of_vector_c(self, Taylor_expansion, u0_nl, u_l):
        C1 = 7042.938
        US = C1*(u0_nl+u_l)
        if Taylor_expansion=='T3':
            print("Case: Cubic Taylor Exp.")
            expr_ufl_S = US+ US**3/6
            expr_ufl_C = 1+US**2/2      
        elif Taylor_expansion=='T11': 
            print("Case: Taylor's 11th Exp.")
            expr_ufl_S = US+np.power(US,3)/6+np.power(US,5)/120+np.power(US,7)/5040+np.power(US,9)/362880+np.power(US,11)/39916800 
            expr_ufl_C = 1+np.power(US,2)/2+np.power(US,4)/24+np.power(US,6)/720+np.power(US,8)/40320+np.power(US,10)/3628800       
        else:
            print("Case: Hyperbolic Sine")
            expr_ufl_S = ufl.sinh(US)
            expr_ufl_C = ufl.cosh(US)  
        NL_Fem_G_S = dolfinx.fem.Function(self.fenics_space)
        NL_Fem_G_C = dolfinx.fem.Function(self.fenics_space)       
        NL_Fem_G_S.interpolate(dolfinx.fem.Expression(expr_ufl_S, self._ipts()))
        NL_Fem_G_C.interpolate(dolfinx.fem.Expression(expr_ufl_C, self._ipts()))
        return NL_Fem_G_S, NL_Fem_G_C
    
    def Scheme_election(self):
        fD_w0,_ = self.Evaluate_fd_dfd('SINH',1,False)    
        print('iter S=%d: norm=%g: w=%g' % (0, abs(fD_w0), 1))  
        print('Evaluate point w0=%g' % (1))     
        ######################### Choice of scheme
        Scheme='T3-HS'
        if np.abs(fD_w0)<100000:
            Bisection_Secant_Method=False
        else:
            Bisection_Secant_Method=True 
            if (math.isnan(fD_w0)==True):  #First overflow correction
                Scheme='T3-T11' 
    ####################### Initial relaxation factor w0_NR for Newton-Raphson
        if np.abs(fD_w0)<=10: 
            w0_NR=1
        else:
            w0_NR=2
        print('Scheme '+Scheme)  
    ############################### Important variables that depend on the scheme used     
        if Scheme=='T3-HS':
            Taylor_expansion_list = ['SINH']  
        elif Scheme=='T3-T11':
            Taylor_expansion_list = ['T11']  
        Iter_Transition =len(Taylor_expansion_list)  
        return Scheme, Taylor_expansion_list, w0_NR, Iter_Transition, Bisection_Secant_Method
    
    def G_ion(self, expr):
        C1 = 7042.938
        C2 = 0.5924850225 
        f = dolfinx.fem.Function(self.fenics_space)
        f.interpolate(dolfinx.fem.Expression(expr, self._ipts()))
        return 0.5*(C2/C1)*self.ep_ex*self.k**2*dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(f*self.dx(2)))
    
    def _ipts(self):
        p = self.fenics_space.element.interpolation_points
        return p() if callable(p) else p
    
    def solve_system(self, gradation=0.1, probe_radius=1.4, solvent_thickness=3.0, algorithm=10, grid_scale=2.0, grid_scale_ratio=2.0, new_mesh=False):
        from coupled_solver import dir_name
        global residuals
        time_init = time.time()
        
        if self.mesh is None: # Verifica si se cargo lo malla antes
            if new_mesh==True: 
                mesh_path = os.path.join(dir_name, 'volumetric_mesh', f'{self.mol_name}.msh')
                boundary_mesh_path = os.path.join(dir_name, 'boundary_mesh', f'{self.mol_name}*')
                if os.path.exists(mesh_path):
                    os.remove(mesh_path)
                for archivo in glob.glob(boundary_mesh_path):
                    if os.path.isfile(archivo):
                        os.remove(archivo)
            self.create_mesh(new_mesh, gradation, probe_radius, solvent_thickness, algorithm, grid_scale, grid_scale_ratio)

        tdim = self.mesh.topology.dim
        fdim = tdim - 1

        self.mesh.topology.create_connectivity(0, tdim)
        self.mesh.topology.create_connectivity(tdim, 0)
        self.mesh.topology.create_entities(fdim)
        self.mesh.topology.create_connectivity(tdim, fdim)  # cell -> facet
        self.mesh.topology.create_connectivity(fdim, tdim)  # facet -> cell
        self.mesh.topology.create_connectivity(fdim, 0)     # facet -> vertex 
        self.mesh.topology.create_entity_permutations()

        ses_facets = self.facet_tags.find(1)
        stern_facets = self.facet_tags.find(2)
        
        self.solute_cells = self.cell_tags.find(1)
        self.stern_cells = self.cell_tags.find(2)

        ses_mesh, ses_nodes = build_boundary_mesh(self.mesh, ses_facets, cell_tags=self.cell_tags, inner_marker=1)
        stern_mesh, stern_nodes = build_boundary_mesh(self.mesh, stern_facets)

        if self.mu.shape != (self.N, 3):
            self.mu = np.zeros((self.N,3))

        #--- Definir espacios funcionales ---#
        
        time_init_spaces = time.time()

        self.fenics_space, self.P1_space = _get_fem_spaces(self.mesh)
        self.dirichlet_space_ses, self.neumann_space_ses        = _get_bem_spaces(ses_mesh)
        self.dirichlet_space_stern, self.neumann_space_stern    = _get_bem_spaces(stern_mesh)
        self.trace_matrix   = _compute_trace_matrix(self.fenics_space, stern_nodes)
        self.trace_ses      = _compute_trace_matrix(self.P1_space, ses_nodes)
        self.trace_stern    = _compute_trace_matrix(self.P1_space, stern_nodes)

        time_final_spaces = time.time()

        print(f'Time to define spaces: {(time_final_spaces - time_init_spaces):.2f} [s]')

        self.fem_ndof = self.fenics_space.dofmap.index_map.size_global
        self.bem_ndof = self.dirichlet_space_stern.global_dof_count
        self.total_ndof = self.fem_ndof + self.bem_ndof

        print("FEM dofs: {0}".format(self.fem_ndof))
        print("BEM dofs: {0}".format(self.bem_ndof))
        print("Total dofs: {0}".format(self.total_ndof))

        self.extract_solute_dofs()

        self.phi0_dofs = np.zeros(self.fem_ndof)

        #--- Operadores potenciales ---#

        time_init_pot = time.time()

        self.slpo_dofs, self.dlpo_dofs = _get_potential_operators(self.dirichlet_space_ses, self.neumann_space_ses, self.x_dofs_solute)
        self.slpo_mult, self.dlpo_mult = _get_potential_operators(self.dirichlet_space_ses, self.neumann_space_ses, self.x_q)

        time_final_pot = time.time()
        print(f'Time to define potential operators: {(time_final_pot - time_init_pot):.2f} [s]')

        # Operadores FEM y BEM

        time_init_operators = time.time()
        
        self.fem_functions()
        self.bem_operators()
        self.build_mass_matrix()

        time_final_operators = time.time()

        print(f'Time to define boundary and FEM operators: {(time_final_operators - time_init_operators):.2f} [s]')

        # Construcción del lado izquierdo

        time_init_lhs = time.time()

        self.build_lhs()

        time_final_lhs = time.time()

        print(f'Time to assemble LHS: {(time_final_lhs - time_init_lhs):.2f} [s]')

        # Parte constante del lado derecho

        time_init_rhs = time.time()

        self.build_perm_rhs()

        time_final_rhs = time.time()

        print(f'Time to assembly permanent part of the RHS: {(time_final_rhs - time_init_rhs):.2f} [s]')

        # Inicio esquema iterativo

        self.sol_prev = np.zeros(self.total_ndof)
 
        maxiter_nl = 100  # Max no of iterations allowed

        u0_nl = dolfinx.fem.Function(self.fenics_space)
        u0_nl.x.array[:] = 0.0
        soln_fem_nl  = np.zeros(self.fem_ndof)
        soln_bem_nl = np.zeros(self.bem_ndof)
        self.soln0_nl = np.concatenate([soln_fem_nl, soln_bem_nl])
        d_soln0_nl = self.soln0_nl

        lim_eps = 0.35
        tol_0 = 1e-3 # Tolerancia inicial de gmres para no lineal
        C1 = 7042.938

        for iter_number in range(self.maxiter):
            print(F"------- Dipole iteration {iter_number + 1} -------")

            time_init_har = time.time()
            
            self.harmonic_component() # Se resuelve la componente armónica

            time_final_har = time.time()
            print(f'Time to solve the harmonic component system: {(time_final_har - time_init_har):.2f} [s]')

            time_gmres_init = time.time()

            self.build_var_rhs()
            self.solve_linear_system() # Se resuelve la parte lineal del sistema

            time_gmres_final = time.time()
            print(F"Compution time for linear system: {time_gmres_final - time_gmres_init} [s]")

            time_nonlinear_init = time.time()

            eps = np.inf
            non_liner_iter = 0
            Taylor_expansion ='T3'

            while (eps > self.ntol) and (non_liner_iter < maxiter_nl):
                residuals = []
                start2 = time.time()
                non_liner_iter += 1
                NL_Fem_G_S, NL_Fem_G_C = self.Taylor_Expansion_of_vector_c(Taylor_expansion, u0_nl, self.u_l)
                c_fem =  dolfinx.fem.assemble_vector(dolfinx.fem.form(self.ep_ex*self.k**2 /C1 * NL_Fem_G_S*self.v*self.dx(2))).array
                c_nlG = np.concatenate([c_fem, self.bem_rhs])
                rhs_nlG = -(self.cterm_lhs_nl*(self.sol_l+self.soln0_nl)-self.cterm_rhs+c_nlG)

                self.dN[0][0] = (fenicsx.FenicsOperator(self.ep_ex*self.k**2*NL_Fem_G_C*self.u*self.v*self.dx(2))).weak_form()
                blocked_dN = BlockedDiscreteOperator(self.dN)

                start1 = time.time()
                self.d_soln_nl, info = gmres(
                                    (self.cterm_lhs_nl+blocked_dN),
                                    rhs_nlG,
                                    x0 = d_soln0_nl,
                                    rtol=tol_0,
                                    atol=self.gmres_atol,
                                    M=self.P_nl,
                                    callback=gmres_callback,
                                    callback_type="pr_norm",
                                    restart=500,
                                    maxiter=2000
                                    ) 
                end1 = time.time()

                curr_time1 = (end1 - start1)
                
                print("Total time in GMRES Nonlineal: {:5.2f} [s]".format(curr_time1))

                if non_liner_iter==1:
                    Scheme, Taylor_expansion_list, w0_NR, Iter_Transition, Bisection_Secant_Method = self.Scheme_election()
                if non_liner_iter<=Iter_Transition:
                    Taylor_expansion=Taylor_expansion_list[non_liner_iter-1]

                if non_liner_iter==1:
                    if Bisection_Secant_Method==True:
                        _,I_w0_NR, w0_NR = self.w_optimal_by_Bisection(Taylor_expansion,2,3,0,Secant_equation=True)
                        print('Iter Total BI-SEC previus w I_w0_NR=%d: w0_NR=%g' % (I_w0_NR, w0_NR)) 
                    w,Iter_w = self.w_optimal_by_Newton_Rapson(Taylor_expansion,w0_NR,0)
                    print('Iter Total NR I_w=%d: w=%g' % (Iter_w, w))
                else:
                    if eps>=lim_eps:
                        w,Iter_w = self.w_optimal_by_Newton_Rapson(Taylor_expansion,1,0)
                        print('Iter Total NR I_w=%d: w=%g' % (Iter_w, w))

                d_soln0_nl = self.d_soln_nl*w
                self.soln0_nl = self.soln0_nl+d_soln0_nl
                eps0 = eps
                eps = C1 * np.linalg.norm(d_soln0_nl, ord=np.inf)
                self.phi_hat_nl = self.soln0_nl[:self.fem_ndof]
                u_nl = dolfinx.fem.Function(self.fenics_space)
                u_nl.x.array[:] = np.ascontiguousarray(self.phi_hat_nl)
                u0_nl.x.array[:] = u_nl.x.array[:]
                print('iter=%d: norm=%g' % (non_liner_iter, eps))

                while ((eps/(1.25*tol_0))//1000==0) and (tol_0>=self.gmres_rtol*10):
                    tol_0=tol_0*0.1

                end2 = time.time() 
                curr_time2 = (end2 - start2)
                print("Total time to solve 1 nonlinear iteration: {:5.2f} [s]".format(curr_time2))

            time_nonlinear_final = time.time()

            print(F"Compution time for non-linear system: {time_nonlinear_final - time_nonlinear_init} [s]")
            
            time_g0_init = time.time()

            self.reaction_potential_dofs()

            time_g0_final = time.time()
            print(F"Harmonic component computation time for the DOFs: {time_g0_final - time_g0_init} [s]")

            time_der_init = time.time()

            dphi_rf, ddphi_rf = _compute_solvent_derivatives(self.fenics_space, self.x_q, self.phi_rf_dofs)

            time_der_final = time.time()
            print(F"Computation time for calculating the derivative of the reaction potential: {time_der_final - time_der_init} [s]")

            mu_prev = self.mu.copy()
            time_mu_init = time.time()

            self.mu = compute_induced_dipole(self.x_q, self.q, self.d, self.mu, self.Q, self.alpha, self.thole, self.polar_group,
                                        self.connections_12, self.pointer_connections_12, self.connections_13, self.pointer_connections_13,
                                        dphi_rf, self.ep_in, self.SOR)

            
            time_mu_final = time.time()
            print(F"Compution time for induced dipole: {time_mu_final - time_mu_init} [s]")

            dipole_diff = np.max(np.sqrt(np.sum((self.mu-mu_prev)**2, axis = 1)))

            if dipole_diff < self.dtol:
                print(F"The induced dipole in dissolved state has converged in {iter_number+1} iterations")
                break

            print(F"Induced dipole residual: {dipole_diff}")

        self.phi_hat    = self.phi_hat_l + self.phi_hat_nl
        phi_hat_mult    = _compute_reaction_potential(self.phi_hat, self.x_q, self.fenics_space) # Componente de corrección sobre los multipolos
        phi0_mult       = self.slpo_mult * self.dphi0dn + self.dlpo_mult * self.G_fun
        phi_rf_mult     = phi_hat_mult + phi0_mult[0]

        self.G_diss_solv = solvation_energy_solvent(self.q, self.d, self.Q, phi_rf_mult, dphi_rf, ddphi_rf)

        time_energy_init = time.time()

        self.G_diss_mult = get_coulomb_energy(self.x_q, self.q, self.d, self.mu, self.Q, self.alpha, self.ep_in, self.thole,
                                              self.polar_group, self.connections_12, self.connections_13,
                                              self.pointer_connections_12, self.pointer_connections_13,
                                              self.p12scale, self.p13scale)

        time_energy_final = time.time()
        print(F"Compution time for Coulomb Energy: {time_energy_final - time_energy_init} seconds")

        if not self.mu_flag:
            p_pol_vacc = np.zeros((self.N,3))
            dipole_diff_vacc = 1.
            iteration = 0

            while dipole_diff_vacc > self.dtol:
                iteration +=1
                p_pol_prev = p_pol_vacc.copy()
                p_pol_vacc = compute_induced_dipole(self.x_q, self.q, self.d, p_pol_vacc, self.Q, self.alpha, self.thole, self.polar_group,
                                                    self.connections_12, self.pointer_connections_12, self.connections_13,
                                                    self.pointer_connections_13, np.zeros((self.N,3)), self.ep_in)

                dipole_diff_vacc = np.max(np.sqrt(np.sum((p_pol_vacc-p_pol_prev)**2, axis = 1)))
                print(F"Induced dipole residual in vacuum: {dipole_diff_vacc}")

            print(F"{iteration} iterations for vacuum induced dipole to converge")
            self.G_vacc = get_coulomb_energy(self.x_q, self.q, self.d, p_pol_vacc, self.Q, self.alpha, self.ep_in, self.thole, self.polar_group, 
                                            self.connections_12, self.connections_13, self.pointer_connections_12, 
                                            self.pointer_connections_13, self.p12scale, self.p13scale)

        time_final = time.time()

        US = dolfinx.fem.Function(self.fenics_space)
        US.x.array[:] = C1*self.phi_hat        

        G_nl = self.G_ion(ufl.sinh(US)*US + 2 - 2*ufl.cosh(US))

        if not np.isfinite(G_nl):
            S = US + US**3/6 + US**5/120 + US**7/5040 + US**9/362880 + US**11/39916800
            C = 1 + US**2/2 + US**4/24 + US**6/720 + US**8/40320 + US**10/3628800
            G_nl = self.G_ion(S*US + 2 - 2*C)

        print(F"Solvent contribution: {self.G_diss_solv} [kcal/Mol]")
        print(F"Solvation energy nonlinear: {G_nl} [kcal/Mol]")
        print(F"Multipoles contribution: {self.G_diss_mult} [kcal/Mol]")
        print(F"Coulomb vacuum energy: {self.G_vacc}")

        print("These values consider the polarization energy")

        self.total_energy = self.G_diss_solv + self.G_diss_mult - self.G_vacc + G_nl

        record_info_path = os.path.join(dir_name, 'final_mu', f'{self.mol_name}.npy')
        os.makedirs(os.path.dirname(record_info_path), exist_ok=True)
        np.save(record_info_path, self.mu)

        print(F"Total solvation energy: {self.total_energy} [kcal/Mol]")
        print(F"Compution time for Coulomb Energy: {time_energy_final - time_energy_init} seconds")
        print(F"Total time: {time_final - time_init} seconds.")