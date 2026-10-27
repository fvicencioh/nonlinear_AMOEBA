import numpy as np
import gmsh
import dolfinx
import ufl
import bempp_cl.api
import os
import trimesh

from bempp_cl.api.external import fenicsx
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI
from coupled_solver.utils.get_data import _tinker_to_xyzr

def _get_mesh_files(xyzr_filepath, grid_scale=1.0, probe_radius=1.4):
    from coupled_solver import dir_name
    
    xyzr_filename = xyzr_filepath.split("/")[-1]
    xyzr_filename_we = xyzr_filename.split(".")[0]
    nanoshaper_dir_name = os.path.join(dir_name, "ExternalSoftware/NanoShaper/")
    mesh_dir_name = os.path.join(nanoshaper_dir_name, "meshs/" + xyzr_filename_we)
    
    if not os.path.exists(nanoshaper_dir_name+"meshs"):
        os.makedirs(nanoshaper_dir_name+"meshs")
        
    if not os.path.exists(mesh_dir_name):
        os.makedirs(mesh_dir_name)
        
    os.system('cp ' + xyzr_filepath + " " + mesh_dir_name)
        
    # Make Changes to Config File
    config_template_file = open(nanoshaper_dir_name+'config', 'r')
    config_file = open(nanoshaper_dir_name + 'surfaceConfiguration.prm', 'w')
    
    for line in config_template_file:
        if 'XYZR_FileName' in line:
            path = os.path.join(mesh_dir_name, xyzr_filename)
            line = 'XYZR_FileName = ' + path + ' \n'
        elif 'Grid_scale' in line:
            line = 'Grid_scale = {:04.1f} \n'.format(grid_scale)
        elif 'Probe_Radius' in line:
            line = 'Probe_Radius = {:03.1f} \n'.format(probe_radius)
            
        config_file.write(line)
        
    config_file.close()
    config_template_file.close()
    
    os.chdir(nanoshaper_dir_name)
    os.system(nanoshaper_dir_name+"NanoShaper")
    
    os.system('mv ' + nanoshaper_dir_name + '*.vert ' + xyzr_filename_we + '.vert')
    os.system('mv ' + nanoshaper_dir_name + '*.face ' + xyzr_filename_we + '.face')
    
    os.system('mv ' + nanoshaper_dir_name + xyzr_filename_we + '.* ' + mesh_dir_name)
    
    os.chdir(dir_name+"/..")
    
def _split_mesh(filename):
    from coupled_solver import dir_name
    faces = np.loadtxt(filename+'.face', dtype=int, skiprows=3, usecols=(0,1,2))
    vertices = np.loadtxt(filename+'.vert', dtype=float, skiprows=3, usecols=(0,1,2))
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces-1)
    
    mesh_split = mesh.split()
    
    vertices_split = mesh_split[0].vertices
    faces_split = mesh_split[0].faces
    
    os.system("rm " + filename +".vert")
    os.system("rm " + filename +".face")
    
    np.savetxt(filename+'.face', faces_split+1, fmt='%i')
    np.savetxt(filename+'.vert', vertices_split, fmt='%1.5f')

    os.makedirs(dir_name + '/boundary_mesh', exist_ok=True)
    os.system('cp ' + filename+'.face ' + dir_name + '/boundary_mesh')
    os.system('cp ' + filename+'.vert ' + dir_name + '/boundary_mesh')

def _read_nodes(name):
    vert_filename = name + '.vert'
    with open(vert_filename, 'r') as file:
        n_verts = len(file.readlines())
    
    nodes = np.zeros((n_verts, 3))
    vert_file = open(vert_filename, 'r')
    i = 0
    for line in vert_file:
        line = line.split()
        nodes[i,:] = line[:3]
        i+=1
    
    vert_file.close()
    return nodes

def _read_faces(name):
    faces_filename = name + '.face'
    with open(faces_filename, 'r') as file:
        n_faces = len(file.readlines())

    faces = np.zeros((n_faces, 3), dtype=int)
    faces_file = open(faces_filename, 'r')
    i = 0
    for line in faces_file:
        line = line.split()
        faces[i,:] = line[:3]
        i+=1

    faces_file.close()
    return faces

def _create_solvent_xyzr(file, thickness):
    solvent_filepath = file.split('.')[0] +'_solvent.xyzr'
    xyzr_file = open(file, 'r')
    solvent_file = open(solvent_filepath, 'w')

    for line in xyzr_file:
        line = line.split()
        if len(line)>0:
            radii = float(line[-1])
            solvent_radii = radii + thickness
            line[-1] = str(solvent_radii)
            solvent_line = '\t'.join(line)+'\n'
            solvent_file.write(solvent_line)

    xyzr_file.close()
    solvent_file.close()

def _create_volumetric_mesh(name, gradation=0.1, probe_radius=1.4, solvent_thickness=3.0, algorithm=10, grid_scale=2.0, grid_scale_ratio=2.0):
    from coupled_solver import dir_name
    base = os.path.dirname(dir_name)
    xyzr_filepath = os.path.join(base, 'molecules', name, f'{name}.xyzr')

    if not os.path.exists(xyzr_filepath):
        print(f"{name}.xyzr file not detected.")
        print(f"Creating {name}.xyzr file")
        xyzr_filepath_we = xyzr_filepath.split('.')[0]
        try:
            _tinker_to_xyzr(xyzr_filepath_we) 
            print("File Created")
        except FileNotFoundError:
            print(f'No se ha encontrado el archivo {xyzr_filepath}, no se ha creado el archivo en formato .xyzr, si esta malla es de una esfera, ignorar este error.')

    mesh_dir = os.path.join(dir_name, "ExternalSoftware/NanoShaper/meshs/" + name + "/")
    boundary_dir = dir_name + '/boundary_mesh/'
    os.makedirs(boundary_dir, exist_ok=True)

    ses_face_path = os.path.join(boundary_dir, name + '_ses.face')
    ses_vert_path = os.path.join(boundary_dir, name + '_ses.vert')
    ses_stl_path = os.path.join(boundary_dir, name + '_ses.stl')

    if not os.path.exists(ses_stl_path):
        if not os.path.exists(ses_face_path) or not os.path.exists(ses_vert_path):
            print(f"Malla SES no encontrada para {name}. Generando con NanoShaper (Probe Radius: {probe_radius}, Grid Scale: {grid_scale})...")
            _get_mesh_files(xyzr_filepath, grid_scale, probe_radius=probe_radius)
            _split_mesh(mesh_dir + name)
            
            os.rename(os.path.join(boundary_dir, name + '.face'), ses_face_path)
            os.rename(os.path.join(boundary_dir, name + '.vert'), ses_vert_path)

        nodes_SES = _read_nodes(os.path.join(boundary_dir, name + '_ses'))
        faces_SES = _read_faces(os.path.join(boundary_dir, name + '_ses'))
        ses_mesh = trimesh.Trimesh(vertices=nodes_SES, faces=faces_SES - 1)
        ses_mesh.export(ses_stl_path)

    solvent_face_path = os.path.join(boundary_dir, name + '_solvent.face')
    solvent_vert_path = os.path.join(boundary_dir, name + '_solvent.vert')
    solvent_stl_path = os.path.join(boundary_dir, name + '_solvent.stl')

    solvent_filepath = os.path.join(base, 'molecules', name, f'{name}_solvent.xyzr')
    solvent_dir = os.path.join(dir_name, "ExternalSoftware/NanoShaper/meshs/" + name + '_solvent' + "/")

    if not os.path.exists(solvent_stl_path):
        if not os.path.exists(solvent_face_path) or not os.path.exists(solvent_vert_path):
            print(f"Malla de solvente no encontrada para {name}. Generando con NanoShaper (Radio: {probe_radius}; Espesor capa de solvente: {solvent_thickness}, Grid Scale: {grid_scale/grid_scale_ratio})...")
            _create_solvent_xyzr(xyzr_filepath, solvent_thickness)
            _get_mesh_files(solvent_filepath, grid_scale/grid_scale_ratio, probe_radius=probe_radius)
            _split_mesh(solvent_dir + name + '_solvent')

        # Leer archivos y convertirlos a STL
        nodes_solvent = _read_nodes(os.path.join(boundary_dir, name + '_solvent'))
        faces_solvent = _read_faces(os.path.join(boundary_dir, name + '_solvent'))
        solvent_mesh = trimesh.Trimesh(vertices=nodes_solvent, faces=faces_solvent - 1)
        solvent_mesh.export(solvent_stl_path)

    if not gmsh.isInitialized():
        gmsh.initialize() 
    
    gmsh.clear() 
    
    gmsh.model.add("molecular_volume_coupled")

    gmsh.merge(ses_stl_path)
    surf_ses_tag = gmsh.model.getEntities(2)[-1][1] 
    
    gmsh.merge(solvent_stl_path)
    surf_solvent_tag = gmsh.model.getEntities(2)[-1][1] 

    # Volumen 1 (Soluto)
    sl_ses = gmsh.model.geo.addSurfaceLoop([surf_ses_tag])
    vol_solute = gmsh.model.geo.addVolume([sl_ses])

    # Volumen 2 (Capa de Solvente)
    sl_solvent_ext = gmsh.model.geo.addSurfaceLoop([surf_solvent_tag])
    vol_solvent = gmsh.model.geo.addVolume([sl_solvent_ext, sl_ses])

    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", algorithm) # 1 = Delaunay, 10 = HXT (Delaunay me tiro error en la generación de malla, x corregir)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [surf_ses_tag])

    surface_size = 1.0 / grid_scale # hmin aproximado en base a triangulos de la ses

    gmsh.model.mesh.field.add("MathEval", 2)
    gmsh.model.mesh.field.setString(2, "F", f"{surface_size} + {gradation} * F1")

    gmsh.model.mesh.field.add("Restrict", 3)
    gmsh.model.mesh.field.setNumber(3, "InField", 2)
    gmsh.model.mesh.field.setNumbers(3, "VolumesList", [vol_solute])
    gmsh.model.mesh.field.setAsBackgroundMesh(3)

    gmsh.model.mesh.field.add("Min", 4)
    gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])

    gmsh.model.mesh.field.setAsBackgroundMesh(4)

    gmsh.model.mesh.generate(3)

    gmsh.model.addPhysicalGroup(3, [vol_solute], tag=1, name="Solute")
    gmsh.model.addPhysicalGroup(3, [vol_solvent], tag=2, name="Solvent")
    gmsh.model.addPhysicalGroup(2, [surf_ses_tag], tag=1, name="SES")
    gmsh.model.addPhysicalGroup(2, [surf_solvent_tag], tag=2, name="Solvent Surface")

    os.makedirs(dir_name+'/volumetric_mesh/', exist_ok=True)
    gmsh.write(dir_name+'/volumetric_mesh/'+ name +'.msh')

def load_dolfin_mesh(molecule, gradation=0.1, probe_radius=1.4, solvent_thickness=3.0, algorithm=10, grid_scale=2.0, grid_scale_ratio=2.0):
    """
    Import or create the volumetric mesh for the input molecule.
    """
    from coupled_solver import dir_name

    mesh_path = os.path.join(dir_name, 'volumetric_mesh', f'{molecule}.msh')
    if not os.path.exists(mesh_path):
        print(f'Volumetric mesh for {molecule} could not be found. Creating...')
        _create_volumetric_mesh(molecule, gradation, probe_radius, solvent_thickness, algorithm, grid_scale, grid_scale_ratio)

    print(f'Loading volumetric mesh from {mesh_path}')
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        mesh_path, 
        MPI.COMM_WORLD, 
        rank=0, 
        gdim=3
        )

    return mesh, cell_tags, facet_tags

def build_boundary_mesh(mesh, facets, cell_tags=None, inner_marker=None):
    """
    Build the boundary mesh from the facets of the volumetric mesh.
    Ensures normals point outwards from the 'inner_marker' domain.
    """

    tdim = mesh.topology.dim
    fdim = tdim - 1

    f2c = mesh.topology.connectivity(fdim, tdim)
    boundary_geo = dolfinx.mesh.entities_to_geometry(mesh, fdim, facets, True)

    tet_indices = []
    for f in facets:
        connected_cells = f2c.links(f)
        if len(connected_cells) == 1:
            # Frontera exterior real, solo hay 1 celda
            tet_indices.append(connected_cells[0])
        elif len(connected_cells) == 2:
            # Frontera interior (SES)
            if cell_tags is None or inner_marker is None:
                raise ValueError("Para fronteras internas, indicar cell_tags e inner_marker.")
            if cell_tags.values[connected_cells[0]] == inner_marker:
                tet_indices.append(connected_cells[0])
            elif cell_tags.values[connected_cells[1]] == inner_marker:
                tet_indices.append(connected_cells[1])
            else:
                raise RuntimeError("Ninguna de las celdas conectadas pertenece al marcador interno proporcionado.")
        else:
            raise RuntimeError("Una faceta está conectada a más de 2 celdas.")
            
    tet_indices = np.array(tet_indices, dtype=np.int32)

    tet_geo = dolfinx.mesh.entities_to_geometry(mesh, tdim, tet_indices, False)
    
    bm_nodes = []
    bm_cells = []
    seen = set()
    
    def add_node(g):
        if g not in seen:
            seen.add(g)
            bm_nodes.append(g)
    
    X = mesh.geometry.x
    
    for tri_geo, tet_g in zip(boundary_geo, tet_geo):
        tri_geo = list(tri_geo)
    
        for g in tri_geo:
            add_node(int(g))
    
        tri_set = set(tri_geo)
        v_opposite = None
        for g in tet_g:
            gg = int(g)
            if gg not in tri_set:
                v_opposite = gg
                break
    
        v0, v1, v2 = X[tri_geo[0]], X[tri_geo[1]], X[tri_geo[2]]
        v3 = X[v_opposite]
    
        normal = np.cross(v1 - v0, v2 - v0)
        to_other = v3 - v0
    
        if np.dot(normal, to_other) > 0:
            tri_geo = [tri_geo[0], tri_geo[2], tri_geo[1]]
    
        bm_cells.append(tri_geo)
    
    node_map = {g: i for i, g in enumerate(bm_nodes)}
    cells = np.array([[node_map[g] for g in tri] for tri in bm_cells], dtype=np.int64).T
    coords = X[np.array(bm_nodes, dtype=np.int64)].T
    
    grid = bempp_cl.api.Grid(coords, cells)
    
    if mesh.comm.rank == 0:
        print(f"Boundary mesh created: {grid.number_of_elements} elements, {grid.number_of_vertices} vertices")
    
    return grid, bm_nodes 