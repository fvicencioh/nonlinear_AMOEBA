import numpy as np
from dolfinx import geometry


def _closest_point_on_segment(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def _closest_point_on_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return np.linalg.norm(p - a)

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return np.linalg.norm(p - b)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        closest = a + v * ab
        return np.linalg.norm(p - closest)

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return np.linalg.norm(p - c)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        closest = a + w * ac
        return np.linalg.norm(p - closest)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        closest = b + w * (c - b)
        return np.linalg.norm(p - closest)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    closest = a + v * ab + w * ac
    return np.linalg.norm(p - closest)


_TET_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
_TET_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def check_charge_mesh_distance(mesh, x_q, cell_tags=None, solute_marker=None,
                                verbose=True):
    tdim = mesh.topology.dim
    x_q = np.asarray(x_q, dtype=np.float64)

    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    coords = mesh.geometry.x

    bb_tree = geometry.bb_tree(mesh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, x_q)
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, x_q)

    solute_cells = None
    if cell_tags is not None and solute_marker is not None:
        solute_cells = set(cell_tags.find(solute_marker).tolist())

    results = []
    for i, p in enumerate(x_q):
        links = colliding_cells.links(i)
        n_candidates = len(cell_candidates.links(i))

        if len(links) == 0:
            print(f"[carga {i}] Ojito: no se encontro celda que "
                  f"contenga el punto {p}. Puede estar fuera del dominio "
                  f"o justo en el borde con tolerancia insuficiente.")
            results.append(None)
            continue

        # si hay filtro de soluto, preferir una celda que este en Omega^-
        cell = links[0]
        if solute_cells is not None:
            for c in links:
                if c in solute_cells:
                    cell = c
                    break

        verts = c2v.links(cell)
        pts = coords[verts]  # (4,3)

        face_dists = [_closest_point_on_triangle(p, pts[a], pts[b], pts[c])
                      for (a, b, c) in _TET_FACES]
        edge_dists = [_closest_point_on_segment(p, pts[a], pts[b])
                      for (a, b) in _TET_EDGES]
        edge_lengths = [np.linalg.norm(pts[a] - pts[b]) for (a, b) in _TET_EDGES]

        min_face = min(face_dists)
        min_edge = min(edge_dists)
        h_local = min(edge_lengths)
        ratio = min_face / h_local if h_local > 0 else np.nan

        results.append(dict(
            cell=cell,
            n_candidates=n_candidates,
            min_dist_face=min_face,
            min_dist_edge=min_edge,
            h_local=h_local,
            ratio_face_over_h=ratio,
        ))

        if verbose:
            flag = "  <-- SOSPECHOSO (revisar)" if ratio < 0.02 else ""
            print(f"[carga {i}] celda={cell}  candidatos_bbox={n_candidates}  "
                  f"dist_min_cara={min_face:.4e}  dist_min_arista={min_edge:.4e}  "
                  f"h_local={h_local:.4e}  ratio={ratio:.4e}{flag}")

    return results


if __name__ == "__main__":
    print(__doc__)
