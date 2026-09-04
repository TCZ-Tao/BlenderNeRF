"""Gaussian Splatting initialization point cloud (points3d.ply).

Volume AABB matches the 3DGS NeRF-synthetic random-box init.
Surface sampling ports the mesh pipeline from scripts/extract_gt_pointcloud.py
(contact-face culling, area-uniform candidates, Poisson-disk filtering) but the
PLY is still 3DGS init: random SH0 colors and zero normals.
"""

from __future__ import annotations

import math
import os

import bpy
import numpy as np

from . import gbuffer


# 3DGS SH0 coefficient used when converting random harmonics to RGB
SPLATS_SH_C0 = 0.28209479177387814

CANDIDATE_OVERSAMPLE = 6.0
RADIUS_SCALE = 0.85
FIT_RADIUS = True
RADIUS_FIT_ITERS = 10
NORMAL_KEEP_DOT = 0.25
MIN_POINTS_PER_MATERIAL = 8
CONTACT_OPPOSE_DOT = -0.5
INSIDE_EPS = 1e-6
AREA_EPS = 1e-18
_INSIDE_RAY_DIR = (
    0.991 / math.sqrt(0.991**2 + 0.087**2 + 0.087**2),
    0.087 / math.sqrt(0.991**2 + 0.087**2 + 0.087**2),
    0.087 / math.sqrt(0.991**2 + 0.087**2 + 0.087**2),
)
_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
)


def _original(obj):
    return obj.original if hasattr(obj, 'original') else obj


def is_exportable_instance(inst, is_object_visible) -> bool:
    '''Render-visible MESH, including Geometry Nodes / collection instances.'''
    obj = inst.object
    if obj is None or obj.type != 'MESH':
        return False

    orig = _original(obj)
    if inst.is_instance:
        parent = inst.parent
        if parent is None:
            return bool(is_object_visible(orig))
        parent_orig = _original(parent)
        return bool(is_object_visible(parent_orig))

    return bool(is_object_visible(orig))


def has_exportable_mesh(is_object_visible) -> bool:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for inst in depsgraph.object_instances:
        if is_exportable_instance(inst, is_object_visible):
            return True
    return False


def _usable_material(mat) -> bool:
    return mat is not None and mat.use_nodes and mat.node_tree is not None


def collect_materials(scene, depsgraph, is_object_visible) -> list:
    mats = list(gbuffer.mesh_materials(scene))
    seen = {mat.name for mat in mats}

    for inst in depsgraph.object_instances:
        if not is_exportable_instance(inst, is_object_visible):
            continue
        for slot in inst.object.material_slots:
            mat = slot.material
            if not _usable_material(mat) or mat.name in seen:
                continue
            seen.add(mat.name)
            mats.append(mat)
    return mats


def material_id_map(materials) -> dict[str, int]:
    explicit = {}
    for mat in materials:
        if 'gt_material_id' in mat:
            explicit[mat.name] = int(mat['gt_material_id'])

    if not explicit:
        return {mat.name: i for i, mat in enumerate(materials, start=1)}

    used = set(explicit.values())
    result = {}
    next_id = 1
    for mat in materials:
        if mat.name in explicit:
            result[mat.name] = explicit[mat.name]
            continue
        while next_id in used:
            next_id += 1
        result[mat.name] = next_id
        used.add(next_id)
        next_id += 1
    return result


def _slot_global_ids(eval_obj, id_map: dict[str, int]) -> np.ndarray:
    ids = []
    for slot in eval_obj.material_slots:
        mat = slot.material
        if _usable_material(mat):
            ids.append(id_map.get(mat.name, 0))
        else:
            ids.append(0)
    if not ids:
        return np.zeros(1, dtype=np.int32)
    return np.asarray(ids, dtype=np.int32)


def _normal_matrix(matrix_world: np.ndarray) -> np.ndarray:
    rotation = matrix_world[:3, :3]
    try:
        return np.linalg.inv(rotation).T
    except np.linalg.LinAlgError:
        return np.linalg.pinv(rotation).T


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, np.maximum(lengths, 1e-12))


def _read_split_normals(mesh, n_tris: int, tri_loop_idx: np.ndarray) -> np.ndarray:
    split = np.empty(n_tris * 9, dtype=np.float32)
    try:
        mesh.loop_triangles.foreach_get('split_normals', split)
        split = split.reshape(n_tris, 3, 3)
        if float(np.max(np.abs(split))) > 1e-8:
            return split
    except Exception:
        pass

    n_loops = len(mesh.loops)
    loop_n = np.empty(n_loops * 3, dtype=np.float32)
    try:
        if hasattr(mesh, 'calc_normals_split'):
            mesh.calc_normals_split()
        mesh.loops.foreach_get('normal', loop_n)
    except Exception:
        loop_n[:] = 0.0
    loop_n = loop_n.reshape(n_loops, 3)
    return loop_n[tri_loop_idx]


def extract_instance_triangles(inst, depsgraph, id_map: dict[str, int]):
    eval_obj = inst.object
    matrix_world = np.array(inst.matrix_world, dtype=np.float64)
    mesh = eval_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()

        n_verts = len(mesh.vertices)
        n_tris = len(mesh.loop_triangles)
        if n_verts == 0 or n_tris == 0:
            return None

        verts_local = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', verts_local)
        verts_local = verts_local.reshape(n_verts, 3).astype(np.float64, copy=False)

        rotation = matrix_world[:3, :3]
        translation = matrix_world[:3, 3]
        verts_world = verts_local @ rotation.T + translation

        tri_vert_idx = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices', tri_vert_idx)
        tri_vert_idx = tri_vert_idx.reshape(n_tris, 3)

        tri_loop_idx = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get('loops', tri_loop_idx)
        tri_loop_idx = tri_loop_idx.reshape(n_tris, 3)

        local_ids = np.empty(n_tris, dtype=np.int32)
        mesh.loop_triangles.foreach_get('material_index', local_ids)
        slot_ids = _slot_global_ids(eval_obj, id_map)
        local_ids = np.clip(local_ids, 0, len(slot_ids) - 1)
        face_material_id = slot_ids[local_ids]

        corners = verts_world[tri_vert_idx]
        edge1 = corners[:, 1] - corners[:, 0]
        edge2 = corners[:, 2] - corners[:, 0]
        cross = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        geom_n = _normalize_rows(cross)

        split_local = _read_split_normals(mesh, n_tris, tri_loop_idx).astype(
            np.float64, copy=False
        )
        nmat = _normal_matrix(matrix_world)
        split_world = split_local.reshape(-1, 3) @ nmat.T
        split_world = _normalize_rows(split_world).reshape(n_tris, 3, 3)

        valid = areas > AREA_EPS
        if not np.any(valid):
            return None

        return (
            corners[valid].astype(np.float32),
            split_world[valid].astype(np.float32),
            areas[valid].astype(np.float64),
            geom_n[valid].astype(np.float32),
            np.ascontiguousarray(face_material_id[valid], dtype=np.int32),
        )
    finally:
        eval_obj.to_mesh_clear()


def collect_triangles(depsgraph, id_map: dict[str, int], is_object_visible):
    corner_chunks: list[np.ndarray] = []
    split_chunks: list[np.ndarray] = []
    area_chunks: list[np.ndarray] = []
    geom_chunks: list[np.ndarray] = []
    mat_chunks: list[np.ndarray] = []
    obj_chunks: list[np.ndarray] = []
    face_chunks: list[np.ndarray] = []
    face_offset = 0
    object_id = 0

    for inst in depsgraph.object_instances:
        if not is_exportable_instance(inst, is_object_visible):
            continue
        result = extract_instance_triangles(inst, depsgraph, id_map)
        if result is None:
            continue

        corners, split_n, areas, geom_n, face_mat = result
        n_tris = len(areas)

        corner_chunks.append(corners)
        split_chunks.append(split_n)
        area_chunks.append(areas)
        geom_chunks.append(geom_n)
        mat_chunks.append(face_mat)
        obj_chunks.append(np.full(n_tris, object_id, dtype=np.int32))
        face_chunks.append(
            np.arange(face_offset, face_offset + n_tris, dtype=np.int32)
        )
        face_offset += n_tris
        object_id += 1

    if not corner_chunks:
        raise RuntimeError(
            'Gaussian Points (Surface) requires at least one visible mesh with triangles.'
        )

    return (
        np.concatenate(corner_chunks, axis=0),
        np.concatenate(split_chunks, axis=0),
        np.concatenate(area_chunks, axis=0),
        np.concatenate(geom_chunks, axis=0),
        np.concatenate(mat_chunks, axis=0),
        np.concatenate(obj_chunks, axis=0),
        np.concatenate(face_chunks, axis=0),
    )


def _iter_id_groups(ids: np.ndarray):
    order = np.argsort(ids, kind='stable')
    sorted_ids = ids[order]
    bounds = np.concatenate(
        (
            [0],
            np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1,
            [len(order)],
        )
    )
    for start, end in zip(bounds[:-1], bounds[1:]):
        yield int(sorted_ids[start]), order[start:end]


def resolve_contact_gap(corners: np.ndarray, contact_gap: float | None) -> float:
    if contact_gap is not None and contact_gap > 0.0:
        return float(contact_gap)
    pts = corners.reshape(-1, 3)
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    gap = max(1e-4, 0.008 * diag)
    print(
        f'[BlenderNeRF] Surface contact-gap auto {gap:.5f} '
        f'(bbox diagonal {diag:.5f} x 0.8%)'
    )
    return gap


def _bvh_from_corners(corners: np.ndarray):
    from mathutils.bvhtree import BVHTree

    n_tris = len(corners)
    if n_tris == 0:
        return None
    verts = np.ascontiguousarray(corners, dtype=np.float64).reshape(-1, 3).tolist()
    faces = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(n_tris)]
    return BVHTree.FromPolygons(verts, faces, all_triangles=True)


def build_object_spatial_index(corners: np.ndarray, object_ids: np.ndarray):
    n_obj = int(object_ids.max()) + 1 if len(object_ids) else 0
    trees: list = [None] * n_obj
    mins = np.zeros((n_obj, 3), dtype=np.float64)
    maxs = np.zeros((n_obj, 3), dtype=np.float64)
    for oid, idx in _iter_id_groups(object_ids):
        sub = corners[idx]
        trees[oid] = _bvh_from_corners(sub)
        pts = sub.reshape(-1, 3)
        mins[oid] = pts.min(axis=0)
        maxs[oid] = pts.max(axis=0)
    return trees, mins, maxs


def _count_ray_hits(tree, origin, direction, max_dist, eps: float = 1e-5) -> int:
    ox, oy, oz = origin
    dx, dy, dz = direction
    remaining = float(max_dist)
    count = 0
    for _ in range(64):
        hit = tree.ray_cast((ox, oy, oz), (dx, dy, dz), remaining)
        if not hit or hit[0] is None:
            break
        dist = float(hit[3])
        count += 1
        step = max(dist, 0.0) + eps
        ox += dx * step
        oy += dy * step
        oz += dz * step
        remaining -= step
        if remaining <= eps:
            break
    return count


def _point_inside_other(tree, pxyz, max_dist: float) -> bool:
    hit = tree.find_nearest(pxyz, max_dist)
    if not hit or hit[0] is None:
        return False
    loc, hit_n, _idx, _dist = hit
    signed = (
        (pxyz[0] - loc[0]) * hit_n[0]
        + (pxyz[1] - loc[1]) * hit_n[1]
        + (pxyz[2] - loc[2]) * hit_n[2]
    )
    if signed < -INSIDE_EPS:
        return True
    if signed > INSIDE_EPS:
        return False
    origin = (
        pxyz[0] + _INSIDE_RAY_DIR[0] * INSIDE_EPS,
        pxyz[1] + _INSIDE_RAY_DIR[1] * INSIDE_EPS,
        pxyz[2] + _INSIDE_RAY_DIR[2] * INSIDE_EPS,
    )
    return _count_ray_hits(tree, origin, _INSIDE_RAY_DIR, max_dist) % 2 == 1


def contact_occluded_mask(
    positions: np.ndarray,
    normals: np.ndarray,
    object_ids: np.ndarray,
    trees: list,
    mins: np.ndarray,
    maxs: np.ndarray,
    gap: float,
    oppose_dot: float,
) -> np.ndarray:
    n = len(positions)
    occluded = np.zeros(n, dtype=bool)
    if n == 0 or not trees:
        return occluded

    pad = max(float(gap), 0.0)
    behind = -0.25 * pad if pad > 0.0 else 0.0
    n_obj = len(trees)
    diags = np.linalg.norm(maxs - mins, axis=1) + 1e-4
    n_contact = 0
    n_inside = 0

    for src_id, idx in _iter_id_groups(object_ids):
        if src_id < 0 or src_id >= n_obj:
            continue
        overlap = np.all(mins <= maxs[src_id] + pad, axis=1) & np.all(
            maxs >= mins[src_id] - pad, axis=1
        )
        if src_id < len(overlap):
            overlap[src_id] = False
        neighbors = np.flatnonzero(overlap)
        if len(neighbors) == 0:
            continue

        pts = np.asarray(positions[idx], dtype=np.float64)
        nrm = np.asarray(normals[idx], dtype=np.float64)
        local = np.zeros(len(idx), dtype=bool)
        for other in neighbors:
            tree = trees[int(other)]
            if tree is None:
                continue
            jmin = mins[other]
            jmax = maxs[other]
            in_aabb = np.all(pts >= jmin, axis=1) & np.all(pts <= jmax, axis=1)
            in_gap = (
                np.all(pts >= jmin - pad, axis=1) & np.all(pts <= jmax + pad, axis=1)
                if pad > 0.0
                else in_aabb
            )
            maybe = (in_aabb | in_gap) & ~local
            other_diag = float(diags[other])
            for k in np.flatnonzero(maybe):
                p = pts[k]
                pxyz = (float(p[0]), float(p[1]), float(p[2]))
                nvec = nrm[k]

                if pad > 0.0 and in_gap[k]:
                    hit = tree.find_nearest(pxyz, pad)
                    if hit and hit[0] is not None:
                        loc, hit_n, _hit_i, _dist = hit
                        if (
                            (loc[0] - p[0]) * nvec[0]
                            + (loc[1] - p[1]) * nvec[1]
                            + (loc[2] - p[2]) * nvec[2]
                            >= behind
                            and hit_n[0] * nvec[0]
                            + hit_n[1] * nvec[1]
                            + hit_n[2] * nvec[2]
                            <= oppose_dot
                        ):
                            local[k] = True
                            n_contact += 1
                            continue

                if in_aabb[k] and _point_inside_other(tree, pxyz, other_diag):
                    local[k] = True
                    n_inside += 1
        occluded[idx] = local

    print(
        f'[BlenderNeRF] Surface occlusion: contact {n_contact}, '
        f'inside {n_inside}, total {int(occluded.sum())}/{n}'
    )
    return occluded


def cull_contact_triangles(
    corners: np.ndarray,
    split_n: np.ndarray,
    areas: np.ndarray,
    geom_n: np.ndarray,
    tri_mat: np.ndarray,
    tri_obj: np.ndarray,
    tri_face: np.ndarray,
    gap: float,
    oppose_dot: float,
):
    trees, mins, maxs = build_object_spatial_index(corners, tri_obj)
    centroids = corners.mean(axis=1)
    occluded = contact_occluded_mask(
        centroids, geom_n, tri_obj, trees, mins, maxs, gap, oppose_dot
    )
    n_all = len(areas)
    n_drop = int(occluded.sum())
    area_drop = float(areas[occluded].sum()) if n_drop else 0.0
    print(
        f'[BlenderNeRF] Surface triangles kept {n_all - n_drop}/{n_all} '
        f'({100.0 * (n_all - n_drop) / max(n_all, 1):.1f}%), '
        f'culled area {area_drop:.4f}/{float(areas.sum()):.4f}, gap={gap:.5f}'
    )
    keep = ~occluded
    if not np.any(keep):
        raise RuntimeError(
            'Gaussian Points (Surface) has no triangles left after contact-face culling.'
        )
    return (
        corners[keep],
        split_n[keep],
        areas[keep],
        geom_n[keep],
        tri_mat[keep],
        tri_obj[keep],
        tri_face[keep],
        trees,
        mins,
        maxs,
    )


def sample_area_uniform(
    corners: np.ndarray,
    split_n: np.ndarray,
    geom_n: np.ndarray,
    areas: np.ndarray,
    material_ids: np.ndarray,
    object_ids: np.ndarray,
    face_ids: np.ndarray,
    n_candidates: int,
    rng: np.random.Generator,
):
    total_area = float(areas.sum())
    if total_area <= 0.0:
        raise RuntimeError('Gaussian Points (Surface): visible triangle area is 0.')

    cum = np.cumsum(areas)
    picks = rng.random(n_candidates) * total_area
    tri_idx = np.searchsorted(cum, picks, side='right')
    np.clip(tri_idx, 0, len(areas) - 1, out=tri_idx)

    r1 = rng.random(n_candidates)
    r2 = rng.random(n_candidates)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = r2 * sqrt_r1
    w = 1.0 - u - v
    bary = np.stack((u, v, w), axis=1).astype(np.float32)

    tri_corners = corners[tri_idx]
    points = (
        bary[:, 0:1] * tri_corners[:, 0]
        + bary[:, 1:2] * tri_corners[:, 1]
        + bary[:, 2:3] * tri_corners[:, 2]
    )

    tri_split = split_n[tri_idx]
    normals = (
        bary[:, 0:1] * tri_split[:, 0]
        + bary[:, 1:2] * tri_split[:, 1]
        + bary[:, 2:3] * tri_split[:, 2]
    )
    geometric = geom_n[tri_idx]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    degenerate = lengths[:, 0] < 1e-12
    normals = np.divide(normals, np.maximum(lengths, 1e-12))
    normals[degenerate] = geometric[degenerate]

    return (
        np.ascontiguousarray(points, dtype=np.float32),
        np.ascontiguousarray(normals, dtype=np.float32),
        np.ascontiguousarray(geometric, dtype=np.float32),
        bary,
        material_ids[tri_idx].astype(np.int32, copy=False),
        object_ids[tri_idx].astype(np.int32, copy=False),
        face_ids[tri_idx].astype(np.int32, copy=False),
        total_area,
    )


def _poisson_one_group(
    points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    min_dot: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    if n == 1 or radius <= 0.0:
        return np.arange(n, dtype=np.int32)

    order = rng.permutation(n)
    inv_r = 1.0 / radius
    r2 = radius * radius
    grid: dict[tuple[int, int, int], list[tuple[float, float, float, float, float, float]]] = {}
    kept: list[int] = []

    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)

    for i in order:
        px, py, pz = pts[i]
        nx, ny, nz = nrm[i]
        cx = int(math.floor(px * inv_r))
        cy = int(math.floor(py * inv_r))
        cz = int(math.floor(pz * inv_r))

        conflict = False
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            bucket = grid.get((cx + dx, cy + dy, cz + dz))
            if not bucket:
                continue
            for qx, qy, qz, qnx, qny, qnz in bucket:
                if nx * qnx + ny * qny + nz * qnz < min_dot:
                    continue
                ddx = px - qx
                ddy = py - qy
                ddz = pz - qz
                if ddx * ddx + ddy * ddy + ddz * ddz < r2:
                    conflict = True
                    break
            if conflict:
                break

        if conflict:
            continue
        grid.setdefault((cx, cy, cz), []).append((px, py, pz, nx, ny, nz))
        kept.append(int(i))

    return np.asarray(kept, dtype=np.int32)


def _iter_surface_groups(object_ids: np.ndarray, material_ids: np.ndarray):
    keys = (object_ids.astype(np.int64, copy=False) << 32) | (
        material_ids.astype(np.int64, copy=False) & np.int64(0xFFFFFFFF)
    )
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    bounds = np.concatenate(
        (
            [0],
            np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1,
            [len(order)],
        )
    )
    for start, end in zip(bounds[:-1], bounds[1:]):
        yield order[start:end]


def poisson_filter_surface_aware(
    points: np.ndarray,
    normals: np.ndarray,
    material_ids: np.ndarray,
    object_ids: np.ndarray,
    radius: float,
    min_dot: float,
    rng: np.random.Generator,
) -> np.ndarray:
    kept_parts: list[np.ndarray] = []
    for idx in _iter_surface_groups(object_ids, material_ids):
        local = _poisson_one_group(points[idx], normals[idx], radius, min_dot, rng)
        if len(local):
            kept_parts.append(idx[local])
    if not kept_parts:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(kept_parts, axis=0)


def fit_poisson_radius(
    points: np.ndarray,
    normals: np.ndarray,
    material_ids: np.ndarray,
    object_ids: np.ndarray,
    target_n: int,
    radius0: float,
    min_dot: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    poisson_seed = int(rng.integers(0, 2**31 - 1))

    def _run(radius: float) -> np.ndarray:
        return poisson_filter_surface_aware(
            points,
            normals,
            material_ids,
            object_ids,
            radius,
            min_dot,
            np.random.default_rng(poisson_seed),
        )

    kept = _run(radius0)
    if not FIT_RADIUS or target_n <= 0:
        return radius0, kept

    best_radius = radius0
    best = kept
    best_err = abs(len(kept) - target_n)
    if best_err <= max(8, int(0.05 * target_n)):
        return best_radius, best

    if len(kept) > target_n:
        lo, hi = radius0, radius0 * 8.0
    else:
        lo, hi = max(radius0 / 8.0, 1e-12), radius0

    for _ in range(RADIUS_FIT_ITERS):
        mid = math.sqrt(lo * hi)
        kept = _run(mid)
        err = abs(len(kept) - target_n)
        if err < best_err:
            best_err = err
            best = kept
            best_radius = mid
        if len(kept) > target_n:
            lo = mid
        else:
            hi = mid
        if err <= max(8, int(0.05 * target_n)):
            break

    return best_radius, best


def ensure_min_points_per_material(
    kept: np.ndarray,
    material_ids: np.ndarray,
    min_count: int,
) -> np.ndarray:
    if min_count <= 0 or len(material_ids) == 0:
        return kept

    kept_set = set(int(i) for i in kept.tolist())
    extra: list[int] = []
    unique_ids = np.unique(material_ids)
    for mat_id in unique_ids.tolist():
        cand = np.flatnonzero(material_ids == mat_id)
        have = sum(1 for i in cand.tolist() if i in kept_set)
        if have >= min_count:
            continue
        need = min_count - have
        for i in cand.tolist():
            if i in kept_set:
                continue
            extra.append(i)
            kept_set.add(i)
            need -= 1
            if need <= 0:
                break

    if not extra:
        return kept
    return np.concatenate((kept, np.asarray(extra, dtype=np.int32)))


def sample_surface_xyz(scene, target_n: int, rng: np.random.Generator, is_object_visible) -> np.ndarray:
    view_layer = bpy.context.view_layer
    if view_layer is not None:
        view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    materials = collect_materials(scene, depsgraph, is_object_visible)
    id_map = material_id_map(materials)
    corners, split_n, areas, geom_n, tri_mat, tri_obj, tri_face = collect_triangles(
        depsgraph, id_map, is_object_visible
    )

    cull_occluded = bool(getattr(scene, 'splats_cull_occluded', True))
    used_gap = 0.0
    occ_trees = occ_mins = occ_maxs = None
    if cull_occluded:
        used_gap = resolve_contact_gap(corners, None)
        (
            corners,
            split_n,
            areas,
            geom_n,
            tri_mat,
            tri_obj,
            tri_face,
            occ_trees,
            occ_mins,
            occ_maxs,
        ) = cull_contact_triangles(
            corners,
            split_n,
            areas,
            geom_n,
            tri_mat,
            tri_obj,
            tri_face,
            used_gap,
            CONTACT_OPPOSE_DOT,
        )
    else:
        print('[BlenderNeRF] Surface occlusion culling off')

    n_candidates = int(math.ceil(target_n * CANDIDATE_OVERSAMPLE))
    print(
        f'[BlenderNeRF] Surface sampling {n_candidates} candidates '
        f'(target={target_n}, oversample={CANDIDATE_OVERSAMPLE:g})'
    )

    (
        cand_points,
        cand_normals,
        cand_geom,
        _cand_bary,
        cand_mat,
        cand_obj,
        _cand_face,
        total_area,
    ) = sample_area_uniform(
        corners,
        split_n,
        geom_n,
        areas,
        tri_mat,
        tri_obj,
        tri_face,
        n_candidates,
        rng,
    )

    if cull_occluded and occ_trees is not None:
        cand_occ = contact_occluded_mask(
            cand_points,
            cand_geom,
            cand_obj,
            occ_trees,
            occ_mins,
            occ_maxs,
            used_gap,
            CONTACT_OPPOSE_DOT,
        )
        n_drop = int(cand_occ.sum())
        if n_drop:
            keep_c = ~cand_occ
            cand_points = cand_points[keep_c]
            cand_normals = cand_normals[keep_c]
            cand_geom = cand_geom[keep_c]
            cand_mat = cand_mat[keep_c]
            cand_obj = cand_obj[keep_c]
            print(
                f'[BlenderNeRF] Surface dropped {n_drop} contact candidates, '
                f'{len(cand_points)} left'
            )
        if len(cand_points) == 0:
            raise RuntimeError(
                'Gaussian Points (Surface) has no candidates left after contact-face culling.'
            )

    radius0 = RADIUS_SCALE * math.sqrt(float(total_area) / float(target_n))
    print(f'[BlenderNeRF] Surface Poisson radius0={radius0:.6f}')

    sample_radius, kept = fit_poisson_radius(
        cand_points,
        cand_normals,
        cand_mat,
        cand_obj,
        target_n,
        radius0,
        NORMAL_KEEP_DOT,
        rng,
    )
    kept = ensure_min_points_per_material(kept, cand_mat, MIN_POINTS_PER_MATERIAL)
    if len(kept) == 0:
        raise RuntimeError('Gaussian Points (Surface): Poisson filtering kept 0 points.')

    points = np.ascontiguousarray(cand_points[kept], dtype=np.float64)
    print(
        f'[BlenderNeRF] Surface kept {len(points)} points '
        f'(target {target_n}, radius {sample_radius:.6f})'
    )
    return points


def sample_volume_xyz(mins, maxs, n: int, rng: np.random.Generator) -> np.ndarray:
    lo = np.array(mins, dtype=np.float64)
    hi = np.array(maxs, dtype=np.float64)
    return rng.random((n, 3)) * (hi - lo) + lo


def write_points3d_ply(filepath: str, xyz: np.ndarray, rng: np.random.Generator) -> None:
    n = len(xyz)
    shs = rng.random((n, 3)) / 255.0
    rgb = np.clip(shs * SPLATS_SH_C0 + 0.5, 0.0, 1.0) * 255.0
    rgb = np.rint(rgb).astype(np.uint8)
    normals = np.zeros((n, 3), dtype=np.float64)

    with open(filepath, 'w', encoding='ascii', newline='\n') as file:
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write(f'element vertex {n}\n')
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        file.write('property float nx\n')
        file.write('property float ny\n')
        file.write('property float nz\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')
        file.write('end_header\n')
        np.savetxt(
            file,
            np.column_stack((xyz, normals, rgb)),
            fmt='%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d',
        )


def save_points3d_ply(scene, directory, is_object_visible, world_aabb) -> str:
    n = int(scene.splats_nb_points)
    rng = np.random.default_rng(scene.seed)
    mode = getattr(scene, 'splats_sample_mode', 'AABB')

    if mode == 'SURFACE':
        xyz = sample_surface_xyz(scene, n, rng, is_object_visible)
    else:
        mins, maxs = world_aabb(scene)
        if mins is None:
            raise RuntimeError(
                'Gaussian Points requires at least one visible mesh to compute the scene AABB.'
            )
        xyz = sample_volume_xyz(mins, maxs, n, rng)

    filepath = os.path.join(directory, 'points3d.ply')
    write_points3d_ply(filepath, xyz, rng)
    return filepath
