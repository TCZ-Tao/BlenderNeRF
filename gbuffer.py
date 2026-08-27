"""G-buffer / PBR map rendering for BlenderNeRF.

Ports the still-image pipeline from scripts/pbr_maps_render.py so each selected
channel can be rendered as a full train/test animation pass.
"""

from __future__ import annotations

import colorsys
import json
import os
import traceback

import bpy


SAMPLES = 1
USE_TRANSPARENT_BG = True
VIEW_Z_TOWARDS_CAMERA = True
MATERIAL_ID_BACKGROUND = 0
ID_FILTER_WIDTH = 0.0
EEVEE_ENGINE = "BLENDER_EEVEE"

HELPER_PREFIX = "_PBRMAP_"
ID_WORLD_NAME = "_PBRMAP_ID_WORLD"

RGBA_CHANNEL = "rgba"

DATA_CHANNELS = (
    "albedo",
    "roughness",
    "metallic",
    "geometric_normal",
    "shading_normal",
    "linear_depth",
    "material_id",
    "material_id_vis",
)

CHANNEL_PROPS = (
    ("gbuffer_rgba", RGBA_CHANNEL),
    ("gbuffer_albedo", "albedo"),
    ("gbuffer_roughness", "roughness"),
    ("gbuffer_metallic", "metallic"),
    ("gbuffer_geometric_normal", "geometric_normal"),
    ("gbuffer_shading_normal", "shading_normal"),
    ("gbuffer_linear_depth", "linear_depth"),
    ("gbuffer_material_id", "material_id"),
    ("gbuffer_material_id_vis", "material_id_vis"),
)

_DATA_CHANNELS = {
    "roughness",
    "metallic",
    "geometric_normal",
    "shading_normal",
    "linear_depth",
    "material_id",
}
_EXR_CHANNELS = {"material_id", "linear_depth"}
_OPAQUE_CHANNELS = {"material_id", "material_id_vis", "linear_depth"}
_NORMAL_CHANNELS = {"geometric_normal", "shading_normal"}
_ID_WORLD_CHANNELS = {"material_id", "material_id_vis", "linear_depth"}

_CHANNEL_SOCKET = {
    "albedo": "Base Color",
    "roughness": "Roughness",
    "metallic": "Metallic",
}

_KELLY_MAX_CONTRAST = (
    (0.953, 0.765, 0.000),
    (0.529, 0.337, 0.573),
    (0.953, 0.518, 0.000),
    (0.631, 0.792, 0.945),
    (0.745, 0.000, 0.196),
    (0.761, 0.698, 0.502),
    (0.518, 0.518, 0.510),
    (0.000, 0.533, 0.337),
    (0.902, 0.561, 0.675),
    (0.000, 0.404, 0.647),
    (0.976, 0.576, 0.475),
    (0.376, 0.306, 0.592),
    (0.965, 0.651, 0.000),
    (0.702, 0.267, 0.424),
    (0.863, 0.827, 0.000),
    (0.533, 0.176, 0.090),
    (0.553, 0.714, 0.000),
    (0.396, 0.271, 0.133),
    (0.886, 0.345, 0.133),
    (0.169, 0.239, 0.149),
)
_GOLDEN_HUE = 0.38196601125

_job: dict = {
    "passes": [],
    "index": 0,
    "render_snapshot": None,
    "overrides": [],
}


def selected_output_channels(scene) -> list[str]:
    """Channels that should be written. When G-buffer is off, RGB is a single implicit pass."""
    if not scene.gbuffer:
        return [RGBA_CHANNEL]
    return [name for prop, name in CHANNEL_PROPS if getattr(scene, prop)]


def needs_mesh_materials(scene) -> bool:
    if not scene.gbuffer:
        return False
    return any(ch != RGBA_CHANNEL for ch in selected_output_channels(scene))


def build_passes(do_train: bool, do_test: bool, scene) -> list[tuple[str, str]]:
    channels = selected_output_channels(scene)
    passes: list[tuple[str, str]] = []
    if do_train:
        passes.extend(("train", ch) for ch in channels)
    if do_test:
        passes.extend(("test", ch) for ch in channels)
    return passes


def current_pass() -> tuple[str, str] | None:
    passes = _job.get("passes") or []
    index = _job.get("index", 0)
    if index < 0 or index >= len(passes):
        return None
    return passes[index]


def advance_pass() -> bool:
    _job["index"] = int(_job.get("index", 0)) + 1
    return current_pass() is not None


def begin_job(scene, passes: list[tuple[str, str]]) -> None:
    end_job(scene)
    repair_disconnected_principleds(scene)
    _job["passes"] = list(passes)
    _job["index"] = 0
    _job["render_snapshot"] = snapshot_render_state(scene)
    _job["overrides"] = []


def end_job(scene) -> None:
    _clear_overrides(_job.get("overrides") or [])
    _sweep_helper_nodes()
    repair_disconnected_principleds(scene)
    snapshot = _job.get("render_snapshot")
    if snapshot is not None:
        try:
            restore_render_state(scene, snapshot)
        except Exception:
            traceback.print_exc()
    _remove_temp_world()
    _job["passes"] = []
    _job["index"] = 0
    _job["render_snapshot"] = None
    _job["overrides"] = []


def apply_pass_settings(scene, channel: str, out_dir: str) -> None:
    """Restore the original render state, then set up this pass's output and overrides."""
    _clear_overrides(_job.get("overrides") or [])
    snapshot = _job.get("render_snapshot")
    if snapshot is not None:
        restore_render_state(scene, snapshot)

    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, "")

    if (not scene.gbuffer) or channel == RGBA_CHANNEL:
        scene.render.filepath = filepath
        return

    _apply_gbuffer_channel(scene, channel, filepath)


def write_material_id_json(scene, output_path: str) -> None:
    if not scene.gbuffer:
        return
    if not (scene.gbuffer_material_id or scene.gbuffer_material_id_vis):
        return
    materials = mesh_materials(scene)
    if not materials:
        return
    _write_id_json(os.path.join(output_path, "material_id.json"), materials)


def mesh_materials(scene) -> list:
    """Collect node materials in object → slot order. First material is ID 1."""
    mats = []
    seen = set()
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.name in seen:
                continue
            if not mat.use_nodes or mat.node_tree is None:
                continue
            seen.add(mat.name)
            mats.append(mat)
    return mats


def _set_emission_color(emission, channel: str, inp) -> None:
    if channel == "albedo":
        emission.inputs["Color"].default_value = tuple(inp.default_value)
        return
    value = float(inp.default_value)
    emission.inputs["Color"].default_value = (value, value, value, 1.0)


def _new_helper(node_tree, bl_idname: str, name: str, location):
    node = node_tree.nodes.new(bl_idname)
    node.name = name[:63]
    node.location = location
    node.hide = True
    return node


def _connect_normal_chain(node_tree, bsdf, emission, index: int, use_shading: bool) -> None:
    origin = (bsdf.location.x + 200, bsdf.location.y)

    geom = _new_helper(node_tree, "ShaderNodeNewGeometry", f"{HELPER_PREFIX}GEO_{index}", origin)
    xform = _new_helper(
        node_tree, "ShaderNodeVectorTransform", f"{HELPER_PREFIX}XF_{index}", (origin[0] + 40, origin[1])
    )
    xform.vector_type = "NORMAL"
    xform.convert_from = "WORLD"
    xform.convert_to = "CAMERA"

    encoded_src = xform.outputs["Vector"]
    loc_x = origin[0] + 80
    if VIEW_Z_TOWARDS_CAMERA:
        flip = _new_helper(
            node_tree, "ShaderNodeVectorMath", f"{HELPER_PREFIX}FLIP_{index}", (loc_x, origin[1])
        )
        flip.operation = "MULTIPLY"
        flip.inputs[1].default_value = (1.0, 1.0, -1.0)
        node_tree.links.new(xform.outputs["Vector"], flip.inputs[0])
        encoded_src = flip.outputs["Vector"]
        loc_x += 40

    mul = _new_helper(
        node_tree, "ShaderNodeVectorMath", f"{HELPER_PREFIX}MUL_{index}", (loc_x, origin[1])
    )
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = (0.5, 0.5, 0.5)

    add = _new_helper(
        node_tree, "ShaderNodeVectorMath", f"{HELPER_PREFIX}ADD_{index}", (loc_x + 40, origin[1])
    )
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)

    src = geom.outputs["Normal"]
    if use_shading:
        normal_in = bsdf.inputs.get("Normal")
        if normal_in is not None and normal_in.is_linked:
            src = normal_in.links[0].from_socket

    node_tree.links.new(src, xform.inputs["Vector"])
    node_tree.links.new(encoded_src, mul.inputs[0])
    node_tree.links.new(mul.outputs["Vector"], add.inputs[0])
    node_tree.links.new(add.outputs["Vector"], emission.inputs["Color"])


def repair_disconnected_principleds(scene=None) -> int:
    """Reconnect a dangling Principled BSDF to Material Output after a failed G-buffer restore."""
    mats = mesh_materials(scene) if scene is not None else [
        mat for mat in bpy.data.materials if mat.use_nodes and mat.node_tree
    ]
    repaired = 0
    for mat in mats:
        nt = mat.node_tree
        if nt is None:
            continue
        outputs = [
            n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output
        ]
        principleds = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"]
        if not outputs or not principleds:
            continue
        bsdf_out = principleds[0].outputs.get("BSDF")
        if bsdf_out is None:
            continue
        for node in outputs:
            surf = node.inputs.get("Surface")
            if surf is None or surf.is_linked:
                continue
            nt.links.new(bsdf_out, surf)
            repaired += 1
    return repaired


def apply_channel_override(node_tree, channel: str) -> list[tuple]:
    """Drive Material Output with an Emission of the chosen PBR channel.

    Always plugs the emission into the active Material Output. Only rewiring
    Principled's existing outgoing links fails when those links are already gone.
    """
    restore_links: list[tuple] = []
    principleds = [n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED"]
    emissions = []

    for i, bsdf in enumerate(principleds):
        bsdf_out = bsdf.outputs.get("BSDF")
        if bsdf_out is None:
            continue

        emission = _new_helper(
            node_tree,
            "ShaderNodeEmission",
            f"{HELPER_PREFIX}EM_{i}_{bsdf.name}",
            (bsdf.location.x + 240, bsdf.location.y),
        )
        emission.label = f"PBR {channel}"
        emission.inputs["Strength"].default_value = 1.0

        if channel in _NORMAL_CHANNELS:
            _connect_normal_chain(
                node_tree, bsdf, emission, i, use_shading=(channel == "shading_normal")
            )
        else:
            socket_name = _CHANNEL_SOCKET.get(channel)
            if socket_name is None or socket_name not in bsdf.inputs:
                node_tree.nodes.remove(emission)
                continue
            inp = bsdf.inputs[socket_name]
            if inp.is_linked:
                node_tree.links.new(inp.links[0].from_socket, emission.inputs["Color"])
            else:
                _set_emission_color(emission, channel, inp)

        emissions.append(emission)

        outgoing = list(bsdf_out.links)
        for link in outgoing:
            to_socket = link.to_socket
            restore_links.append((bsdf_out, to_socket))
            node_tree.links.remove(link)
            node_tree.links.new(emission.outputs["Emission"], to_socket)

    if not emissions:
        return restore_links

    emission_nodes = set(emissions)
    for node in node_tree.nodes:
        if node.type != "OUTPUT_MATERIAL" or not node.is_active_output:
            continue
        surf = node.inputs.get("Surface")
        if surf is None:
            continue
        if surf.is_linked and surf.links[0].from_node in emission_nodes:
            continue
        if surf.is_linked:
            restore_links.append((surf.links[0].from_socket, surf))
            node_tree.links.remove(surf.links[0])
        node_tree.links.new(emissions[0].outputs["Emission"], surf)

    return restore_links


def rgb_to_hex(rgb: tuple[float, ...]) -> str:
    r, g, b = rgb[:3]
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255.0)),
        int(round(g * 255.0)),
        int(round(b * 255.0)),
    )


def id_to_viz_rgba(mat_id: int) -> tuple[float, float, float, float]:
    if mat_id <= 0:
        return (0.0, 0.0, 0.0, 1.0)
    idx = mat_id - 1
    if idx < len(_KELLY_MAX_CONTRAST):
        r, g, b = _KELLY_MAX_CONTRAST[idx]
        return (r, g, b, 1.0)
    hue = ((idx - len(_KELLY_MAX_CONTRAST)) * _GOLDEN_HUE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (r, g, b, 1.0)


def apply_id_override(node_tree, color: tuple[float, ...]) -> list[tuple]:
    restore_links: list[tuple] = []
    r, g, b = color[0], color[1], color[2]
    a = color[3] if len(color) > 3 else 1.0

    emission = _new_helper(node_tree, "ShaderNodeEmission", f"{HELPER_PREFIX}ID_EM", (400, 0))
    emission.label = "ID"
    emission.inputs["Color"].default_value = (r, g, b, a)
    emission.inputs["Strength"].default_value = 1.0

    for node in node_tree.nodes:
        if node.type != "OUTPUT_MATERIAL" or not node.is_active_output:
            continue
        surf = node.inputs.get("Surface")
        if surf is None:
            continue
        if surf.is_linked:
            restore_links.append((surf.links[0].from_socket, surf))
            node_tree.links.remove(surf.links[0])
        node_tree.links.new(emission.outputs["Emission"], surf)

    return restore_links


def apply_depth_override(node_tree) -> list[tuple]:
    restore_links: list[tuple] = []

    outputs = [
        node
        for node in node_tree.nodes
        if node.type == "OUTPUT_MATERIAL" and node.is_active_output
    ]

    cam = _new_helper(node_tree, "ShaderNodeCameraData", f"{HELPER_PREFIX}CAM", (200, 0))
    abs_z = _new_helper(node_tree, "ShaderNodeMath", f"{HELPER_PREFIX}ABSZ", (320, 0))
    abs_z.operation = "ABSOLUTE"
    z_socket = cam.outputs.get("View Z Depth") or cam.outputs.get("View Distance")
    node_tree.links.new(z_socket, abs_z.inputs[0])

    emission = _new_helper(node_tree, "ShaderNodeEmission", f"{HELPER_PREFIX}DEPTH_EM", (460, 0))
    emission.label = "Linear Depth"
    emission.inputs["Strength"].default_value = 1.0
    node_tree.links.new(abs_z.outputs["Value"], emission.inputs["Color"])

    for node in outputs:
        surf = node.inputs.get("Surface")
        if surf is None:
            continue
        if surf.is_linked:
            restore_links.append((surf.links[0].from_socket, surf))
            node_tree.links.remove(surf.links[0])
        node_tree.links.new(emission.outputs["Emission"], surf)

    return restore_links


def restore_node_tree(node_tree, restore_links: list[tuple]) -> None:
    helpers = [n for n in node_tree.nodes if n.name.startswith(HELPER_PREFIX)]
    for node in helpers:
        node_tree.nodes.remove(node)
    for from_socket, to_socket in restore_links:
        try:
            if from_socket.id_data is not None and to_socket.id_data is not None:
                node_tree.links.new(from_socket, to_socket)
        except Exception:
            traceback.print_exc()


def _snapshot_view(view) -> dict:
    return {
        "view_transform": view.view_transform,
        "look": view.look,
        "exposure": view.exposure,
        "gamma": view.gamma,
    }


def _restore_view(view, state: dict) -> None:
    view.view_transform = state["view_transform"]
    view.look = state["look"]
    view.exposure = state["exposure"]
    view.gamma = state["gamma"]


def _eevee_get(scene, name: str, default=None):
    eevee = getattr(scene, "eevee", None)
    if eevee is None:
        return default
    return getattr(eevee, name, default)


def _eevee_set(scene, name: str, value) -> None:
    eevee = getattr(scene, "eevee", None)
    if eevee is None or not hasattr(eevee, name):
        return
    setattr(eevee, name, value)


def snapshot_render_state(scene) -> dict:
    cycles = scene.cycles
    img = scene.render.image_settings
    return {
        "filepath": scene.render.filepath,
        "engine": scene.render.engine,
        "film_transparent": scene.render.film_transparent,
        "use_compositing": scene.render.use_compositing,
        "use_sequencer": scene.render.use_sequencer,
        "dither_intensity": scene.render.dither_intensity,
        "use_motion_blur": scene.render.use_motion_blur,
        "filter_size": scene.render.filter_size,
        "file_format": img.file_format,
        "color_mode": img.color_mode,
        "color_depth": img.color_depth,
        "color_management": img.color_management,
        "exr_codec": img.exr_codec,
        "samples": cycles.samples,
        "use_adaptive_sampling": cycles.use_adaptive_sampling,
        "use_denoising": cycles.use_denoising,
        "pixel_filter_type": cycles.pixel_filter_type,
        "filter_width": cycles.filter_width,
        "eevee_taa_render_samples": _eevee_get(scene, "taa_render_samples"),
        "eevee_use_taa_reprojection": _eevee_get(scene, "use_taa_reprojection"),
        "eevee_use_shadows": _eevee_get(scene, "use_shadows"),
        "eevee_use_raytracing": _eevee_get(scene, "use_raytracing"),
        "eevee_use_fast_gi": _eevee_get(scene, "use_fast_gi"),
        "scene_view": _snapshot_view(scene.view_settings),
        "scene_display": scene.display_settings.display_device,
        "file_view": _snapshot_view(img.view_settings),
        "file_display": img.display_settings.display_device,
        "world": scene.world,
    }


def restore_render_state(scene, state: dict) -> None:
    scene.render.filepath = state["filepath"]
    scene.render.engine = state["engine"]
    scene.render.film_transparent = state["film_transparent"]
    scene.render.use_compositing = state["use_compositing"]
    scene.render.use_sequencer = state["use_sequencer"]
    scene.render.dither_intensity = state["dither_intensity"]
    scene.render.use_motion_blur = state["use_motion_blur"]
    scene.render.filter_size = state["filter_size"]
    img = scene.render.image_settings
    img.file_format = state["file_format"]
    img.color_mode = state["color_mode"]
    img.color_depth = state["color_depth"]
    img.color_management = state["color_management"]
    img.exr_codec = state["exr_codec"]
    scene.cycles.samples = state["samples"]
    scene.cycles.use_adaptive_sampling = state["use_adaptive_sampling"]
    scene.cycles.use_denoising = state["use_denoising"]
    scene.cycles.pixel_filter_type = state["pixel_filter_type"]
    scene.cycles.filter_width = state["filter_width"]
    if state["eevee_taa_render_samples"] is not None:
        _eevee_set(scene, "taa_render_samples", state["eevee_taa_render_samples"])
    if state["eevee_use_taa_reprojection"] is not None:
        _eevee_set(scene, "use_taa_reprojection", state["eevee_use_taa_reprojection"])
    if state["eevee_use_shadows"] is not None:
        _eevee_set(scene, "use_shadows", state["eevee_use_shadows"])
    if state["eevee_use_raytracing"] is not None:
        _eevee_set(scene, "use_raytracing", state["eevee_use_raytracing"])
    if state["eevee_use_fast_gi"] is not None:
        _eevee_set(scene, "use_fast_gi", state["eevee_use_fast_gi"])
    _restore_view(scene.view_settings, state["scene_view"])
    scene.display_settings.display_device = state["scene_display"]
    _restore_view(img.view_settings, state["file_view"])
    img.display_settings.display_device = state["file_display"]
    scene.world = state["world"]


def _apply_color_management(scene, channel: str) -> None:
    view_transform = "Standard" if channel in {"albedo", "material_id_vis"} else "Raw"
    scene.display_settings.display_device = "sRGB"
    try:
        scene.view_settings.view_transform = view_transform
    except TypeError:
        pass
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    img = scene.render.image_settings
    img.color_management = "OVERRIDE"
    img.display_settings.display_device = "sRGB"
    try:
        img.view_settings.view_transform = view_transform
    except TypeError:
        pass
    img.view_settings.look = "None"
    img.view_settings.exposure = 0.0
    img.view_settings.gamma = 1.0


def prepare_render(scene, channel: str, filepath: str) -> None:
    is_exr = channel in _EXR_CHANNELS
    is_opaque = channel in _OPAQUE_CHANNELS
    transparent = USE_TRANSPARENT_BG and not is_opaque

    scene.render.engine = EEVEE_ENGINE
    scene.render.filepath = filepath
    scene.render.film_transparent = transparent
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.dither_intensity = 0.0
    scene.render.use_motion_blur = False
    _eevee_set(scene, "taa_render_samples", max(1, SAMPLES))
    _eevee_set(scene, "use_taa_reprojection", False)
    _eevee_set(scene, "use_shadows", False)
    _eevee_set(scene, "use_raytracing", False)
    _eevee_set(scene, "use_fast_gi", False)
    scene.render.filter_size = ID_FILTER_WIDTH

    img = scene.render.image_settings
    if is_exr:
        img.file_format = "OPEN_EXR"
        img.color_mode = "RGB"
        img.color_depth = "32"
        img.exr_codec = "ZIP"
    else:
        img.file_format = "PNG"
        img.color_mode = "RGBA" if transparent else "RGB"
        img.color_depth = "8"

    _apply_color_management(scene, channel)


def _make_id_world():
    world = bpy.data.worlds.get(ID_WORLD_NAME)
    if world is None:
        world = bpy.data.worlds.new(ID_WORLD_NAME)
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return world


def _remove_temp_world() -> None:
    world = bpy.data.worlds.get(ID_WORLD_NAME)
    if world is not None:
        bpy.data.worlds.remove(world)


def _write_id_json(path: str, materials) -> None:
    payload = {
        "background": MATERIAL_ID_BACKGROUND,
        "format": "OPEN_EXR",
        "color_depth": 32,
        "encoding": "RGB linear float, R=G=B=id; 0=background, 1=first material, ...",
        "filter": f"EEVEE raster, TAA=1, film filter_size={ID_FILTER_WIDTH} (nearest / no AA blend)",
        "viz_palette": (
            "Kelly 1965 twenty-two colors of maximum contrast "
            "(black/white omitted; background is black). "
            "IDs beyond the table use golden-angle HSV."
        ),
        "materials": [
            {
                "id": i,
                "name": mat.name,
                "viz_color": rgb_to_hex(id_to_viz_rgba(i)),
            }
            for i, mat in enumerate(materials, start=1)
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _clear_overrides(overrides: list) -> None:
    for nt, links in overrides:
        restore_node_tree(nt, links)
    overrides.clear()


def _sweep_helper_nodes() -> None:
    """Remove leftover helper nodes even if override records were lost (interrupted job / reload)."""
    for mat in bpy.data.materials:
        nt = getattr(mat, "node_tree", None)
        if nt is None:
            continue
        helpers = [n for n in nt.nodes if n.name.startswith(HELPER_PREFIX)]
        for node in helpers:
            nt.nodes.remove(node)


def _apply_gbuffer_channel(scene, channel: str, filepath: str) -> None:
    materials = mesh_materials(scene)
    overrides = _job.setdefault("overrides", [])

    if channel in _ID_WORLD_CHANNELS:
        scene.world = _make_id_world()

    if channel == "material_id":
        for mat_id, mat in enumerate(materials, start=1):
            value = float(mat_id)
            links = apply_id_override(mat.node_tree, (value, value, value, 1.0))
            overrides.append((mat.node_tree, links))
    elif channel == "material_id_vis":
        for mat_id, mat in enumerate(materials, start=1):
            links = apply_id_override(mat.node_tree, id_to_viz_rgba(mat_id))
            overrides.append((mat.node_tree, links))
    elif channel == "linear_depth":
        for mat in materials:
            links = apply_depth_override(mat.node_tree)
            overrides.append((mat.node_tree, links))
    else:
        for mat in materials:
            links = apply_channel_override(mat.node_tree, channel)
            overrides.append((mat.node_tree, links))

    prepare_render(scene, channel, filepath)
