"""Headless BlenderNeRF entry point.

Blender executes flags left to right. Load the .blend BEFORE --python, and
pass --addons BlenderNeRF when using --factory-startup.

    unset DISPLAY WAYLAND_DISPLAY
    blender -b -noaudio --factory-startup --addons BlenderNeRF \\
      scene.blend --python-exit-code 1 --python cli.py -- \\
      --method COS --cycles-device CUDA --save-path /data/out

    blender -b -noaudio --factory-startup --addons BlenderNeRF \\
      scene.blend --python-exit-code 1 --python cli.py -- \\
      --mode relight --method COS --envmap /data/hdris/bridge.hdr \\
      --cycles-device CUDA --save-path /data/out --name lego
"""

from __future__ import annotations

import argparse
import os
import sys

import bpy


OPERATORS = {
    'SOF': 'object.subset_of_frames',
    'TTC': 'object.train_test_cameras',
    'COS': 'object.camera_on_sphere',
}


def _argv_after_double_dash():
    argv = sys.argv
    if '--' in argv:
        return argv[argv.index('--') + 1:]
    return []


def _addon_ready():
    '''True once register() has attached our Scene properties.'''
    return hasattr(bpy.types.Scene, 'save_path')


def ensure_addon_enabled():
    addon_dir = os.path.dirname(os.path.realpath(__file__))
    module_name = os.path.basename(addon_dir)

    if _addon_ready():
        return module_name

    try:
        bpy.ops.preferences.addon_enable(module=module_name)
    except Exception as exc:
        print(f'[BlenderNeRF] addon_enable({module_name}) failed: {exc}', file=sys.stderr)

    if _addon_ready():
        return module_name

    parent = os.path.dirname(addon_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    import importlib
    if module_name in sys.modules:
        mod = importlib.reload(sys.modules[module_name])
    else:
        mod = importlib.import_module(module_name)

    if hasattr(mod, 'register') and not _addon_ready():
        mod.register()

    if not _addon_ready():
        raise RuntimeError(
            'BlenderNeRF did not register scene.save_path. '
            'With --factory-startup, also pass --addons BlenderNeRF, '
            f'or keep cli.py inside the add-on folder (module {module_name!r}).'
        )
    return module_name


def _dataset_name(scene, method):
    if method == 'SOF':
        return scene.sof_dataset_name
    if method == 'TTC':
        return scene.ttc_dataset_name
    return scene.cos_dataset_name


def _call_operator(method):
    if method == 'SOF':
        return bpy.ops.object.subset_of_frames()
    if method == 'TTC':
        return bpy.ops.object.train_test_cameras()
    return bpy.ops.object.camera_on_sphere()


def apply_cycles_device(device: str) -> None:
    '''Switch Cycles to CPU / CUDA / OPTIX / HIP / METAL / ONEAPI.'''
    device = device.strip().upper()
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    if device == 'CPU':
        scene.cycles.device = 'CPU'
        print('[BlenderNeRF] Cycles device=CPU')
        return

    scene.cycles.device = 'GPU'
    addon = bpy.context.preferences.addons.get('cycles')
    if addon is None:
        raise RuntimeError('Cycles add-on is not enabled; cannot set --cycles-device')
    prefs = addon.preferences
    try:
        prefs.compute_device_type = device
    except TypeError as exc:
        raise RuntimeError(f'Unsupported Cycles device {device!r}: {exc}') from exc

    refresh = getattr(prefs, 'refresh_devices', None) or getattr(prefs, 'get_devices', None)
    if callable(refresh):
        refresh()

    enabled = []
    listed = []
    for dev in getattr(prefs, 'devices', []):
        listed.append(f'{dev.name} ({dev.type}, use={bool(getattr(dev, "use", False))})')
        use = (getattr(dev, 'type', '') == device)
        dev.use = use
        if use:
            enabled.append(f'{dev.name} ({dev.type})')
    print(f'[BlenderNeRF] Cycles devices: {listed or ["(none)"]}')
    if not enabled:
        print(
            f'WARNING: no {device} devices found; Cycles may fall back to CPU. '
            'Check NVIDIA drivers, CUDA_VISIBLE_DEVICES, and that --cycles-device is after --.',
            file=sys.stderr,
        )
    print(
        f'[BlenderNeRF] Cycles device={device} '
        f'scene.cycles.device={scene.cycles.device} gpu={enabled or ["none"]}'
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Export a BlenderNeRF dataset in background mode (blender -b).',
        epilog='Put scene.blend BEFORE --. Only script flags go after --.',
    )
    parser.add_argument('--mode', choices=('export', 'relight'), default='export',
                        help='export writes the dataset (default); relight swaps a World HDRI and renders test views')
    parser.add_argument('--method', choices=sorted(OPERATORS), required=True,
                        help='SOF, TTC, or COS (must match how the .blend was set up)')
    parser.add_argument('--save-path', default='',
                        help='Output directory (overrides the path stored in the .blend)')
    parser.add_argument('--name', default='',
                        help='Dataset folder name (overrides the method Name field; no SOF/TTC/COS prefix)')
    parser.add_argument('--envmap', default='',
                        help='HDRI file used as World lighting (--mode relight)')
    parser.add_argument('--no-render', action='store_true',
                        help='Write transforms/JSON only, skip image renders (export mode)')
    parser.add_argument('--engine', default='',
                        help='Render engine override, e.g. CYCLES')
    parser.add_argument('--cycles-device', default='',
                        metavar='DEVICE',
                        help='Cycles device: CPU, CUDA, OPTIX, HIP, METAL, or ONEAPI')
    args, unknown = parser.parse_known_args(argv)
    blends = [a for a in unknown if a.lower().endswith('.blend')]
    if blends:
        parser.error(
            f'{blends[0]!r} was passed after -- ; Blender never loads it. '
            'Put the .blend before --python / -- , e.g. '
            'blender -b scene.blend --python cli.py -- --method COS ...'
        )
    if unknown:
        parser.error(f'unrecognized arguments: {" ".join(unknown)}')
    if args.mode == 'relight':
        if not (args.envmap or '').strip():
            parser.error('--mode relight requires --envmap PATH')
        if args.no_render:
            parser.error('--no-render cannot be used with --mode relight')
    elif args.envmap:
        parser.error('--envmap requires --mode relight')
    return args


def _has_images(directory):
    if not os.path.isdir(directory):
        return False
    for root, _dirs, files in os.walk(directory):
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr', '.tif', '.tiff')) for f in files):
            return True
    return False


def _envmap_stem(filepath):
    return bpy.path.clean_name(os.path.splitext(os.path.basename(filepath))[0])


def main():
    argv = _argv_after_double_dash()
    if not argv:
        print(
            'Usage: blender -b -noaudio --factory-startup --addons BlenderNeRF '
            'scene.blend --python cli.py -- --method COS --cycles-device CUDA --save-path /data/out',
            file=sys.stderr,
        )
        return 2

    args = parse_args(argv)

    scene = bpy.context.scene
    print(
        f'[BlenderNeRF] blend={bpy.data.filepath!r} scene={scene.name!r} '
        f'res={scene.render.resolution_x}x{scene.render.resolution_y} '
        f'@ {scene.render.resolution_percentage}% engine={scene.render.engine} '
        f'camera={getattr(scene.camera, "name", None)}'
    )
    if not bpy.data.filepath:
        print(
            'ERROR: no .blend is loaded; this is the factory default scene '
            '(wrong size/content, Cycles CPU). Blender runs arguments left to right — '
            'put the .blend BEFORE --python:\n'
            '  blender -b -noaudio --factory-startup --addons BlenderNeRF \\\n'
            '    "wood chair.blend" --python cli.py -- --method COS --cycles-device CUDA ...',
            file=sys.stderr,
        )
        return 1

    ensure_addon_enabled()

    # Drop the one-shot handler so it cannot clobber --save-path when COS creates cameras.
    for handler in list(bpy.app.handlers.depsgraph_update_post):
        if getattr(handler, '__name__', '') == 'set_init_props':
            bpy.app.handlers.depsgraph_update_post.remove(handler)

    bpy.context.view_layer.update()

    scene = bpy.context.scene
    if args.save_path:
        scene.save_path = os.path.abspath(os.path.expanduser(args.save_path))
    if args.name:
        if args.method == 'SOF':
            scene.sof_dataset_name = args.name
        elif args.method == 'TTC':
            scene.ttc_dataset_name = args.name
        else:
            scene.cos_dataset_name = args.name
    if args.no_render:
        scene.render_frames = False
    if args.engine:
        scene.render.engine = args.engine
    if args.cycles_device:
        apply_cycles_device(args.cycles_device)

    envmap = ''
    if args.mode == 'relight':
        envmap = os.path.abspath(os.path.expanduser(args.envmap))
        if not os.path.isfile(envmap):
            print(f'ERROR: environment map file not found: {envmap}', file=sys.stderr)
            return 1
        scene.relight_method = args.method
        scene.relight_envmap = envmap

    save_path = bpy.path.abspath(scene.save_path)
    if not save_path:
        print('ERROR: save path is empty. Pass --save-path.', file=sys.stderr)
        return 1
    if sys.platform != 'win32' and len(save_path) >= 2 and save_path[1] == ':':
        print(
            f'WARNING: save_path looks like a Windows path: {save_path!r}. '
            'Pass --save-path with a Linux directory.',
            file=sys.stderr,
        )

    os.makedirs(save_path, exist_ok=True)
    scene.save_path = save_path
    # Re-pin after abspath: DIR_PATH can store // and a later depsgraph must not win.
    save_path = bpy.path.abspath(scene.save_path)
    scene.save_path = save_path

    dataset_name = _dataset_name(scene, args.method)
    print(
        f'[BlenderNeRF] mode={args.mode} method={args.method} save_path={save_path} '
        f'name={dataset_name!r} '
        f'render_frames={bool(scene.render_frames)} engine={scene.render.engine} '
        f'background={bool(bpy.app.background)}'
        + (f' envmap={envmap!r}' if envmap else '')
    )

    if args.mode == 'relight':
        try:
            result = bpy.ops.object.blendernerf_relight()
        except RuntimeError as exc:
            print(f'ERROR: relight failed: {exc}', file=sys.stderr)
            return 1
        print(f'[BlenderNeRF] operator result={result}')

        output_dir = bpy.path.clean_name(dataset_name)
        actual_save = bpy.path.abspath(scene.save_path)
        relight_dir = os.path.join(actual_save, output_dir, 'test_rli', _envmap_stem(envmap))
        if not _has_images(relight_dir):
            print(f'ERROR: no relight images written to {relight_dir}', file=sys.stderr)
            return 1
        return 0

    result = _call_operator(args.method)
    print(f'[BlenderNeRF] operator result={result}')

    output_dir = bpy.path.clean_name(dataset_name)
    actual_save = bpy.path.abspath(scene.save_path)
    output_path = os.path.join(actual_save, output_dir)
    zip_path = output_path + '.zip'
    if scene.compress_to_zip:
        wrote = os.path.isfile(zip_path)
        marker = zip_path
    else:
        wrote = os.path.isdir(output_path)
        marker = output_path
    if actual_save.rstrip('\\/') != save_path.rstrip('\\/'):
        print(
            f'[BlenderNeRF] note: scene save_path resolved to {actual_save} '
            f'(requested {save_path})',
            file=sys.stderr,
        )
    if not wrote:
        print(f'ERROR: dataset was not written to {marker}', file=sys.stderr)
        return 1

    if scene.render_frames and not scene.compress_to_zip:
        if not _has_images(output_path):
            print(f'WARNING: no rendered images found under {output_path}', file=sys.stderr)
            print('G-buffer images are in train/rgba, train/albedo, ... not in train/ itself.', file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
