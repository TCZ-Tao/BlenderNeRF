# BlenderNeRF

Whether a VFX artist, a research fellow or a graphics amateur, **BlenderNeRF** is the easiest and fastest way to create synthetic NeRF and Gaussian Splatting datasets within Blender. Obtain renders and camera parameters with a single click, while having full user control over the 3D scene and camera!

<p align='center'>
  <a href="https://youtu.be/C8YuDoU11cg"><img src="https://img.youtube.com/vi/C8YuDoU11cg/maxresdefault.jpg" width='90%'></a>
  <br>
  Are you ready to NeRF? Start with a single click in Blender by checking out <a href="https://youtu.be/C8YuDoU11cg">this tutorial</a>!
</p>


## Neural Radiance Fields

**Neural Radiance Fields ([NeRF](https://www.matthewtancik.com/nerf))** aim at representing a 3D scene as a view dependent volumetric object from 2D images only, alongside their respective camera information. The 3D scene is reverse engineered from the training images with help of a simple neural network.

[**Gaussian Splatting**](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) is a follow-up method for rendering radiance fields in a point-based manner. This representation is highly optimised for GPU rendering and leverages more traditional graphics techniques to achieve high frame rates.

I recommend watching [this YouTube video](https://www.youtube.com/watch?v=YX5AoaWrowY) by **Corridor Crew** for a thrilling investigation on a few use cases and future potential applications of NeRFs.


## Motivation

Rendering is an expensive computation. Photorealistic scenes can take seconds to hours to render depending on the scene complexity, hardware and available software resources.

NeRFs and Gaussian splats can speed up this process, but require camera information typically extracted via cumbersome code. This plugin enables anyone to get renders and cameras with a single click in Blender.

<p align='center'>
  <img src='https://maximeraafat.github.io/assets/posts/blendernerf/BlenderNeRF_compressed.gif' width='90%'/>
</p>


## Installation

1. Download this repository as a **ZIP** file
2. Open Blender (4.0.0 or above)
3. In Blender, head to **Edit > Preferences > Add-ons**, and select **Install From Disk** under the drop icon
4. Select the downloaded **ZIP** file

Although release versions of **BlenderNeRF** are available for download, they are primarily intended for tracking major code changes and for citation purposes. I recommend downloading the current repository directly, since minor changes or bug fixes might not be included in a release right away.

After editing the add-on sources, **Reload Scripts** (F3) reloads every BlenderNeRF submodule so UI and operator changes show up without restarting Blender.


## Setting

**BlenderNeRF** consists of 3 methods discussed in the sub-sections below. Each method can create **training** data and **testing** data: camera poses in `transforms_train.json` / `transforms_test.json`, plus rendered images when `Render Frames` is on. A TensoIR-style `metadata.json` (resolution, environment map, samples per pixel) is always written next to the transforms.

By default the dataset is a folder under **Save Path**. Enable `Compress to ZIP` to archive that folder into a single ZIP and delete the uncompressed copy.

Training data can then be used by a NeRF model to learn the 3D scene representation. Once trained, the model may be evaluated (or tested) on the testing data to obtain novel renders.

### Subset of Frames

**Subset of Frames (SOF)** renders every **N** frames from a camera animation, and utilises the rendered subset of frames as NeRF training data. The registered testing data spans over all frames of the same camera animation, including training frames. When trained, the NeRF model can render the full camera animation and is consequently well suited for interpolating or rendering large animations of static scenes.

<p align='center'>
  <img src='https://maximeraafat.github.io/assets/posts/blendernerf/SOF.gif' width='90%'/>
</p>

### Train and Test Cameras

**Train and Test Cameras (TTC)** registers training and testing data from two separate user defined cameras. A NeRF model can then be fitted with the data extracted from the training camera, and be evaluated on the testing data.

<p align='center'>
  <img src='https://maximeraafat.github.io/assets/posts/blendernerf/TTC.gif' width='90%'/>
</p>

### Camera on Sphere

**Camera on Sphere (COS)** renders training frames by uniformly sampling random camera views directed at the center from a user controlled sphere. Testing data is taken from a selected **Test Camera**. Use `Apply Spherical Spiral` to keyframe that camera along the NeRF-synthetic two-revolution spherical spiral (sample count follows `Test Frames`).

<p align='center'>
  <img src='https://maximeraafat.github.io/assets/posts/blendernerf/COS.gif' width='90%'/>
</p>


## How to use the Methods

The add-on properties panel is available under `3D View > N panel > BlenderNeRF` (the **N panel** is accessible under the 3D viewport when pressing `N`). All 3 methods (**SOF**, **TTC** and **COS**) share a common tab called `BlenderNeRF shared UI` with the below listed controllable properties.

* `Train` (activated by default) : whether to register training data (renderings + camera information)
* `Test` (activated by default) : whether to register testing data (camera poses and, if `Render Frames` is on, images)
* `iNGP AABB` (by default set to **4**) : aabb scale parameter as described in Instant NGP (more details below). Enable `Show` to preview the origin-centered cube in the viewport
* `Render Frames` (activated by default) : whether to render train/test images. If off, only the `transforms_*.json` files are written
* `Hide Render View` (deactivated by default) : keep the current workspace instead of opening the render result window
* `Save Log File` (deactivated by default) : whether to save a log file containing reproducibility information on the **BlenderNeRF** run
* `File Format` (**NGP** by default) : whether to export the camera files in the Instant NGP or defaut NeRF file format convention
* `Gaussian Points` (deactivated by default) : whether to export a `points3d.ply` file for Gaussian Splatting
* `Points` (by default set to **100000**) : number of random initialization points sampled uniformly in the scene AABB (only with `Gaussian Points`)
* `Dummy Test Camera File` (deactivated by default) : write an empty `transforms_test.json` and skip test-image rendering, even if `Test` is on
* `G-buffer Maps` (deactivated by default) : whether to render extra unlit maps into per-channel folders (see [G-buffer Maps](#g-buffer-maps))
* `Save Path` (empty by default) : path to the output directory in which the dataset will be created
* `Compress to ZIP` (deactivated by default) : archive the dataset as a ZIP and delete the uncompressed folder. Leave off to keep files under `<save path>/<name>`

If the `Gaussian Points` property is active, **BlenderNeRF** writes a `points3d.ply` of uniformly random points inside the world-space AABB of all render-visible meshes (modifiers included). This matches the 3D Gaussian Splatting initialization used for NeRF synthetic datasets (default 100000 points, random SH0 colors, zero normals). It does **not** use mesh vertices, and it does **not** use the `iNGP AABB` panel value.

The [**Gaussian Splatting**](https://github.com/graphdeco-inria/gaussian-splatting) repository natively supports **NeRF** datasets, but requires both train and test data. `Dummy Test Camera File` writes an empty `transforms_test.json` (`frames: []`) so that loader can run without real test views, and skips test-image rendering even when `Test` is on. Leave it off to export the full test camera poses; with `Test` and `Render Frames` on, the `test` folder is rendered in the same run.

`iNGP AABB` is restricted to be an integer power of 2, it defines the side length of the bounding box volume in which NeRF will trace rays. The cube is centered at the world origin: with the default value **4**, it spans `[-2, 2]` on each axis. Enable `Show` next to the property to draw a viewport-only `BlenderNeRF AABB` empty. The property was introduced with **NVIDIA's [Instant NGP](https://github.com/NVlabs/instant-ngp)** version of NeRF.

The `File Format` property can either be **NGP** or **NeRF**. The **NGP** file format convention is the same as the **NeRF** one, with a few additional parameters which can be accessed by Instant NGP.

Notice that each method has its distinctive `Name` property (by default set to `dataset`) corresponding to the dataset folder name (and ZIP filename if `Compress to ZIP` is on). Please note that unsupported characters, such as spaces, `#` or `/`, will automatically be replaced by an underscore.

### Dataset layout

```text
<save path>/<name>/
  metadata.json              # scene, imw, imh, envmap, envmap_inten, spp
  transforms_train.json
  transforms_test.json
  train/                     # RGB when G-buffer is off
  test/
  test_rli/<envmap>/         # Relight Mode RGB (folder name = HDRI filename without extension)
  points3d.ply               # if Gaussian Points
  log.txt                    # if Save Log File
  material_id.json           # if Material ID / Material ID Viz
```

With **G-buffer Maps** on, RGB and extra channels are nested per split:

```text
train/rgba/   train/albedo/   train/roughness/   …
test/rgba/    test/albedo/    …
```

`transforms_*.json` `file_path` values then point at `train/rgba/` (or `test/rgba/`) instead of `train/` / `test/`.

`metadata.json` records the dataset name, output resolution, World environment-map filename and background strength, and the render sample count (Cycles `samples`, or EEVEE TAA samples).

Below are described the properties specific to each method (the `Name` property is left out, since already discussed above).

### How to SOF

* `Frame Step` (by default set to **3**) : **N** (as defined in the [Setting](#setting) section) = frequency at which the training frames are registered
* `Camera` (always set to the active camera) : camera used for registering training and testing data
* `PLAY SOF` : play the **Subset of Frames** method operator to export NeRF data

### How to TTC

* `Frames` (by default set to **100**) : number of training frames used from the training camera
* `Train Cam` (empty by default) : camera used for registering the training data
* `Test Cam` (empty by default) : camera used for registering the testing data
* `PLAY TTC` : play the **Train and Test Cameras** method operator to export NeRF data

`Frames` amount of training frames will be captured using the `Train Cam` object, starting from the scene start frame. Testing uses `Test Cam` over the scene frame range.

### How to COS

* `Test Camera` (always set to the active camera) : camera used for registering the testing data
* `Location` (by default set to **0 m** vector) : center position of the training sphere from which camera views are sampled
* `Rotation` (by default set to **0°** vector) : rotation of the training sphere from which camera views are sampled
* `Scale` (by default set to **1** vector) : scale vector of the training sphere in xyz axes
* `Radius` (by default set to **4 m**) : radius scalar of the training sphere
* `Lens` (by default set to **50 mm**) : focal length of the training camera
* `Seed` (by default set to **0**) : seed to initialize the random camera view sampling procedure
* `Train Frames` (by default set to **100**) : number of training frames sampled and rendered from the training sphere
* `Test Frames` (by default set to **100**) : number of testing frames captured from the **Test Camera**
* `Sphere` (deactivated by default) : whether to show the training sphere from which random views will be sampled
* `Camera` (deactivated by default) : whether to show the camera used for registering the training data
* `Upper Views` (deactivated by default) : whether to sample views from the upper training hemisphere only (rotation variant)
* `Outwards` (deactivated by default) : whether to point the camera outwards of the training sphere
* `Apply Spherical Spiral` : keyframe the selected **Test Camera** along a two-revolution spherical spiral on the BlenderNeRF Sphere (the NeRF synthetic test path). The frame count follows `Test Frames`
* `PLAY COS` : play the **Camera on Sphere** method operator to export NeRF data

Note that activating the `Sphere` and `Camera` properties creates a `BlenderNeRF Sphere` empty object and a `BlenderNeRF Camera` camera object respectively. `Show` next to `iNGP AABB` creates a viewport-only `BlenderNeRF AABB` cube empty. `Apply Spherical Spiral` also creates a viewport-only `BlenderNeRF Spiral Path` curve. Please do not create any objects with these names manually, since this might break the add-on functionalities.

`Train Frames` amount of training frames will be captured using the `BlenderNeRF Camera` object, starting from the scene start frame. The training camera is locked in place and cannot manually be moved. After PLAY COS, the scene camera is restored to the **Test Camera** (never left on `BlenderNeRF Camera`).

`Apply Spherical Spiral` cannot target `BlenderNeRF Camera`; pick the COS **Test Camera** instead. It needs `Test Frames` ≥ 2 and a non-flat sphere scale.


## G-buffer Maps

When `G-buffer Maps` is on, PLAY SOF / TTC / COS runs an extra unlit pass per selected channel (EEVEE, 1 sample, no shadows / raytracing / motion blur). The original RGB(A) render still uses the scene engine (typically Cycles). Channels can be toggled independently; all default to on once G-buffer is enabled:

| Channel | Folder | Format | Notes |
|---|---|---|---|
| RGBA | `rgba` | scene format | Original lit RGB(A) frames |
| Albedo | `albedo` | PNG | Principled Base Color |
| Roughness | `roughness` | PNG | Principled Roughness |
| Metallic | `metallic` | PNG | Principled Metallic |
| Geometric Normal | `geometric_normal` | PNG | Camera-space geometric normals (no bump maps); view Z towards camera |
| Shading Normal | `shading_normal` | PNG | Camera-space shading normals (with bump / normal maps) |
| Linear Depth | `linear_depth` | OpenEXR 32-bit | Camera-space linear depth |
| Material ID | `material_id` | OpenEXR 32-bit | Numeric IDs (`R=G=B=id`; `0` = background, `1` = first material, …) |
| Material ID Viz | `material_id_vis` | PNG | Colorized ID visualization (Kelly 1965 palette) |

If Material ID or Material ID Viz is selected, a `material_id.json` mapping (`id`, material `name`, `viz_color`) is written at the dataset root.


## Relight Mode

Relight Mode renders **test-camera** frames under a different World HDRI for inverse-rendering / relighting GT. It does **not** re-export transforms, G-buffer maps, or train views, and it does **not** zip the dataset.

In the shared panel, enable `Relight Mode`:

* `Method` : which test-camera timeline to use (`SOF` / `TTC` / `COS`). Output uses that method's **Name** field as-is (no `SOF_` / `COS_` prefix). Only one method runs per job.
* `Environment Map` : path to a single HDRI (`.hdr`, `.exr`, …)
* `Render Relight` : swap the World Environment Texture to that file (scene lamps and Film Transparent stay unchanged), render the test sequence, write RGB into `<save path>/<name>/test_rli/<envmap>/`, then restore the original World

Frame names follow the scene render output (`frame_path`), same as `test/`.

Headless example:

```bash
blender -b -noaudio --factory-startup --addons BlenderNeRF \
  scene.blend --python-exit-code 1 --python cli.py -- \
  --mode relight --method COS --envmap /data/hdris/bridge.hdr \
  --cycles-device CUDA --save-path /data/out --name lego
```

That writes `/data/out/lego/test_rli/bridge/`. `--name` overrides the method Name field; it is not prefixed with the method id.


## Tips for Optimal Results

NVIDIA provides a few helpful tips on how to train a NeRF model using [Instant NGP](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md). Feel free to visit their repository for further help. Below are some quick tips for optimal **nerfing** gained from personal experience.

* NeRF trains best with 50 to 150 images
* Testing views should not deviate too much from training views
* Scene movement, motion blur or blurring artefacts can degrade the reconstruction quality
* The captured scene should be at least one Blender unit away from the camera
* Keep `iNGP AABB` as tight as possible to the scene scale, higher values will slow down training
* If the reconstruction quality appears blurry, start by adjusting `iNGP AABB` while keeping it a power of 2
* Avoid adjusting the camera focal lengths during the animation, the vanilla NeRF methods do not support multiple focal lengths
* Avoid extreme focal lengths, values between 30 mm and 70 mm work well in practice
* A `Vertical` camera sensor fit sometimes leads to distorted NeRF volumes, avoid it if possible


## How to NeRF

If you have access to an NVIDIA GPU, you might want to install [Instant NGP](https://github.com/NVlabs/instant-ngp#installation) on your own device for an optimal user experience, by following the instructions provided on their repository. Otherwise, you can run NeRF in a COLAB notebook on Google GPUs for free with a Google account.

Open this [COLAB notebook](https://colab.research.google.com/drive/1dQInHx0Eg5LZUpnhEfoHDP77bCMwAPab?usp=sharing) (also downloadable [here](https://gist.github.com/maximeraafat/122a63c81affd6d574c67d187b82b0b0)) and follow the instructions.


## Remarks

This add-on is being developed as a fun side project over the course of multiple months and versions of Blender, mainly on macOS. If you encounter any issues with the plugin functionalities, feel free to open a GitHub issue with a clear description of the problem, which **BlenderNeRF** version the issues have been experienced with, and any further information if relevant.

### Real World Data

While this extension is intended for synthetic datasets creation, existing tools for importing motion tracking data from real world cameras are available. One such example is **[Tracky](https://github.com/Shopify/tracky)** by **Shopify**, an open source iOS app and an adjacent Blender plugin recording motion tracking data from an ARKit session on iPhone. Keep in mind however that tracking data can be subject to drifts and inaccuracies, which might affect the resulting NeRF reconstruction quality.


## Headless rendering (Linux server)

Configure the scene in the Blender UI (method, cameras, resolution, G-buffer, samples), pack textures (**File → External Data → Pack Resources**), then save the `.blend`. Copy this add-on to:

```text
~/.config/blender/<BLENDER_VERSION>/scripts/addons/BlenderNeRF/
```

`<BLENDER_VERSION>` must match the Blender you run (e.g. `5.2`).

Blender reads arguments **left to right**. The `.blend` must be loaded **before** `--python`, and `--addons BlenderNeRF` is required with `--factory-startup` (factory mode does not enable user add-ons).

| Before `--` | After `--` |
|---|---|
| Blender flags, **`--addons BlenderNeRF`**, **`scene.blend`**, `--python cli.py` | `cli.py` flags only: `--method`, `--save-path`, `--cycles-device`, … |

Example `run_blender.sh`:

```bash
#!/bin/bash
set -euo pipefail

# SSH/VS Code often sets DISPLAY to a dead X11 session; -b still hangs if it is set.
unset DISPLAY WAYLAND_DISPLAY

export CUDA_VISIBLE_DEVICES=7          # pick one GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID    # indices match nvidia-smi

BLEND="wood chair.blend"
CLI="$HOME/.config/blender/5.2/scripts/addons/BlenderNeRF/cli.py"
OUT="$PWD/dining_chair"

blender -b -noaudio --factory-startup --addons BlenderNeRF \
  "$BLEND" \
  --python-exit-code 1 \
  --python "$CLI" -- \
  --method COS --cycles-device CUDA --save-path "$OUT"
```

`--method` is `SOF`, `TTC`, or `COS` (must match how the `.blend` was set up). `--cycles-device` accepts `CPU`, `CUDA`, `OPTIX`, `HIP`, `METAL`, or `ONEAPI`.

Optional `cli.py` flags:

* `--mode` : `export` (default dataset write) or `relight` (test views under a new HDRI)
* `--save-path` : override the **Save Path** stored in the `.blend`
* `--name` : override the method **Name** (dataset folder name; not prefixed with SOF/TTC/COS)
* `--envmap` : HDRI file; required with `--mode relight`
* `--no-render` : write transforms / JSON only, skip image renders (`export` mode)
* `--engine` : render engine override, e.g. `CYCLES`
* `--cycles-device` : Cycles compute device (see above)

Check the first log line: `blend='.../wood chair.blend'` and `res=WxH` must match the file. If `blend=''`, the `.blend` was passed after `--` or after `--python`, and you rendered the factory default cube on CPU.

With **G-buffer Maps** enabled, RGB frames are in `train/rgba/` and `test/rgba/`, not in `train/` itself.

Do **not** put `--cycles-device` after `--` *and* forget `--addons BlenderNeRF`. Typical failures:

* Hang right after the version string → `unset DISPLAY WAYLAND_DISPLAY`, add `-noaudio --factory-startup`.
* `Scene` has no `save_path` → missing `--addons BlenderNeRF`.
* `unrecognized arguments: foo.blend` → the `.blend` was after `--`; move it before `--python`.
* CPU-only / wrong resolution / default cube → `.blend` was after `--python`; swap the order as above.


## Citation

If you find this repository useful in your research, please consider citing **BlenderNeRF** using the dedicated GitHub button above. If you made use of this extension for your artistic projects, feel free to share some of your work using the `#blendernerf` hashtag on social media! :)