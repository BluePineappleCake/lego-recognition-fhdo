"""Render single LEGO parts from the LDraw library, in the style of the
internet renders (one brick lying on a grey floor, soft shadow, tight crop).

Run it with Blender (no add-on needed, tested with Blender 5.3):

    blender -b -P scripts/render_ldraw.py -- --ldraw C:/ldraw --parts 4073

or with the bpy package (pip install bpy, same Python version as Blender):

    python scripts/render_ldraw.py --ldraw C:/ldraw --parts 4073 3062b 3005

What happens per part:

    1. load_ldraw: read <part>.dat and all sub-files (studs, cylinders ...)
       from the LDraw library into one triangle mesh. The geometry is exact.
    2. for every pose: pick a colour, drop the brick onto the floor with the
       rigid body simulation, so it lies the way a real brick lies (on top,
       bottom or side, never balanced on an edge).
    3. render VIEWS camera positions around the resting brick (Cycles),
       crop tightly around the brick, save as JPEG.

Closer to real photos than plain LDraw renders:

    - round parts with 48 instead of 16 segments (p/48, off with --lowres)
    - rounded edges (Bevel shader), uneven surface and roughness (noise),
      a little subsurface scattering (ABS is not fully opaque)
    - floors in the colours of our capture template, with paper grain
    - two lights (key + fill) with a random colour temperature, depth of
      field, random focal length, camera up to almost straight from above

Camera noise, motion blur and JPEG artefacts are better added as training
augmentation: there they are new for every epoch.

Output, same naming scheme as the internet renders:

    <out>/<folder>/<part>_<colour>_<view>_<id>.jpeg

All views of one pose share the id prefix (id = pose id + view), so a split by
pose keeps them together. <out>/renders_manifest.csv lists every image with
its part, colour, pose and seed.

Colours: the colour names of the internet renders (LEGO names), looked up in
LDConfig.ldr of the LDraw library. --trans-share sets how many poses get a
transparent colour (glass material) - the internet renders have none, but our
own 4073 bricks are transparent.

Keep in mind: if only one class is rendered with this script, the model may
learn "this render style = this part". Render the look-alikes (or all parts)
with the same script.
"""

import argparse
import csv
import math
import random
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

# folder names of our data set (data/raw/internet/renders, photos_own_cutout)
FOLDERS = {
    "2654": "2654 plate 2x2",
    "3001": "3001 brick 2x4",
    "3002": "3002 brick 2x3",
    "3003": "3003 brick 2x2",
    "3004": "3004 brick 1x2",
    "3005": "3005 brick 1x1",
    "3009": "3009 brick 1x6",
    "3010": "3010 brick 1x4",
    "3020": "3020 plate 2x4",
    "3021": "3021 plate 2x3",
    "3022": "3022 plate 2x2",
    "3023": "3023 plate 1x2",
    "3062": "3062 round brick 1x1",
    "3062b": "3062 round brick 1x1",
    "3622": "3622 brick 1x3",
    "3623": "3623 plate 1x3",
    "3666": "3666 plate 1x6",
    "3710": "3710 plate 1x4",
    "4073": "4073 round plate 1x1",
    "6215": "6215 brick curved 2x3",
}

# colour names used in the internet renders
OPAQUE_COLOURS = [
    "Aqua", "Black", "Brick Yellow", "Bright Blue", "Bright Bluish Green",
    "Bright Green", "Bright Orange", "Bright Purple", "Bright Red",
    "Bright Reddish Violet", "Bright Yellow", "Bright Yellowish-Green",
    "Cool Yellow", "Dark Azur", "Dark Brown", "Dark Green", "Dark Orange",
    "Dark Red", "Dark Stone Grey", "Earth Blue", "Earth Green",
    "Flame Yellowish Orange", "Lavender", "Light Nougat", "Light Purple",
    "Light Royal Blue", "Light Stone Grey", "Medium Azur", "Medium Blue",
    "Medium Lavender", "Medium Nougat", "Medium Stone Grey",
    "Medium-Yellowish green", "Nougat", "Olive Green", "Reddish Brown",
    "Sand Blue", "Sand Green", "Sand Yellow", "Spring Yellowish Green", "White",
]  # fmt: skip
TRANS_COLOURS = [
    "Transparent", "Transparent Light Blue", "Transparent Blue",
    "Transparent Red", "Transparent Yellow", "Transparent Green",
]  # fmt: skip
# LEGO names that LDConfig.ldr lists under another name
COLOUR_ALIASES = {"dark red": "new dark red"}

ENERGY_MIN, ENERGY_MAX = (
    15,
    50,
)  # area light (W); floor about as bright as the internet renders
FSTOP_RANGE = (4.0, 22.0)  # small = shallow depth of field (scene is 25x real size)
# floors (name, colour, weight): grey like the internet renders, the colours of
# our capture template (make_capture_template.py) and white paper
FLOORS = [
    ("grey", None, 0.3),
    ("petrol", "#005869", 0.17),
    ("altrosa", "#8D5B58", 0.17),
    ("mint", "#0CAF93", 0.16),
    ("white", "#FFFFFF", 0.2),
]
LDU = 0.01  # metres per LDraw unit (real: 0.0004; larger is easier for the physics)
MANIFEST_HEADER = ["file", "part", "colour", "floor", "pose_id", "view", "seed"]
SEARCH_DIRS = ("parts", "p", "unofficial/parts", "unofficial/p")


# ---------------------------------------------------------------- LDraw


def find_ldraw_file(root: Path, name: str, hires: bool = True) -> Path:
    """Look a file up like LDraw programs do. With hires, round primitives come
    from p/48 (48 segments per circle instead of 16), if there is such a file."""
    name = name.replace("\\", "/")
    folders = (("p/48",) if hires and "/" not in name else ()) + SEARCH_DIRS
    for folder in folders:
        for candidate in (name, name.lower()):
            path = root / folder / candidate
            if path.is_file():
                return path
    raise FileNotFoundError(f"{name} not found in {root} ({', '.join(SEARCH_DIRS)})")


def load_ldraw(root: Path, part: str, hires: bool = True):
    """<part>.dat -> (vertices (N x 3, LDraw units), triangles (M x 3), description).

    Follows sub-file references (line type 1) with their 4x4 matrices and the
    BFC winding rules, so all faces point outwards. Lines (type 2, 5) are
    skipped.
    """
    cache = {}
    vertices, triangles = [], []
    description = ""

    def read(path):
        if path not in cache:
            cache[path] = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        return cache[path]

    def walk(path, matrix, invert, depth):
        nonlocal description
        ccw, invert_next = True, False
        for line in read(path):
            tokens = line.split()
            if not tokens:
                continue
            kind = tokens[0]
            if kind == "0":
                # an alias ("~Moved to 6141") takes the description of the part it points to
                if depth <= 1 and description.startswith("~"):
                    description = ""
                skip = ("!", "Name:", "Author:", "BFC", "//")
                if (
                    depth <= 1
                    and not description
                    and len(tokens) > 1
                    and not tokens[1].startswith(skip)
                ):
                    description = " ".join(tokens[1:])
                if len(tokens) > 1 and tokens[1] == "BFC":
                    words = tokens[2:]
                    if "CW" in words:
                        ccw = False
                    if "CCW" in words:
                        ccw = True
                    if "INVERTNEXT" in words:
                        invert_next = True
            elif kind == "1" and len(tokens) >= 15:
                x, y, z, a, b, c, d, e, f, g, h, i = map(float, tokens[2:14])
                local = np.array(
                    [[a, b, c, x], [d, e, f, y], [g, h, i, z], [0, 0, 0, 1]]
                )
                child = find_ldraw_file(root, " ".join(tokens[14:]), hires)
                walk(child, matrix @ local, invert != invert_next, depth + 1)
                invert_next = False
            elif kind in ("3", "4"):
                n = int(kind)
                points = np.array(tokens[2 : 2 + 3 * n], float).reshape(n, 3)
                points = (matrix[:3, :3] @ points.T).T + matrix[:3, 3]
                flip = (not ccw) != invert != (np.linalg.det(matrix[:3, :3]) < 0)
                if flip:
                    points = points[::-1]
                start = len(vertices)
                vertices.extend(points)
                triangles.append((start, start + 1, start + 2))
                if n == 4:
                    triangles.append((start, start + 2, start + 3))
                invert_next = False

    walk(find_ldraw_file(root, f"{part}.dat"), np.eye(4), False, 0)
    return np.array(vertices), np.array(triangles), description


def read_colours(root: Path) -> dict[str, tuple[str, bool]]:
    """LEGO colour name (lower case) -> (hex value, transparent?) from LDConfig.ldr."""
    lines = (root / "LDConfig.ldr").read_text(encoding="utf-8").splitlines()
    colours = {}
    for i, line in enumerate(lines[:-1]):
        lego = re.match(r"0\s+//\s+LEGOID\s+[\d /]+-\s*(.+)$", line.strip())
        ldraw = re.match(
            r"0 !COLOUR \S+\s+CODE\s+\d+\s+VALUE\s+(#\w+)(.*)", lines[i + 1]
        )
        if lego and ldraw:
            for name in lego[1].split("/"):
                key = name.strip().lower().replace("-", " ")
                colours.setdefault(key, (ldraw[1], "ALPHA" in ldraw[2]))
    return colours


def colour_value(colours, name):
    key = name.lower().replace("-", " ")
    key = COLOUR_ALIASES.get(key, key)
    if key not in colours:
        raise KeyError(f"colour {name!r} not in LDConfig.ldr")
    return colours[key]


# ---------------------------------------------------------------- scene


def srgb_to_linear(hex_value: str) -> tuple[float, float, float, float]:
    rgb = [int(hex_value[k : k + 2], 16) / 255 for k in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return (*lin, 1.0)


def principled(name):
    material = bpy.data.materials.new(name)
    if material.node_tree is None:
        material.use_nodes = True
    return material, material.node_tree.nodes["Principled BSDF"]


def plastic_material(name):
    """Principled BSDF plus what real bricks have and LDraw meshes do not:

    - rounded edges (Bevel node: edges catch the light like real moulded edges)
    - slightly uneven surface (noise -> bump: wear, scratches, fingerprints)
    - roughness that varies over the surface (noise -> roughness)
    Returns (material, {name: node}) for set_colour.
    """
    material, bsdf = principled(name)
    tree = material.node_tree
    nodes, links = tree.nodes, tree.links
    coords = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"  # W (set per pose) gives every pose another pattern
    noise.inputs["Scale"].default_value = 40
    noise.inputs["Detail"].default_value = 8
    links.new(coords.outputs["Object"], noise.inputs["Vector"])
    roughness = nodes.new("ShaderNodeMapRange")
    links.new(noise.outputs["Fac"], roughness.inputs["Value"])
    links.new(roughness.outputs["Result"], bsdf.inputs["Roughness"])
    bevel = nodes.new("ShaderNodeBevel")
    bevel.inputs["Radius"].default_value = 0.3 * LDU  # real edge radius ~0.1 mm
    bump = nodes.new("ShaderNodeBump")
    links.new(bevel.outputs["Normal"], bump.inputs["Normal"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material, {
        "bsdf": bsdf,
        "roughness": roughness,
        "bump": bump,
        "noise": noise,
    }


def paper_material(name):
    """Floor: a colour plus a fine paper grain (noise -> bump and brightness)."""
    material, bsdf = principled(name)
    tree = material.node_tree
    nodes, links = tree.nodes, tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 300
    noise.inputs["Detail"].default_value = 6
    bump = nodes.new("ShaderNodeBump")
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.08
    links.new(noise.outputs["Color"], mix.inputs["B"])
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    return material, {"bsdf": bsdf, "colour": mix, "bump": bump}


def kelvin_to_rgb(kelvin):
    """Colour of a light with this colour temperature (approximation by T. Helland)."""
    t = kelvin / 100
    r = 255 if t <= 66 else 329.7 * (t - 60) ** -0.1332
    g = 99.47 * math.log(t) - 161.12 if t <= 66 else 288.12 * (t - 60) ** -0.0755
    b = 255 if t >= 66 else (0 if t <= 19 else 138.52 * math.log(t - 10) - 305.04)
    return tuple(min(max(c, 0), 255) / 255 for c in (r, g, b))


def setup_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 90
    scene.render.use_border = True  # render only the box around the brick
    scene.render.use_crop_to_border = True
    scene.view_settings.view_transform = "Standard"
    use_gpu(scene)

    world = bpy.data.worlds.new("world")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True

    bpy.ops.mesh.primitive_plane_add(size=200)
    floor = bpy.context.object
    floor_material, floor_nodes = paper_material("floor")
    floor.data.materials.append(floor_material)
    bpy.ops.rigidbody.world_add()
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    floor.rigid_body.collision_shape = "BOX"
    floor.rigid_body.friction = 0.8

    world_rb = scene.rigidbody_world
    world_rb.substeps_per_frame = 20
    world_rb.solver_iterations = 20
    world_rb.point_cache.frame_end = args.frames

    lights = []
    for _ in range(2):  # key and fill light (we photograph with two LED panels)
        bpy.ops.object.light_add(type="AREA")
        lights.append(bpy.context.object)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.dof.use_dof = True
    scene.camera = camera
    return scene, floor_nodes, lights, camera


def use_gpu(scene):
    """Cycles on the GPU if there is one (OptiX/CUDA on NVIDIA), else CPU."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return
    for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == kind]
        if devices:
            for device in prefs.devices:
                device.use = device.type == kind
            scene.cycles.device = "GPU"
            print(f"[render] GPU: {kind}, {', '.join(d.name for d in devices)}")
            return
    print("[render] no GPU found, rendering on the CPU (slow)")


def make_part_object(vertices, triangles, name):
    """LDraw mesh -> Blender object (z up, metres, smooth with sharp edges)."""
    import bmesh

    # LDraw: y points down. Blender: z points up. (x, y, z) -> (x, z, -y)
    v = vertices[:, [0, 2, 1]] * np.array([1, 1, -1]) * LDU
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(v.tolist(), [], triangles.tolist())
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.02 * LDU)
    bm.to_mesh(mesh)
    bm.free()
    mesh.shade_smooth()
    mesh.set_sharp_from_angle(angle=math.radians(35))

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    # origin at the centre of the bounding box
    centre = (v.min(axis=0) + v.max(axis=0)) / 2
    mesh.transform(Matrix.Translation(Vector(-centre)))

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    obj.rigid_body.collision_shape = "CONVEX_HULL"
    obj.rigid_body.friction = 0.6
    obj.rigid_body.restitution = 0.2
    obj.rigid_body.collision_margin = 0.0
    obj.select_set(False)

    material, nodes = plastic_material(f"{name}_material")
    mesh.materials.append(material)
    return obj, nodes


def set_colour(nodes, hex_value, transparent, rng):
    bsdf = nodes["bsdf"]
    bsdf.inputs["Base Color"].default_value = srgb_to_linear(hex_value)
    bsdf.inputs["IOR"].default_value = 1.55
    bsdf.inputs["Transmission Weight"].default_value = 1.0 if transparent else 0.0
    # ABS lets a little light in: light colours glow slightly at the edges
    bsdf.inputs["Subsurface Weight"].default_value = (
        0.0 if transparent else rng.uniform(0, 0.15)
    )
    bsdf.inputs["Subsurface Radius"].default_value = (0.5 * LDU,) * 3
    bsdf.inputs["Subsurface Scale"].default_value = 1.0
    base = rng.uniform(0.0, 0.06) if transparent else rng.uniform(0.04, 0.2)
    spread = rng.uniform(0.0, 0.15)  # new brick: 0, worn brick: up to 0.15
    nodes["roughness"].inputs["To Min"].default_value = base
    nodes["roughness"].inputs["To Max"].default_value = base + spread
    nodes["bump"].inputs["Strength"].default_value = rng.uniform(0.0, 0.08)
    nodes["noise"].inputs["W"].default_value = rng.uniform(0, 100)


def drop(scene, obj, rng, frames):
    """Let the brick fall onto the floor from a random orientation.

    Returns its resting matrix. The simulation is reset at frame 1.
    """
    scene.frame_set(1)
    obj.location = (0, 0, obj.dimensions.length * rng.uniform(0.8, 1.5))
    obj.rotation_euler = [rng.uniform(0, 2 * math.pi) for _ in range(3)]
    scene.rigidbody_world.point_cache.frame_end = frames
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
    return obj.matrix_world.copy()


def randomise_light_and_floor(scene, floor_nodes, lights, rng, floors):
    """Floor colour and grain, two lights with one colour temperature. Returns the floor name."""
    floors = [f for f in FLOORS if f[0] in floors]
    name, hex_value, _ = rng.choices(floors, weights=[w for *_, w in floors])[0]
    if hex_value is None:  # grey like the internet renders
        grey = rng.uniform(0.35, 0.6)
        colour = (grey, grey, grey * rng.uniform(0.97, 1.0), 1)
    else:
        colour = srgb_to_linear(hex_value)
    floor_nodes["colour"].inputs["A"].default_value = colour
    floor_nodes["bsdf"].inputs["Roughness"].default_value = rng.uniform(0.5, 0.95)
    floor_nodes["bump"].inputs["Strength"].default_value = rng.uniform(0.0, 0.3)

    # white balance: all lights of one photo have the same colour temperature
    # (only a hint of it: phones correct the white balance)
    tint = tuple(0.4 * c + 0.6 for c in kelvin_to_rgb(rng.uniform(3200, 6500)))
    for index, light in enumerate(lights):
        elevation = math.radians(rng.uniform(35, 85))
        azimuth = rng.uniform(0, 2 * math.pi)
        distance = 3.0
        light.location = (
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        )
        light.rotation_euler = (
            (Vector((0, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()
        )
        light.data.size = rng.uniform(0.5, 2.0)  # bigger = softer shadow
        light.data.color = tint
        key = index == 0
        light.data.energy = rng.uniform(ENERGY_MIN, ENERGY_MAX) * (
            1 if key else rng.uniform(0, 0.6)
        )

    # a white floor should look white (like a phone photo of white paper), not grey
    scene.view_settings.exposure = rng.uniform(0.0, 0.45) if name == "white" else 0.0

    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (*tint, 1)
    background.inputs["Strength"].default_value = rng.uniform(0.2, 0.8)
    return name


def place_camera(camera, target, radius, elevation, azimuth):
    fov = camera.data.angle
    distance = radius / math.sin(fov / 2) * 1.1
    camera.location = target + Vector(
        (
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        )
    )
    camera.rotation_euler = (
        (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.dof.focus_distance = distance
    bpy.context.view_layer.update()  # otherwise camera.matrix_world is still the old one


def randomise_camera(camera, rng):
    """Focal length and depth of field (close-up photos are only partly sharp)."""
    camera.data.lens = rng.uniform(50, 100)
    camera.data.dof.aperture_fstop = rng.uniform(*FSTOP_RANGE)


def set_crop(scene, camera, obj, margin):
    """Render only a box around the brick (tight crop like the internet renders)."""
    world = obj.matrix_world
    points = [
        world_to_camera_view(scene, camera, world @ v.co) for v in obj.data.vertices
    ]
    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])
    pad = margin * max(xs.max() - xs.min(), ys.max() - ys.min())
    render = scene.render
    render.border_min_x = max(xs.min() - pad, 0.0)
    render.border_max_x = min(xs.max() + pad, 1.0)
    render.border_min_y = max(ys.min() - pad, 0.0)
    render.border_max_y = min(ys.max() + pad, 1.0)


# ---------------------------------------------------------------- main


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ldraw",
        required=True,
        help="LDraw library folder (contains parts/, p/, LDConfig.ldr)",
    )
    parser.add_argument(
        "--parts", nargs="+", required=True, help="LDraw part numbers, e.g. 4073 3062b"
    )
    parser.add_argument("--out", default="data/raw/own_renders")
    parser.add_argument(
        "--poses",
        type=int,
        default=2000,
        help="poses per part (images = poses x views)",
    )
    parser.add_argument(
        "--views", type=int, default=1, help="camera positions per pose"
    )
    parser.add_argument("--trans-share", type=float, default=0.3)
    parser.add_argument(
        "--elevation",
        type=float,
        nargs=2,
        default=(25, 88),
        help="camera height range in degrees (90 = straight from above, like our sheet photos)",
    )
    parser.add_argument(
        "--floors",
        nargs="+",
        default=["white"],
        choices=[name for name, *_ in FLOORS],
        help="floor colours to choose from, e.g. --floors white grey petrol altrosa mint",
    )
    parser.add_argument(
        "--lowres", action="store_true", help="16-segment circles instead of p/48"
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument(
        "--frames", type=int, default=120, help="physics frames per drop"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    root = Path(args.ldraw)
    colours = read_colours(root)
    for name in OPAQUE_COLOURS + TRANS_COLOURS:
        colour_value(colours, name)  # fail early on an unknown colour

    if not 1 <= args.views <= 10:
        raise ValueError(
            "--views must be 1..10 (the views of a pose share an id range of 10)"
        )
    # absolute: Blender resolves a relative render path against the drive root,
    # not against the current folder
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[render] output: {out}")
    manifest_path = out / "renders_manifest.csv"
    new_manifest = not manifest_path.exists()
    if not new_manifest:
        with open(manifest_path, encoding="utf-8") as file:
            header = file.readline().strip().split(",")
        if header != MANIFEST_HEADER:
            raise ValueError(
                f"{manifest_path} has other columns ({header}): delete it or use another --out"
            )
    scene, floor_nodes, lights, camera = setup_scene(args)

    with open(manifest_path, "a", newline="", encoding="utf-8") as manifest:
        writer = csv.writer(manifest, lineterminator="\n")
        if new_manifest:
            writer.writerow(
                ["file", "part", "colour", "floor", "pose_id", "view", "seed"]
            )
        for part_index, part in enumerate(args.parts):
            render_part(
                args,
                root,
                colours,
                part_index,
                part,
                out,
                scene,
                floor_nodes,
                lights,
                camera,
                manifest,
                writer,
            )


def render_part(
    args,
    root,
    colours,
    part_index,
    part,
    out,
    scene,
    floor_nodes,
    lights,
    camera,
    manifest,
    writer,
):
    vertices, triangles, description = load_ldraw(root, part, hires=not args.lowres)
    label = part.rstrip("abcdefghijklmnopqrstuvwxyz")  # 3062b -> 3062
    folder = out / FOLDERS.get(part, f"{label} {description.lower()}")
    folder.mkdir(parents=True, exist_ok=True)
    obj, part_nodes = make_part_object(vertices, triangles, part)
    print(f"[part] {part} ({description}): {len(triangles)} triangles -> {folder}")

    for pose in range(args.poses):
        seed = args.seed * 1_000_000 + part_index * 10_000 + pose
        rng = random.Random(seed)
        floor = randomise_light_and_floor(scene, floor_nodes, lights, rng, args.floors)
        transparent = rng.random() < args.trans_share
        choices = TRANS_COLOURS if transparent else OPAQUE_COLOURS
        if floor == "white":  # a white brick on white paper is invisible in the render
            choices = [c for c in choices if c != "White"]
        colour = rng.choice(choices)
        hex_value, _ = colour_value(colours, colour)
        set_colour(part_nodes, hex_value, transparent, rng)
        randomise_camera(camera, rng)
        drop(scene, obj, rng, args.frames)

        target = Vector(
            np.mean([obj.matrix_world @ c for c in map(Vector, obj.bound_box)], axis=0)
        )
        radius = obj.dimensions.length / 2
        elevation = math.radians(rng.uniform(*args.elevation))
        azimuth = rng.uniform(0, 2 * math.pi)
        pose_id = 3_000_000_000 + seed * 10
        for view in range(args.views):
            place_camera(
                camera,
                target,
                radius,
                min(elevation + math.radians(rng.uniform(-6, 6)), math.radians(89)),
                azimuth + math.radians(view * rng.uniform(5, 12)),
            )
            set_crop(scene, camera, obj, rng.uniform(0.04, 0.15))
            name = f"{label}_{colour}_{view}_{pose_id + view}.jpeg"
            scene.render.filepath = str(folder / name)
            bpy.ops.render.render(write_still=True)
            writer.writerow(
                [f"{folder.name}/{name}", label, colour, floor, pose_id, view, seed]
            )
        manifest.flush()
        print(f"[pose] {part} {pose + 1}/{args.poses}: {colour}")

    bpy.data.objects.remove(obj)


if __name__ == "__main__":
    main()
