"""Export one authored terrace/ladder step as a reusable origin-local GLB.

Run with Blender 5.1 or newer:

  blender --background --factory-startup --python \
    arts/export_ladder_step_cube.py -- \
    --input arts/stylized_fantasy_terrain.blend \
    --output assets/models/stylized_fantasy_terrain_parts/ladder_step_cube.glb \
    --overwrite

The source .blend is opened read-only and is never saved. TerraceStep_00 is the
representative because all six authored TerraceStep meshes have identical cube
geometry and the same Warm Timber material. Its authored orientation is baked
into the exported mesh, then the mesh is anchored at its bottom-center.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


SOURCE_OBJECT = "TerraceStep_00"
STEP_OBJECTS = tuple(f"TerraceStep_{index:02d}" for index in range(6))
EXPECTED_MATERIAL = "Warm Timber"
EXPECTED_VERTICES = 8
EXPECTED_POLYGONS = 6
EXPECTED_EVALUATED_VERTICES = 56
EXPECTED_EVALUATED_POLYGONS = 54
EXPECTED_TRIANGLES = 108
BOUNDS_TOLERANCE = 1.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one authored ladder step cube as a standalone GLB."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def rounded(value: float) -> float:
    result = round(float(value), 9)
    return 0.0 if abs(result) < 5.0e-10 else result


def vector_values(vector: Vector) -> list[float]:
    return [rounded(component) for component in vector]


def bounds_record(minimum: Vector, maximum: Vector) -> dict[str, list[float]]:
    return {
        "min": vector_values(minimum),
        "max": vector_values(maximum),
        "size": vector_values(maximum - minimum),
        "center": vector_values((minimum + maximum) * 0.5),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounds_for_mesh(mesh: bpy.types.Mesh) -> tuple[Vector, Vector]:
    if not mesh.vertices:
        raise RuntimeError("Cannot calculate bounds for an empty mesh.")
    minimum = Vector(
        (
            min(vertex.co.x for vertex in mesh.vertices),
            min(vertex.co.y for vertex in mesh.vertices),
            min(vertex.co.z for vertex in mesh.vertices),
        )
    )
    maximum = Vector(
        (
            max(vertex.co.x for vertex in mesh.vertices),
            max(vertex.co.y for vertex in mesh.vertices),
            max(vertex.co.z for vertex in mesh.vertices),
        )
    )
    return minimum, maximum


def stabilize_uv_layers(mesh: bpy.types.Mesh) -> None:
    """Remove sub-micro UV noise left by evaluated bevel interpolation."""
    for layer in mesh.uv_layers:
        for loop in layer.data:
            loop.uv = (round(float(loop.uv.x), 6), round(float(loop.uv.y), 6))


def bounds_for_objects(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not points:
        raise RuntimeError("Cannot calculate bounds for an empty object set.")
    minimum = Vector(
        (
            min(point.x for point in points),
            min(point.y for point in points),
            min(point.z for point in points),
        )
    )
    maximum = Vector(
        (
            max(point.x for point in points),
            max(point.y for point in points),
            max(point.z for point in points),
        )
    )
    return minimum, maximum


def assert_close(actual: Vector, expected: Vector, label: str) -> None:
    error = max(abs(actual[index] - expected[index]) for index in range(3))
    if error > BOUNDS_TOLERANCE:
        raise RuntimeError(
            f"{label} differs by {error:.9g}; actual={tuple(actual)}, "
            f"expected={tuple(expected)}"
        )


def glb_summary(path: Path) -> dict[str, object]:
    with path.open("rb") as glb:
        header = glb.read(12)
        if len(header) != 12:
            raise RuntimeError(f"{path.name} has a truncated GLB header.")
        magic, version, total_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise RuntimeError(f"{path.name} is not a glTF 2 binary file.")
        json_length, json_type = struct.unpack("<I4s", glb.read(8))
        if json_type != b"JSON":
            raise RuntimeError(f"{path.name} has no leading JSON chunk.")
        document = json.loads(glb.read(json_length).decode("utf-8"))
    if total_length != path.stat().st_size:
        raise RuntimeError(f"{path.name} has an inconsistent GLB byte length.")
    return {
        "bytes": total_length,
        "sha256": sha256(path),
        "nodes": len(document.get("nodes", [])),
        "meshes": len(document.get("meshes", [])),
        "materials": len(document.get("materials", [])),
        "materialNames": [
            material.get("name", "") for material in document.get("materials", [])
        ],
        "cameras": len(document.get("cameras", [])),
        "animations": len(document.get("animations", [])),
    }


def round_trip_bounds(path: Path) -> tuple[Vector, Vector, int, list[str]]:
    objects_before = set(bpy.data.objects)
    collections_before = set(bpy.data.collections)
    meshes_before = set(bpy.data.meshes)
    materials_before = set(bpy.data.materials)
    images_before = set(bpy.data.images)

    bpy.ops.import_scene.gltf(filepath=str(path))
    imported_objects = [obj for obj in bpy.data.objects if obj not in objects_before]
    imported_meshes = [obj for obj in imported_objects if obj.type == "MESH"]
    minimum, maximum = bounds_for_objects(imported_meshes)
    material_names = sorted(
        {
            re.sub(r"\.\d{3}$", "", material.name)
            for obj in imported_meshes
            for material in obj.data.materials
            if material is not None
        }
    )

    for obj in imported_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in [item for item in bpy.data.collections if item not in collections_before]:
        bpy.data.collections.remove(collection)
    for mesh in [item for item in bpy.data.meshes if item not in meshes_before]:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in [item for item in bpy.data.materials if item not in materials_before]:
        if material.users == 0:
            bpy.data.materials.remove(material)
    for image in [item for item in bpy.data.images if item not in images_before]:
        if image.users == 0:
            bpy.data.images.remove(image)

    return minimum, maximum, len(imported_meshes), material_names


def verify_authored_steps() -> bpy.types.Object:
    candidates: list[bpy.types.Object] = []
    for name in STEP_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise RuntimeError(f"Missing expected mesh object: {name}")
        if len(obj.data.vertices) != EXPECTED_VERTICES:
            raise RuntimeError(f"{name} is not the expected 8-vertex cube.")
        if len(obj.data.polygons) != EXPECTED_POLYGONS:
            raise RuntimeError(f"{name} is not the expected 6-face cube.")
        materials = [material.name for material in obj.data.materials if material]
        if materials != [EXPECTED_MATERIAL]:
            raise RuntimeError(f"{name} has unexpected materials: {materials}")
        candidates.append(obj)

    reference_coordinates = [vertex.co.copy() for vertex in candidates[0].data.vertices]
    for candidate in candidates[1:]:
        coordinates = [vertex.co.copy() for vertex in candidate.data.vertices]
        if len(coordinates) != len(reference_coordinates):
            raise RuntimeError(f"{candidate.name} geometry differs from {SOURCE_OBJECT}.")
        for index, (actual, expected) in enumerate(zip(coordinates, reference_coordinates)):
            assert_close(actual, expected, f"{candidate.name} vertex {index}")
    return candidates[0]


def export_glb(path: Path, obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.hide_render = False
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=False,
        export_cameras=False,
        export_lights=False,
        export_animations=False,
        export_materials="EXPORT",
    )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise RuntimeError(f"Input Blender file does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise RuntimeError(f"Output exists (use --overwrite): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(input_path), load_ui=False)
    source = verify_authored_steps()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = source.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(
        evaluated, preserve_all_data_layers=True, depsgraph=depsgraph
    )
    mesh.name = "LadderStepCubeMesh"
    stabilize_uv_layers(mesh)

    authored_orientation = source.matrix_world.copy()
    authored_orientation.translation = Vector((0.0, 0.0, 0.0))
    mesh.transform(authored_orientation)
    oriented_minimum, oriented_maximum = bounds_for_mesh(mesh)
    bottom_center = Vector(
        (
            (oriented_minimum.x + oriented_maximum.x) * 0.5,
            (oriented_minimum.y + oriented_maximum.y) * 0.5,
            oriented_minimum.z,
        )
    )
    mesh.transform(Matrix.Translation(-bottom_center))

    collection = bpy.data.collections.new("__EXPORT_LADDER_STEP_CUBE")
    bpy.context.scene.collection.children.link(collection)
    export_object = bpy.data.objects.new("LadderStepCube", mesh)
    collection.objects.link(export_object)
    export_object.matrix_world = Matrix.Identity(4)

    expected_minimum, expected_maximum = bounds_for_mesh(mesh)
    mesh.calc_loop_triangles()
    triangle_count = len(mesh.loop_triangles)
    if len(mesh.vertices) != EXPECTED_EVALUATED_VERTICES:
        raise RuntimeError(
            f"Expected {EXPECTED_EVALUATED_VERTICES} evaluated vertices, "
            f"found {len(mesh.vertices)}."
        )
    if len(mesh.polygons) != EXPECTED_EVALUATED_POLYGONS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVALUATED_POLYGONS} evaluated polygons, "
            f"found {len(mesh.polygons)}."
        )
    if triangle_count != EXPECTED_TRIANGLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRIANGLES} triangles, found {triangle_count}."
        )

    export_glb(output_path, export_object)
    summary = glb_summary(output_path)
    if summary["nodes"] != 1 or summary["meshes"] != 1:
        raise RuntimeError(f"Unexpected GLB node/mesh counts: {summary}")
    if summary["materials"] != 1:
        raise RuntimeError(f"Expected one GLB material: {summary}")
    if summary["materialNames"] != [EXPECTED_MATERIAL]:
        raise RuntimeError(f"Unexpected GLB material name: {summary}")
    if summary["cameras"] or summary["animations"]:
        raise RuntimeError(f"GLB contains a camera or animation: {summary}")

    imported_minimum, imported_maximum, imported_mesh_count, imported_materials = (
        round_trip_bounds(output_path)
    )
    assert_close(imported_minimum, expected_minimum, "round-trip minimum")
    assert_close(imported_maximum, expected_maximum, "round-trip maximum")
    if imported_mesh_count != 1:
        raise RuntimeError(
            f"Expected one round-trip mesh, found {imported_mesh_count}."
        )
    if imported_materials != [EXPECTED_MATERIAL]:
        raise RuntimeError(
            f"Round-trip material mismatch: expected {EXPECTED_MATERIAL}, "
            f"found {imported_materials}."
        )

    receipt = {
        "status": "verified",
        "input": input_path.as_posix(),
        "inputSha256": sha256(input_path),
        "output": output_path.as_posix(),
        "sourceObject": SOURCE_OBJECT,
        "equivalentSourceObjects": list(STEP_OBJECTS),
        "material": EXPECTED_MATERIAL,
        "vertices": len(mesh.vertices),
        "polygons": len(mesh.polygons),
        "triangles": triangle_count,
        "anchor": "bottom-center",
        "bounds": bounds_record(expected_minimum, expected_maximum),
        "glb": summary,
        "roundTrip": {
            "verified": True,
            "meshCount": imported_mesh_count,
            "materials": imported_materials,
            "bounds": bounds_record(imported_minimum, imported_maximum),
            "tolerance": BOUNDS_TOLERANCE,
        },
    }
    print("CODEX_LADDER_STEP_EXPORT=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
