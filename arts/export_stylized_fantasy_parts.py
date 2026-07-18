"""Export the fantasy terrain as seven reusable, origin-local GLB prototypes.

The source Blender scene contains 28 placed terrain/building/tree/sheep groups.
This exporter keeps all 28 source placements in the manifest, but writes only
seven GLBs: terrain, cottage, castle, watchtower, pine, broad tree, and sheep.
Root translation, yaw, and scale are moved to the GDevelop instances so every
copy can share its family's GLB.

Run with Blender 5.1 or newer:

  blender --background --factory-startup --python \
    arts/export_stylized_fantasy_parts.py -- \
    --input arts/stylized_fantasy_terrain.blend \
    --output-dir assets/models/stylized_fantasy_terrain_parts \
    --overwrite

The source .blend is opened read-only and is never saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


EXPECTED_SOURCE_MESHES = 654
EXPECTED_TERRAIN_MESHES = 450
EXPECTED_BUILDING_MESHES = 120
EXPECTED_TREE_MESHES = 60
EXPECTED_SHEEP_MESHES = 24
EXPECTED_SOURCE_INSTANCES = 28
EXPECTED_PROTOTYPES = 7
EXPECTED_SOURCE_TRIANGLES = 29476
EXPECTED_OUTPUT_MESHES = 540
EXPECTED_OUTPUT_TRIANGLES = 23492

GDEVELOP_ANCHOR = Vector((640.0, 360.0, 0.0))
GDEVELOP_TARGET_WIDTH = 4680.0
BOUNDS_TOLERANCE = 1.0e-4
ROTATION_TOLERANCE_DEGREES = 1.0e-4

PROTOTYPE_DEFINITIONS = {
    "terrain": {
        "filename": "terrain.glb",
        "gdevelopName": "Terrain",
        "representative": "terrain",
    },
    "cottage": {
        "filename": "building_cottage.glb",
        "gdevelopName": "BuildingCottage",
        "representative": "cottage_a",
    },
    "castle": {
        "filename": "building_castle.glb",
        "gdevelopName": "BuildingCastle",
        "representative": "castle",
    },
    "watchtower": {
        "filename": "building_watchtower.glb",
        "gdevelopName": "BuildingWatchtower",
        "representative": "western_watchtower",
    },
    "pine": {
        "filename": "tree_pine.glb",
        "gdevelopName": "TreePine",
        "representative": "pine_01",
    },
    "broad_tree": {
        "filename": "tree_broad.glb",
        "gdevelopName": "TreeBroad",
        "representative": "broad_tree_04",
    },
    "sheep": {
        "filename": "sheep.glb",
        "gdevelopName": "Sheep",
        "representative": "sheep_01",
    },
}

# These families contain small authored mesh/material variations. Consolidation
# deliberately standardizes them to the representative named above.
APPROXIMATE_GEOMETRY_INSTANCES = {
    "cottage_b",
    "cottage_d",
    "cottage_e",
    "southern_watchtower",
    "broad_tree_00",
    "broad_tree_01",
    "broad_tree_02",
    "broad_tree_03",
    "broad_tree_05",
    "broad_tree_06",
    "broad_tree_07",
}
CANONICAL_MATERIAL_SUBSTITUTIONS = {
    "broad_tree_01",
    "broad_tree_03",
    "broad_tree_05",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export seven reusable GLBs and all 28 source placements."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rounded(value: float) -> float:
    result = round(float(value), 9)
    return 0.0 if abs(result) < 5.0e-10 else result


def normalized_scale(value: float) -> float:
    """Snap float noise around the scene's authored 0.05 scale increments."""
    authored_increment = round(float(value) * 20.0) / 20.0
    return authored_increment if abs(float(value) - authored_increment) < 1.0e-5 else float(value)


def vector_values(vector: Vector) -> list[float]:
    return [rounded(component) for component in vector]


def matrix_values(matrix: Matrix) -> list[list[float]]:
    return [[rounded(value) for value in row] for row in matrix]


def bounds_record(minimum: Vector, maximum: Vector) -> dict[str, list[float]]:
    return {
        "min": vector_values(minimum),
        "max": vector_values(maximum),
        "size": vector_values(maximum - minimum),
        "center": vector_values((minimum + maximum) * 0.5),
    }


def bounds_for_objects(
    objects: list[bpy.types.Object], transform: Matrix | None = None
) -> tuple[Vector, Vector]:
    outer = transform if transform is not None else Matrix.Identity(4)
    points = [
        outer @ obj.matrix_world @ Vector(corner)
        for obj in objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not points:
        raise RuntimeError("Cannot calculate bounds for an empty mesh set.")
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


def expanded_to_origin(minimum: Vector, maximum: Vector) -> tuple[Vector, Vector]:
    return (
        Vector((min(minimum.x, 0.0), min(minimum.y, 0.0), min(minimum.z, 0.0))),
        Vector((max(maximum.x, 0.0), max(maximum.y, 0.0), max(maximum.z, 0.0))),
    )


def assert_close(actual: Vector, expected: Vector, label: str) -> None:
    error = max(abs(actual[index] - expected[index]) for index in range(3))
    if error > BOUNDS_TOLERANCE:
        raise RuntimeError(
            f"{label} differs by {error:.9g}; actual={tuple(actual)}, "
            f"expected={tuple(expected)}"
        )


def glb_summary(path: Path) -> dict[str, int | str]:
    with path.open("rb") as glb:
        header = glb.read(12)
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
        "cameras": len(document.get("cameras", [])),
        "animations": len(document.get("animations", [])),
    }


def round_trip_bounds(path: Path) -> tuple[Vector, Vector, int]:
    objects_before = set(bpy.data.objects)
    collections_before = set(bpy.data.collections)
    meshes_before = set(bpy.data.meshes)
    materials_before = set(bpy.data.materials)
    images_before = set(bpy.data.images)

    bpy.ops.import_scene.gltf(filepath=str(path))
    imported_objects = [obj for obj in bpy.data.objects if obj not in objects_before]
    imported_meshes = [obj for obj in imported_objects if obj.type == "MESH"]
    minimum, maximum = bounds_for_objects(imported_meshes)

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

    return minimum, maximum, len(imported_meshes)


def mesh_triangles(objects: list[bpy.types.Object]) -> int:
    triangles = 0
    for obj in objects:
        obj.data.calc_loop_triangles()
        triangles += len(obj.data.loop_triangles)
    return triangles


def evaluated_duplicates(
    sources: list[bpy.types.Object], basis_world: Matrix, slug: str
) -> tuple[bpy.types.Collection, list[bpy.types.Object]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    export_collection = bpy.data.collections.new(f"__EXPORT_{slug}")
    bpy.context.scene.collection.children.link(export_collection)
    duplicates: list[bpy.types.Object] = []
    inverse_basis = basis_world.inverted_safe()

    for source in sorted(sources, key=lambda item: item.name):
        evaluated = source.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(
            evaluated, preserve_all_data_layers=True, depsgraph=depsgraph
        )
        duplicate = bpy.data.objects.new(source.name, mesh)
        export_collection.objects.link(duplicate)
        duplicate.matrix_world = inverse_basis @ source.matrix_world
        duplicates.append(duplicate)

    return export_collection, duplicates


def remove_duplicates(
    collection: bpy.types.Collection, duplicates: list[bpy.types.Object]
) -> None:
    meshes = [obj.data for obj in duplicates]
    for obj in duplicates:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    bpy.data.collections.remove(collection)


def select_only(objects: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def export_glb(path: Path, objects: list[bpy.types.Object]) -> None:
    select_only(objects)
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


def collection_objects(name: str) -> set[bpy.types.Object]:
    collection = bpy.data.collections.get(name)
    if collection is None:
        raise RuntimeError(f"Missing expected collection: {name}")
    return set(collection.all_objects)


def object_named(name: str) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError(f"Missing expected object: {name}")
    return obj


def make_source_instances(meshes: list[bpy.types.Object]) -> list[dict[str, object]]:
    mesh_set = set(meshes)
    buildings_collection = collection_objects("04_Buildings")
    instances: list[dict[str, object]] = []
    claimed: set[bpy.types.Object] = set()

    def add(
        key: str,
        kind: str,
        prototype_key: str,
        sources: list[bpy.types.Object],
        root_object: bpy.types.Object | None = None,
        basis_world: Matrix | None = None,
    ) -> None:
        source_set = set(sources)
        if not source_set or not source_set <= mesh_set:
            raise RuntimeError(f"{key} resolved to an invalid mesh set.")
        overlap = claimed & source_set
        if overlap:
            raise RuntimeError(
                f"{key} overlaps already claimed meshes: "
                + ", ".join(sorted(item.name for item in overlap))
            )
        if prototype_key not in PROTOTYPE_DEFINITIONS:
            raise RuntimeError(f"{key} has unknown prototype {prototype_key}.")
        claimed.update(source_set)
        root_matrix = (
            basis_world.copy()
            if basis_world is not None
            else root_object.matrix_world.copy()
        )
        instances.append(
            {
                "key": key,
                "kind": kind,
                "prototypeKey": prototype_key,
                "sources": sorted(source_set, key=lambda item: item.name),
                "rootObject": root_object.name if root_object is not None else None,
                "rootMatrix": root_matrix,
            }
        )

    for letter in "ABCDE":
        root = object_named(f"Cottage_{letter}")
        add(
            f"cottage_{letter.lower()}",
            "building",
            "cottage",
            [obj for obj in meshes if obj.parent == root],
            root_object=root,
        )

    castle_sources = [
        obj
        for obj in meshes
        if obj in buildings_collection
        and obj.name.startswith("Castle_")
        and not obj.name.startswith("Castle_Terrace_")
    ]
    add(
        "castle",
        "building",
        "castle",
        castle_sources,
        root_object=object_named("Castle_Keep"),
    )

    for direction in ("Western", "Southern"):
        add(
            f"{direction.lower()}_watchtower",
            "building",
            "watchtower",
            [
                obj
                for obj in meshes
                if obj in buildings_collection
                and obj.name.startswith(f"{direction}_Watchtower_")
            ],
            root_object=object_named(f"{direction}_Watchtower_Body"),
        )

    for index in range(7):
        add(
            f"pine_{index:02d}",
            "tree",
            "pine",
            [obj for obj in meshes if obj.name.startswith(f"Pine_{index:02d}_")],
            root_object=object_named(f"Pine_{index:02d}_Trunk"),
        )

    for index in range(8):
        add(
            f"broad_tree_{index:02d}",
            "tree",
            "broad_tree",
            [obj for obj in meshes if obj.name.startswith(f"BroadTree_{index:02d}_")],
            root_object=object_named(f"BroadTree_{index:02d}_Trunk"),
        )

    for index in range(1, 5):
        root = object_named(f"Sheep_{index:02d}")
        add(
            f"sheep_{index:02d}",
            "sheep",
            "sheep",
            [obj for obj in meshes if obj.parent == root],
            root_object=root,
        )

    terrain = sorted(mesh_set - claimed, key=lambda item: item.name)
    add(
        "terrain",
        "terrain",
        "terrain",
        terrain,
        basis_world=Matrix.Identity(4),
    )
    terrain_record = instances.pop()
    instances.insert(0, terrain_record)

    kind_counts = {
        kind: sum(len(instance["sources"]) for instance in instances if instance["kind"] == kind)
        for kind in ("terrain", "building", "tree", "sheep")
    }
    expected_counts = {
        "terrain": EXPECTED_TERRAIN_MESHES,
        "building": EXPECTED_BUILDING_MESHES,
        "tree": EXPECTED_TREE_MESHES,
        "sheep": EXPECTED_SHEEP_MESHES,
    }
    if len(instances) != EXPECTED_SOURCE_INSTANCES:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_INSTANCES} source instances, found {len(instances)}."
        )
    if claimed != mesh_set:
        raise RuntimeError("Source partition does not cover every mesh exactly once.")
    if kind_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected partition counts: {kind_counts}; expected {expected_counts}."
        )
    return instances


def make_prototypes(instances: list[dict[str, object]]) -> list[dict[str, object]]:
    instances_by_key = {str(instance["key"]): instance for instance in instances}
    prototypes: list[dict[str, object]] = []
    for prototype_key, definition in PROTOTYPE_DEFINITIONS.items():
        representative_key = str(definition["representative"])
        representative = instances_by_key.get(representative_key)
        if representative is None:
            raise RuntimeError(
                f"Prototype {prototype_key} is missing representative {representative_key}."
            )
        if representative["prototypeKey"] != prototype_key:
            raise RuntimeError(
                f"Representative {representative_key} maps to the wrong prototype."
            )
        prototypes.append(
            {
                "key": prototype_key,
                "kind": representative["kind"],
                "filename": definition["filename"],
                "gdevelopName": definition["gdevelopName"],
                "representativeKey": representative_key,
                "sources": representative["sources"],
                "rootMatrix": representative["rootMatrix"],
            }
        )
    if len(prototypes) != EXPECTED_PROTOTYPES:
        raise RuntimeError(
            f"Expected {EXPECTED_PROTOTYPES} prototypes, found {len(prototypes)}."
        )
    mapped = {str(instance["prototypeKey"]) for instance in instances}
    if mapped != set(PROTOTYPE_DEFINITIONS):
        raise RuntimeError(f"Prototype mapping mismatch: {sorted(mapped)}")
    return prototypes


def previous_manifest_glbs(manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        return set()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = document.get("assets", document.get("prototypes", []))
    filenames: set[str] = set()
    for record in records:
        filename = record.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise RuntimeError("Prior manifest contains an unsafe output filename.")
        if filename.lower().endswith(".glb"):
            filenames.add(filename)
    return filenames


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = (
        args.manifest.resolve() if args.manifest else output_dir / "manifest.json"
    )
    if not input_path.is_file():
        raise RuntimeError(f"Input Blender file does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_candidates = previous_manifest_glbs(manifest_path)

    bpy.ops.wm.open_mainfile(filepath=str(input_path), load_ui=False)
    source_scene = bpy.context.scene
    meshes = [obj for obj in source_scene.objects if obj.type == "MESH"]
    if len(meshes) != EXPECTED_SOURCE_MESHES:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_MESHES} source meshes, found {len(meshes)}."
        )

    instances = make_source_instances(meshes)
    prototypes = make_prototypes(instances)

    full_collection, full_duplicates = evaluated_duplicates(
        meshes, Matrix.Identity(4), "full_bounds"
    )
    full_minimum, full_maximum = bounds_for_objects(full_duplicates)
    full_center = (full_minimum + full_maximum) * 0.5
    uniform_scale = GDEVELOP_TARGET_WIDTH / (full_maximum.x - full_minimum.x)
    remove_duplicates(full_collection, full_duplicates)

    project_root = input_path.parent.parent
    source_relative = input_path.relative_to(project_root).as_posix()
    try:
        output_relative = output_dir.relative_to(project_root).as_posix()
    except ValueError:
        output_relative = output_dir.as_posix()

    prototype_records: list[dict[str, object]] = []
    prototype_by_key: dict[str, dict[str, object]] = {}
    output_mesh_count = 0
    output_triangle_count = 0

    for prototype_index, prototype in enumerate(prototypes, start=1):
        output_path = output_dir / str(prototype["filename"])
        if output_path.exists() and not args.overwrite:
            raise RuntimeError(
                f"Output exists (use --overwrite): {output_path.as_posix()}"
            )

        collection, duplicates = evaluated_duplicates(
            prototype["sources"], prototype["rootMatrix"], str(prototype["key"])
        )
        try:
            local_minimum, local_maximum = bounds_for_objects(duplicates)
            origin_minimum, origin_maximum = expanded_to_origin(
                local_minimum, local_maximum
            )
            triangle_count = mesh_triangles(duplicates)
            export_glb(output_path, duplicates)
        finally:
            remove_duplicates(collection, duplicates)

        summary = glb_summary(output_path)
        expected_meshes = len(prototype["sources"])
        if summary["cameras"] or summary["animations"]:
            raise RuntimeError(f"{output_path.name} contains a camera or animation.")
        if summary["meshes"] != expected_meshes:
            raise RuntimeError(
                f"{output_path.name} exported {summary['meshes']} meshes; "
                f"expected {expected_meshes}."
            )

        imported_minimum, imported_maximum, imported_mesh_count = round_trip_bounds(
            output_path
        )
        assert_close(imported_minimum, local_minimum, f"{output_path.name} min")
        assert_close(imported_maximum, local_maximum, f"{output_path.name} max")
        if imported_mesh_count != expected_meshes:
            raise RuntimeError(
                f"{output_path.name} round trip found {imported_mesh_count} meshes; "
                f"expected {expected_meshes}."
            )

        local_size = origin_maximum - origin_minimum
        default_size = [
            rounded(uniform_scale * float(local_size.x)),
            rounded(uniform_scale * float(local_size.y)),
            rounded(uniform_scale * float(local_size.z)),
        ]
        record = {
            "index": prototype_index - 1,
            "key": prototype["key"],
            "kind": prototype["kind"],
            "filename": prototype["filename"],
            "resourcePath": f"{output_relative}/{prototype['filename']}",
            "gdevelopObjectName": prototype["gdevelopName"],
            "representativeKey": prototype["representativeKey"],
            "sourceMeshCount": expected_meshes,
            "sourceTriangleCount": triangle_count,
            "sourceObjects": [obj.name for obj in prototype["sources"]],
            "localBounds": bounds_record(local_minimum, local_maximum),
            "localBoundsIncludingOrigin": bounds_record(origin_minimum, origin_maximum),
            "gdevelopDefaultSize": default_size,
            "glb": summary,
            "roundTrip": {
                "verified": True,
                "meshCount": imported_mesh_count,
                "localBounds": bounds_record(imported_minimum, imported_maximum),
                "tolerance": BOUNDS_TOLERANCE,
            },
        }
        prototype_records.append(record)
        prototype_by_key[str(prototype["key"])] = record
        output_mesh_count += expected_meshes
        output_triangle_count += triangle_count
        print(
            f"EXPORTED_PROTOTYPE {prototype_index:02d}/{EXPECTED_PROTOTYPES} "
            f"{output_path.name} meshes={expected_meshes} "
            f"triangles={triangle_count} bytes={summary['bytes']}"
        )

    if output_mesh_count != EXPECTED_OUTPUT_MESHES:
        raise RuntimeError(
            f"Expected {EXPECTED_OUTPUT_MESHES} output meshes, found {output_mesh_count}."
        )
    if output_triangle_count != EXPECTED_OUTPUT_TRIANGLES:
        raise RuntimeError(
            f"Expected {EXPECTED_OUTPUT_TRIANGLES} output triangles, "
            f"found {output_triangle_count}."
        )

    instance_records: list[dict[str, object]] = []
    source_triangle_count = 0
    for instance_index, instance in enumerate(instances):
        root_matrix: Matrix = instance["rootMatrix"]
        location, rotation, root_scale = root_matrix.decompose()
        euler = rotation.to_euler("XYZ")
        euler_degrees = [rounded(math.degrees(component)) for component in euler]
        if abs(euler_degrees[0]) > ROTATION_TOLERANCE_DEGREES or abs(euler_degrees[1]) > ROTATION_TOLERANCE_DEGREES:
            raise RuntimeError(
                f"{instance['key']} has unsupported X/Y root rotation: {euler_degrees}"
            )

        collection, duplicates = evaluated_duplicates(
            instance["sources"], root_matrix, f"audit_{instance['key']}"
        )
        try:
            source_local_minimum, source_local_maximum = bounds_for_objects(duplicates)
            raw_minimum, raw_maximum = bounds_for_objects(duplicates, root_matrix)
            triangle_count = mesh_triangles(duplicates)
        finally:
            remove_duplicates(collection, duplicates)

        source_triangle_count += triangle_count
        prototype_record = prototype_by_key[str(instance["prototypeKey"])]
        default_size = prototype_record["gdevelopDefaultSize"]
        source_origin_minimum, source_origin_maximum = expanded_to_origin(
            source_local_minimum, source_local_maximum
        )
        source_origin_size = source_origin_maximum - source_origin_minimum
        prototype_origin_size = prototype_record["localBoundsIncludingOrigin"]["size"]
        applied_scale = [
            normalized_scale(
                abs(float(root_scale[index]))
                * float(source_origin_size[index])
                / float(prototype_origin_size[index])
            )
            for index in range(3)
        ]
        gdevelop_position = [
            rounded(float(GDEVELOP_ANCHOR.x) + uniform_scale * float(location.x)),
            rounded(float(GDEVELOP_ANCHOR.y) - uniform_scale * float(location.y)),
            rounded(
                float(GDEVELOP_ANCHOR.z)
                + uniform_scale * (float(location.z) - float(full_center.z))
            ),
        ]
        gdevelop_size = [
            rounded(float(default_size[index]) * applied_scale[index])
            for index in range(3)
        ]
        layout_yaw = rounded(-euler_degrees[2])
        geometry_policy = (
            "canonical-substitution"
            if instance["key"] in APPROXIMATE_GEOMETRY_INSTANCES
            else "equivalent"
        )
        material_policy = (
            "canonical-substitution"
            if instance["key"] in CANONICAL_MATERIAL_SUBSTITUTIONS
            else "preserved"
        )
        record = {
            "index": instance_index,
            "key": instance["key"],
            "kind": instance["kind"],
            "prototypeKey": instance["prototypeKey"],
            "gdevelopObjectName": prototype_record["gdevelopObjectName"],
            "sourceMeshCount": len(instance["sources"]),
            "sourceTriangleCount": triangle_count,
            "sourceObjects": [obj.name for obj in instance["sources"]],
            "rootObject": instance["rootObject"],
            "rootMatrix": matrix_values(root_matrix),
            "pivot": vector_values(location),
            "pivotEulerDegrees": euler_degrees,
            "rootScale": vector_values(root_scale),
            "rawBounds": bounds_record(raw_minimum, raw_maximum),
            "sourceLocalBounds": bounds_record(
                source_local_minimum, source_local_maximum
            ),
            "sourceLocalBoundsIncludingOrigin": bounds_record(
                source_origin_minimum, source_origin_maximum
            ),
            "reusePolicy": {
                "geometry": geometry_policy,
                "materials": material_policy,
                "acceptedByConsolidationRequest": True,
            },
            "gdevelop": {
                "position": gdevelop_position,
                "size": gdevelop_size,
                "layoutRotation": [0.0, 0.0, layout_yaw],
                "appliedScale": [rounded(value) for value in applied_scale],
                "rootTransformBakedIntoGlb": False,
            },
        }
        instance_records.append(record)

    if source_triangle_count != EXPECTED_SOURCE_TRIANGLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_TRIANGLES} source triangles, "
            f"found {source_triangle_count}."
        )

    current_filenames = {str(record["filename"]) for record in prototype_records}
    stale_filenames = sorted(stale_candidates - current_filenames)
    removed_stale: list[str] = []
    for filename in stale_filenames:
        stale_path = (output_dir / filename).resolve()
        if stale_path.parent != output_dir:
            raise RuntimeError(f"Refusing to remove output outside {output_dir}.")
        if stale_path.is_file():
            stale_path.unlink()
            removed_stale.append(filename)

    manifest = {
        "schemaVersion": 2,
        "source": {
            "file": source_relative,
            "sha256": sha256(input_path),
            "blenderVersion": bpy.app.version_string,
            "scene": source_scene.name,
            "objectCount": len(source_scene.objects),
            "meshCount": len(meshes),
            "excludedObjects": sorted(
                obj.name for obj in source_scene.objects if obj.type != "MESH"
            ),
            "evaluatedBounds": bounds_record(full_minimum, full_maximum),
        },
        "partition": {
            "instanceCount": len(instance_records),
            "meshCount": sum(record["sourceMeshCount"] for record in instance_records),
            "triangleCount": source_triangle_count,
            "countsByKind": {
                kind: sum(1 for record in instance_records if record["kind"] == kind)
                for kind in ("terrain", "building", "tree", "sheep")
            },
            "meshesByKind": {
                kind: sum(
                    record["sourceMeshCount"]
                    for record in instance_records
                    if record["kind"] == kind
                )
                for kind in ("terrain", "building", "tree", "sheep")
            },
            "exhaustive": True,
            "pairwiseDisjoint": True,
        },
        "outputs": {
            "prototypeCount": len(prototype_records),
            "meshCount": output_mesh_count,
            "triangleCount": output_triangle_count,
            "removedPriorManifestOutputs": removed_stale,
            "countsByKind": {
                kind: sum(1 for record in prototype_records if record["kind"] == kind)
                for kind in ("terrain", "building", "tree", "sheep")
            },
        },
        "gdevelopIntegration": {
            "anchor": vector_values(GDEVELOP_ANCHOR),
            "fullSceneCenter": vector_values(full_center),
            "uniformScale": rounded(uniform_scale),
            "positionFormula": [
                "x = anchorX + scale * blenderX",
                "y = anchorY - scale * blenderY",
                "z = anchorZ + scale * (blenderZ - fullSceneCenterZ)",
            ],
            "rotationFormula": "gdevelopZ = -blenderZ",
            "content": {
                "originLocation": "ModelOrigin",
                "centerLocation": "ModelOrigin",
                "rotationX": 90.0,
                "rotationY": 0.0,
                "rotationZ": 0.0,
                "keepAspectRatio": True,
                "materialType": "KeepOriginal",
            },
        },
        "consolidation": {
            "policy": "one shared prototype per semantic model family",
            "geometrySubstitutionInstances": sorted(APPROXIMATE_GEOMETRY_INSTANCES),
            "materialSubstitutionInstances": sorted(CANONICAL_MATERIAL_SUBSTITUTIONS),
        },
        "prototypes": prototype_records,
        "instances": instance_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "EXPORT_SUMMARY "
        f"prototypes={len(prototype_records)} instances={len(instance_records)} "
        f"sourceMeshes={manifest['partition']['meshCount']} "
        f"outputMeshes={output_mesh_count} sourceTriangles={source_triangle_count} "
        f"outputTriangles={output_triangle_count} failures=0 "
        f"manifest={manifest_path.as_posix()}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"EXPORT_FAILURE {type(error).__name__}: {error}", file=sys.stderr)
        raise
