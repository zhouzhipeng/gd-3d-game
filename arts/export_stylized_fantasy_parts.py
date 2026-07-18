"""Export the stylized fantasy scene into origin-local, per-instance GLBs.

Run with Blender 5.1 or newer:

  blender --background --factory-startup --python \
    arts/export_stylized_fantasy_parts.py -- \
    --input arts/stylized_fantasy_terrain.blend \
    --output-dir assets/models/stylized_fantasy_terrain_parts \
    --overwrite

The source .blend is opened read-only and is never saved. Each output is made
from evaluated mesh copies, so render-enabled modifiers are baked without
mutating the authored scene. The generated manifest records the exact Blender
pivot and the GDevelop placement/size mapping used by this project.
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
EXPECTED_ASSETS = 28

GDEVELOP_ANCHOR = Vector((640.0, 360.0, 0.0))
GDEVELOP_TARGET_WIDTH = 4680.0
BOUNDS_TOLERANCE = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split stylized_fantasy_terrain.blend into dedicated GLBs."
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


def vector_values(vector: Vector) -> list[float]:
    return [rounded(component) for component in vector]


def bounds_record(minimum: Vector, maximum: Vector) -> dict[str, list[float]]:
    return {
        "min": vector_values(minimum),
        "max": vector_values(maximum),
        "size": vector_values(maximum - minimum),
        "center": vector_values((minimum + maximum) * 0.5),
    }


def bounds_for_objects(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [
        obj.matrix_world @ Vector(corner)
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
    sources: list[bpy.types.Object], pivot: Vector, slug: str
) -> tuple[bpy.types.Collection, list[bpy.types.Object]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    export_collection = bpy.data.collections.new(f"__EXPORT_{slug}")
    bpy.context.scene.collection.children.link(export_collection)
    duplicates: list[bpy.types.Object] = []

    for source in sorted(sources, key=lambda item: item.name):
        evaluated = source.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(
            evaluated, preserve_all_data_layers=True, depsgraph=depsgraph
        )
        duplicate = bpy.data.objects.new(source.name, mesh)
        export_collection.objects.link(duplicate)
        duplicate.matrix_world = Matrix.Translation(-pivot) @ source.matrix_world
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


def make_specs(meshes: list[bpy.types.Object]) -> list[dict[str, object]]:
    mesh_set = set(meshes)
    buildings_collection = collection_objects("04_Buildings")

    specs: list[dict[str, object]] = []
    claimed: set[bpy.types.Object] = set()

    def add(
        key: str,
        kind: str,
        filename: str,
        gdevelop_name: str,
        sources: list[bpy.types.Object],
        pivot_object: bpy.types.Object | None = None,
        pivot: Vector | None = None,
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
        claimed.update(source_set)
        root_pivot = (
            pivot.copy()
            if pivot is not None
            else pivot_object.matrix_world.translation.copy()
        )
        root_euler = (
            pivot_object.matrix_world.to_euler("XYZ")
            if pivot_object is not None
            else Vector((0.0, 0.0, 0.0))
        )
        specs.append(
            {
                "key": key,
                "kind": kind,
                "filename": filename,
                "gdevelopName": gdevelop_name,
                "sources": sorted(source_set, key=lambda item: item.name),
                "pivot": root_pivot,
                "pivotEulerDegrees": [
                    rounded(math.degrees(component)) for component in root_euler
                ],
            }
        )

    for letter in "ABCDE":
        root = object_named(f"Cottage_{letter}")
        children = [obj for obj in meshes if obj.parent == root]
        add(
            f"cottage_{letter.lower()}",
            "building",
            f"building_cottage_{letter.lower()}.glb",
            f"BuildingCottage{letter}",
            children,
            pivot_object=root,
        )

    castle = [
        obj
        for obj in meshes
        if obj in buildings_collection
        and obj.name.startswith("Castle_")
        and not obj.name.startswith("Castle_Terrace_")
    ]
    add(
        "castle",
        "building",
        "building_castle.glb",
        "BuildingCastle",
        castle,
        pivot_object=object_named("Castle_Keep"),
    )

    for direction in ("Western", "Southern"):
        sources = [
            obj
            for obj in meshes
            if obj in buildings_collection
            and obj.name.startswith(f"{direction}_Watchtower_")
        ]
        add(
            f"{direction.lower()}_watchtower",
            "building",
            f"building_{direction.lower()}_watchtower.glb",
            f"Building{direction}Watchtower",
            sources,
            pivot_object=object_named(f"{direction}_Watchtower_Body"),
        )

    for index in range(7):
        prefix = f"Pine_{index:02d}_"
        sources = [obj for obj in meshes if obj.name.startswith(prefix)]
        add(
            f"pine_{index:02d}",
            "tree",
            f"tree_pine_{index:02d}.glb",
            f"TreePine{index:02d}",
            sources,
            pivot_object=object_named(f"Pine_{index:02d}_Trunk"),
        )

    for index in range(8):
        prefix = f"BroadTree_{index:02d}_"
        sources = [obj for obj in meshes if obj.name.startswith(prefix)]
        add(
            f"broad_tree_{index:02d}",
            "tree",
            f"tree_broad_{index:02d}.glb",
            f"TreeBroad{index:02d}",
            sources,
            pivot_object=object_named(f"BroadTree_{index:02d}_Trunk"),
        )

    for index in range(1, 5):
        root = object_named(f"Sheep_{index:02d}")
        children = [obj for obj in meshes if obj.parent == root]
        add(
            f"sheep_{index:02d}",
            "sheep",
            f"sheep_{index:02d}.glb",
            f"Sheep{index:02d}",
            children,
            pivot_object=root,
        )

    terrain = sorted(mesh_set - claimed, key=lambda item: item.name)
    specs.insert(
        0,
        {
            "key": "terrain",
            "kind": "terrain",
            "filename": "terrain.glb",
            "gdevelopName": "Terrain",
            "sources": terrain,
            "pivot": Vector((0.0, 0.0, 0.0)),
            "pivotEulerDegrees": [0.0, 0.0, 0.0],
        },
    )
    claimed.update(terrain)

    kind_counts = {
        kind: sum(len(spec["sources"]) for spec in specs if spec["kind"] == kind)
        for kind in ("terrain", "building", "tree", "sheep")
    }
    expected_counts = {
        "terrain": EXPECTED_TERRAIN_MESHES,
        "building": EXPECTED_BUILDING_MESHES,
        "tree": EXPECTED_TREE_MESHES,
        "sheep": EXPECTED_SHEEP_MESHES,
    }
    if len(specs) != EXPECTED_ASSETS:
        raise RuntimeError(f"Expected {EXPECTED_ASSETS} assets, found {len(specs)}.")
    if claimed != mesh_set:
        raise RuntimeError("Asset partition does not cover every source mesh exactly once.")
    if kind_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected partition counts: {kind_counts}; expected {expected_counts}."
        )
    return specs


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

    bpy.ops.wm.open_mainfile(filepath=str(input_path), load_ui=False)
    source_scene = bpy.context.scene
    meshes = [obj for obj in source_scene.objects if obj.type == "MESH"]
    if len(meshes) != EXPECTED_SOURCE_MESHES:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_MESHES} source meshes, found {len(meshes)}."
        )

    specs = make_specs(meshes)

    # Evaluated full-scene bounds establish the anchor used by the current
    # combined terrain object. Temporary copies include every render modifier.
    full_collection, full_duplicates = evaluated_duplicates(
        meshes, Vector((0.0, 0.0, 0.0)), "full_bounds"
    )
    full_minimum, full_maximum = bounds_for_objects(full_duplicates)
    full_center = (full_minimum + full_maximum) * 0.5
    uniform_scale = GDEVELOP_TARGET_WIDTH / (full_maximum.x - full_minimum.x)
    remove_duplicates(full_collection, full_duplicates)

    project_root = input_path.parent.parent
    source_relative = input_path.relative_to(project_root).as_posix()
    output_relative = output_dir.relative_to(project_root).as_posix()
    asset_records: list[dict[str, object]] = []
    total_triangles = 0

    for asset_index, spec in enumerate(specs, start=1):
        output_path = output_dir / spec["filename"]
        if output_path.exists() and not args.overwrite:
            raise RuntimeError(
                f"Output exists (use --overwrite): {output_path.as_posix()}"
            )

        collection, duplicates = evaluated_duplicates(
            spec["sources"], spec["pivot"], str(spec["key"])
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
        if summary["cameras"] or summary["animations"]:
            raise RuntimeError(f"{output_path.name} contains a camera or animation.")
        if summary["meshes"] != len(spec["sources"]):
            raise RuntimeError(
                f"{output_path.name} exported {summary['meshes']} meshes; "
                f"expected {len(spec['sources'])}."
            )

        imported_minimum, imported_maximum, imported_mesh_count = round_trip_bounds(
            output_path
        )
        assert_close(imported_minimum, local_minimum, f"{output_path.name} min")
        assert_close(imported_maximum, local_maximum, f"{output_path.name} max")
        if imported_mesh_count != len(spec["sources"]):
            raise RuntimeError(
                f"{output_path.name} round trip found {imported_mesh_count} meshes; "
                f"expected {len(spec['sources'])}."
            )

        pivot: Vector = spec["pivot"]
        # Keep the integration numbers as Python doubles. mathutils.Vector
        # stores float32 components, which is appropriate for Blender geometry
        # but needlessly rounds the authored GDevelop positions and dimensions.
        gdevelop_position = [
            rounded(float(GDEVELOP_ANCHOR.x) + uniform_scale * float(pivot.x)),
            rounded(float(GDEVELOP_ANCHOR.y) - uniform_scale * float(pivot.y)),
            rounded(
                float(GDEVELOP_ANCHOR.z)
                + uniform_scale * (float(pivot.z) - float(full_center.z))
            ),
        ]
        local_size = origin_maximum - origin_minimum
        gdevelop_size = [
            rounded(uniform_scale * float(local_size.x)),
            rounded(uniform_scale * float(local_size.y)),
            rounded(uniform_scale * float(local_size.z)),
        ]
        raw_minimum = local_minimum + pivot
        raw_maximum = local_maximum + pivot
        total_triangles += triangle_count

        record = {
            "index": asset_index - 1,
            "key": spec["key"],
            "kind": spec["kind"],
            "filename": spec["filename"],
            "resourcePath": f"{output_relative}/{spec['filename']}",
            "gdevelopObjectName": spec["gdevelopName"],
            "sourceMeshCount": len(spec["sources"]),
            "sourceTriangleCount": triangle_count,
            "sourceObjects": [obj.name for obj in spec["sources"]],
            "pivot": vector_values(pivot),
            "pivotEulerDegrees": spec["pivotEulerDegrees"],
            "rawBounds": bounds_record(raw_minimum, raw_maximum),
            "localBounds": bounds_record(local_minimum, local_maximum),
            "localBoundsIncludingOrigin": bounds_record(
                origin_minimum, origin_maximum
            ),
            "gdevelop": {
                "position": gdevelop_position,
                "size": gdevelop_size,
                "layoutRotation": [0.0, 0.0, 0.0],
                "rootRotationBakedIntoGlb": True,
            },
            "glb": summary,
            "roundTrip": {
                "verified": True,
                "meshCount": imported_mesh_count,
                "localBounds": bounds_record(imported_minimum, imported_maximum),
                "tolerance": BOUNDS_TOLERANCE,
            },
        }
        asset_records.append(record)
        print(
            f"EXPORTED {asset_index:02d}/{EXPECTED_ASSETS} "
            f"{output_path.name} meshes={len(spec['sources'])} "
            f"triangles={triangle_count} bytes={summary['bytes']}"
        )

    manifest = {
        "schemaVersion": 1,
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
            "assetCount": len(asset_records),
            "meshCount": sum(record["sourceMeshCount"] for record in asset_records),
            "triangleCount": total_triangles,
            "countsByKind": {
                kind: sum(1 for record in asset_records if record["kind"] == kind)
                for kind in ("terrain", "building", "tree", "sheep")
            },
            "meshesByKind": {
                kind: sum(
                    record["sourceMeshCount"]
                    for record in asset_records
                    if record["kind"] == kind
                )
                for kind in ("terrain", "building", "tree", "sheep")
            },
            "exhaustive": True,
            "pairwiseDisjoint": True,
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
        "assets": asset_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "EXPORT_SUMMARY "
        f"assets={len(asset_records)} meshes={manifest['partition']['meshCount']} "
        f"triangles={total_triangles} failures=0 manifest={manifest_path.as_posix()}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"EXPORT_FAILURE {type(error).__name__}: {error}", file=sys.stderr)
        raise
