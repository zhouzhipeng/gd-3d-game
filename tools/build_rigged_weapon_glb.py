#!/usr/bin/env python3
"""Build a small weapon GLB that can reuse a character GLB's animations.

The output keeps the weapon meshes/materials/binary data, copies only the
character's joint hierarchy, and parents the weapon scene roots to a named
attachment joint. GDevelop can then map animations from the character GLB as a
shared animation model resource.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"{path} is too small to be a GLB")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise ValueError(f"{path} is not a valid GLB 2.0 file")

    offset = 12
    document: dict[str, Any] | None = None
    binary = b""
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == JSON_CHUNK:
            document = json.loads(chunk.rstrip(b" \x00").decode("utf-8"))
        elif chunk_type == BIN_CHUNK:
            binary = chunk

    if document is None:
        raise ValueError(f"{path} has no JSON chunk")
    return document, binary


def write_glb(path: Path, document: dict[str, Any], binary: bytes) -> None:
    json_bytes = json.dumps(
        document, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)

    chunks = [struct.pack("<II", len(json_bytes), JSON_CHUNK), json_bytes]
    if binary:
        chunks.extend([struct.pack("<II", len(binary), BIN_CHUNK), binary])
    body = b"".join(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[row][k] * b[k][column] for k in range(4)) for column in range(4)]
        for row in range(4)
    ]


def node_matrix(node: dict[str, Any]) -> list[list[float]]:
    if "matrix" in node:
        values = node["matrix"]
        return [[float(values[column * 4 + row]) for column in range(4)] for row in range(4)]

    tx, ty, tz = (node.get("translation") or [0.0, 0.0, 0.0])
    sx, sy, sz = (node.get("scale") or [1.0, 1.0, 1.0])
    x, y, z, w = (node.get("rotation") or [0.0, 0.0, 0.0, 1.0])
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length:
        x, y, z, w = x / length, y / length, z / length, w / length

    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    scale = [
        [float(sx), 0.0, 0.0, 0.0],
        [0.0, float(sy), 0.0, 0.0],
        [0.0, 0.0, float(sz), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    transform = multiply(rotation, scale)
    transform[0][3], transform[1][3], transform[2][3] = float(tx), float(ty), float(tz)
    return transform


def transform_point(matrix: list[list[float]], point: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def scene_bounds(document: dict[str, Any]) -> tuple[list[float], list[float]]:
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    accessors = document.get("accessors", [])
    scene_index = document.get("scene", 0)
    scene = document.get("scenes", [])[scene_index]
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]

    def visit(node_index: int, parent_matrix: list[list[float]]) -> None:
        node = nodes[node_index]
        world = multiply(parent_matrix, node_matrix(node))
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            for primitive in meshes[mesh_index].get("primitives", []):
                position_index = primitive.get("attributes", {}).get("POSITION")
                if position_index is None:
                    continue
                accessor = accessors[position_index]
                if "min" not in accessor or "max" not in accessor:
                    continue
                low, high = accessor["min"], accessor["max"]
                for x in (low[0], high[0]):
                    for y in (low[1], high[1]):
                        for z in (low[2], high[2]):
                            point = transform_point(world, (x, y, z))
                            for axis in range(3):
                                minimum[axis] = min(minimum[axis], point[axis])
                                maximum[axis] = max(maximum[axis], point[axis])
        for child_index in node.get("children", []):
            visit(child_index, world)

    for root_index in scene.get("nodes", []):
        visit(root_index, identity_matrix())

    if any(math.isinf(value) for value in minimum + maximum):
        raise ValueError("The GLB scene has no mesh POSITION bounds")
    return minimum, maximum


def append_invisible_bounds(
    document: dict[str, Any],
    binary: bytes,
    minimum: list[float],
    maximum: list[float],
) -> bytes:
    """Add transparent points so different mount GLBs normalize identically."""
    existing = [
        index
        for index, node in enumerate(document.get("nodes", []))
        if node.get("name") == "__GDevelopWeaponMountBounds"
    ]
    if existing:
        raise ValueError("The GLB already contains a weapon-mount bounds helper")

    binary += b"\x00" * ((-len(binary)) % 4)
    byte_offset = len(binary)
    corners = [
        (x, y, z)
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    binary += b"".join(struct.pack("<fff", *corner) for corner in corners)

    buffer_views = document.setdefault("bufferViews", [])
    buffer_view_index = len(buffer_views)
    buffer_views.append(
        {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(corners) * 12,
            "target": 34962,
        }
    )

    accessors = document.setdefault("accessors", [])
    accessor_index = len(accessors)
    accessors.append(
        {
            "bufferView": buffer_view_index,
            "componentType": 5126,
            "count": len(corners),
            "type": "VEC3",
            "min": minimum,
            "max": maximum,
        }
    )

    materials = document.setdefault("materials", [])
    material_index = len(materials)
    materials.append(
        {
            "name": "__GDevelopInvisibleMountBounds",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.0, 0.0, 0.0, 0.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "alphaMode": "BLEND",
            "doubleSided": True,
        }
    )

    meshes = document.setdefault("meshes", [])
    mesh_index = len(meshes)
    meshes.append(
        {
            "name": "__GDevelopWeaponMountBounds",
            "primitives": [
                {
                    "attributes": {"POSITION": accessor_index},
                    "material": material_index,
                    "mode": 0,
                }
            ],
        }
    )

    nodes = document.setdefault("nodes", [])
    node_index = len(nodes)
    nodes.append(
        {
            "name": "__GDevelopWeaponMountBounds",
            "mesh": mesh_index,
            "extras": {"purpose": "Keep character and weapon model bounds identical"},
        }
    )
    scene_index = document.get("scene", 0)
    document["scenes"][scene_index].setdefault("nodes", []).append(node_index)
    document.setdefault("buffers", [{"byteLength": 0}])[0]["byteLength"] = len(binary)
    return binary


def copy_joint_hierarchy(
    character: dict[str, Any], output: dict[str, Any], joint_name: str
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    skins = character.get("skins", [])
    if not skins:
        raise ValueError("The character GLB has no skin/joint hierarchy")
    joint_indices = set(skins[0].get("joints", []))
    nodes = character.get("nodes", [])
    matching = [index for index in joint_indices if nodes[index].get("name") == joint_name]
    if len(matching) != 1:
        raise ValueError(f"Expected one joint named {joint_name!r}, found {len(matching)}")

    ordered = sorted(joint_indices)
    remap = {old: new for new, old in enumerate(ordered)}
    copied: list[dict[str, Any]] = []
    for old_index in ordered:
        source = nodes[old_index]
        target = {
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key in {"name", "matrix", "translation", "rotation", "scale", "weights", "extras"}
        }
        children = [remap[child] for child in source.get("children", []) if child in joint_indices]
        if children:
            target["children"] = children
        copied.append(target)

    parented_joints = {
        child for old in ordered for child in nodes[old].get("children", []) if child in joint_indices
    }
    roots = [remap[old] for old in ordered if old not in parented_joints]
    container_index = len(copied)
    copied.append({"name": "Rig_Medium_WeaponMount", "children": roots})

    # GLTFLoader only creates THREE.Bone objects for nodes referenced by a skin.
    # GDevelop's shared-animation compatibility check deliberately compares those
    # Bone objects, so a copied joint hierarchy without this lightweight skin
    # declaration looks like it has no rig at runtime. No inverse bind matrices
    # are needed because the weapon mesh itself is rigidly parented to a hand
    # joint rather than skinned.
    source_skin = skins[0]
    copied_skin: dict[str, Any] = {
        "name": f"{source_skin.get('name', 'CharacterRig')}_WeaponMount",
        "joints": [remap[index] for index in source_skin.get("joints", [])],
    }
    skeleton_index = source_skin.get("skeleton")
    if skeleton_index in remap:
        copied_skin["skeleton"] = remap[skeleton_index]

    return copied, remap[matching[0]], copied_skin


def build_weapon(
    character_path: Path,
    weapon_path: Path,
    output_path: Path,
    joint_name: str,
    target_depth: float,
    envelope_source_path: Path | None,
    character_output_path: Path | None,
) -> None:
    character, character_binary = read_glb(character_path)
    weapon, weapon_binary = read_glb(weapon_path)
    output = copy.deepcopy(weapon)

    skeleton_nodes, target_joint_index, copied_skin = copy_joint_hierarchy(
        character, output, joint_name
    )
    skeleton_count = len(skeleton_nodes)
    weapon_nodes = copy.deepcopy(weapon.get("nodes", []))
    for node in weapon_nodes:
        if "children" in node:
            node["children"] = [skeleton_count + child for child in node["children"]]

    weapon_scene_index = weapon.get("scene", 0)
    weapon_roots = weapon.get("scenes", [])[weapon_scene_index].get("nodes", [])
    mounted_roots = [skeleton_count + root for root in weapon_roots]
    target_joint = skeleton_nodes[target_joint_index]
    target_joint["children"] = target_joint.get("children", []) + mounted_roots

    output["nodes"] = skeleton_nodes + weapon_nodes
    output["scenes"] = [{"name": "RiggedWeapon", "nodes": [skeleton_count - 1]}]
    output["scene"] = 0
    output.pop("animations", None)
    output["skins"] = [copied_skin]
    output.setdefault("asset", {})["generator"] = (
        f"{output.get('asset', {}).get('generator', 'glTF')} + GDevelop shared-rig weapon mount"
    )

    envelope_document = character
    if envelope_source_path is not None:
        envelope_document, _ = read_glb(envelope_source_path)
    envelope_min, envelope_max = scene_bounds(envelope_document)
    weapon_binary = append_invisible_bounds(output, weapon_binary, envelope_min, envelope_max)

    if character_output_path is not None:
        character_output = copy.deepcopy(character)
        character_output.setdefault("asset", {})["generator"] = (
            f"{character_output.get('asset', {}).get('generator', 'glTF')}"
            " + GDevelop weapon-mount sizing envelope"
        )
        character_binary = append_invisible_bounds(
            character_output, character_binary, envelope_min, envelope_max
        )
        write_glb(character_output_path, character_output, character_binary)

    write_glb(output_path, output, weapon_binary)
    round_trip, _ = read_glb(output_path)
    if character_output_path is not None:
        character_round_trip, _ = read_glb(character_output_path)
    else:
        character_round_trip = character
    character_min, character_max = scene_bounds(character_round_trip)
    weapon_min, weapon_max = scene_bounds(round_trip)
    character_size = [character_max[i] - character_min[i] for i in range(3)]
    weapon_size = [weapon_max[i] - weapon_min[i] for i in range(3)]
    uniform_scale = target_depth / character_size[1]

    def dimensions(size: list[float]) -> tuple[float, float, float]:
        return (
            uniform_scale * size[0],
            uniform_scale * size[2],
            uniform_scale * size[1],
        )

    knight_dimensions = dimensions(character_size)
    weapon_dimensions = dimensions(weapon_size)
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")
    if character_output_path is not None:
        print(
            f"Wrote {character_output_path} "
            f"({character_output_path.stat().st_size} bytes)"
        )
    print(f"Attachment joint: {joint_name}")
    print(f"Character bounds: min={character_min}, max={character_max}")
    print(f"Weapon bounds: min={weapon_min}, max={weapon_max}")
    print(
        "Suggested GDevelop knight dimensions (width x height x depth): "
        f"{knight_dimensions[0]:.6f} x {knight_dimensions[1]:.6f} x {knight_dimensions[2]:.6f}"
    )
    print(
        "Suggested GDevelop weapon dimensions (width x height x depth): "
        f"{weapon_dimensions[0]:.6f} x {weapon_dimensions[1]:.6f} x {weapon_dimensions[2]:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True, type=Path)
    parser.add_argument("--weapon", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--joint", default="handslot.r")
    parser.add_argument("--target-depth", type=float, default=120.0)
    parser.add_argument(
        "--envelope-source",
        type=Path,
        help="GLB whose complete bind-pose bounds should normalize every mount object",
    )
    parser.add_argument(
        "--character-output",
        type=Path,
        help="Optional character copy with the same invisible sizing envelope",
    )
    args = parser.parse_args()
    build_weapon(
        args.character,
        args.weapon,
        args.output,
        args.joint,
        args.target_depth,
        args.envelope_source,
        args.character_output,
    )


if __name__ == "__main__":
    main()
