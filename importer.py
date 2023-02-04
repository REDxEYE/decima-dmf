import json
import logging
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import cast, Optional, Dict, List

import bpy
import numpy as np
import numpy.typing as npt
from mathutils import Vector, Quaternion, Matrix

from .dmf import (DMFMaterial, DMFMesh, DMFModel, DMFNode,
                  DMFNodeType, DMFModelGroup, DMFLodModel,
                  DMFPrimitive, DMFSceneFile, DMFSkeleton,
                  DMFSemantic, DMFComponentType, DMFInstance)
from .material_utils import (clear_nodes, Nodes, create_node,
                             connect_nodes, create_texture_node,
                             create_material)


def get_logger(name) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s]--[%(name)s]: %(message)s", level=logging.INFO)
    logger = logging.getLogger(name)
    return logger


LOGGER = get_logger("DMF::Loader")

CONTEXT = {'collections': [], "instances": {}}


def _convert_quat(quat):
    return quat[3], quat[0], quat[1], quat[2]


def _get_or_create_collection(name, parent: bpy.types.Collection) -> bpy.types.Collection:
    new_collection = (bpy.data.collections.get(name, None) or
                      bpy.data.collections.new(name))
    if new_collection.name not in parent.children:
        parent.children.link(new_collection)
    new_collection.name = name
    return new_collection


def _add_uv(mesh_data: bpy.types.Mesh, uv_name: str, uv_data: npt.NDArray[float]):
    uv_layer = mesh_data.uv_layers.new(name=uv_name)
    uv_layer_data = uv_data.copy()
    uv_layer.data.foreach_set('uv', uv_layer_data.flatten())


def _convert_type_and_size(semantic: DMFSemantic, input_dtype_array: npt.NDArray, output_dtype: npt.DTypeLike,
                           element_start: Optional[int] = None, element_end: Optional[int] = None):
    input_array = input_dtype_array[semantic.name]

    if element_start is None:
        element_start = 0
    if element_end is None:
        element_end = input_array.shape[-1]

    def _convert(source_array):
        input_dtype = source_array.dtype
        meta_type = input_dtype_array.dtype.metadata[semantic.name]
        if meta_type == DMFComponentType.X10Y10Z10W2NORMALIZED.name:
            x = (source_array >> 0 & 1023 ^ 512) - 512
            y = (source_array >> 10 & 1023 ^ 512) - 512
            z = (source_array >> 20 & 1023 ^ 512) - 512
            w = (source_array >> 30 & 1)

            vector_length = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            x = x.astype(np.float32) / vector_length
            y = y.astype(np.float32) / vector_length
            z = z.astype(np.float32) / vector_length
            return np.dstack([x, y, z, w])[0].astype(output_dtype)

        if input_dtype == output_dtype:
            return source_array.copy()

        if output_dtype == np.float32:
            float_array = source_array.copy().astype(np.float32)
            if input_dtype == np.int16:
                return float_array / 0x7FFF
            elif input_dtype == np.uint16:
                return float_array / 0xFFFF
            elif input_dtype == np.uint8:
                return float_array / 0xFF
            elif input_dtype == np.int8:
                return float_array / 0x7F
            elif input_dtype == np.float16:
                return float_array
        raise NotImplementedError(f"Cannot convert {input_dtype} to {output_dtype}")

    return _convert(input_array)[:, element_start:element_end]


def import_dmf_skeleton(skeleton: DMFSkeleton, name: str):
    arm_data = bpy.data.armatures.new(name + "_ARMDATA")
    arm_obj = bpy.data.objects.new(name + "_ARM", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)

    arm_obj.show_in_front = True
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bones = []
    for bone in skeleton.bones:
        bl_bone = arm_data.edit_bones.new(bone.name)
        bl_bone.tail = Vector([0, 0, 0.1]) + bl_bone.head
        bones.append(bl_bone)

        if bone.parent_id != -1:
            bl_bone.parent = bones[bone.parent_id]

        bone_pos = bone.transform.position
        bone_rot = bone.transform.rotation

        bone_pos = Vector(bone_pos)
        # noinspection PyTypeChecker
        bone_rot = Quaternion(_convert_quat(bone_rot))
        mat = Matrix.Translation(bone_pos) @ bone_rot.to_matrix().to_4x4()
        if bone.local_space and bl_bone.parent:
            bl_bone.matrix = bl_bone.parent.matrix @ mat
        else:
            bl_bone.matrix = mat

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.collection.objects.unlink(arm_obj)
    return arm_obj


def _load_texture(scene: DMFSceneFile, texture_id: int):
    texture = scene.textures[texture_id]
    if bpy.data.images.get(texture.name, None) is not None:
        return
    buffer = scene.buffers[texture.buffer_id]
    image = bpy.data.images.new(texture.name, width=1, height=1)
    image.pack(data=buffer.get_data(scene.buffers_path), data_len=buffer.size)
    image.source = 'FILE'
    image.use_fake_user = True
    image.alpha_mode = 'CHANNEL_PACKED'


def _get_texture(scene, texture_id):
    texture = scene.textures[texture_id]
    image = bpy.data.images.get(texture.name, None)
    if image is None:
        print(f"Texture {texture.name} not found")
    return image


def build_material(material: DMFMaterial, bl_material, scene: DMFSceneFile):
    LOGGER.debug(f"Creating material \"{material.name}\"")

    if bl_material.get("LOADED", False):
        return
    bl_material["LOADED"] = True

    if not (material.texture_ids or material.texture_descriptors):
        return

    for texture_id in material.texture_ids.values():
        if texture_id == -1:
            continue
        _load_texture(scene, texture_id)

    sorted_descriptors = sorted(material.texture_descriptors, key=lambda a: a.texture_id)

    for texture_id, _ in groupby(sorted_descriptors, key=lambda a: a.texture_id):
        if texture_id == -1:
            continue
        _load_texture(scene, texture_id)

    bl_material.use_nodes = True
    clear_nodes(bl_material)
    output_node = create_node(bl_material, Nodes.ShaderNodeOutputMaterial)
    bsdf_node = create_node(bl_material, Nodes.ShaderNodeBsdfPrincipled)
    connect_nodes(bl_material, output_node.inputs[0], bsdf_node.outputs[0])

    for semantic in material.texture_ids.keys():
        texture_id = material.texture_ids.get(semantic, None)
        if texture_id is None:
            continue
        create_texture_node(bl_material, _get_texture(scene, texture_id), semantic)

    for texture_id, texture_descriptors in groupby(sorted_descriptors, key=lambda a: a.texture_id):
        if texture_id == -1:
            continue
        texture_description = []
        for texture_descriptor in texture_descriptors:
            texture_description.append(f"{texture_descriptor.usage_type}_{texture_descriptor.channels}")
        create_texture_node(bl_material, _get_texture(scene, texture_id),
                            " | ".join(texture_description))


def import_dmf_model(model: DMFModel, scene: DMFSceneFile, parent_collection: bpy.types.Collection):
    LOGGER.debug(f"Loading \"{model.name}\" model")
    if model.skeleton_id is not None:
        skeleton = import_dmf_skeleton(scene.skeletons[model.skeleton_id], model.name)
    else:
        skeleton = None

    primitives = _load_primitives(model, scene, skeleton)

    if skeleton is not None:
        parent = skeleton
        for primitive in primitives:
            modifier = primitive.modifiers.new(type="ARMATURE", name="Armature")
            modifier.object = skeleton
    else:
        parent = bpy.data.objects.new(model.name + "_ROOT", None)

    for primitive in primitives:
        primitive.parent = parent

    for child in model.children:
        grouper = bpy.data.objects.new(model.name + "_CHILDREN", None)
        grouper.parent = parent
        if model.transform:
            grouper.location = model.transform.position
            grouper.rotation_mode = "QUATERNION"
            grouper.rotation_quaternion = _convert_quat(model.transform.rotation)
            grouper.scale = model.transform.scale
        parent_collection.objects.link(grouper)
        # for collection_id in model.collection_ids:
        #     CONTEXT["collections"][collection_id].objects.link(grouper)

        children = import_dmf_node(child, scene, parent_collection)
        if children is not None:
            children.parent = grouper

    if model.transform is not None:
        parent.location = model.transform.position
        parent.rotation_mode = "QUATERNION"
        parent.rotation_quaternion = _convert_quat(model.transform.rotation)
        parent.scale = model.transform.scale
    if skeleton is not None:
        parent_collection.objects.link(skeleton)
        # for collection_id in model.collection_ids:
        #     CONTEXT["collections"][collection_id].objects.link(skeleton)
    for primitive in primitives:
        parent_collection.objects.link(primitive)
        # for collection_id in model.collection_ids:
        #     CONTEXT["collections"][collection_id].objects.link(primitive)

    return parent


def _load_primitives(model: DMFModel, scene: DMFSceneFile, skeleton: bpy.types.Object):
    primitives = []
    primitive_groups: Dict[int, List[DMFPrimitive]] = defaultdict(list)
    for primitive in model.mesh.primitives:
        primitive_groups[primitive.grouping_id].append(primitive)

    for primitive_group in primitive_groups.values():
        mesh_data = bpy.data.meshes.new(model.name + f"_MESH")
        mesh_obj = bpy.data.objects.new(model.name, mesh_data)
        material_ids = np.zeros(primitive_group[0].index_count // 3, np.int32)
        material_id = 0
        vertex_data = primitive_group[0].get_vertices(scene)
        total_indices: List[npt.NDArray[np.uint32]] = []
        primitive_0 = primitive_group[0]
        for primitive in primitive_group:
            material = scene.materials[primitive.material_id]

            build_material(material, create_material(material.name, mesh_obj), scene)
            material_ids[primitive.index_start // 3:primitive.index_end // 3] = material_id
            material_id += 1

            indices = primitive.get_indices(scene)
            total_indices.append(indices)

        all_indices = np.vstack(total_indices)

        position_data = _convert_type_and_size(DMFSemantic.POSITION, vertex_data, np.float32, 0, 3)
        mesh_data.from_pydata(position_data, [], all_indices)
        mesh_data.update(calc_edges=True, calc_edges_loose=True)

        vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
        mesh_data.loops.foreach_get('vertex_index', vertex_indices)
        t_vertex_data = vertex_data[vertex_indices]
        for uv_layer_id in range(7):
            semantic = DMFSemantic(f"TEXCOORD_{uv_layer_id}")
            uv_layer_name = f"UV{uv_layer_id}"
            if primitive_0.has_attribute(semantic):
                _add_uv(mesh_data, uv_layer_name, _convert_type_and_size(semantic, t_vertex_data, np.float32))

        if primitive_0.has_attribute(DMFSemantic.COLOR_0):
            vertex_colors = mesh_data.vertex_colors.new(name="COLOR")
            vertex_colors_data = vertex_colors.data
            color_data = _convert_type_and_size(DMFSemantic.COLOR_0, t_vertex_data, np.float32)
            vertex_colors_data.foreach_set('color', color_data.flatten())

        mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
        mesh_data.use_auto_smooth = True
        if primitive_0.has_attribute(DMFSemantic.NORMAL):
            normal_data = _convert_type_and_size(DMFSemantic.NORMAL, vertex_data, np.float32, element_end=3)
            mesh_data.normals_split_custom_set_from_vertices(normal_data)

        mesh_data.polygons.foreach_set('material_index', material_ids[:all_indices.shape[0]])

        if skeleton is not None:
            _add_skinning(scene.skeletons[model.skeleton_id], mesh_obj, model.mesh, primitive_0, vertex_data)

        primitives.append(mesh_obj)
    return primitives


def _add_skinning(skeleton: DMFSkeleton, mesh_obj: bpy.types.Object, mesh: DMFMesh,
                  primitive_0: DMFPrimitive, vertex_data: npt.NDArray):
    weight_groups = [mesh_obj.vertex_groups.new(name=bone.name) for bone in skeleton.bones]
    elem_count = 0
    for j in range(3):
        if DMFSemantic(f"JOINTS_{j}") not in primitive_0.vertex_attributes:
            break
        elem_count += 1
    blend_weights = np.zeros((primitive_0.vertex_count, elem_count * 4), np.float32)
    blend_indices = np.zeros((primitive_0.vertex_count, elem_count * 4), np.int32)
    for j in range(3):
        if DMFSemantic(f"JOINTS_{j}") not in primitive_0.vertex_attributes:
            continue
        blend_indices[:, 4 * j:4 * (j + 1)] = vertex_data[f"JOINTS_{j}"].copy()
        weight_semantic = DMFSemantic(f"WEIGHTS_{j}")
        if primitive_0.has_attribute(weight_semantic):
            weight_data = _convert_type_and_size(weight_semantic, vertex_data, np.float32)
            blend_weights[:, 4 * j:4 * (j + 1)] = weight_data

    np_remap_table = np.full(max(mesh.bone_remap_table.keys()) + 1, -1, np.int32)
    np_remap_table[list(mesh.bone_remap_table.keys())] = list(mesh.bone_remap_table.values())
    remapped_indices = np_remap_table[blend_indices]

    for n, (bone_indices, bone_weights) in enumerate(zip(remapped_indices, blend_weights)):
        total = bone_weights.sum()
        weight_groups[bone_indices[0]].add([n], 1 - total, "ADD")
        if total > 0.0:
            for i, bone_index in enumerate(bone_indices[1:]):
                weight_groups[bone_index].add([n], bone_weights[i], "ADD")


def import_dmf_model_group(model_group: DMFModelGroup, scene: DMFSceneFile, parent_collection: bpy.types.Collection):
    LOGGER.debug(f"Loading \"{model_group.name}\" model group")
    # group_collection = bpy.data.collections.new(model_group.name)
    # parent_collection.children.link(group_collection)

    group_obj = bpy.data.objects.new(model_group.name, None)
    if model_group.transform:
        group_obj.location = model_group.transform.position
        group_obj.rotation_mode = "QUATERNION"
        group_obj.rotation_quaternion = _convert_quat(model_group.transform.rotation)
        group_obj.scale = model_group.transform.scale

    for child in model_group.children:
        obj = import_dmf_node(child, scene, parent_collection)
        if obj:
            obj.parent = group_obj
    parent_collection.objects.link(group_obj)
    # for collection_id in model_group.collection_ids:
    #     collection = CONTEXT["collections"][collection_id]
    #     collection.objects.link(group_obj)
    return group_obj


def import_dmf_lod(lod_model: DMFLodModel, scene: DMFSceneFile, parent_collection: bpy.types.Collection):
    group_obj = bpy.data.objects.new(lod_model.name, None)
    parent_collection.objects.link(group_obj)
    if lod_model.transform:
        group_obj.location = lod_model.transform.position
        group_obj.rotation_mode = "QUATERNION"
        group_obj.rotation_quaternion = _convert_quat(lod_model.transform.rotation)
        group_obj.scale = lod_model.transform.scale
    if lod_model.lods:
        obj = import_dmf_node(lod_model.lods[0].model, scene, parent_collection)
        if obj:
            obj.parent = group_obj

        for child in lod_model.children:
            c_obj = import_dmf_node(child, scene, parent_collection)
            if c_obj:
                c_obj.parent = group_obj
    return group_obj


def import_dmf_instance(instance_model: DMFInstance, scene: DMFSceneFile, parent_collection: bpy.types.Collection):
    obj = bpy.data.objects.new(instance_model.name, None)
    obj.empty_display_size = 4

    if instance_model.instance_id != -1:
        instance_name = CONTEXT["instances"][instance_model.instance_id]
        obj.instance_type = 'COLLECTION'
        obj.instance_collection = bpy.data.collections[instance_name]

    if instance_model.transform:
        obj.location = instance_model.transform.position
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = _convert_quat(instance_model.transform.rotation)
        obj.scale = instance_model.transform.scale

    parent_collection.objects.link(obj)
    return obj


def import_dmf_node(node: DMFNode, scene: DMFSceneFile, parent_collection: bpy.types.Collection):
    if node is None:
        return None
    if node.type == DMFNodeType.Model:
        return import_dmf_model(cast(DMFModel, node), scene, parent_collection)
    elif node.type == DMFNodeType.LOD:
        return import_dmf_lod(cast(DMFLodModel, node), scene, parent_collection)
    elif node.type == DMFNodeType.Instance:
        return import_dmf_instance(cast(DMFInstance, node), scene, parent_collection)
    elif node.type == DMFNodeType.ModelGroup:
        model_group = cast(DMFModelGroup, node)
        if not model_group.children:
            return None
        return import_dmf_model_group(model_group, scene, parent_collection)


def _collect_view_collections(parent):
    result = [parent]
    for child in parent.children:
        result += _collect_view_collections(child)
    return result


def import_dmf(scene: DMFSceneFile):
    CONTEXT["instances"].clear()
    CONTEXT["collections"].clear()

    if scene.meta_data.version != 1:
        raise ValueError(f"Version {scene.meta_data.version} is not supported!")

    # collections = CONTEXT["collections"] = []

    # for collection_desc in scene.collections:
    #     collection = bpy.data.collections.new(collection_desc.name)
    #     if collection_desc.parent is not None:
    #         parent = collections[collection_desc.parent]
    #         parent.children.link(collection)
    #     else:
    #         bpy.context.scene.collection.children.link(collection)
    #     collections.append(collection)
    #     for layer_collection in _collect_view_collections(bpy.context.scene.view_layers[0].layer_collection):
    #         if layer_collection.collection.name == collection.name:
    #             layer_collection.exclude = not collection_desc.enabled

    instances_collection = bpy.data.collections.new("INSTANCES")
    bpy.context.scene.collection.children.link(instances_collection)

    for layer_collection in _collect_view_collections(bpy.context.scene.view_layers[0].layer_collection):
        if layer_collection.collection.name == instances_collection.name:
            layer_collection.exclude = True

    LOGGER.info(f"Loading {len(scene.instances)} instances")
    for i, node in enumerate(scene.instances):
        if i % 100 == 0:
            LOGGER.info(f"Load progress {i + 1}/{len(scene.instances)}")
        instance_collection = bpy.data.collections.new(node.name)
        instances_collection.children.link(instance_collection)
        import_dmf_node(node, scene, instance_collection)
        CONTEXT["instances"][i] = instance_collection.name

    master_collection = bpy.data.collections.new("MASTER")
    bpy.context.scene.collection.children.link(master_collection)

    LOGGER.info(f"Loading {len(scene.models)} Objects")
    for i, node in enumerate(scene.models):
        if i % 100 == 0:
            LOGGER.info(f"Load progress {i + 1}/{len(scene.models)}")
        import_dmf_node(node, scene, master_collection)


def import_dmf_from_path(file: Path):
    with file.open('r') as f:
        scene = DMFSceneFile.from_json(json.load(f))
        scene.set_buffers_path(file.parent / 'dbuffers')
        import_dmf(scene)
