import zlib
from base64 import b64decode
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Protocol, Tuple, Optional, List

import numpy as np
import numpy.typing as npt


class DMFSemantic(Enum):
    POSITION = "POSITION"
    TANGENT = "TANGENT"
    NORMAL = "NORMAL"
    COLOR_0 = "COLOR_0"
    COLOR_1 = "COLOR_1"
    COLOR_2 = "COLOR_2"
    COLOR_3 = "COLOR_3"
    TEXCOORD_0 = "TEXCOORD_0"
    TEXCOORD_1 = "TEXCOORD_1"
    TEXCOORD_2 = "TEXCOORD_2"
    TEXCOORD_3 = "TEXCOORD_3"
    TEXCOORD_4 = "TEXCOORD_4"
    TEXCOORD_5 = "TEXCOORD_5"
    TEXCOORD_6 = "TEXCOORD_6"
    JOINTS_0 = "JOINTS_0"
    JOINTS_1 = "JOINTS_1"
    JOINTS_2 = "JOINTS_2"
    JOINTS_3 = "JOINTS_3"
    WEIGHTS_0 = "WEIGHTS_0"
    WEIGHTS_1 = "WEIGHTS_1"
    WEIGHTS_2 = "WEIGHTS_2"
    WEIGHTS_3 = "WEIGHTS_3"


class DMFComponentType(Enum):
    SIGNED_SHORT = "SIGNED_SHORT", np.int16
    UNSIGNED_SHORT = "UNSIGNED_SHORT", np.uint16
    SIGNED_SHORT_NORMALIZED = "SIGNED_SHORT_NORMALIZED", np.int16
    UNSIGNED_SHORT_NORMALIZED = "UNSIGNED_SHORT_NORMALIZED", np.uint16
    UNSIGNED_BYTE = "UNSIGNED_BYTE", np.uint8
    UNSIGNED_BYTE_NORMALIZED = "UNSIGNED_BYTE_NORMALIZED", np.uint8
    FLOAT = "FLOAT", np.float32
    HALF_FLOAT = "HALF_FLOAT", np.float16
    X10Y10Z10W2NORMALIZED = "X10Y10Z10W2NORMALIZED", np.int32

    def __new__(cls, a, b):
        entry = object.__new__(cls)
        entry._value_ = a  # set the value, and the extra attribute
        entry.dtype = b
        return entry


class DMFNodeType(Enum):
    NODE = "NODE"
    LOD = "LOD"
    MODEL = "MODEL"
    MODEL_GROUP = "MODEL_GROUP"
    SKINNED_MODEL = "SKINNED_MODEL"
    INSTANCE = "INSTANCE"
    ATTACHMENT = "ATTACHMENT"


class JsonSerializable(Protocol):

    def to_json(self):
        raise NotImplementedError()

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        raise NotImplementedError()


@dataclass(slots=True)
class DMFTextureDescriptor(JsonSerializable):
    texture_id: int
    channels: str
    usage_type: str

    def to_json(self):
        return {"textureId": self.texture_id, "channels": self.channels, "usageType": self.usage_type}

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(data["textureId"], data["channels"], data["usageType"])


@dataclass(slots=True)
class DMFSceneMetaData(JsonSerializable):
    generator: str
    version: int

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(**data)


@dataclass(slots=True)
class DMFTexture(JsonSerializable):
    name: str
    buffer_id: int
    usage_type: int
    metadata: Dict[str, str]

    def to_json(self):
        return {
            "name": self.name,
            "bufferId": self.buffer_id,
            "usageType": self.usage_type,
            "metadata": self.metadata
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(data["name"], data["bufferId"], data.get("usageType", 0), data.get("metadata", {}))


@dataclass(slots=True)
class DMFCollection(JsonSerializable):
    name: str
    enabled: bool = field(default=True)
    parent: Optional[int] = field(default=None)

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(data["name"], data.get("enabled", True), data.get("parent", None))


@dataclass(slots=True)
class DMFTransform(JsonSerializable):
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(**data)

    @classmethod
    def identity(cls):
        return cls((0, 0, 0), (0, 0, 0), (0, 0, 0, 0))


@dataclass(slots=True)
class DMFBone(JsonSerializable):
    name: str
    transform: DMFTransform
    parent_id: int
    local_space: bool

    def to_json(self):
        return {
            "name": self.name,
            "transform": self.transform.to_json(),
            "parentId": self.parent_id,
            "localSpace": self.local_space
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(
            data["name"],
            DMFTransform.from_json(data["transform"]),
            data["parentId"],
            data.get("localSpace", False)
        )


@dataclass(slots=True)
class DMFBuffer(JsonSerializable):
    name: str
    size: int
    path: Optional[str]
    data: Optional[str]

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(data.get("name", "<NONE>"), data["size"], data.get("path", None), data.get("data", None))

    def to_json(self):
        return {
            "name": self.name,
            "size": self.size,
            "data": self.data,
            "path": self.path
        }

    def get_data(self, data_path: Path):
        if self.path is not None:
            with (data_path / self.path).open("rb") as f:
                data = f.read()
        elif self.data is not None:
            data = zlib.decompress(b64decode(self.data))
        else:
            raise ValueError("Data/Path are missing")
        assert len(data) == self.size, f"Buffer {self.name} has wrong size. Expected {self.size}, got {len(data)}"
        return data


@dataclass(slots=True)
class DMFBufferView(JsonSerializable):
    buffer_id: int
    offset: int
    size: int

    def to_json(self):
        return {"bufferId": self.buffer_id, "offset": self.offset, "size": self.size}

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(data["bufferId"], data["offset"], data["size"])


@dataclass(slots=True)
class DMFSkeleton(JsonSerializable):
    bones: List[DMFBone]
    transform: Optional[DMFTransform]

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        transform = DMFTransform.from_json(data["transform"]) if "transform" in data else None
        return cls([DMFBone.from_json(item) for item in data.get("bones", [])], transform)


@dataclass(slots=True)
class DMFMaterial(JsonSerializable):
    name: str
    type: str
    texture_ids: Dict[str, int] = field(default_factory=dict)
    texture_descriptors: List[DMFTextureDescriptor] = field(default_factory=list)

    def to_json(self):
        return {
            "name": self.name,
            "type": self.type,
            "textureIds": self.texture_ids,
            "textureDescriptors": [desc.to_json() for desc in self.texture_descriptors]
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(data["name"], data.get("type", "UNKNOWN"), data.get("textureIds", []),
                   [DMFTextureDescriptor.from_json(desc) for desc in data.get("textureDescriptors", [])])


@dataclass(slots=True)
class DMFNode(JsonSerializable):
    type: DMFNodeType
    name: Optional[str]
    collection_ids: List[int]
    transform: Optional[DMFTransform]
    children: List['DMFNode']
    visible: bool

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        if data is None:
            return None
        node_type = DMFNodeType(data["type"])
        name = data.get("name", node_type.name)
        collection_ids = list(set(data.get('collectionIds', [])))
        transform = DMFTransform.from_json(data["transform"]) if "transform" in data else DMFTransform.identity()
        children = [cls.from_json(item) for item in data.get("children", [])]
        visible = data.get("visible", True)

        if node_type == DMFNodeType.MODEL:
            return DMFModel(node_type, name, collection_ids, transform, children, visible,
                            DMFMesh.from_json(data["mesh"]), data.get("skeletonId", None))
        if node_type == DMFNodeType.SKINNED_MODEL:
            return DMFSkinnedModel(node_type, name, collection_ids, transform, children, visible,
                                   data.get("skeletonId", None))
        elif node_type == DMFNodeType.MODEL_GROUP:
            return DMFModelGroup(node_type, name, collection_ids, transform, children, visible)
        elif node_type == DMFNodeType.LOD:
            return DMFLodModel(node_type, name, collection_ids, transform, children, visible,
                               [DMFLod.from_json(lod_data) for lod_data in data.get("lods", [])])
        elif node_type == DMFNodeType.INSTANCE:
            return DMFInstance(node_type, name, collection_ids, transform, children, visible, data["instanceId"])
        elif node_type == DMFNodeType.ATTACHMENT:
            return DMFAttachment(node_type, name, collection_ids, transform, children, visible, data["boneName"])
        else:
            return DMFNode(node_type, name, collection_ids, transform, children, visible)


@dataclass(slots=True)
class DMFSceneFile(JsonSerializable):
    meta_data: DMFSceneMetaData
    collections: List[DMFCollection]
    models: List[DMFNode]
    skeletons: List[DMFSkeleton]
    buffers: List[DMFBuffer]
    buffer_views: List[DMFBufferView]
    materials: List[DMFMaterial]
    textures: List[DMFTexture]
    instances: List[DMFNode]

    _buffers_path: Optional[Path] = field(default=None)

    def to_json(self):
        return {
            "metadata": self.meta_data.to_json(),
            "collections": [item.to_json() for item in self.collections],
            "models": [item.to_json() for item in self.models],
            "buffers": [item.to_json() for item in self.buffers],
            "bufferViews": [item.to_json() for item in self.buffer_views],
            "materials": [item.to_json() for item in self.materials],
            "textures": [item.to_json() for item in self.textures],
            "instances": [item.to_json() for item in self.instances],
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(
            DMFSceneMetaData.from_json(data["metadata"]),
            [DMFCollection.from_json(item) for item in data.get("collections", [])],
            [DMFNode.from_json(item) for item in data.get("models", [])],
            [DMFSkeleton.from_json(item) for item in data.get("skeletons", [])],
            [DMFBuffer.from_json(item) for item in data.get("buffers", [])],
            [DMFBufferView.from_json(item) for item in data.get("bufferViews", [])],
            [DMFMaterial.from_json(item) for item in data.get("materials", [])],
            [DMFTexture.from_json(item) for item in data.get("textures", [])],
            [DMFNode.from_json(item) for item in data.get("instances", [])],
        )

    def set_buffers_path(self, buffers_path: Path):
        self._buffers_path = buffers_path

    @property
    def buffers_path(self):
        assert self._buffers_path
        return self._buffers_path


@dataclass(slots=True)
class DMFVertexAttribute(JsonSerializable):
    semantic: DMFSemantic
    element_count: int
    element_type: DMFComponentType
    size: int
    stride: Optional[int]
    offset: Optional[int]
    buffer_view_id: int

    def to_json(self):
        return {
            "semantic": self.semantic,
            "elementCount": self.element_count,
            "elementyType": self.element_type,
            "size": self.size,
            "stride": self.stride,
            "offset": self.offset,
            "bufferViewId": self.buffer_view_id
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(
            DMFSemantic(data["semantic"]),
            data["elementCount"],
            DMFComponentType(data["elementType"]),
            data["size"],
            data.get("stride", None),
            data.get("offset", 0),
            data["bufferViewId"]
        )

    @classmethod
    def supported(cls, data: Dict[str, Any]):
        semantic = data["semantic"]
        supported_semantic = semantic in [item.name for item in list(DMFSemantic)]
        element_type = data["elementType"]
        supported_element_type = element_type in [item.name for item in list(DMFComponentType)]
        return supported_semantic and supported_element_type

    def convert(self, scene) -> npt.NDArray:
        buffer = scene.buffers_views[self.buffer_view_id].get_data(scene)[self.offset:]
        data = np.frombuffer(buffer, self.element_type.dtype).reshape((-1, self.element_count))
        return data


class DMFBufferType(Enum):
    MULTI_BUFFER = "MULTI_BUFFER"
    SINGLE_BUFFER = "SINGLE_BUFFER"


@dataclass(slots=True)
class DMFPrimitive(JsonSerializable):
    grouping_id: int
    vertex_count: int
    vertex_start: int
    vertex_end: int
    vertex_attributes: Dict[DMFSemantic, DMFVertexAttribute]
    vertex_type: DMFBufferType
    index_count: int
    index_start: int
    index_end: int
    index_size: int
    index_buffer_view_id: int

    material_id: Optional[int]

    _dtype: npt.DTypeLike = None

    def to_json(self):
        return {
            "groupingId": self.grouping_id,
            "vertexCount": self.vertex_count,
            "vertexStart": self.vertex_start,
            "vertexEnd": self.vertex_end,
            "vertexAttributes": {semantic.name: item.to_json() for semantic, item in self.vertex_attributes.items()},
            "vertexType": self.vertex_type,
            "indexCount": self.index_count,
            "indexStart": self.index_start,
            "indexEnd": self.index_end,
            "indexSize": self.index_size,
            "indexBufferViewId": self.index_buffer_view_id,
            "materialId": self.material_id
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(
            data["groupingId"],
            data["vertexCount"],
            data["vertexStart"],
            data["vertexEnd"],
            {DMFSemantic(name): DMFVertexAttribute.from_json(item) for name, item in
             data.get("vertexAttributes", {}).items() if DMFVertexAttribute.supported(item)},
            DMFBufferType(data["vertexType"]),
            data["indexCount"],
            data["indexStart"],
            data["indexEnd"],
            data["indexSize"],
            data["indexBufferViewId"],
            data.get("materialId", None),
        )

    def has_attribute(self, semantic: DMFSemantic):
        return semantic in self.vertex_attributes


@dataclass(slots=True)
class DMFMesh(JsonSerializable):
    primitives: List[DMFPrimitive]
    bone_remap_table: Dict[int, int]

    def to_json(self):
        return {
            "primitives": [item.to_json() for item in self.primitives],
            "boneRemapTable": self.bone_remap_table
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        remap_table = {int(k): v for k, v in data.get("boneRemapTable", {}).items()}

        return cls([DMFPrimitive.from_json(item) for item in data.get("primitives", [])], remap_table)


@dataclass(slots=True)
class DMFModelGroup(DMFNode):
    pass


@dataclass(slots=True)
class DMFModel(DMFNode):
    mesh: DMFMesh
    skeleton_id: int


@dataclass(slots=True)
class DMFSkinnedModel(DMFNode):
    skeleton_id: int


@dataclass(slots=True)
class DMFAttachment(DMFNode):
    bone_name: str


@dataclass(slots=True)
class DMFLod(JsonSerializable):
    model: DMFNode
    lod_id: int
    distance: float

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(DMFNode.from_json(data.get("model", None)), data["id"], data["distance"])

    def to_json(self):
        return {"model": self.model.to_json() if self.model else None, "id": self.lod_id, "distance": self.distance}


@dataclass(slots=True)
class DMFLodModel(DMFNode):
    lods: List[DMFLod]


@dataclass(slots=True)
class DMFInstance(DMFNode):
    instance_id: int
