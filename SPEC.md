### DMFSceneFile

`DMFSceneFile` is the root class for the 3D model format. It contains all the information needed to fully describe a 3D scene.

#### Field Specification

- `metadata` ([DMFSceneMetaData](#dmfscenemetadata)): The metadata of the scene.
- `collections` (Array of [DMFCollection](#dmfcollection)): A list of collections present in the scene.
- `models` (Array of [DMFNode](#dmfnode)): A list of models present in the scene.
- `skeletons` (Array of [DMFSkeleton](#dmfskeleton)): A list of skeletons present in the scene.
- `buffers` (Array of [DMFBuffer](#dmfbuffer)): A list of buffers containing raw binary data.
- `bufferViews` (Array of [DMFBufferView](#dmfbufferview)): A list of views into the buffers.
- `materials` (Array of [DMFMaterial](#dmfmaterial)): A list of materials used in the scene.
- `textures` (Array of [DMFTexture](#dmftexture)): A list of textures used in the scene.
- `instances` (Array of [DMFNode](#dmfnode)): A list of instances of models in the scene.

Please note that the data in this class will be used for future references as some other classes may reference data from its arrays. 

#### Structure

A `DMFSceneFile` object has the following structure:

```json
{
    "metadata": DMFSceneMetaData,
    "collections": Array[DMFCollection],
    "models": Array[DMFNode],
    "skeletons": Array[DMFSkeleton],
    "buffers": Array[DMFBuffer],
    "bufferViews": Array[DMFBufferView],
    "materials": Array[DMFMaterial],
    "textures": Array[DMFTexture],
    "instances": Array[DMFNode]
}
```
#### JSON Example
Here's an example of a DMFSceneFile object:

```json
{
    "metadata": {
        "generator": "Example Exporter (v1.0, abc123)",
        "version": 1
    },
    "collections": [...],  // List of DMFCollection JSON objects
    "models": [...],  // List of DMFNode JSON objects
    "skeletons": [...],  // List of DMFSkeleton JSON objects
    "buffers": [...],  // List of DMFBuffer JSON objects
    "bufferViews": [...],  // List of DMFBufferView JSON objects
    "materials": [...],  // List of DMFMaterial JSON objects
    "textures": [...],  // List of DMFTexture JSON objects
    "instances": [...]  // List of DMFNode JSON objects
}
```

Please note that the `[...]` in the example above should be replaced with the corresponding JSON representations of each class type. The details for these representations can be found in their respective class specifications.


### DMFSceneMetaData

`DMFSceneMetaData` contains the metadata information about the 3D scene.

#### Field Specification

- `generator` (String): The name of the exporter generating the scene. This is generally in the format "{appTitle} ({appVersion}, {buildCommit})".
- `version` (Integer): The version of the format. As of now, the current version is 1.

#### Structure

A `DMFSceneMetaData` object has the following structure:

```json
{
    "generator": String,
    "version": Integer
}
```

#### JSON Example
Here's an example of a `DMFSceneMetaData` object:
```json
{
    "generator": "Example Exporter (v1.0, abc123)",
    "version": 1
}
```

### DMFCollection

`DMFCollection` represents a collection of objects within the 3D scene.

#### Field Specification

- `name` (String): The name of the collection.
- `enabled` (Boolean, optional): Defines if the collection is enabled or not. By default, this field is set to `True`.
- `parent` (Integer, optional): The index of the parent collection object within the `collections` array of the [DMFSceneFile](#dmfscenefile) class. If this field is not provided, it defaults to `None`, indicating that the collection has no parent.

#### Structure

A `DMFCollection` object has the following structure:

```json
{
    "name": String,
    "enabled": Boolean,
    "parent": Integer
}
```
#### JSON Example
Here's an example of a `DMFCollection` object:
```json
{
    "name": "ExampleCollection",
    "enabled": true,
    "parent": 0
}
```

### DMFTransform

`DMFTransform` represents the transformation of an object in the 3D space.

#### Field Specification

- `position` (Tuple of Floats): The position of the object in 3D space, represented as a tuple of three float values (x, y, z).
- `scale` (Tuple of Floats): The scale of the object in 3D space, represented as a tuple of three float values (x, y, z).
- `rotation` (Tuple of Floats): The rotation of the object in 3D space, represented as a tuple of four float values (x, y, z, w) for a quaternion.

#### Structure

A `DMFTransform` object has the following structure:

```json
{
    "position": [Float, Float, Float],
    "scale": [Float, Float, Float],
    "rotation": [Float, Float, Float, Float]
}
```
#### JSON Example
Here's an example of a `DMFTransform` object:
```json
{
    "position": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0],
    "rotation": [0.0, 0.0, 0.0, 1.0]
}
```

### DMFBone

`DMFBone` represents a single bone within a skeleton in the 3D scene.

#### Field Specification

- `name` (String): The name of the bone.
- `transform` ([DMFTransform](#DMFTransform)): The transformation of the bone in the 3D space.
- `parent_id` (Integer): The index of the parent bone in the skeleton's bone array. 
- `local_space` (Boolean): Indicates whether the bone's transformation is in local space (relative to the parent bone) or in world space.

#### Structure

A `DMFBone` object has the following structure:

```json
{
    "name": String,
    "transform": DMFTransform,
    "parentId": Integer,
    "localSpace": Boolean
}
```

#### JSON Example
Here's an example of a `DMFBone` object:
```json
{
    "name": "Bone_1",
    "transform": {
        "position": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0.0, 0.0, 0.0, 1.0]
    },
    "parentId": 0,
    "localSpace": true
}
```

***Note*** `parentId` of `-1` means bone have no parent

### DMFBuffer

`DMFBuffer` represents a data buffer used in the 3D scene.

#### Field Specification

- `name` (String): The name of the buffer.
- `size` (Integer): The original size of the buffer.
- `path` (Optional String): The relative path to the buffer if it is external.
- `data` (Optional String): The internal data of the buffer if it is internal. This data is compressed using zlib deflate and base64 encoded.

**Note:** A buffer can be either internal or external. If the `data` field is provided, the buffer is considered internal. If the `path` field is provided, the buffer is considered external. Providing both `path` and `data` is illegal and will yield undefined behaviour.

#### Structure

A `DMFBuffer` object has the following structure:

```json
{
    "name": String,
    "size": Integer,
    "path": Optional String,
    "data": Optional String
}
```

#### JSON Example
Here's an example of an `DMFBuffer` object (internal buffer):
```json
{
    "name": "Buffer_1",
    "size": 1024,
    "data": "compressedDataString..."
}
```
And an example of an `DMFBuffer` object (external buffer):
```json
{
    "name": "Buffer_2",
    "size": 2048,
    "path": "path/to/externalBuffer.bin"
}
```

### DMFBufferView

`DMFBufferView` represents a view into a buffer object.

#### Field Specification

- `buffer_id` (Integer): The index into the `buffers` array in the [DMFSceneFile](#dmfscenefile) object that this view references.
- `offset` (Integer): The offset in bytes from the start of the buffer.
- `size` (Integer): The length of the buffer view in bytes.

#### Structure

A `DMFBufferView` object has the following structure:

```json
{
    "bufferId": Integer,
    "offset": Integer,
    "size": Integer
}
```

#### JSON Example
Here's an example of a `DMFBufferView` object:
```json
{
    "bufferId": 0,
    "offset": 1024,
    "size": 256
}
```
In this example, the buffer view references the buffer at index 0 in the [DMFSceneFile](#dmfscenefile) object's buffers array, and starts at an offset of 1024 bytes from the start of the buffer, with a length of 256 bytes.

### DMFTexture

`DMFTexture` represents a texture used in the 3D scene.

#### Field Specification

- `name` (String): The name of the texture.
- `buffer_id` (Integer): The index into the `buffers` array in the [DMFSceneFile](#dmfscenefile) object that this texture references.
- `usage_type` (String): The usage type of the texture descriptor. This field is of type `DMFUsageType`.
- `metadata` (Dict): Additional metadata about the texture. The format of this field is not defined.

#### Structure

A `DMFTexture` object has the following structure:

```json
{
    "name": String,
    "bufferId": Integer,
    "usageType": Integer,
    "metadata": Object
}
```

#### JSON Example
Here's an example of a `DMFTexture` object:
```json
{
    "name": "Texture_1",
    "bufferId": 0,
    "metadata": {
        "format": "PNG",
        "width": "1024",
        "height": "1024"
    }
}
```

In this example, the texture is named "Texture_1" and references the buffer at index 0 in the [DMFSceneFile](#dmfscenefile) object's `buffers` array. The metadata contains additional information about the texture, including its format, width, and height.

***Notes***: 
* The usage_type field is a DMFUsageType string, which is a complex case not yet defined in version 1 of the format.
* The texture buffer must contain the texture in a general format, such as PNG, TGA, EXR, TIFF, and so on.

### DMFTextureDescriptor

`DMFTextureDescriptor` represents a texture descriptor used in the 3D scene.

#### Field Specification

- `texture_id` (Integer): The index into the `textures` array in the [DMFSceneFile](#dmfscenefile) object that this texture descriptor references.
- `channels` (String): The channels used by the texture. Valid values are "R", "G", "B", "A", "RG", "RGB", and "RGBA".
- `usage_type` (String): The usage type of the texture descriptor. This field is of type `DMFUsageType`.

#### Structure

A `DMFTextureDescriptor` object has the following structure:

```json
{
    "textureId": Integer,
    "channels": String,
    "usageType": String
}
```

#### JSON Example
Here's an example of a `DMFTextureDescriptor` object:
```json
{
    "textureId": 0,
    "channels": "RGBA",
    "usageType": "Unknown"
}
```
In this example, the texture descriptor references the texture at index 0 in the [DMFSceneFile](#dmfscenefile) object's `textures` array, and the texture has RGBA channels. The usage type is "Unknown", as it is not yet defined in the DMFUsageType enum.

***Note***: The usage_type field is a DMFUsageType string, which is a complex case not yet defined in version 1 of the format.

### DMFSkeleton

`DMFSkeleton` represents a skeleton in the 3D scene.

#### Field Specification

- `bones` (List of [DMFBone](#dmfbone)): A list of bones that make up the skeleton.
- `transform` ([DMFTransform](#dmftransform), Optional): The transform of the skeleton.

#### Structure

A `DMFSkeleton` object has the following structure:

```json
{
    "bones": [DMFBone, DMFBone, ...],
    "transform": DMFTransform
}
```

#### JSON Example
Here's an example of a `DMFSkeleton` object:
```json
{
    "bones": [
        {
            "name": "Bone_1",
            "transform": {
                "position": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "rotation": [0.0, 0.0, 0.0, 1.0]
            },
            "parentId": -1,
            "localSpace": true
        },
        {
            "name": "Bone_2",
            "transform": {
                "position": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "rotation": [0.0, 0.0, 0.0, 1.0]
            },
            "parentId": 0,
            "localSpace": true
        }
    ],
    "transform": {
        "position": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0.0, 0.0, 0.0, 1.0]
    }
}
```

In this example, the skeleton has two bones, named "Bone_1" and "Bone_2", respectively. "Bone_1" is the root bone and has a parentId of -1, indicating that it has no parent. "Bone_2" has a parentId of 0, indicating that it is a child of "Bone_1". The [DMFTransform](#dmftransform) object in the transform field describes the transform of the skeleton itself.

### DMFMaterial

`DMFMaterial` represents a material used in the 3D scene.

#### Field Specification

- `name` (String): The name of the material.
- `type` (String): The type of the material. This field has no particular format.
- `texture_ids` (Dict): A dictionary that maps texture channel names to the index of the texture in the [DMFSceneFile](#dmfscenefile) object's `textures` array.
- `texture_descriptors` (List): A list of `DMFTextureDescriptor` objects that provide additional information about the textures used by the material.

#### Structure

A `DMFMaterial` object has the following structure:

```json
{
    "name": String,
    "type": String,
    "textureIds": Object,
    "textureDescriptors": Array
}
```

#### JSON Example
Here's an example of a `DMFMaterial` object:
```json
{
    "name": "Material_1",
    "type": "PBR",
    "textureIds": {
        "BaseColor": 0,
        "Metallic": 1,
        "Roughness": 2
    },
    "textureDescriptors": [
        {
            "textureId": 0,
            "channels": "RGBA",
            "usageType": "BASE_COLOR"
        },
        {
            "textureId": 1,
            "channels": "R",
            "usageType": "METALLIC"
        },
        {
            "textureId": 2,
            "channels": "R",
            "usageType": "ROUGHNESS"
        }
    ]
}
```

In this example, the material is named "Material_1" and is of type "PBR". It uses three textures: "BaseColor", "Metallic", and "Roughness". These textures are stored at indices 0, 1, and 2 in the [DMFSceneFile](#dmfscenefile) object's `textures` array. The `textureDescriptors` array provides additional information about the textures, including the channels they use and their usage types.

### DMFNodeType

`DMFNodeType` is an enumeration of the different types of nodes that can be present in the 3D scene.

#### Possible Values

- `NODE`: A regular node in the scene hierarchy.
- `LOD`: A node representing a level-of-detail object.
- `MODEL`: A node representing a 3D model.
- `MODEL_GROUP`: A node representing a group of 3D models.
- `SKINNED_MODEL`: A node representing a skinned 3D model.
- `INSTANCE`: A node representing an instance of another node in the scene.
- `ATTACHMENT`: A node representing an attachment to another node in the scene.

**Note**: Using unsupported node types will cause loading errors.

### DMFNode

`DMFNode` represents a node in the 3D scene.

#### Field Specification

- `type` ([DMFNodeType](#dmfnodetype)): The type of the node.
- `name` (Optional String): The name of the node.
- `collection_ids` (List of Integer): The indices of the [collections](#dmfcollection) that this node belongs to.
- `transform` (Optional [DMFTransform](#dmftransform)): The transformation of the node.
- `children` (List of [DMFNode](#dmfnode)): The child nodes of the current node.
- `visible` (Boolean): Whether the node is visible in the scene.

#### Structure

A `DMFNode` object has the following structure:

```json
{
    "type": String,
    "name": Optional String,
    "collectionIds": List of Integer,
    "transform": Optional DMFTransform,
    "children": List of DMFNode,
    "visible": Boolean
}
```

#### JSON Example
Here's an example of a `DMFNode` object:
```json
{
    "type": "NODE",
    "name": "MyNode",
    "collectionIds": [0, 1],
    "transform": {
        "position": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0.0, 0.0, 0.0, 1.0]
    },
    "children": [],
    "visible": true
}
```

In this example, the node has a type of `NODE`, a name of MyNode, belongs to the collections at indices 0 and 1, has a transformation, has no children, and is visible in the scene.

### DMFBufferType

`DMFBufferType` is an enumeration that defines whether vertex attributes in a mesh are stored in a single buffer or multiple buffers.

#### Enumeration Members

- `MULTI_BUFFER`: Each vertex attribute has its own buffer.
- `SINGLE_BUFFER`: All vertex attributes are packed into a single buffer.

### DMFSemantic

`DMFSemantic` is an enumeration of all currently supported vertex semantics.

#### Supported Semantics

- `POSITION`: The vertex position.
- `TANGENT`: The vertex tangent vector.
- `NORMAL`: The vertex normal vector.
- `COLOR_0`: The vertex color for the first color channel.
- `COLOR_1`: The vertex color for the second color channel.
- `COLOR_2`: The vertex color for the third color channel.
- `COLOR_3`: The vertex color for the fourth color channel.
- `TEXCOORD_0`: The first texture coordinate.
- `TEXCOORD_1`: The second texture coordinate.
- `TEXCOORD_2`: The third texture coordinate.
- `TEXCOORD_3`: The fourth texture coordinate.
- `TEXCOORD_4`: The fifth texture coordinate.
- `TEXCOORD_5`: The sixth texture coordinate.
- `TEXCOORD_6`: The seventh texture coordinate.
- `JOINTS_0`: The first 4 joint indices for skinning.
- `JOINTS_1`: The second 4 joint indices for skinning.
- `JOINTS_2`: The third 4 joint indices for skinning.
- `JOINTS_3`: The fourth 4 joint indices for skinning.
- `WEIGHTS_0`: The weight for the first 4 joint indices for skinning.
- `WEIGHTS_1`: The weight for the second 4 joint indices for skinning.
- `WEIGHTS_2`: The weight for the third 4 joint indices for skinning.
- `WEIGHTS_3`: The weight for the fourth 4 joint indices for skinning.

#### Notes

- Adding semantics that are not listed here will not be loaded.

### DMFVertexAttribute

`DMFVertexAttribute` represents an attribute of a vertex.

#### Field Specification

- `semantic` (`DMFSemantic`): The semantic of the attribute.
- `element_count` (Integer): The number of elements in the attribute.
- `element_type` (`DMFComponentType`): The type of each element in the attribute.
- `size` (Integer): The size of the attribute in bytes.
- `stride` (Optional Integer): The number of bytes between the start of consecutive attributes.
- `offset` (Optional Integer): The offset in bytes of the first element of the attribute in the buffer.
- `buffer_view_id` (Integer): The index into the `buffer_views` array in the `DMFSceneFile` object that this attribute references.

#### Structure

A `DMFVertexAttribute` object has the following structure:

```json
{
    "semantic": String,
    "elementCount": Integer,
    "elementType": String,
    "size": Integer,
    "stride": Optional Integer,
    "offset": Optional Integer,
    "bufferViewId": Integer
}
```

#### JSON Example
Here's an example of a DMFVertexAttribute object:
```json
{
    "semantic": "POSITION",
    "elementCount": 3,
    "elementType": "FLOAT",
    "size": 12,
    "stride": 24,
    "offset": 0,
    "bufferViewId": 0
}
```

In this example, the vertex attribute is of the "POSITION" semantic and contains 3 floats. Each float is 4 bytes, so the size field is 12 bytes. The attribute is stored in the `buffer` view at index 0 in the [DMFSceneFile](#dmfscenefile) object's `buffer_views` array, with an offset of 0 bytes and a stride of 24 bytes (which means that the next attribute starts 24 bytes after the start of this one).

### DMFPrimitive

`DMFPrimitive` represents a primitive (a part of a mesh) in a 3D model.

#### Field Specification

- `grouping_id` (Integer): An identifier used when multiple primitives use the same vertex and index buffer. When each primitive uses its own buffer, each primitive must have a unique grouping ID (incremental).
- `vertex_count` (Integer): The number of vertices in the primitive.
- `vertex_start` (Integer): The index of the first vertex in the vertex buffer that belongs to this primitive.
- `vertex_end` (Integer): The index of the last vertex in the vertex buffer that belongs to this primitive.
- `vertex_attributes` (Dictionary): A dictionary of vertex attributes (DMFSemantic -> DMFVertexAttribute).
- `vertex_type` (DMFBufferType): The data type of the vertex buffer.
- `index_count` (Integer): The number of indices in the primitive.
- `index_start` (Integer): The index of the first index in the index buffer that belongs to this primitive.
- `index_end` (Integer): The index of the last index in the index buffer that belongs to this primitive.
- `index_size` (Integer): The size of each index in bytes.
- `index_buffer_view_id` (Integer): The index into the `bufferViews` array in the `DMFSceneFile` object that this primitive's index buffer view references.
- `material_id` (Integer or null): The index into the `materials` array in the `DMFSceneFile` object that this primitive references, or null if it does not reference a material.

#### Structure

A `DMFPrimitive` object has the following structure:

```json
{
  "groupingId": Integer,
  "vertexCount": Integer,
  "vertexStart": Integer,
  "vertexEnd": Integer,
  "vertexAttributes": Dictionary of "DMFSemantic" to "DMFVertexAttribute",
  "vertexType": DMFBufferType,
  "indexCount": Integer,
  "indexStart": Integer,
  "indexEnd": Integer,
  "indexSize": Integer,
  "indexBufferViewId": Integer,
  "materialId": Integer or null
```

#### JSON Example
Here's an example of a DMFPrimitive object:
```json
{
    "groupingId": 0,
    "vertexCount": 8,
    "vertexStart": 0,
    "vertexEnd": 8,
    "vertexAttributes": {...},
    "vertexType": "SINGLE_BUFFER",
    "indexCount": 36,
    "indexStart": 0,
    "indexEnd": 36,
    "indexSize": 2,
    "indexBufferViewId": 3,
    "materialId": 0
}
```

### DMFMesh

`DMFMesh` represents a mesh in the 3D scene.

#### Field Specification

- `primitives` (List): A list of `DMFPrimitive` objects that define the geometry of the mesh.
- `bone_remap_table` (Dictionary): A dictionary that maps local bone IDs to skeleton bone array.

#### Structure

A `DMFMesh` object has the following structure:

```json
{
    "primitives": [
        {...},
        {...},
        ...
    ],
    "boneRemapTable": {
        "0": 2,
        "1": 0,
        ...
    }
}
```

#### JSON Example
Here's an example of a `DMFMesh` object:
```json
{
    "primitives": [
        {
            "indices": {"bufferViewId": 0, "componentType": "UNSIGNED_SHORT"},
            "materialId": 0,
            "attributes": {
                "POSITION": {"bufferViewId": 1, "componentType": "FLOAT", "count": 512},
                "NORMAL": {"bufferViewId": 2, "componentType": "FLOAT", "count": 512},
                "TEXCOORD_0": {"bufferViewId": 3, "componentType": "FLOAT", "count": 512}
            }
        }
    ],
    "boneRemapTable": {
        "0": 35,
        "1": 38,
        "2": 37
    }
}
```
In this example, the mesh has a single primitive with indices stored in a buffer view at index 0 in the [DMFSceneFile](#dmfscenefile) object's `buffer_views` array, and attribute data stored in buffer views at indices 1, 2, and 3 in the `buffer_views` array. The `boneRemapTable` maps local bone IDs to the index of the bone in the [DMFSkeleton](#dmfskeleton) object's bones array.

### DMFModel

`DMFModel` represents a 3D model in the scene.

#### Field Specification

- Fields that are inherited from [DMFNode](#dmfnode)
- `mesh` (`DMFMesh`): The mesh of the model.
- `skeleton_id` (Integer): The index of the skeleton in the [DMFSceneFile](#dmfscenefile) object's `skeletons` array that is used to animate the model. If the model is not skinned, this field is set to -1.
#### Structure

A `DMFModel` object has the following structure:

```json
{
  "mesh": DMFMesh,
  "type": "MODEL",
  "name": String,
  "children": List of DMFNode,
  "collectionIds": List of Integers,
  "transform": DMFTransfrom
}
```

#### JSON Example
Here's an example of a `DMFModel` object:
```json
      "mesh": {...},
      "type": "MODEL",
      "name": "body",
      "children": [],
      "collectionIds": []
      "transform": {...}
```

### DMFModelGroup

`DMFModelGroup` represents a group of models.

#### Field Specification

`DMFModelGroup` inherits all fields from the [DMFNode](#dmfnode) class.

#### Structure

A `DMFModelGroup` object has the same structure as a [DMFNode](#dmfnode) object.


### DMFAttachment

`DMFAttachment` represents an attachment that can be attached to a bone in a skeleton.

#### Field Specification

- Fields that are inherited from [DMFNode](#dmfnode)
- `bone_name` (String): The name of the bone in the parent skeleton that this attachment is attached to.

#### Structure

A `DMFAttachment` object has the following structure:

```json
{
  "type": "ATTACHMENT",
  "name": String,
  "children": List of DMFNode,
  "collectionIds": List of Integers,
  "transform": DMFTransfrom
  "bone_name": String
}
```

### DMFInstance

`DMFInstance` represents an instance of a model in the 3D scene.

#### Field Specification

- Fields that are inherited from [DMFNode](#dmfnode)
- `instance_id` (Integer): The index into the `instances` array in the [DMFSceneFile](#dmfscenefile) object.

#### Structure

A `DMFInstance` object has the same structure as a [DMFNode](#dmfnode) object, with the addition of the `instance_id` field:

```json
{
  "type": "INSTANCE",
  "name": String,
  "children": List of DMFNode,
  "collectionIds": List of Integers,
  "transform": DMFTransfrom,
  "instance_id": Integer
}
```
