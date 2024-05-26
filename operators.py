import time
from pathlib import Path

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty, FloatProperty

from .importer import import_dmf_from_path


class DMF_OT_DMFImport(bpy.types.Operator):
    bl_idname = "dmf.dmf_import"
    bl_label = "Import Decima DMF file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.dmf", options={'HIDDEN'})

    def execute(self, context):
        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for file in self.files:
            file = directory / file.name
            _start = time.time()
            import_dmf_from_path(file)
            print(f'Imported {file.name} took {time.time() - _start:.2f} seconds')
            self.report({'INFO'}, f"Import Complete. Took {time.time()- _start:.2f} seconds")
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}
