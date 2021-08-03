import bpy
from mathutils import Vector

mesh = bpy.context.object.data

bpy.ops.object.mode_set(mode='EDIT')

selected_verts = [v for v in mesh.vertices if v.select]

normal = Vector((0,0,0))
for v in selected_verts:    
    normal += v.normal

normal.z = -normal.z

print(normal.normalized())