import trimesh
import os
import open3d as o3d
import pyVHACD
import numpy as np

def test_convexity():
    # Load a mesh from a file
    mesh = trimesh.load_mesh('/home/zhaodong/code/gello_software/third_party/mujoco_menagerie/franka_emika_panda/assets/SPOON.obj')

    # Check if the mesh is convex
    is_convex = mesh.is_convex

    # Assert that the mesh is convex
    assert is_convex, "The mesh should be convex"
    
    # Optionally, print the result
    print(f"The mesh is {'convex' if is_convex else 'not convex'}.")

def decompose_mesh():
    """
    Decomposes a mesh into its convex components.
    """
    input_mesh_path = "/home/zhaodong/code/gello_software/third_party/mujoco_menagerie/franka_emika_panda/assets/SPOON.obj"
    output_dir = "spoon_parts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original mesh
    # mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    # mesh.compute_vertex_normals()
    mesh = trimesh.load_mesh(input_mesh_path)

    # Decompose the mesh into convex parts
    out  = trimesh.decomposition.convex_decomposition(mesh)

    # Save each convex part as a separate mesh file
    for i, part in enumerate(out):
        part_mesh = trimesh.Trimesh(vertices=part['vertices'], faces=part['faces'])
        part_mesh.export(os.path.join(output_dir, f"part_{i}.obj"))
        print(f"Saved part {i} to {os.path.join(output_dir, f'part_{i}.obj')}")

def generate_spoon_xml(mesh_dir, output_path, base_position=(0, 0, 0)):
    obj_files = sorted([f for f in os.listdir(mesh_dir) if f.endswith(".obj")])

    xml_lines = []
    xml_lines.append('<?xml version="1.0" encoding="utf-8"?>')
    xml_lines.append('<mujoco model="spoon_parts">')
    xml_lines.append('  <asset>')

    for i, obj_file in enumerate(obj_files):
        name = f"spoon_part_{i}"
        xml_lines.append(f'    <mesh name="{name}" file="{os.path.join(mesh_dir, obj_file)}" scale="0.002 0.002 0.002"/>')

    xml_lines.append('  </asset>')
    xml_lines.append('  <worldbody>')
    xml_lines.append(f'    <body name="spoon" pos="{base_position[0]} {base_position[1]} {base_position[2]}">')

    for i in range(len(obj_files)):
        name = f"spoon_part_{i}"
        xml_lines.append(f'      <geom name="geom_{name}" type="mesh" mesh="{name}" rgba="0.8 0.8 0.8 1" />')

    xml_lines.append('    </body>')
    xml_lines.append('  </worldbody>')
    xml_lines.append('</mujoco>')

    with open(output_path, "w") as f:
        f.write("\n".join(xml_lines))

    print(f"MuJoCo XML written to {output_path}")

if __name__ == "__main__":
    print(o3d.__file__)
    try:
        test_convexity()
    except:
        # decompose_mesh()
        mesh_dir = "spoon_parts"
        output_path = "spoon_parts.xml"
        generate_spoon_xml(mesh_dir, output_path)
    