import math
import numpy as np

import skimpy
import skimpy_3d
from skimpy_blox import minecraft, colors

import tkinter as tk
from tkinter import filedialog

from typing import List
from dataclasses import dataclass

from direct.showbase.ShowBase import ShowBase
from direct.task.Task import Task
from pandac.PandaModules import WindowProperties
from panda3d.core import KeyboardButton
from panda3d.core import AmbientLight, DirectionalLight, LVector3
from panda3d.core import Geom, GeomNode, GeomTriangles, GeomVertexArrayFormat, GeomVertexFormat, GeomVertexData
from direct.gui.DirectGui import *


def create_geometry_from_mesh(mesh):
    # Define the individual array formats for vertex attributes.
    vertex_position_format = GeomVertexArrayFormat("vertex", 3, Geom.NT_float32, Geom.C_point)
    vertex_normal_format = GeomVertexArrayFormat("normal", 3, Geom.NT_float32, Geom.C_normal)
    vertex_color_format = GeomVertexArrayFormat("color", 4, Geom.NT_uint8, Geom.C_color)

    # Define the vertex data format with positions and colors.
    vertex_format = GeomVertexFormat()
    vertex_format.addArray(vertex_position_format)
    vertex_format.addArray(vertex_normal_format)
    vertex_format.addArray(vertex_color_format)
    vertex_format = GeomVertexFormat.registerFormat(vertex_format)

    # Populate the vertex position and color arrays.
    vertex_data = GeomVertexData("mesh_vertices", vertex_format, Geom.UH_static)
    vertex_data.modifyArrayHandle(0).copyDataFrom(np.array(mesh.positions, dtype=np.float32))
    vertex_data.modifyArrayHandle(1).copyDataFrom(np.array(mesh.normals, dtype=np.float32))
    vertex_data.modifyArrayHandle(2).copyDataFrom(np.array(mesh.colors, dtype=np.uint8))

    # Populate the triangle indices.
    triangles = GeomTriangles(Geom.UH_static)
    triangles.setIndexType(Geom.NT_uint32)
    triangles.modifyVertices().modifyHandle().copyDataFrom(np.array(mesh.triangles, dtype=np.uint32))

    geometry = Geom(vertex_data)
    geometry.add_primitive(triangles)
    return geometry


@dataclass
class Eye:
    yaw: float = 0.0
    pitch: float = -30.0
    pos: np.ndarray = np.array([-8, 0, 4], np.float32)


class Viewer(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        base.setBackgroundColor(0, 0, 0)

        # Disable default camera control.
        base.disableMouse()

        # Set relative mode and hide the cursor:
        self.toggle_mouse(True)
        self.accept('e', lambda: self.toggle_mouse(not self.capture_mouse))

        # Add an event handler to prompt for a model to load.
        self.accept('f', self.prompt_for_model)

        # Initialize the camera and add a task to control it.
        self.eye = Eye()
        self.update_camera()
        self.taskMgr.add(self.control_camera, "control_camera_task")

        self.load_default_model()
        self.setup_lights()

    def toggle_mouse(self, capture=True):
        win_props = WindowProperties()
        win_props.setCursorHidden(capture)
        base.win.requestProperties(win_props)
        self.capture_mouse = capture
        win_x, win_y = base.win.getXSize(), base.win.getYSize()
        base.win.movePointer(0, win_x // 2, win_y // 2)

    def update_camera(self):
        self.camera.setPos(*self.eye.pos)
        self.camera.setHpr(self.eye.yaw - 90.0, self.eye.pitch, 0)

    def control_camera(self, task):
        theta = self.eye.yaw / 180 * math.pi
        phi = self.eye.pitch / 180 * math.pi
        eye_dir = np.array([
            math.cos(theta) * math.cos(phi),
            math.sin(theta) * math.cos(phi),
            math.sin(phi),
        ])
        up_dir = np.array([0.0, 0.0, 1.0])
        side_dir = np.cross(eye_dir, up_dir)

        # Update camera based on
        move_speed = 10.0 * globalClock.get_dt()
        if base.mouseWatcherNode.is_button_down(KeyboardButton.shift()):
            move_speed *= 10.0
        if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('w')):
            self.eye.pos[:] += move_speed * eye_dir
        if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('s')):
            self.eye.pos[:] -= move_speed * eye_dir
        if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('d')):
            self.eye.pos[:] += move_speed * side_dir
        if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('a')):
            self.eye.pos[:] -= move_speed * side_dir

        if self.capture_mouse:
            view_speed = 10.0 * globalClock.get_dt()
            win_x, win_y = base.win.getXSize(), base.win.getYSize()
            cursor = base.win.getPointer(0)
            dx = cursor.getX() - base.win.getXSize() // 2
            dy = cursor.getY() - base.win.getYSize() // 2
            self.eye.yaw = (self.eye.yaw - view_speed * dx) % 360.0
            self.eye.pitch = min(85.0, max(-85.0, self.eye.pitch - move_speed * dy))
            base.win.movePointer(0, win_x // 2, win_y // 2)

        self.update_camera()

        return Task.cont

    def load_default_model(self):
        # Initialize the voxel geometry.
        t = skimpy.Tensor(shape=(3, 3, 3), val=0)
        t[0, 0, 0] = 1
        t[1, 0, 0] = 2
        t[1, 1, 0] = 1
        t[1, 2, 0] = 2

        # Define the color of each voxel type.
        config = skimpy_3d.VoxelConfig()
        config[0] = skimpy_3d.EmptyVoxel()
        config[1] = skimpy_3d.ColorVoxel(128, 0, 0)
        config[2] = skimpy_3d.ColorVoxel(0, 0, 128)

        # Map the tensor onto a mesh.
        mesh = skimpy_3d.generate_mesh(config, t._tensor)

        # Load the mesh into a panda geometry node.
        self.model = GeomNode('mesh_node')
        self.model.addGeom(create_geometry_from_mesh(mesh))
        render.attachNewNode(self.model)

    def prompt_for_model(self):
        # Prompt for a model to load.
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()

        # Load the corresponding level into a tensor.
        print(f"Loading level: {file_path}")
        level = minecraft.SkimpyMinecraftLevel.load(file_path)
        print(f"Generating megatensor...")
        tensor = level.megatensor()

        # Create the corresponding color config.
        config = skimpy_3d.VoxelConfig()
        config[0] = skimpy_3d.EmptyVoxel()
        for i in range(1, 256):
            r, g, b = colors.color_for_id(i)
            config[i] = skimpy_3d.ColorVoxel(r, g, b)

        # Map the tensor onto a mesh.
        print(f"Building geometry...")
        mesh = skimpy_3d.generate_mesh(config, tensor._tensor)

        # Load the mesh into a panda geometry node.
        self.model.removeAllGeoms()
        self.model.addGeom(create_geometry_from_mesh(mesh))
        print(f"Done.")

    def setup_lights(self):
        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor((0.3, 0.3, 0.3, 1))
        directional_light = DirectionalLight("directional_light")
        directional_light.setDirection(LVector3(0, 8, -2.5))
        directional_light.setColor((0.9, 0.8, 0.9, 1))
        render.setLight(render.attachNewNode(directional_light))
        render.setLight(render.attachNewNode(ambient_light))


if __name__ == "__main__":
    Viewer().run()
