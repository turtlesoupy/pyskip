import numpy as np
import pythreejs as three

from IPython.display import display


class Scene:
    def __init__(self, eye_pos, obj_pos, obj, up=(0, 1, 0)):
        self.camera = three.PerspectiveCamera(
            position=tuple(eye_pos),
            lookAt=tuple(obj_pos),
            fov=30,
        )
        self.camera.up = up
        self.light = three.PointLight(
            color="white",
            position=(-eye_pos[0], eye_pos[1], eye_pos[2]),
        )
        self.obj = obj
        self.scene = three.Scene(children=[self.obj, self.light, self.camera])


def render_scene(scene: Scene):
    renderer = three.Renderer(
        camera=scene.camera,
        background="white",
        background_opacity=1,
        scene=scene.scene,
        controls=[three.OrbitControls(controlling=scene.camera)],
        width=984,
        height=984,
    )

    return display(renderer)


def build_geometry(triangles, positions, normals=None, colors=None):
    attributes = {
        "position": three.BufferAttribute(
            np.array(positions, dtype="float32").reshape(-1, 3),
            normalized=False,
        ),
        "index": three.BufferAttribute(
            np.array(triangles, dtype="uint32"),
            normalized=False,
        ),
    }
    if normals:
        attributes["normal"] = three.BufferAttribute(
            np.array(normals, dtype="float32").reshape(-1, 3),
            normalized=False,
        )
    if colors:
        attributes["color"] = three.BufferAttribute(
            np.array(colors, dtype="uint8").reshape(-1, 4),
            normalized=False,
        )
    geometry = three.BufferGeometry(attributes=attributes)

    # Compute normals from face geometry if they were not specified.
    if not normals:
        geometry.exec_three_obj_method("computeVertexNormals")

    return geometry


def render_mesh(mesh, up_vector=(0, 1, 0)):
    geometry = build_geometry(
        mesh.triangles,
        mesh.positions,
        normals=getattr(mesh, "normals", None),
        colors=getattr(mesh, "colors", None),
    )

    if hasattr(mesh, "colors"):
        material = three.MeshLambertMaterial(vertexColors="VertexColors")
    else:
        material = three.MeshLambertMaterial()

    obj = three.Mesh(
        geometry=geometry,
        material=material,
        position=[0, 0, 0],
    )

    center = np.array(mesh.positions).reshape(-1, 3).mean(axis=0)
    offset = 8 * np.array(mesh.positions).reshape(-1, 3).std(axis=0)
    return render_scene(Scene(eye_pos=center + offset, obj_pos=center, obj=obj, up=up_vector))


def render_wireframe(mesh, up_vector=(0, 1, 0)):
    geometry = build_geometry(
        mesh.triangles,
        mesh.positions,
        normals=getattr(mesh, "normals", None),
        colors=getattr(mesh, "colors", None),
    )

    obj = three.LineSegments(
        geometry=three.WireframeGeometry(geometry),
        material=three.LineBasicMaterial(color="green"),
        position=[0, 0, 0],
    )

    center = np.array(mesh.positions).reshape(-1, 3).mean(axis=0)
    offset = 8 * np.array(mesh.positions).reshape(-1, 3).std(axis=0)
    return render_scene(Scene(eye_pos=center + offset, obj_pos=center, obj=obj, up=up_vector))
