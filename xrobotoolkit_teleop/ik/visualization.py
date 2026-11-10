"""Shared browser renderer; consumes snapshots and never imports an IK backend."""

from pathlib import Path
import io
import subprocess
import webbrowser
import xml.etree.ElementTree as ET

import meshcat
import meshcat.geometry as geometry
import meshcat.transformations as tf
import numpy as np
import zmq
from PIL import Image, ImageDraw


def _origin(element):
    origin = element.find("origin")
    if origin is None:
        return np.eye(4)
    angles = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    result = tf.euler_matrix(*angles)
    result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    return result


class IKViewer:
    def __init__(self, urdf_path, viewer=None, open_browser=True):
        self._owns_viewer = viewer is None
        self._closed = False
        descriptions = []
        self.visuals = []
        self._status = None
        self._triads = set()
        path = Path(urdf_path)
        for link in ET.parse(path).getroot().findall("link"):
            for index, visual in enumerate(link.findall("visual")):
                shape = list(visual.find("geometry"))[0]
                transform = _origin(visual)
                if shape.tag == "mesh":
                    filename = Path(shape.get("filename"))
                    if not filename.is_absolute():
                        filename = path.parent / filename
                    suffix = filename.suffix.lower()
                    loader = {
                        ".stl": geometry.StlMeshGeometry,
                        ".obj": geometry.ObjMeshGeometry,
                        ".dae": geometry.DaeMeshGeometry,
                    }.get(suffix)
                    if loader is None:
                        raise ValueError(f"Unsupported visualization mesh: {filename}")
                    mesh = loader.from_file(str(filename))
                    transform[:3, :3] = transform[:3, :3] @ np.diag(np.fromstring(shape.get("scale", "1 1 1"), sep=" "))
                elif shape.tag == "box":
                    mesh = geometry.Box(np.fromstring(shape.get("size"), sep=" "))
                elif shape.tag == "sphere":
                    mesh = geometry.Sphere(float(shape.get("radius")))
                elif shape.tag == "cylinder":
                    mesh = geometry.Cylinder(float(shape.get("length")), float(shape.get("radius")))
                    transform = transform @ tf.rotation_matrix(np.pi / 2, [1, 0, 0])
                else:
                    raise ValueError(f"Unsupported visualization geometry: {shape.tag}")
                descriptions.append((link.get("name"), index, mesh, transform))
        # Load and validate assets before starting a browser server.
        self.viewer = viewer if viewer is not None else meshcat.Visualizer()
        try:
            for link, index, mesh, transform in descriptions:
                node = self.viewer[f"robot/{link}/{index}"]
                node.set_object(mesh, geometry.MeshLambertMaterial(color=0xA4AFBD))
                self.visuals.append((node, link, transform))
            if open_browser:
                webbrowser.open(self.viewer.url())
        except Exception:
            self.close()
            raise

    def _triad(self, path, pose, scale):
        if path not in self._triads:
            self.viewer[path].set_object(geometry.triad(scale=scale))
            self._triads.add(path)
        self.viewer[path].set_transform(pose)

    def update(self, snapshot):
        for node, link, local in self.visuals:
            node.set_transform(snapshot.link_poses[link] @ local)
        for name, pose in snapshot.targets.items():
            self._triad(f"targets/{name}", pose, 0.08)
        for name, pose in snapshot.end_effectors.items():
            self._triad(f"end_effectors/{name}", pose, 0.06)
        if snapshot.status != self._status:
            image = Image.new("RGB", (512, 32), "white")
            ImageDraw.Draw(image).text((8, 8), snapshot.status[:80], fill="black")
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            material = geometry.MeshBasicMaterial(map=geometry.ImageTexture(geometry.PngImage(stream.getvalue())))
            self.viewer["status"].set_object(geometry.Box([1.6, 0.001, 0.1]), material)
            self.viewer["status"].set_transform(tf.translation_matrix([0, 0, 1.5]))
            self._status = snapshot.status

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._owns_viewer:
                # Meshcat normally waits forever for a server reply. A dead
                # server must not prevent the backend and XR client closing.
                socket = self.viewer.window.zmq_socket
                socket.setsockopt(zmq.SNDTIMEO, 250)
                socket.setsockopt(zmq.RCVTIMEO, 250)
            self.viewer.delete()
        finally:
            if self._owns_viewer:
                window = self.viewer.window
                try:
                    window.zmq_socket.close(linger=0)
                finally:
                    if window.server_proc is not None:
                        window.server_proc.terminate()
                        try:
                            window.server_proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            window.server_proc.kill()
                            window.server_proc.wait(timeout=2)
