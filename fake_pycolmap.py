"""
Fake pycolmap module that reads COLMAP binary files directly.
Provides the minimal API used by NeRFICG's MipNeRF360 dataset loader.
"""
import struct
import numpy as np
from pathlib import Path
from enum import IntEnum
from collections import OrderedDict

class CameraModelId(IntEnum):
    SIMPLE_PINHOLE = 0
    PINHOLE = 1
    SIMPLE_RADIAL = 2
    RADIAL = 3

class Rigid3d:
    """Minimal Rigid3d class matching pycolmap's API."""
    def __init__(self, qvec=None, tvec=None, rotation=None, translation=None):
        if rotation is not None:
            self.rotation = np.array(rotation, dtype=np.float64)
        elif qvec is not None:
            self.rotation = self._qvec2rotmat(qvec)
        else:
            self.rotation = np.eye(3)
        
        if translation is not None:
            self.translation = np.array(translation, dtype=np.float64)
        elif tvec is not None:
            self.translation = np.array(tvec, dtype=np.float64)
        else:
            self.translation = np.zeros(3)

    @staticmethod
    def _qvec2rotmat(qvec):
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = qvec
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float64)
        return R

    def matrix(self):
        """Return 3x4 transformation matrix [R | t]."""
        return np.hstack([self.rotation, self.translation.reshape(3, 1)])

    def inverse(self):
        """Return inverse transformation."""
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return Rigid3d(rotation=R_inv, translation=t_inv)


class Camera:
    def __init__(self, camera_id, model_id, width, height, params):
        self.camera_id = camera_id
        self.model = CameraModelId(model_id)
        self.width = width
        self.height = height
        self.params = np.array(params, dtype=np.float64)

class Image:
    def __init__(self, image_id, qvec, tvec, camera_id, name):
        self.image_id = image_id
        self.qvec = np.array(qvec, dtype=np.float64)
        self.tvec = np.array(tvec, dtype=np.float64)
        self.camera_id = camera_id
        self.name = name
        self.camera = None  # will be set later

    def cam_from_world(self):
        """Return world-to-camera transformation as Rigid3d."""
        return Rigid3d(qvec=self.qvec, tvec=self.tvec)

class Point3D:
    def __init__(self, point_id, xyz, rgb, error):
        self.point_id = point_id
        self.xyz = np.array(xyz, dtype=np.float64)
        self.color = np.array(rgb, dtype=np.float64)  # pycolmap uses 'color' not 'rgb'
        self.rgb = self.color  # alias
        self.error = error

class Reconstruction:
    def __init__(self, path):
        self.path = Path(path)
        self.cameras = OrderedDict()
        self.images = OrderedDict()
        self.points3D = OrderedDict()
        self._load()

    def _read_next_bytes(self, fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    def _load(self):
        # Read cameras.bin
        cameras_path = self.path / "cameras.bin"
        if not cameras_path.exists():
            # Try text format
            cameras_txt = self.path / "cameras.txt"
            if cameras_txt.exists():
                self._load_cameras_text(cameras_txt)
            else:
                raise FileNotFoundError(f"No cameras file found in {self.path}")
        else:
            self._load_cameras_binary(cameras_path)

        # Read images.bin
        images_path = self.path / "images.bin"
        if not images_path.exists():
            images_txt = self.path / "images.txt"
            if images_txt.exists():
                self._load_images_text(images_txt)
            else:
                raise FileNotFoundError(f"No images file found in {self.path}")
        else:
            self._load_images_binary(images_path)

        # Read points3D.bin
        points_path = self.path / "points3D.bin"
        if points_path.exists():
            self._load_points3D_binary(points_path)
        else:
            points_txt = self.path / "points3D.txt"
            if points_txt.exists():
                self._load_points3D_text(points_txt)

        # Link cameras to images
        for img in self.images.values():
            if img.camera_id in self.cameras:
                img.camera = self.cameras[img.camera_id]

    def _load_cameras_binary(self, path):
        with open(path, "rb") as fid:
            num_cameras = self._read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_cameras):
                cam_id, model_id = self._read_next_bytes(fid, 8, "ii")
                width, height = self._read_next_bytes(fid, 16, "QQ")
                # Determine number of params based on model
                if model_id == 0:  # SIMPLE_PINHOLE
                    params = self._read_next_bytes(fid, 24, "3d")
                elif model_id == 1:  # PINHOLE
                    params = self._read_next_bytes(fid, 32, "4d")
                elif model_id == 2:  # SIMPLE_RADIAL
                    params = self._read_next_bytes(fid, 32, "4d")
                elif model_id == 3:  # RADIAL
                    params = self._read_next_bytes(fid, 40, "5d")
                else:
                    # Read remaining bytes as doubles
                    num_params = CameraModelId(model_id)
                    params = self._read_next_bytes(fid, 8 * 4, "4d")
                self.cameras[cam_id] = Camera(cam_id, model_id, width, height, params)

    def _load_images_binary(self, path):
        with open(path, "rb") as fid:
            num_images = self._read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_images):
                image_id = self._read_next_bytes(fid, 4, "i")[0]
                qvec = self._read_next_bytes(fid, 32, "4d")
                tvec = self._read_next_bytes(fid, 24, "3d")
                camera_id = self._read_next_bytes(fid, 4, "i")[0]
                # Read image name (null-terminated string)
                img_name = b""
                while True:
                    ch = fid.read(1)
                    if ch == b"\0" or ch == b"":
                        break
                    img_name += ch
                # Read num_points2D
                num_points2D = self._read_next_bytes(fid, 8, "Q")[0]
                # Skip point data
                if num_points2D > 0:
                    fid.read(num_points2D * 24)  # x, y (2*8 bytes) + point3D_id (8 bytes)
                self.images[image_id] = Image(image_id, qvec, tvec, camera_id, img_name.decode("utf-8"))

    def _load_points3D_binary(self, path):
        with open(path, "rb") as fid:
            num_points = self._read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_points):
                xyz = self._read_next_bytes(fid, 24, "3d")
                point_id = self._read_next_bytes(fid, 8, "q")[0]
                rgb = self._read_next_bytes(fid, 3, "3B")
                error = self._read_next_bytes(fid, 8, "d")[0]
                track_len = self._read_next_bytes(fid, 8, "Q")[0]
                if track_len > 0:
                    fid.read(track_len * 8)  # image_id (4) + point2D_idx (4)
                self.points3D[point_id] = Point3D(point_id, xyz, rgb, error)

    def _load_cameras_text(self, path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                model_name = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                model_map = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3}
                model_id = model_map.get(model_name, 1)
                self.cameras[cam_id] = Camera(cam_id, model_id, width, height, params)

    def _load_images_text(self, path):
        with open(path) as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            parts = line.split()
            image_id = int(parts[0])
            qvec = [float(p) for p in parts[1:5]]
            tvec = [float(p) for p in parts[5:8]]
            camera_id = int(parts[8])
            name = parts[9]
            self.images[image_id] = Image(image_id, qvec, tvec, camera_id, name)
            i += 2  # skip the points2D line

    def _load_points3D_text(self, path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                point_id = int(parts[0])
                xyz = [float(p) for p in parts[1:4]]
                rgb = [int(p) for p in parts[4:7]]
                error = float(parts[7])
                self.points3D[point_id] = Point3D(point_id, xyz, rgb, error)

    def camera(self, camera_id):
        return self.cameras[camera_id]

    def summary(self):
        return f"Reconstruction with {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} points3D"

    @property
    def num_images(self):
        return len(self.images)

    @property
    def num_points3D(self):
        return len(self.points3D)
