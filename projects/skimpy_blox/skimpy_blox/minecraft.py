import gzip
import struct
from itertools import permutations
from .nbt_helper.world import WorldFolder
from .nbt_helper.chunk import AnvilChunk, McRegionChunk
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import tempfile
import pickle

import skimpy
import skimpy.reduce
from typing import Tuple, List
from dataclasses import dataclass
from zipfile import ZipFile

from PIL import Image
from . import colors


@dataclass
class SkimpyMinecraftChunk:
    coord: Tuple[int, int, int]
    tensor: skimpy.Tensor

    def to_numpy(self):
        return NumpyMinecraftChunk(
            self.coord,
            self.tensor.to_numpy(),
        )

    def to_skimpy(self):
        return self


@dataclass
class NumpyMinecraftChunk:
    coord: Tuple[int, int, int]
    tensor: np.ndarray

    def to_numpy(self):
        return self

    def to_skimpy(self):
        return SkimpyMinecraftChunk(
            self.coord,
            skimpy.Tensor.from_numpy(self.tensor),
        )


class SkimpyMinecraftLevel:
    def __init__(
        self,
        chunk_list: List[SkimpyMinecraftChunk],
        bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        column_order: Tuple[int, int, int],
    ):
        self.chunk_list = chunk_list
        self.bbox = bbox
        self.column_order = column_order

    def dense_dimensions(self):
        width_span = self.bbox[0]
        height_span = self.bbox[1]
        depth_span = self.bbox[2]
        return (
            abs(width_span[1] - width_span[0]),
            abs(height_span[1] - height_span[0]),
            abs(depth_span[1] - depth_span[0]),
        )

    def dump(self, path):
        with gzip.open(path, "wb") as f:
            dumpable = SkimpyMinecraftLevel(
                chunk_list=[e.to_numpy() for e in tqdm(self.chunk_list)],
                bbox=self.bbox,
                column_order=self.column_order,
            )
            pickle.dump(dumpable, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(self, path):
        with gzip.open(path, "rb") as f:
            ret = pickle.load(f)
            ret.chunk_list = [e.to_skimpy() for e in ret.chunk_list]
            return ret

    def num_nonzero_voxels(self):
        return sum((len(chunk.tensor) - skimpy.reduce.sum((chunk.tensor == 0).to(int))) for chunk in self.chunk_list)

    def num_runs(self):
        return sum(chunk.tensor.rle_length() for chunk in self.chunk_list)

    def minecraft_representation_size(self):
        return sum(len(chunk.tensor) for chunk in self.chunk_list)

    def sparse_compression_ratio(self):
        dense_size = 0
        sparse_size = 0

        for chunk in self.chunk_list:
            dense_size += len(chunk.tensor)
            sparse_size += 2 * (len(chunk.tensor) - skimpy.reduce.sum((chunk.tensor == 0).to(int)))

        return dense_size / sparse_size

    def rle_compression_ratio(self):
        dense_size = 0
        rle_size = 0

        for chunk in self.chunk_list:
            dense_size += len(chunk.tensor)
            rle_size += 2 * chunk.tensor.rle_length()

        return dense_size / rle_size

    def megatensor(self):
        dim_1, dim_2, dim_3 = self._xyz_to_skimpy_col(self.dense_dimensions(), self.column_order)
        megatensor = skimpy.Tensor(shape=(dim_1, dim_2, dim_3), dtype=int)
        for chunk in self.chunk_list:
            start_x = chunk.coord[0] - self.bbox[0][0]
            start_y = chunk.coord[1] - self.bbox[1][0]
            start_z = chunk.coord[2] - self.bbox[2][0]

            xyz_tensor_shape = self._skimpy_col_to_xyz(chunk.tensor.shape, self.column_order)
            end_x = start_x + xyz_tensor_shape[0]
            end_y = start_y + xyz_tensor_shape[1]
            end_z = start_z + xyz_tensor_shape[2]
            rng = (slice(start_x, end_x, 1), slice(start_y, end_y, 1), slice(start_z, end_z, 1))
            megatensor[self._xyz_to_skimpy_col(rng, self.column_order)] = chunk.tensor
        return megatensor

    @classmethod
    def approx_best_column_order(cls, world_folder, num_chunks):
        scores = {}
        for column_order in permutations((0, 1, 2)):
            scores[column_order] = cls.from_world(world_folder, column_order=column_order,
                                                  num_chunks=num_chunks).rle_compression_ratio()

        best_order = max(scores.items(), key=lambda x: x[1])[0]
        return (best_order, scores)

    @classmethod
    def from_world_infer_order(cls, world_folder, num_chunks=200):
        best_order, stats = cls.approx_best_column_order(world_folder, num_chunks)
        print(stats)
        return cls.from_world(world_folder, column_order=best_order)

    @classmethod
    def from_world(cls, world_folder, as_numpy=False, column_order=(0, 1, 2), num_chunks=None):
        world_folder = Path(world_folder)
        chunk_list = []
        with tempfile.TemporaryDirectory() as tmpdir:
            if world_folder.is_dir():
                world = WorldFolder(world_folder)
            else:
                with ZipFile(world_folder, "r") as zippy:
                    zippy.extractall(tmpdir)
                world = WorldFolder(tmpdir)

            world.chunkclass = AnvilChunk  # NBT library bug workaroun

            bb = world.get_boundingbox()
            bbox_x = (bb.minx * 16, bb.minx * 16 + bb.lenx() * 16)
            bbox_y = (0, 0)
            bbox_z = (bb.minz * 16, bb.minz * 16 + bb.lenz() * 16)

            for i, chunk in enumerate(tqdm(world.iter_chunks(), total=world.chunk_count())):
                if num_chunks is not None and i >= num_chunks:
                    break
                x, z = chunk.get_coords()
                x *= 16
                z *= 16
                y = 0
                bbox_y = (
                    min(bbox_y[0], 0),
                    max(bbox_y[1],
                        chunk.get_max_height() + 1),
                )

                if as_numpy:
                    chunk_list.append(cls.chunk_to_numpy((x, y, z), chunk, column_order=column_order))
                else:
                    chunk_list.append(cls.chunk_to_skimpy((x, y, z), chunk, column_order=column_order))

            return cls(
                chunk_list,
                (bbox_x, bbox_y, bbox_z),
                column_order,
            )

    @classmethod
    def _block_at(cls, chunk, x, y, z) -> int:
        if isinstance(chunk, AnvilChunk):
            sy, by = divmod(y, 16)
            section = chunk.get_section(sy)
            if section is None:
                return None

            # block = section.get_block(x, by, z)
            i = by * 256 + z * 16 + x
            return section.names[section.indexes[i]]  # HACK: Due to monkey patch, this will be an integer
        elif isinstance(chunk, McRegionChunk):
            return chunk.blocks.get_block(x, y, z)

    @classmethod
    def _xyz_to_skimpy_col(cls, item, column_order):
        x_col = next(i for i, v in enumerate(column_order) if v == 0)
        y_col = next(i for i, v in enumerate(column_order) if v == 1)
        z_col = next(i for i, v in enumerate(column_order) if v == 2)
        return (item[x_col], item[y_col], item[z_col])

    @classmethod
    def _xyz_to_numpy_col(cls, item, column_order):
        return tuple(reversed(cls._xyz_to_skimpy_col(item, column_order)))

    @classmethod
    def _skimpy_col_to_xyz(cls, item, column_order):
        remap_x = item[column_order[0]]
        remap_y = item[column_order[1]]
        remap_z = item[column_order[2]]
        return (remap_x, remap_y, remap_z)

    @classmethod
    def _numpy_col_to_xyz(cls, item, column_order):
        return tuple(reversed(cls._numpy_col_to_xyz(item, column_order)))

    @classmethod
    def chunk_to_numpy(cls, coord, chunk, column_order=(0, 1, 2)):
        max_x = 16
        max_z = 16
        max_y = chunk.get_max_height() + 1

        arr = np.ndarray(shape=cls._xyz_to_numpy_col((max_x, max_y, max_z), column_order), dtype=np.int32)

        for z in range(max_z):
            for x in range(max_x):
                for y in range(max_y):
                    block_int_id = cls._block_at(chunk, x, y, z) or 0
                    arr[cls._xyz_to_numpy_col((x, y, z), column_order)] = block_int_id

        return NumpyMinecraftChunk(coord, arr)

    @classmethod
    def chunk_to_skimpy(cls, coord, chunk, column_order=(0, 1, 2)):
        numpy_chunk = cls.chunk_to_numpy(coord, chunk, column_order)
        tensor = skimpy.Tensor.from_numpy(numpy_chunk.tensor)
        return SkimpyMinecraftChunk(numpy_chunk.coord, tensor)

    @classmethod
    def map_for_chunk(cls, chunk, column_order):
        pixels = b""
        max_x, max_y, max_z = cls._skimpy_col_to_xyz(chunk.tensor.shape, column_order)
        chunk_numpy = chunk.to_numpy()

        for z in range(max_z):
            for x in range(max_x):
                for y in range(max_y - 1, -1, -1):
                    numpy_coords = cls._xyz_to_numpy_col((x, y, z), column_order)
                    block_id = chunk_numpy.tensor[numpy_coords]
                    if block_id is not None:
                        if (block_id != 0 or y == 0):
                            break

                rgb = colors.color_for_id(block_id)
                pixels += struct.pack("BBB", rgb[0], rgb[1], rgb[2])
        im = Image.frombytes('RGB', (16, 16), pixels)
        return im

    def to_map_image(self):
        width = self.bbox[0][1] - self.bbox[0][0]
        depth = self.bbox[2][1] - self.bbox[2][0]
        minx = self.bbox[0][0]
        minz = self.bbox[2][0]

        world_map = Image.new('RGB', (width, depth))
        for chunk in tqdm(self.chunk_list):
            chunkmap = self.map_for_chunk(chunk, self.column_order)
            x, y, z = chunk.coord
            world_map.paste(chunkmap, (x - minx, z - minz))

        return world_map
