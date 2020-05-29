import nbt.chunk
from itertools import permutations
from nbt.world import WorldFolder
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import tempfile

import skimpy
import skimpy.reduce
from typing import Tuple, List
from dataclasses import dataclass
import colorsys
from zipfile import ZipFile


# Upstream library prints here which causes overflow
def _monkey_patch_block_id_to_name(bid):
    try:
        name = nbt.chunk.block_ids[bid]
    except KeyError:
        name = 'unknown_%d' % (bid,)
    return name


nbt.chunk.block_id_to_name = _monkey_patch_block_id_to_name

block_colors = {
    "acacia_leaves": {
        "h": 114,
        "s": 64,
        "l": 22
    },
    "acacia_log": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "air": {
        "h": 0,
        "s": 0,
        "l": 0
    },
    "andesite": {
        "h": 0,
        "s": 0,
        "l": 32
    },
    "azure_bluet": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "bedrock": {
        "h": 0,
        "s": 0,
        "l": 10
    },
    "birch_leaves": {
        "h": 114,
        "s": 64,
        "l": 22
    },
    "birch_log": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "blue_orchid": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "bookshelf": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "brown_mushroom": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "brown_mushroom_block": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "cactus": {
        "h": 126,
        "s": 61,
        "l": 20
    },
    "cave_air": {
        "h": 0,
        "s": 0,
        "l": 0
    },
    "chest": {
        "h": 0,
        "s": 100,
        "l": 50
    },
    "clay": {
        "h": 7,
        "s": 62,
        "l": 23
    },
    "coal_ore": {
        "h": 0,
        "s": 0,
        "l": 10
    },
    "cobblestone": {
        "h": 0,
        "s": 0,
        "l": 25
    },
    "cobblestone_stairs": {
        "h": 0,
        "s": 0,
        "l": 25
    },
    "crafting_table": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "dandelion": {
        "h": 60,
        "s": 100,
        "l": 60
    },
    "dark_oak_leaves": {
        "h": 114,
        "s": 64,
        "l": 22
    },
    "dark_oak_log": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "dark_oak_planks": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "dead_bush": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "diorite": {
        "h": 0,
        "s": 0,
        "l": 32
    },
    "dirt": {
        "h": 27,
        "s": 51,
        "l": 15
    },
    "end_portal_frame": {
        "h": 0,
        "s": 100,
        "l": 50
    },
    "farmland": {
        "h": 35,
        "s": 93,
        "l": 15
    },
    "fire": {
        "h": 55,
        "s": 100,
        "l": 50
    },
    "flowing_lava": {
        "h": 16,
        "s": 100,
        "l": 48
    },
    "flowing_water": {
        "h": 228,
        "s": 50,
        "l": 23
    },
    "glass_pane": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "granite": {
        "h": 0,
        "s": 0,
        "l": 32
    },
    "grass": {
        "h": 94,
        "s": 42,
        "l": 25
    },
    "grass_block": {
        "h": 94,
        "s": 42,
        "l": 32
    },
    "gravel": {
        "h": 21,
        "s": 18,
        "l": 20
    },
    "ice": {
        "h": 240,
        "s": 10,
        "l": 95
    },
    "infested_stone": {
        "h": 320,
        "s": 100,
        "l": 50
    },
    "iron_ore": {
        "h": 22,
        "s": 65,
        "l": 61
    },
    "iron_bars": {
        "h": 22,
        "s": 65,
        "l": 61
    },
    "ladder": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "lava": {
        "h": 16,
        "s": 100,
        "l": 48
    },
    "lilac": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "lily_pad": {
        "h": 114,
        "s": 64,
        "l": 18
    },
    "lit_pumpkin": {
        "h": 24,
        "s": 100,
        "l": 45
    },
    "mossy_cobblestone": {
        "h": 115,
        "s": 30,
        "l": 50
    },
    "mushroom_stem": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "oak_door": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "oak_fence": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "oak_fence_gate": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "oak_leaves": {
        "h": 114,
        "s": 64,
        "l": 22
    },
    "oak_log": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "oak_planks": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "oak_pressure_plate": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "oak_stairs": {
        "h": 114,
        "s": 64,
        "l": 22
    },
    "peony": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "pink_tulip": {
        "h": 0,
        "s": 0,
        "l": 0
    },
    "poppy": {
        "h": 0,
        "s": 100,
        "l": 50
    },
    "pumpkin": {
        "h": 24,
        "s": 100,
        "l": 45
    },
    "rail": {
        "h": 33,
        "s": 81,
        "l": 50
    },
    "red_mushroom": {
        "h": 0,
        "s": 50,
        "l": 20
    },
    "red_mushroom_block": {
        "h": 0,
        "s": 50,
        "l": 20
    },
    "rose_bush": {
        "h": 0,
        "s": 0,
        "l": 100
    },
    "sugar_cane": {
        "h": 123,
        "s": 70,
        "l": 50
    },
    "sand": {
        "h": 53,
        "s": 22,
        "l": 58
    },
    "sandstone": {
        "h": 48,
        "s": 31,
        "l": 40
    },
    "seagrass": {
        "h": 94,
        "s": 42,
        "l": 25
    },
    "sign": {
        "h": 114,
        "s": 64,
        "l": 22
    },
    "spruce_leaves": {
        "h": 114,
        "s": 64,
        "l": 22
    },
    "spruce_log": {
        "h": 35,
        "s": 93,
        "l": 30
    },
    "stone": {
        "h": 0,
        "s": 0,
        "l": 32
    },
    "stone_slab": {
        "h": 0,
        "s": 0,
        "l": 32
    },
    "tall_grass": {
        "h": 94,
        "s": 42,
        "l": 25
    },
    "tall_seagrass": {
        "h": 94,
        "s": 42,
        "l": 25
    },
    "torch": {
        "h": 60,
        "s": 100,
        "l": 50
    },
    "snow": {
        "h": 240,
        "s": 10,
        "l": 85
    },
    "spawner": {
        "h": 180,
        "s": 100,
        "l": 50
    },
    "vine": {
        "h": 114,
        "s": 64,
        "l": 18
    },
    "wall_torch": {
        "h": 60,
        "s": 100,
        "l": 50
    },
    "water": {
        "h": 228,
        "s": 50,
        "l": 23
    },
    "wheat": {
        "h": 123,
        "s": 60,
        "l": 50
    },
    "white_wool": {
        "h": 0,
        "s": 0,
        "l": 100
    },
}


@dataclass
class SkimpyMinecraftChunk:
    coord: Tuple[int, int, int]
    tensor: skimpy.Tensor


@dataclass
class NumpyMinecraftChunk:
    coord: Tuple[int, int, int]
    tensor: np.ndarray


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

    def sparse_compression_ratio(self):
        dense_size = 0
        sparse_size = 0

        for chunk in self.chunk_list:
            dense_size += len(chunk.tensor)
            sparse_size += (len(chunk.tensor) - skimpy.reduce.sum((chunk.tensor == 0).to(int)))

        return dense_size / sparse_size

    def rle_compression_ratio(self):
        dense_size = 0
        rle_size = 0

        for chunk in self.chunk_list:
            dense_size += len(chunk.tensor)
            rle_size += 2 * chunk.tensor.rle_length()

        return dense_size / rle_size

    def megatensor(self):
        dim_x = self.bbox[0][1] - self.bbox[0][0]
        dim_y = self.bbox[1][1] - self.bbox[1][0]
        dim_z = self.bbox[2][1] - self.bbox[2][0]

        megatensor = skimpy.Tensor(shape=(dim_x, dim_y, dim_z), dtype=int)
        for chunk in tqdm(self.chunk_list):
            start_x = chunk.coord[0] - self.bbox[0][0]
            start_y = chunk.coord[1] - self.bbox[1][0]
            start_z = chunk.coord[2] - self.bbox[2][0]
            end_x = start_x + chunk.tensor.shape[0]
            end_y = start_y + chunk.tensor.shape[1]
            end_z = start_z + chunk.tensor.shape[2]
            megatensor[start_x:end_x, start_y:end_y, start_z:end_z] = chunk.tensor
        return megatensor

    @classmethod
    def block_color(cls, id):
        hsl = block_colors[nbt.chunk.block_id_to_name(id)]
        return colorsys.hls_to_rgb(hsl["h"], hsl["l"], hsl["s"])

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
        best_order, _ = cls.approx_best_column_order(world_folder, num_chunks)
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

            world.chunkclass = nbt.chunk.AnvilChunk  # NBT library bug workaroun

            bb = world.get_boundingbox()
            bbox_x = (bb.minx * 16, bb.minx * 16 + bb.lenx() * 16)
            bbox_y = (0, 0)
            bbox_z = (bb.minz * 16, bb.minz * 16 + bb.lenz() * 16)

            for i, chunk in enumerate(tqdm(world.iter_chunks(), total=world.chunk_count())):
                if num_chunks is not None and i >= num_chunks:
                    break
                x, z = chunk.get_coords()
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
        if isinstance(chunk, nbt.chunk.AnvilChunk):
            sy, by = divmod(y, 16)
            section = chunk.get_section(sy)
            if section is None:
                return None

            # block = section.get_block(x, by, z)
            i = by * 256 + z * 16 + x
            return section.indexes[i]
        elif isinstance(chunk, nbt.chunk.McRegionChunk):
            return chunk.blocks.get_block(x, y, z)

    @classmethod
    def _column_remap(cls, item, column_order):
        remap_x = item[column_order[0]]
        remap_y = item[column_order[1]]
        remap_z = item[column_order[2]]
        return (remap_x, remap_y, remap_z)

    @classmethod
    def chunk_to_numpy(cls, coord, chunk, column_order=(0, 1, 2)):
        max_x = 16
        max_z = 16
        max_y = chunk.get_max_height() + 1

        arr = np.ndarray(shape=cls._column_remap((max_x, max_y, max_z), column_order))

        for x in range(max_x):
            for y in range(max_y):
                for z in range(max_z):
                    block_int_id = cls._block_at(chunk, x, y, z) or 0
                    arr[cls._column_remap((x, y, z), column_order)] = block_int_id

        return NumpyMinecraftChunk(coord, arr)

    @classmethod
    def chunk_to_skimpy(cls, coord, chunk, column_order=(0, 1, 2)):
        numpy_chunk = cls.chunk_to_numpy(coord, chunk, column_order)
        tensor = skimpy.Tensor.from_numpy(numpy_chunk.tensor)
        return SkimpyMinecraftChunk(numpy_chunk.coord, tensor)
        """
        max_x = 16
        max_z = 16
        max_y = chunk.get_max_height() + 1

        builder = skimpy.Tensor.builder(cls._column_remap((max_x, max_y, max_z), column_order))

        for x in range(max_x):
            for y in range(max_y):
                for z in range(max_z):
                    block_int_id = cls._block_at(chunk, x, y, z) or 0
                    builder[cls._column_remap((x, y, z), column_order)] = block_int_id

        tensor = builder.build()
        return SkimpyMinecraftChunk(coord, tensor)
        """