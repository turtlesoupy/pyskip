import nbt

block_colors = {
    'acacia_leaves': {
        'h': 114,
        's': 64,
        'l': 22
    },
    'acacia_log': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'air': {
        'h': 0,
        's': 0,
        'l': 0
    },
    'andesite': {
        'h': 0,
        's': 0,
        'l': 32
    },
    'azure_bluet': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'bedrock': {
        'h': 0,
        's': 0,
        'l': 10
    },
    'birch_leaves': {
        'h': 114,
        's': 64,
        'l': 22
    },
    'birch_log': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'blue_orchid': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'bookshelf': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'brown_mushroom': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'brown_mushroom_block': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'cactus': {
        'h': 126,
        's': 61,
        'l': 20
    },
    'cave_air': {
        'h': 0,
        's': 0,
        'l': 0
    },
    'chest': {
        'h': 0,
        's': 100,
        'l': 50
    },
    'clay': {
        'h': 7,
        's': 62,
        'l': 23
    },
    'coal_ore': {
        'h': 0,
        's': 0,
        'l': 10
    },
    'cobblestone': {
        'h': 0,
        's': 0,
        'l': 25
    },
    'cobblestone_stairs': {
        'h': 0,
        's': 0,
        'l': 25
    },
    'crafting_table': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'dandelion': {
        'h': 60,
        's': 100,
        'l': 60
    },
    'dark_oak_leaves': {
        'h': 114,
        's': 64,
        'l': 22
    },
    'dark_oak_log': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'dark_oak_planks': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'dead_bush': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'diorite': {
        'h': 0,
        's': 0,
        'l': 32
    },
    'dirt': {
        'h': 27,
        's': 51,
        'l': 15
    },
    'end_portal_frame': {
        'h': 0,
        's': 100,
        'l': 50
    },
    'farmland': {
        'h': 35,
        's': 93,
        'l': 15
    },
    'fire': {
        'h': 55,
        's': 100,
        'l': 50
    },
    'flowing_lava': {
        'h': 16,
        's': 100,
        'l': 48
    },
    'flowing_water': {
        'h': 228,
        's': 50,
        'l': 23
    },
    'glass_pane': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'granite': {
        'h': 0,
        's': 0,
        'l': 32
    },
    'grass': {
        'h': 94,
        's': 42,
        'l': 25
    },
    'grass_block': {
        'h': 94,
        's': 42,
        'l': 32
    },
    'gravel': {
        'h': 21,
        's': 18,
        'l': 20
    },
    'ice': {
        'h': 240,
        's': 10,
        'l': 95
    },
    'infested_stone': {
        'h': 320,
        's': 100,
        'l': 50
    },
    'iron_ore': {
        'h': 22,
        's': 65,
        'l': 61
    },
    'iron_bars': {
        'h': 22,
        's': 65,
        'l': 61
    },
    'ladder': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'lava': {
        'h': 16,
        's': 100,
        'l': 48
    },
    'lilac': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'lily_pad': {
        'h': 114,
        's': 64,
        'l': 18
    },
    'lit_pumpkin': {
        'h': 24,
        's': 100,
        'l': 45
    },
    'mossy_cobblestone': {
        'h': 115,
        's': 30,
        'l': 50
    },
    'mushroom_stem': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'oak_door': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'oak_fence': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'oak_fence_gate': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'oak_leaves': {
        'h': 114,
        's': 64,
        'l': 22
    },
    'oak_log': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'oak_planks': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'oak_pressure_plate': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'oak_stairs': {
        'h': 114,
        's': 64,
        'l': 22
    },
    'peony': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'pink_tulip': {
        'h': 0,
        's': 0,
        'l': 0
    },
    'poppy': {
        'h': 0,
        's': 100,
        'l': 50
    },
    'pumpkin': {
        'h': 24,
        's': 100,
        'l': 45
    },
    'rail': {
        'h': 33,
        's': 81,
        'l': 50
    },
    'red_mushroom': {
        'h': 0,
        's': 50,
        'l': 20
    },
    'red_mushroom_block': {
        'h': 0,
        's': 50,
        'l': 20
    },
    'rose_bush': {
        'h': 0,
        's': 0,
        'l': 100
    },
    'sugar_cane': {
        'h': 123,
        's': 70,
        'l': 50
    },
    'sand': {
        'h': 53,
        's': 22,
        'l': 58
    },
    'sandstone': {
        'h': 48,
        's': 31,
        'l': 40
    },
    'seagrass': {
        'h': 94,
        's': 42,
        'l': 25
    },
    'sign': {
        'h': 114,
        's': 64,
        'l': 22
    },
    'spruce_leaves': {
        'h': 114,
        's': 64,
        'l': 22
    },
    'spruce_log': {
        'h': 35,
        's': 93,
        'l': 30
    },
    'stone': {
        'h': 0,
        's': 0,
        'l': 32
    },
    'stone_slab': {
        'h': 0,
        's': 0,
        'l': 32
    },
    'tall_grass': {
        'h': 94,
        's': 42,
        'l': 25
    },
    'tall_seagrass': {
        'h': 94,
        's': 42,
        'l': 25
    },
    'torch': {
        'h': 60,
        's': 100,
        'l': 50
    },
    'snow': {
        'h': 240,
        's': 10,
        'l': 85
    },
    'spawner': {
        'h': 180,
        's': 100,
        'l': 50
    },
    'vine': {
        'h': 114,
        's': 64,
        'l': 18
    },
    'wall_torch': {
        'h': 60,
        's': 100,
        'l': 50
    },
    'water': {
        'h': 228,
        's': 50,
        'l': 23
    },
    'wheat': {
        'h': 123,
        's': 60,
        'l': 50
    },
    'white_wool': {
        'h': 0,
        's': 0,
        'l': 100
    },
}


def color_for_id(block_id):
    if block_id is not None:
        if block_id in nbt.chunk.block_ids:
            name = nbt.chunk.block_ids[block_id]
            color = block_colors[name] if name in block_colors else {'h': 0, 's': 0, 'l': 100}
        elif isinstance(block_id, str) and block_id in block_colors:
            color = block_colors[block_id]
        else:
            color = {'h': 0, 's': 0, 'l': 100}
    else:
        color = {'h': 0, 's': 0, 'l': 0}

    final_color = {'h': color['h'], 's': color['s'], 'l': color['l']}
    if final_color['l'] > 100:
        final_color['l'] = 100
    if final_color['l'] < 0:
        final_color['l'] = 0
    rgb = hsl2rgb(final_color['h'], final_color['s'], final_color['l'])
    return rgb


# From http://www.easyrgb.com/index.php?X=MATH&H=19#text19
def hsl2rgb(H, S, L):
    H = H / 360.0
    S = S / 100.0  # Turn into a percentage
    L = L / 100.0
    if (S == 0):
        return (int(L * 255), int(L * 255), int(L * 255))
    var_2 = L * (1 + S) if (L < 0.5) else (L + S) - (S * L)
    var_1 = 2 * L - var_2

    def hue2rgb(v1, v2, vH):
        if (vH < 0):
            vH += 1
        if (vH > 1):
            vH -= 1
        if ((6 * vH) < 1):
            return v1 + (v2 - v1) * 6 * vH
        if ((2 * vH) < 1):
            return v2
        if ((3 * vH) < 2):
            return v1 + (v2 - v1) * (2 / 3.0 - vH) * 6
        return v1

    R = int(255 * hue2rgb(var_1, var_2, H + (1.0 / 3)))
    G = int(255 * hue2rgb(var_1, var_2, H))
    B = int(255 * hue2rgb(var_1, var_2, H - (1.0 / 3)))
    return (R, G, B)
