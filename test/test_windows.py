"""Window/split invariants of gen_data on a small synthetic image: every window lies wholly
in one split, train and val share no pixel, and the summed-area-table counts match a
brute-force scan. Run as `uv run python -m test.test_windows`.
"""

import numpy as np

from dataset.gen_data import block_split_mask, window_corners

SEED = 0
HEIGHT, WIDTH = 37, 41  # not multiples of the block size, so edge blocks are partial
BLOCK_SIZE = 10
PATCH_SIZE = 6
VAL_FRACTION = 0.3
STRIDES = [1, 3, PATCH_SIZE]


def brute_force_corners(val_mask, patch_size, stride):
    height, width = val_mask.shape
    train, val = [], []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            window = val_mask[y:y + patch_size, x:x + patch_size]
            if not window.any():
                train.append((y, x))
            elif window.all():
                val.append((y, x))
    return train, val


def covered_pixels(corners, patch_size, shape):
    covered = np.zeros(shape, dtype=bool)
    for y, x in corners:
        covered[y:y + patch_size, x:x + patch_size] = True
    return covered


def check_stride(val_mask, stride):
    train, val = window_corners(val_mask, PATCH_SIZE, stride)
    expected_train, expected_val = brute_force_corners(val_mask, PATCH_SIZE, stride)
    assert [tuple(corner) for corner in train.tolist()] == expected_train, f"stride {stride}: train corners differ"
    assert [tuple(corner) for corner in val.tolist()] == expected_val, f"stride {stride}: val corners differ"

    train_pixels = covered_pixels(train, PATCH_SIZE, val_mask.shape)
    val_pixels = covered_pixels(val, PATCH_SIZE, val_mask.shape)
    assert not (train_pixels & val_pixels).any(), f"stride {stride}: train and val share pixels"
    assert not (train_pixels & val_mask).any(), f"stride {stride}: a train window touches val"
    assert not (val_pixels & ~val_mask).any(), f"stride {stride}: a val window touches train"
    return len(train), len(val)


def main():
    rng = np.random.default_rng(SEED)
    val_mask = block_split_mask(HEIGHT, WIDTH, BLOCK_SIZE, VAL_FRACTION, rng)
    num_blocks = -(-HEIGHT // BLOCK_SIZE) * -(-WIDTH // BLOCK_SIZE)
    assert 0 < val_mask.sum() < val_mask.size, "val mask should be neither empty nor full"
    print(f"val mask: {int(val_mask.sum())}/{val_mask.size} px over {num_blocks} blocks")
    for stride in STRIDES:
        num_train, num_val = check_stride(val_mask, stride)
        print(f"stride {stride}: train={num_train} val={num_val} OK")
    print("all window invariants hold")


if __name__ == "__main__":
    main()
