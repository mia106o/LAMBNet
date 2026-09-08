"""Verify SOS dataset layout and optionally binarize mask values (0/255 -> 0/1)."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def collect_mask_dirs(data_root: Path):
    dirs = [
        data_root / 'train' / 'palsar',
        data_root / 'train' / 'sentinel',
        data_root / 'test' / 'palsar' / 'gt',
        data_root / 'test' / 'sentinel' / 'gt',
    ]
    return [d for d in dirs if d.is_dir()]


def check_pairing(data_root: Path):
    pairs = [
        ('train/palsar', 'train/palsar'),
        ('train/sentinel', 'train/sentinel'),
        ('test/palsar/sat', 'test/palsar/gt'),
        ('test/sentinel/sat', 'test/sentinel/gt'),
    ]
    img_suffix, mask_suffix = '_sat.jpg', '_mask.png'
    for img_rel, ann_rel in pairs:
        img_dir = data_root / img_rel
        ann_dir = data_root / ann_rel
        if not img_dir.is_dir():
            print(f'[WARN] missing: {img_dir}')
            continue
        imgs = sorted(p.stem.replace('_sat', '') for p in img_dir.glob(f'*{img_suffix}'))
        masks = sorted(p.stem.replace('_mask', '') for p in ann_dir.glob(f'*{mask_suffix}'))
        missing = set(imgs) - set(masks)
        print(f'{img_rel}: {len(imgs)} images, {len(masks)} masks, missing={len(missing)}')
        if missing:
            print(f'  example missing ids: {sorted(missing)[:5]}')


def fix_masks(data_root: Path, dry_run: bool = True):
    for mask_dir in collect_mask_dirs(data_root):
        for mask_path in mask_dir.glob('*_mask.png'):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f'[ERR] cannot read {mask_path}')
                continue
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            uniq = np.unique(mask)
            if mask.ndim == 2 and set(uniq.tolist()).issubset({0, 1}):
                continue
            binary = (mask > 127).astype(np.uint8)
            print(f'fix {mask_path} shape={mask.shape} unique={uniq.tolist()} -> single-channel {{0,1}}')
            if not dry_run:
                cv2.imwrite(str(mask_path), binary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str, help='Path to SOS root (contains train/ and test/)')
    parser.add_argument('--fix-masks', action='store_true', help='Convert 0/255 masks to 0/1')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    assert (data_root / 'train').is_dir(), f'train/ not found under {data_root}'
    assert (data_root / 'test').is_dir(), f'test/ not found under {data_root}'

    print('=== Pairing check ===')
    check_pairing(data_root)
    print('=== Mask value check ===')
    fix_masks(data_root, dry_run=not args.fix_masks)
    if not args.fix_masks:
        print('Run with --fix-masks to binarize 0/255 masks to 0/1.')


if __name__ == '__main__':
    main()
