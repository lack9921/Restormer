#!/usr/bin/env python3
"""
Split LoViF dataset into train/val sets.
Moves last N images per track from Train/ to Test/ for validation.

Usage:
  python split_lovif_val.py --train_dir /path/to/Train --val_size 50

Structure:
  Train/Blur/GT/{0001..4900}.jpg  →  Train/Blur/GT/{0001..4850}.jpg
                                  →  Test/Blur/GT/{4851..4900}.jpg
"""

import os
import shutil
import argparse
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Split LoViF dataset into train/val')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to Train directory (e.g., /media/wwl/.../lyx/Train)')
    parser.add_argument('--val_size', type=int, default=50,
                        help='Number of validation images per track (default: 50)')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Output test dir (default: <train_dir>_val/Test)')
    parser.add_argument('--move', action='store_true',
                        help='Move instead of copy (default: copy)')
    return parser.parse_args()


def main():
    args = parse_args()
    tracks = ['Blur', 'Haze', 'Lowlight', 'Rain', 'Snow']
    test_dir = args.test_dir or os.path.join(os.path.dirname(args.train_dir), 'Test')

    for track in tracks:
        gt_dir = os.path.join(args.train_dir, track, 'GT')
        lq_dir = os.path.join(args.train_dir, track, 'LQ')

        # Check source dirs exist
        if not os.path.isdir(gt_dir) or not os.path.isdir(lq_dir):
            print(f'[SKIP] {track}: source not found')
            continue

        # Get sorted files
        gt_files = sorted(os.listdir(gt_dir))
        lq_files = sorted(os.listdir(lq_dir))

        if not gt_files:
            print(f'[SKIP] {track}: no files found')
            continue

        val_gt_files = gt_files[-args.val_size:]
        val_lq_files = lq_files[-args.val_size:]

        # Create output dirs
        out_gt = os.path.join(test_dir, track, 'GT')
        out_lq = os.path.join(test_dir, track, 'LQ')
        os.makedirs(out_gt, exist_ok=True)
        os.makedirs(out_lq, exist_ok=True)

        # Copy/move files
        fn = shutil.move if args.move else shutil.copy2
        for src_name in val_gt_files:
            fn(os.path.join(gt_dir, src_name), os.path.join(out_gt, src_name))
        for src_name in val_lq_files:
            fn(os.path.join(lq_dir, src_name), os.path.join(out_lq, src_name))

        print(f'[{track}] Moved {len(val_gt_files)} val images → {out_gt}')

    print(f'\nDone! Test data saved to: {test_dir}')
    print(f'\nFor Restormer:')
    print(f'  python basicsr/train.py -opt LoViF_Restoration/Options/LoViF_Restoration_Restormer_workstation.yml')
    print(f'\nFor AdaIR (--elvis_val_last_n={args.val_size}):')
    print(f'  python train.py --elvis_mode --elvis_train_dir {args.train_dir}')


if __name__ == '__main__':
    main()
