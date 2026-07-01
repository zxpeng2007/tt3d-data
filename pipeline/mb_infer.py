"""Standalone MotionBERT 2D->3D inference (batch-friendly).

Mirrors MotionBERT/infer_wild.py but: runs both players in one process, uses
num_workers=0 (Windows-safe), and skips the slow X3D.mp4 render. Must be launched
with cwd = the MotionBERT repo (so `configs/...` resolves) and that repo on
PYTHONPATH (so `import lib...` resolves). Writes <out_dir>/player_<focus>.npy,
each shape (n_frames, 17, 3) in the camera frame.
"""
import argparse
import os
import sys
from pathlib import Path

# Make MotionBERT's `lib` importable regardless of how we were launched.
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--players", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--clip_len", type=int, default=243)
    opts = ap.parse_args()

    args = get_config(opts.config)
    model = load_backbone(args)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = nn.DataParallel(model).cuda()
    ckpt = torch.load(opts.ckpt, map_location=lambda s, l: s, weights_only=False)
    model.load_state_dict(ckpt["model_pos"], strict=True)
    model.eval()

    out_dir = Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for focus in opts.players:
        ds = WildDetDataset(opts.json, clip_len=opts.clip_len, scale_range=[1, 1], focus=focus)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_cuda)
        results = []
        with torch.no_grad():
            for batch_input in loader:
                if use_cuda:
                    batch_input = batch_input.cuda()
                if args.no_conf:
                    batch_input = batch_input[:, :, :, :2]
                if args.flip:
                    bi_flip = flip_data(batch_input)
                    p1 = model(batch_input)
                    p2 = flip_data(model(bi_flip))
                    pred = (p1 + p2) / 2.0
                else:
                    pred = model(batch_input)
                if args.rootrel:
                    pred[:, :, 0, :] = 0
                else:
                    pred[:, 0, 0, 2] = 0
                if args.gt_2d:
                    pred[..., :2] = batch_input[..., :2]
                results.append(pred.cpu().numpy())
        if not results:
            print(f"[mb_infer] WARNING: no frames for player {focus}")
            np.save(out_dir / f"player_{focus}.npy", np.zeros((0, 17, 3), dtype=np.float32))
            continue
        arr = np.concatenate(np.hstack(results))  # (n_frames, 17, 3)
        np.save(out_dir / f"player_{focus}.npy", arr)
        print(f"[mb_infer] player {focus}: {arr.shape}")


if __name__ == "__main__":
    main()
