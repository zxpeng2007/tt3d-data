# tt3d-data

**Scaled 3D table-tennis data generation.** This repo turns large amounts of real table-tennis
match footage into a training dataset of **ball position**, **human body pose**, and **table
position** in 3D, by running the [TT3D](https://github.com/cogsys-tuebingen/tt3d) reconstruction
method (Gossard, Ziegler & Zell, *TT3D: Table Tennis 3D Reconstruction*, CVPR-W 2025) over many
broadcast matches instead of a single example rally.

Where upstream TT3D ships a one-rally demo, this repo adds the missing pieces to run it **at scale**:

1. **Source** — bulk-download full WTT/ITTF broadcast matches with `yt-dlp`.
2. **Segment** — automatically split each match into individual **rally clips** (gameplay-only).
3. **Reconstruct** — run the full TT3D pipeline per rally:
   - **Table** — table segmentation + monocular camera calibration.
   - **Body** — RTMPose (2D) → MotionBERT (2D→3D) → world-frame alignment, per player.
   - **Ball** — [BlurBall](https://github.com/cogsys-tuebingen/blurball) 2D detection → physics-based
     3D trajectory reconstruction.
4. **Aggregate** — collect every rally's outputs into one versioned dataset with a manifest.

> ⚠️ **Footage is copyrighted.** Downloaded videos are used only as private inputs for research
> training data and are **git-ignored** — this repo never redistributes broadcast footage. Only the
> code and (optionally) derived numeric annotations are versioned.

---

## What each rally produces

For every reconstructed rally you get (see [`docs/DATASET_SCHEMA.md`](docs/DATASET_SCHEMA.md)):

| File | Contents |
| --- | --- |
| `camera.yaml` | Calibrated camera intrinsics + extrinsics (table→camera) |
| `ball_traj_2D.csv` | Per-frame 2D ball detections (+ motion-blur cue) from BlurBall |
| `ball_traj_3D.csv` | Reconstructed 3D ball trajectory (position, velocity, spin) |
| `p0_3d.npy`, `p1_3d.npy` | 3D body pose (17 joints) per player, in the table/world frame |
| `meta.json` | Source video id, rally span, fps, quality flags |

---

## Status

This repo is under active build-out. See [`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md) for the current
state of each stage (environment, weights, per-stage validation, bulk run).

## Setup

Full, machine-specific setup lives in [`docs/SETUP.md`](docs/SETUP.md). In short:

```bash
# 1. Create the isolated Python 3.12 environment and install the stack
python scripts/setup_env.py

# 2. Fetch upstream method repos (TT3D, BlurBall, MotionBERT) and model weights
python scripts/download_weights.py

# 3. Validate the pipeline end-to-end on the bundled sample rally
python scripts/run_pipeline.py --rally-dir data/dataset/samples/demo_rally --stages all
```

## Generating data at scale

```bash
# Download N full matches from the configured WTT/ITTF sources
python scripts/download_videos.py --config configs/sources.yaml --limit 100

# Split every downloaded match into rally clips
python scripts/segment_rallies.py --videos data/videos --out data/rallies

# Run the full reconstruction over every rally (resumable, GPU)
python scripts/batch_generate.py --rallies data/rallies --out data/dataset

# Build the consolidated dataset + manifest
python scripts/aggregate_dataset.py --dataset data/dataset
```

## Attribution

This project is a data-scaling wrapper around research by others. Please cite the original work:

```bibtex
@InProceedings{gossard2025,
  author    = {Gossard, Thomas and Ziegler, Andreas and Zell, Andreas},
  title     = {TT3D: Table Tennis 3D Reconstruction},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2025}
}
```

Upstream components: [TT3D](https://github.com/cogsys-tuebingen/tt3d),
[BlurBall](https://github.com/cogsys-tuebingen/blurball),
[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose),
[MotionBERT](https://github.com/Walter0807/MotionBERT). Each retains its own license; see
[`third_party/README.md`](third_party/README.md).
