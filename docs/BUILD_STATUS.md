# Build status

The pipeline is validated end-to-end on real WTT footage and generating data.

| # | Stage | State | Notes |
|---|-------|-------|-------|
| 0-1 | GitHub repo (`zxpeng2007/tt3d-data`) + scaffold | ✅ done | |
| 2-4 | Env: py3.12 venv, torch cu128 GPU, deps | ✅ done | onnxruntime-gpu CUDA provider |
| 5-7 | Vendored upstream, patches, weights | ✅ done | table-seg, BlurBall, MotionBERT, RTMPose |
| 8 | Pipeline code (all stages) | ✅ done | |
| 9 | **Table** (calibration) | ✅ validated | camera.yaml, sensible extrinsics |
| 10 | **Body** (pose→3D→world) | ✅ validated | p0_3d/p1_3d (frames,17,3), metric |
| 11 | **Ball** 2D (BlurBall) | ✅ validated | up to 92% coverage on clean rallies |
| 11b | **Ball** 3D trajectory | ✅ validated | succeeds on clean single rallies (r0001) |
| 12 | Rally segmentation (keyframe timeline) | ✅ validated | Sun match: 911 kf → 502 gameplay → 98 clips |
| 13 | Singles-only + 5 h/player download budget | ✅ done | doubles excluded; per-player cap verified |
| 14 | Bulk match download (WTT/ITTF singles) | 🔄 running | background, capped at 5 h/player |
| 15 | Batch generation over rally clips | 🔄 running | 98 clips, ~2 min/clip, resumable |
| 16 | Dataset aggregation + manifest | ✅ working | manifest.parquet + dataset_card.json |

## Proof: one fully reconstructed rally (KCGdLksBH7s/r0001)
- `camera.yaml` — calibrated camera (table = world origin)
- `p0_3d.npy`, `p1_3d.npy` — both players, (97, 17, 3), metric world coords
- `ball_traj_2D.csv` — 92% ball coverage (pixel + motion blur)
- `ball_traj_3D.csv` — 3D trajectory: ball crosses table (y −2.85→+0.2 m), descends (z 0.38→0.03 m)

## To extend the dataset
```bash
python scripts/run_bulk.py            # download → segment → reconstruct → aggregate
```
All stages are resumable; re-run as more matches download.

Legend: ✅ done · 🔄 in progress
