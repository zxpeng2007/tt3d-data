# Build status

Live checklist for the scaled TT3D data pipeline.

| # | Stage | State | Notes |
|---|-------|-------|-------|
| 0 | GitHub repo (`zxpeng2007/tt3d-data`, public) | ✅ done | |
| 1 | Repo scaffold + docs | ✅ done | |
| 2-4 | Env: py3.12 venv, torch cu128 GPU, deps | ✅ done | onnxruntime-gpu CUDA provider |
| 5-7 | Vendored upstream, patches, weights | ✅ done | table-seg, BlurBall, MotionBERT, RTMPose |
| 8 | Pipeline code (all stages) | ✅ done | |
| 9 | Validate — **table** (calibration) | ✅ done | camera.yaml, sensible extrinsics |
| 10 | Validate — **body** (pose→3D→world) | ✅ done | p0_3d/p1_3d (frames,17,3), metric |
| 11 | Validate — **ball** (BlurBall 2D) | ✅ done | ball_traj_2D, ~45% coverage |
| 11b | Ball 3D reconstruction | ⚠️ best-effort | needs clean single-rally clips; non-fatal |
| 12 | Rally segmentation (keyframe gameplay timeline) | ✅ done | Sun match: 911 kf → 502 gameplay → 98 segments |
| 13 | Singles-only + 5h/player download budget | ✅ done | doubles excluded; per-player cap |
| 14 | Bulk match download (WTT/ITTF singles) | 🔄 running | background, capped |
| 15 | Batch generation over rally clips | 🔄 next | resumable, GPU |
| 16 | Dataset aggregation + manifest | ⬜ pending | |

**Validated end-to-end on real WTT footage:** table (`camera.yaml`), 3D body pose
(`p0_3d.npy`/`p1_3d.npy`), ball position (`ball_traj_2D.csv`), plus 3D ball
trajectory on clean rallies. Rally segmentation now splits full matches into
gameplay clips via a fast one-pass keyframe table-presence timeline.

Legend: ✅ done · 🔄 in progress · ⬜ pending · ⚠️ partial
