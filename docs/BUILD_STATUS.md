# Build status

Live checklist for the scaled TT3D data pipeline.

| # | Stage | State | Notes |
|---|-------|-------|-------|
| 0 | GitHub repo (`zxpeng2007/tt3d-data`, public) | ✅ done | |
| 1 | Repo scaffold + docs | ✅ done | |
| 2 | Python 3.12 venv | ✅ done | |
| 3 | torch cu128 (Blackwell sm_120) | ✅ done | torch 2.11.0+cu128, GPU verified |
| 4 | Pipeline deps (rtmlib, casadi, smp 0.3.3, …) | ✅ done | onnxruntime-gpu w/ CUDA provider |
| 5 | Vendored upstream (TT3D, BlurBall, MotionBERT) | ✅ done | |
| 6 | Upstream patches | ✅ done | rally fps/int64, segmenter loss, save_camcal, WASB_ROOT |
| 7 | Model weights | ✅ done | table-seg, BlurBall (WebDAV), MotionBERT (HF), RTMPose (auto) |
| 8 | Pipeline code (all stages) | ✅ done | |
| 9 | Validate — **table** (calibration) | ✅ done | camera.yaml, sensible extrinsics |
| 10 | Validate — **body** (pose→3D→world) | ✅ done | p0_3d/p1_3d (frames,17,3), metric scale |
| 11 | Validate — **ball** (BlurBall 2D) | ✅ done | ball_traj_2D, ~45% coverage on pilot |
| 11b | Ball 3D reconstruction | ⚠️ best-effort | needs clean single-rally clips; non-fatal |
| 12 | Pilot match download (WTT/ITTF) | ✅ done | + 12 singles matches available |
| 13 | Rally segmentation | 🔄 validating | scene-cut + table-presence |
| 14 | Bulk match download | 🔄 running | singles, background, slow link |
| 15 | Batch generation | ⬜ pending | resumable, GPU |
| 16 | Dataset aggregation + manifest | ⬜ pending | |

**Validated end-to-end on real WTT footage:** a rally reconstructs to table
(`camera.yaml`), 3D body pose (`p0_3d.npy`/`p1_3d.npy`), and ball position
(`ball_traj_2D.csv`), with 3D ball trajectory when the clip is a clean rally.

Legend: ✅ done · 🔄 in progress · ⬜ pending · ⚠️ partial
