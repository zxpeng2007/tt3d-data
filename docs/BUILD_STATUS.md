# Build status

Live checklist for bringing the scaled TT3D data pipeline online. Updated as stages are validated.

| # | Stage | State | Notes |
|---|-------|-------|-------|
| 0 | GitHub repo created (`zxpeng2007/tt3d-data`, public) | ✅ done | |
| 1 | Repo scaffold (README, license, docs, .gitignore) | ✅ done | |
| 2 | Python 3.12 venv | 🔄 in progress | Blackwell sm_120 → needs torch ≥ cu128 build |
| 3 | ML stack install (torch, rtmlib, casadi, seg-models…) | ⬜ pending | recipe from research workflow |
| 4 | Model weights (table seg, BlurBall, RTMPose, MotionBERT) | ⬜ pending | |
| 5 | Vendored upstream (TT3D, BlurBall, MotionBERT) | ⬜ pending | |
| 6 | Stage validation on sample rally — **table** | ⬜ pending | |
| 7 | Stage validation on sample rally — **body** | ⬜ pending | |
| 8 | Stage validation on sample rally — **ball** | ⬜ pending | |
| 9 | Video download (WTT/ITTF via yt-dlp) | ⬜ pending | |
| 10 | Rally segmentation | ⬜ pending | |
| 11 | Pilot: 1 match → rallies → full reconstruction | ⬜ pending | end-to-end proof |
| 12 | Batch orchestrator (resumable) | ⬜ pending | |
| 13 | Bulk generation started (background) | ⬜ pending | target 100+ matches |
| 14 | Dataset aggregation + manifest | ⬜ pending | |

Legend: ✅ done · 🔄 in progress · ⬜ pending · ⚠️ blocked
