# Build status

Live checklist for bringing the scaled TT3D data pipeline online.

| # | Stage | State | Notes |
|---|-------|-------|-------|
| 0 | GitHub repo (`zxpeng2007/tt3d-data`, public) | ✅ done | |
| 1 | Repo scaffold (README, license, docs, .gitignore) | ✅ done | |
| 2 | Python 3.12 venv | ✅ done | `.venv` (CPython 3.12.13) |
| 3 | torch cu128 (Blackwell sm_120) | ✅ done | torch 2.11.0+cu128, GPU verified on RTX 5080 |
| 4 | Pipeline requirements (rtmlib, casadi, smp, …) | 🔄 installing | via uv |
| 5 | Vendored upstream (TT3D, BlurBall, MotionBERT) | ✅ done | cloned into third_party/ |
| 6 | Upstream patches (rally fps/int64, BlurBall root) | ⬜ pending | `scripts/patch_upstream.py` |
| 7 | Model weights (BlurBall, MotionBERT) | ⬜ pending | table seg present; BlurBall/MB download |
| 8 | Pipeline code (all stages + orchestration) | ✅ done | committed |
| 9 | Validate — **table** (calibration) | ⬜ pending | on data/rallies/_demo |
| 10 | Validate — **body** (pose→3D→world) | ⬜ pending | needs MotionBERT weight + a rally |
| 11 | Validate — **ball** (BlurBall→3D) | ⬜ pending | needs BlurBall weight + a rally |
| 12 | Pilot match download (WTT) | 🔄 running | 1 full match, 720p, background |
| 13 | Rally segmentation | ⬜ pending | scene-cut + table presence |
| 14 | Pilot: match → rallies → reconstruction | ⬜ pending | end-to-end proof |
| 15 | Bulk generation (background) | ⬜ pending | target 100+ matches |
| 16 | Dataset aggregation + manifest | ⬜ pending | |

Legend: ✅ done · 🔄 in progress · ⬜ pending · ⚠️ blocked
