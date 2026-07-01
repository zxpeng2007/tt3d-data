# Setup

Verified on Windows 11, NVIDIA RTX 5080 (Blackwell, sm_120), CUDA driver 13.1.

## 1. Python 3.12 environment

The ML stack (MotionBERT, TT3D, table segmenter) targets Python ≤3.12; 3.13/3.14
are too new. `scripts/setup_env.py` creates an isolated `.venv` using a Python 3.12
interpreter (the `py -V:Astral/CPython3.12*` launcher or any `python3.12` on PATH).

```bash
python scripts/setup_env.py
```

This will:
1. Create `.venv` (Python 3.12).
2. Install **torch 2.11 + torchvision (cu128)** — the Blackwell GPU needs a
   cu128 build; older torch (≤2.6) lacks sm_120 kernels. Installed via `uv` when
   available so the 2.75 GB wheel is cached for future reuse, else pip with retries.
3. Install `requirements.txt`.
4. Apply upstream compatibility patches (`scripts/patch_upstream.py`).

Verify the GPU:
```bash
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# torch 2.11.0+cu128 True NVIDIA GeForce RTX 5080 Laptop GPU
```

### 2D pose backend
We use **rtmlib** (RTMDet + RTMPose over ONNXRuntime) instead of the full
mmpose/mmcv stack — mmcv is painful to build on new Python/torch/Blackwell. rtmlib
auto-downloads its ONNX models on first inference. If the CUDA ONNXRuntime provider
is unavailable, it falls back to CPU (slower but functional).

## 2. Model weights

```bash
python scripts/download_weights.py
```

| Weight | Source | Path |
| --- | --- | --- |
| Table segmentation | ships with upstream TT3D | `third_party/tt3d/weights/table_segmentation.ckpt` |
| BlurBall | [Nextcloud](https://cloud.cs.uni-tuebingen.de/index.php/s/6Z8TpM3sXRKHzGC) | `weights/blurball/blurball.pth.tar` |
| MotionBERT (lite) | [MotionBERT model zoo](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md) (OneDrive) | `third_party/MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin` |
| RTMPose/RTMDet | auto (rtmlib) | rtmlib cache |

BlurBall and MotionBERT may need a manual download step (Nextcloud / OneDrive) —
the script prints exact instructions and accepts `BLURBALL_URL` / `MB_CKPT_URL`
env overrides for direct links.

## 3. Validate end-to-end

```bash
# Table stage on the bundled calibration clip
python scripts/run_pipeline.py --rally-dir data/rallies/_demo --stages table
```

## 4. Generate data at scale

```bash
python scripts/download_videos.py --config configs/sources.yaml --limit 100
python scripts/segment_rallies.py --videos data/videos --out data/rallies
python scripts/batch_generate.py --rallies data/rallies --out data/dataset
python scripts/aggregate_dataset.py --dataset data/dataset
```

## Notes / gotchas
- Upstream `rally.py` hardcodes 25 fps and writes the 3D index as `uint8`
  (overflows past ~10 s). `patch_upstream.py` makes fps configurable (`$TT3D_FPS`)
  and widens the index. All clips are normalized to 25 fps canonically.
- `rally.py`, `filter.py`, `align.py` call matplotlib; the pipeline runs them with
  `MPLBACKEND=Agg` so `plt.show()` is a no-op (non-blocking batch).
- Every rally is reconstructed on a single de-duplicated canonical video so ball,
  pose and camera share one frame index.
