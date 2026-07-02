"""Central paths and constants for the tt3d-data pipeline.

All paths resolve relative to the repository root so the pipeline is portable.
Override any of these via environment variables (TT3D_* / see below) or by editing
configs/pipeline.yaml, which `load_config()` overlays on top of these defaults.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- Repository layout -------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "third_party"
TT3D_DIR = THIRD_PARTY / "tt3d"
BLURBALL_DIR = THIRD_PARTY / "blurball"
MOTIONBERT_DIR = THIRD_PARTY / "MotionBERT"

WEIGHTS_DIR = REPO_ROOT / "weights"
DATA_DIR = REPO_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
RALLIES_DIR = DATA_DIR / "rallies"
DATASET_DIR = DATA_DIR / "dataset"
LOGS_DIR = REPO_ROOT / "logs"

# Python interpreter of the pipeline venv (used to launch upstream scripts as
# subprocesses with the correct working directory / sys.path).
VENV_PY = REPO_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PYTHON = str(VENV_PY if VENV_PY.exists() else Path(sys.executable))

# --- Model weight locations --------------------------------------------------
# Table segmentation ships inside upstream TT3D.
TABLE_SEG_CKPT = TT3D_DIR / "weights" / "table_segmentation.ckpt"
# BlurBall model (downloaded from the BlurBall Nextcloud into weights/).
BLURBALL_CKPT = WEIGHTS_DIR / "blurball" / "blurball_best"
# MotionBERT lite checkpoint + config (downloaded into MotionBERT/checkpoint).
MOTIONBERT_CONFIG = MOTIONBERT_DIR / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml"
MOTIONBERT_CKPT = (
    MOTIONBERT_DIR / "checkpoint" / "pose3d"
    / "FT_MB_lite_MB_ft_h36m_global_lite" / "best_epoch.bin"
)

# --- Canonical video parameters ---------------------------------------------
# The upstream TT3D calibration/pose/ball code assumes 25 fps in several places
# (rally.py fps=25, align.py 1/25). We therefore normalize every rally clip to a
# single canonical fps and de-duplicate frames so ball / pose / camera indices
# align 1:1 across all stages.
CANONICAL_FPS = 25
MIN_RALLY_SECONDS = 1.5     # drop clips shorter than this
MAX_RALLY_SECONDS = 40.0    # split/skip anything implausibly long for one rally

# --- ITTF regulation table (metres) -----------------------------------------
# Table is the world/origin frame; camera.yaml stores the camera pose relative
# to it. Coordinates follow TT3D's convention (long axis = Y).
TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
TABLE_HEIGHT = 0.76        # playing surface height above floor
NET_HEIGHT = 0.1525


@dataclass
class PipelineConfig:
    """Runtime knobs, overlaid from configs/pipeline.yaml when present."""
    canonical_fps: int = CANONICAL_FPS
    min_rally_seconds: float = MIN_RALLY_SECONDS
    max_rally_seconds: float = MAX_RALLY_SECONDS

    # BlurBall inference.
    # score_threshold 0.5 (not the README's 0.7): fast balls blur into weak
    # heatmap peaks, and 0.7 drops exactly those; the tracker gate + physics
    # fit reject the extra false positives. max_disp raised for 25fps clips
    # (per-frame displacement is 2x the native 50fps broadcast).
    blurball_step: int = 1
    blurball_score_threshold: float = 0.5
    blurball_max_disp: int = 600
    # bridge detection gaps up to this many missing frames with a local
    # quadratic fit (marked Interp=1 in ball_traj_2D.csv)
    ball_bridge_max_gap: int = 5

    # 3D ball refinement: reject reconstructed points that stray from the
    # (trusted, smooth) 2D track, cap physically impossible speeds, and
    # Savitzky-Golay smooth each flight arc.
    ball3d_max_reproj_px: float = 40.0
    ball3d_max_speed: float = 40.0      # m/s; fastest recorded smashes ~32

    # rtmlib 2D pose
    rtmpose_mode: str = "balanced"      # 'performance' | 'balanced' | 'lightweight'
    det_score_threshold: float = 0.45
    min_player_track_frames: int = 15   # a valid player must persist this long

    # Rally segmentation thresholds
    table_presence_min: float = 0.02    # min table-mask frame fraction for gameplay
    table_presence_max: float = 0.15    # above this it's a close-up/replay, not the wide shot
    gameplay_min_seconds: float = 2.0
    gameplay_merge_gap: float = 5.0     # bridge cutaways up to this many seconds

    # Quality gates for keeping a reconstructed rally
    min_ball_coverage: float = 0.25     # fraction of frames with a ball detection
    require_two_players: bool = True

    # Rendering (expensive) — off by default for bulk generation
    render_debug_video: bool = False

    extra: dict = field(default_factory=dict)


def load_config(path: Path | None = None) -> PipelineConfig:
    """Load PipelineConfig, overlaying configs/pipeline.yaml if it exists."""
    cfg = PipelineConfig()
    path = path or (REPO_ROOT / "configs" / "pipeline.yaml")
    if path.exists():
        import yaml  # local import; PyYAML is a light dependency
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                cfg.extra[key] = value
    return cfg


def ensure_dirs() -> None:
    for d in (WEIGHTS_DIR, DATA_DIR, VIDEOS_DIR, RALLIES_DIR, DATASET_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
