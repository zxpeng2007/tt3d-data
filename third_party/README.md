# Third-party components

This project orchestrates several upstream research repositories. They are **not** committed here;
`scripts/download_weights.py` / `scripts/setup_env.py` clone them into this folder at pinned commits.
Each retains its own license and citation requirements.

| Component | Purpose in our pipeline | Upstream |
| --- | --- | --- |
| **TT3D** | Camera calibration, pose alignment, 3D ball reconstruction, rendering | https://github.com/cogsys-tuebingen/tt3d |
| **BlurBall** | 2D table-tennis ball detection with motion-blur cue | https://github.com/cogsys-tuebingen/blurball |
| **MotionBERT** | 2D→3D human pose lifting | https://github.com/Walter0807/MotionBERT |
| **RTMPose** (via `rtmlib`) | 2D human pose (COCO-17 keypoints) | https://github.com/open-mmlab/mmpose |

Pinned commits are recorded in `configs/third_party.lock` after `setup_env.py` runs.

Downloaded broadcast footage (WTT/ITTF) is copyrighted by the respective rights holders and is used
here solely as private input for research training data. It is git-ignored and never redistributed.
