"""Apply small, idempotent compatibility patches to the vendored upstream repos.

Fixes real issues that block batch use:
  * TT3D rally.py: fps hardcoded to 25, and the 3D index written as uint8
    (`(all_idx*25).astype(np.uint8)`) which OVERFLOWS for rallies longer than
    ~10 s. We make fps read from $TT3D_FPS and widen the index to int64.
  * BlurBall src/configs/global.yaml: WASB_ROOT points at the author's Linux
    path; we set it to the local vendored BlurBall directory.

Safe to run multiple times.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TT3D = REPO / "third_party" / "tt3d"
BLURBALL = REPO / "third_party" / "blurball"


def patch_rally() -> None:
    f = TT3D / "tt3d" / "rally" / "rally.py"
    if not f.exists():
        print(f"skip rally.py (missing {f})")
        return
    src = f.read_text(encoding="utf-8")
    orig = src
    src = src.replace(
        '    fps = 25  # HACK: Change',
        '    import os as _os\n    fps = int(_os.environ.get("TT3D_FPS", 25))  # patched: configurable',
    )
    src = src.replace(
        "(all_idx * 25).astype(np.uint8)",
        "(all_idx * fps).astype(np.int64)",
    )
    if src != orig:
        f.write_text(src, encoding="utf-8")
        print("patched rally.py (fps + int64 index)")
    else:
        print("rally.py already patched / patterns not found")


def patch_blurball_root() -> None:
    f = BLURBALL / "src" / "configs" / "global.yaml"
    if not f.exists():
        print(f"skip blurball global.yaml (missing {f})")
        return
    root = str(BLURBALL.resolve()).replace("\\", "/")
    f.write_text(f"WASB_ROOT: {root}\n", encoding="utf-8")
    print(f"set BlurBall WASB_ROOT -> {root}")


def main() -> None:
    patch_rally()
    patch_blurball_root()


if __name__ == "__main__":
    main()
