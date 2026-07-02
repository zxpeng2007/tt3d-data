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
    # The per-segment plotting block curve_fits a 3-param polynomial; segments
    # shorter than that crash scipy before the actual reconstruction runs.
    src = src.replace(
        "    for i in range(len(q_sol) - 1):\n"
        "        t_temp = np.linspace(ts_exact[i], ts_exact[i + 1], 10)",
        "    for i in range(len(q_sol) - 1):\n"
        "        if q_sol[i + 1] - q_sol[i] < 4:\n"
        "            continue  # patched: too few points to fit the plot polynomial\n"
        "        t_temp = np.linspace(ts_exact[i], ts_exact[i + 1], 10)",
    )
    if src != orig:
        f.write_text(src, encoding="utf-8")
        print("patched rally.py (fps + int64 index)")
    else:
        print("rally.py already patched / patterns not found")


def patch_table_segmenter() -> None:
    """Fix a loss-branch bug: the checkpoint's loss='DICE' matches the first `if`
    but then falls through a separate `if/elif` chain to `raise ValueError`.
    Making the second block an `elif` unifies them into one chain."""
    f = TT3D / "tt3d" / "calibration" / "table_segmenter.py"
    if not f.exists():
        print(f"skip table_segmenter.py (missing {f})")
        return
    src = f.read_text(encoding="utf-8")
    if '        if loss == "BCE+DICE":' in src:
        src = src.replace('        if loss == "BCE+DICE":',
                          '        elif loss == "BCE+DICE":', 1)
        f.write_text(src, encoding="utf-8")
        print("patched table_segmenter.py (loss if->elif)")
    else:
        print("table_segmenter.py already patched / pattern not found")


def patch_calibration_utils() -> None:
    """save_camcal() declares an 'error' column but, when errors is None, never
    appends to it -> pandas 'All arrays must be of the same length'. Append NaN."""
    f = TT3D / "tt3d" / "calibration" / "utils.py"
    if not f.exists():
        print(f"skip calibration/utils.py (missing {f})")
        return
    src = f.read_text(encoding="utf-8")
    anchor = ('            data["f"].append(f)\n\n'
              "    # Create a DataFrame and save it to a CSV file")
    if anchor in src:
        src = src.replace(
            anchor,
            '            data["f"].append(f)\n            data["error"].append(np.nan)\n\n'
            "    # Create a DataFrame and save it to a CSV file",
            1,
        )
        f.write_text(src, encoding="utf-8")
        print("patched calibration/utils.py (save_camcal error column)")
    else:
        print("calibration/utils.py already patched / pattern not found")


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
    patch_table_segmenter()
    patch_calibration_utils()
    patch_blurball_root()


if __name__ == "__main__":
    main()
