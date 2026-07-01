"""Create the pipeline environment.

Steps (idempotent):
  1. Ensure a Python 3.12 venv at .venv (uses the `py -V:Astral/CPython3.12*`
     launcher or an existing python3.12 on PATH).
  2. Install torch+torchvision for the Blackwell GPU (cu128) via `uv` when
     available (uv caches the 2.7 GB wheel so future runs are instant), else pip.
  3. Install pipeline requirements.txt.
  4. Apply small compatibility patches to the vendored upstream repos.

Run:  python scripts/setup_env.py            (from the repo root, any interpreter)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV = REPO / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def sh(cmd: list[str], **kw) -> int:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw).returncode


def _find_python312() -> list[str]:
    """Return a command that launches CPython 3.12."""
    if os.name == "nt":
        # Prefer the uv-managed Astral CPython, else any 3.12 the launcher knows.
        for tag in ("-V:Astral/CPython3.12.13", "-3.12"):
            try:
                out = subprocess.run(["py", tag, "--version"], capture_output=True, text=True)
                if out.returncode == 0 and "3.12" in out.stdout:
                    return ["py", tag]
            except FileNotFoundError:
                break
    for exe in ("python3.12", "python3", "python"):
        p = shutil.which(exe)
        if p:
            v = subprocess.run([p, "--version"], capture_output=True, text=True)
            if "3.12" in v.stdout:
                return [p]
    raise SystemExit("No Python 3.12 found. Install it (e.g. `uv python install 3.12`).")


def ensure_venv() -> None:
    if VENV_PY.exists():
        print(f"venv exists: {VENV_PY}")
        return
    py312 = _find_python312()
    if sh([*py312, "-m", "venv", str(VENV)]) != 0:
        raise SystemExit("venv creation failed")
    sh([str(VENV_PY), "-m", "pip", "install", "-U", "pip"])


def install_torch() -> None:
    # Already installed?
    check = subprocess.run(
        [str(VENV_PY), "-c", "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        print("torch (cuda) already installed")
        return
    uv = shutil.which("uv")
    if uv:
        rc = sh([uv, "pip", "install", "--python", str(VENV_PY),
                 "torch", "torchvision", "--index-url", TORCH_INDEX])
    else:
        rc = sh([str(VENV_PY), "-m", "pip", "install", "--retries", "15", "--timeout", "300",
                 "torch", "torchvision", "--index-url", TORCH_INDEX])
    if rc != 0:
        raise SystemExit("torch install failed")


def install_requirements() -> None:
    sh([str(VENV_PY), "-m", "pip", "install", "-r", str(REPO / "requirements.txt")])


def apply_patches() -> None:
    patch = REPO / "scripts" / "patch_upstream.py"
    if patch.exists():
        sh([str(VENV_PY), str(patch)])


def main() -> None:
    ensure_venv()
    install_torch()
    install_requirements()
    apply_patches()
    print("\nEnvironment ready. Next: python scripts/download_weights.py")


if __name__ == "__main__":
    main()
