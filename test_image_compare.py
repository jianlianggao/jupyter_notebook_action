# tests/test_image_compare.py
import os
import subprocess
from pathlib import Path
import hashlib
import pytest
from PIL import Image
from matplotlib.testing.compare import compare_images
import numpy as np

# Tolerances: images and numeric
IMG_TOL = float(os.environ.get("IMG_TOL", "0.0"))          # image RMS tolerance
XY_ABS_TOL = float(os.environ.get("XY_ABS_TOL", "0.0"))    # absolute tolerance for numeric
XY_REL_TOL = float(os.environ.get("XY_REL_TOL", "0.0"))    # relative tolerance for numeric

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@pytest.fixture(scope="session", autouse=True)
def execute_notebooks():
    Path("results").mkdir(exist_ok=True)
    notebooks = sorted([p for p in Path(".").glob("*.ipynb")])
    for nb in notebooks:
        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=300",
            str(nb)
        ], check=True)

# ---------- Image comparisons ----------
def discover_baseline_pngs():
    base_dir = Path("baseline")
    return sorted(p.name for p in base_dir.glob("*.png")) if base_dir.exists() else []

@pytest.mark.parametrize("png_name", discover_baseline_pngs())
def test_png_matches_baseline(png_name):
    expected = Path("baseline") / png_name
    actual = Path("results") / png_name

    assert expected.exists(), f"Missing baseline: {expected}"
    assert actual.exists(), f"No output from notebook: {actual}"

    # Print hashes for debugging
    print(f"Comparing image {png_name}")
    print(f"  baseline: {expected} sha256={sha256(expected)}")
    print(f"  actual:   {actual}   sha256={sha256(actual)}")

    ew, eh = Image.open(expected).size
    aw, ah = Image.open(actual).size
    assert (ew, eh) == (aw, ah), (
        f"Image sizes differ for {png_name}: baseline={(ew, eh)}, actual={(aw, ah)}"
    )

    err = compare_images(str(expected), str(actual), tol=IMG_TOL)
    assert err is None, f"Image mismatch for {png_name} (tol={IMG_TOL}). Details: {err}"

# ---------- TXT (x,y) comparisons ----------
def discover_baseline_txts():
    base_dir = Path("baseline")
    return sorted(p.name for p in base_dir.glob("*_xy.txt")) if base_dir.exists() else []

@pytest.mark.parametrize("txt_name", discover_baseline_txts())
def test_xy_matches_baseline(txt_name):
    expected = Path("baseline") / txt_name
    actual = Path("results") / txt_name

    assert expected.exists(), f"Missing baseline TXT: {expected}"
    assert actual.exists(), f"No TXT output from notebook: {actual}"

    # Load numeric arrays (ignore header), robust to tabs/spaces
    exp = np.loadtxt(expected, delimiter="\t")   # shape: (N, 2)
    act = np.loadtxt(actual, delimiter="\t")     # shape: (N, 2)

    assert exp.shape == act.shape, (
        f"Shape mismatch for {txt_name}: baseline={exp.shape}, actual={act.shape}"
    )

    # Element-wise tolerance check:
    # |diff| <= XY_ABS_TOL + XY_REL_TOL * |baseline|
    diff = np.abs(act - exp)
    tol = XY_ABS_TOL + XY_REL_TOL * np.abs(exp)
    ok = (diff <= tol)

    if not np.all(ok):
        # Report max diff and first failing index for clarity
        max_diff = float(diff.max())
        idx = np.argwhere(~ok)[0]
        b_val = float(exp[tuple(idx)])
        a_val = float(act[tuple(idx)])
        d_val = float(diff[tuple(idx)])
        t_val = float(tol[tuple(idx)])
        pytest.fail(
            f"Numeric mismatch in {txt_name}: max_diff={max_diff} "
            f"at index {tuple(idx)} (baseline={b_val}, actual={a_val}, "
            f"diff={d_val}, tol={t_val}, abs_tol={XY_ABS_TOL}, rel_tol={XY_REL_TOL})"
        )