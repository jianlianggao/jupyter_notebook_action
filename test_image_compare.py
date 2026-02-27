# tests/test_image_compare.py
import os
import subprocess
from pathlib import Path
import hashlib
import pytest
from PIL import Image
from matplotlib.testing.compare import compare_images
import numpy as np

# Root anchored to current working directory; override with REPO_ROOT if needed
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path.cwd())).resolve()
BASELINE_DIR = REPO_ROOT / "baseline"
RESULTS_DIR = REPO_ROOT / "results"

# Baseline stems (single expected for many actuals)
BASELINE_IMAGE = BASELINE_DIR / "test000.png"
# Support either baseline/test000.txt or baseline/test000_xy.txt
BASELINE_TXT = (BASELINE_DIR / "test000_xy.txt") if (BASELINE_DIR / "test000_xy.txt").exists() else (BASELINE_DIR / "test000_xy.txt")

# Tolerances
IMG_TOL = float(os.environ.get("IMG_TOL", "0.0"))
XY_ABS_TOL = float(os.environ.get("XY_ABS_TOL", "0.0"))
XY_REL_TOL = float(os.environ.get("XY_REL_TOL", "0.0"))

print(f"[DEBUG] REPO_ROOT={REPO_ROOT}")
print(f"[DEBUG] IMG_TOL={IMG_TOL} XY_ABS_TOL={XY_ABS_TOL} XY_REL_TOL={XY_REL_TOL}")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# Optional: execute notebooks here. Remove this fixture if you execute notebooks in your workflow step.
@pytest.fixture(scope="session", autouse=True)
def execute_notebooks():
    RESULTS_DIR.mkdir(exist_ok=True)
    notebooks = sorted(REPO_ROOT.glob("*.ipynb"))
    nb_dir = REPO_ROOT / "notebooks"
    if nb_dir.exists():
        notebooks += [p for p in nb_dir.rglob("*.ipynb") if ".ipynb_checkpoints" not in str(p)]
    notebooks = [p for p in notebooks if p.name not in ("baseline.ipynb",) and "tests" not in p.parts]
    print("[DEBUG] Executing notebooks:", [str(p) for p in notebooks])
    for nb in notebooks:
        env = os.environ.copy()
        env.setdefault("NOTEBOOK_STEM", nb.stem)  # notebooks can use this for filenames
        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=300",
            str(nb)
        ], check=True, env=env, cwd=REPO_ROOT)
    print("[DEBUG] Results:", [p.name for p in RESULTS_DIR.glob("*")])

def _load_xy(path: Path):
    # Robust loader: works with commented headers or plain headers
    try:
        return np.loadtxt(path, delimiter="\t")
    except ValueError:
        return np.loadtxt(path, delimiter="\t", comments=None, skiprows=1)

def test_baseline_presence():
    #assert BASELINE_IMAGE.exists(), f"Missing baseline image: {BASELINE_IMAGE}"
    assert BASELINE_TXT.exists(), f"Missing baseline TXT: {BASELINE_TXT}"
'''
# ---------- Image comparisons: compare each actual to the single baseline ----------
@pytest.mark.parametrize("actual_img", sorted(RESULTS_DIR.glob("test*.png")))
def test_each_image_against_baseline(actual_img: Path):
    expected = BASELINE_IMAGE
    actual = actual_img

    assert expected.exists(), f"Missing baseline: {expected}"
    assert actual.exists(), f"No output image: {actual}"

    print(f"Comparing image {actual.name} -> {expected.name}")
    print(f"  baseline sha256={sha256(expected)}")
    print(f"  actual   sha256={sha256(actual)}")

    ew, eh = Image.open(expected).size
    aw, ah = Image.open(actual).size
    assert (ew, eh) == (aw, ah), (
        f"Size mismatch: baseline={(ew, eh)}, actual={(aw, ah)} for {actual.name}"
    )

    err = compare_images(str(expected), str(actual), tol=IMG_TOL)
    assert err is None, f"Image mismatch for {actual.name} (tol={IMG_TOL}). Details: {err}"
'''
# ---------- TXT comparisons: compare each actual to the single baseline ----------
@pytest.mark.parametrize("actual_txt", sorted(RESULTS_DIR.glob("test*.txt")))
def test_each_txt_against_baseline(actual_txt: Path):
    expected = BASELINE_TXT
    actual = actual_txt

    assert expected.exists(), f"Missing baseline TXT: {expected}"
    assert actual.exists(), f"No output TXT: {actual}"

    exp = _load_xy(expected)
    act = _load_xy(actual)

    assert exp.shape == act.shape, (
        f"Shape mismatch ({actual.name} vs {expected.name}): baseline={exp.shape}, actual={act.shape}"
    )

    # |diff| <= XY_ABS_TOL + XY_REL_TOL * |baseline|
    diff = np.abs(act - exp)
    tol = XY_ABS_TOL + XY_REL_TOL * np.abs(exp)
    ok = (diff <= tol)

    if not np.all(ok):
        max_diff = float(diff.max())
        idx = tuple(np.argwhere(~ok)[0])
        b_val = float(exp[idx])
        a_val = float(act[idx])
        d_val = float(diff[idx])
        t_val = float(tol[idx])
        pytest.fail(
            f"Numeric mismatch in {actual.name}: max_diff={max_diff} at index {idx} "
            f"(baseline={b_val}, actual={a_val}, diff={d_val}, tol={t_val}, "
            f"abs_tol={XY_ABS_TOL}, rel_tol={XY_REL_TOL})"
        )
    else:
        print(f"[OK] Numeric match: {actual.name} vs {expected.name}")
