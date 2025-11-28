# tests/test_image_compare.py
import os
import subprocess
from pathlib import Path
import pytest
from matplotlib.testing.compare import compare_images

# Optional: generate a diff image (requires Pillow + numpy)
def _save_diff(expected_path, actual_path, diff_out_path):
    try:
        from PIL import Image
        import numpy as np
        e = Image.open(expected_path).convert("RGBA")
        a = Image.open(actual_path).convert("RGBA")
        # Align sizes if needed
        if e.size != a.size:
            # pad/truncate the smaller to match larger; here we simply return without diff
            return
        e_arr = np.asarray(e, dtype=np.int16)
        a_arr = np.asarray(a, dtype=np.int16)
        diff = np.clip(np.abs(a_arr - e_arr), 0, 255).astype("uint8")
        Image.fromarray(diff).save(diff_out_path)
    except Exception:
        pass  # diff generation is best-effort; failures won't block tests

@pytest.fixture(scope="session", autouse=True)
def execute_notebooks():
    """
    Execute all.ipynb at repo root (or extend to subfolders) so figures are saved to results/.
    Runs once per test session.
    """
    Path("results").mkdir(exist_ok=True)
    # Find notebooks — adjust patterns as needed
    notebooks = []
    for pattern in ["*.ipynb", "notebooks/*.ipynb"]:
        notebooks.extend(Path(".").glob(pattern))

    if not notebooks:
        return

    # Execute each notebook in place, with a reasonable timeout
    for nb in notebooks:
        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=300",
            str(nb)
        ], check=True)

def discover_baselines():
    """Return a list of baseline PNG filenames (relative names only)."""
    base_dir = Path("baseline")
    if not base_dir.exists():
        return []
    return sorted([p.name for p in base_dir.glob("*.png")])

# Allow tolerance to be set via env var (default 0.0 for pixel-perfect)
TOL = float(os.environ.get("IMG_TOL", "0.0"))

@pytest.mark.parametrize("png_name", discover_baselines())
def test_png_matches_baseline(png_name):
    expected = Path("baseline") / png_name
    actual = Path("results") / png_name

    assert expected.exists(), f"Missing baseline file: {expected}"
    assert actual.exists(), f"Notebook did not produce: {actual}"

    err = compare_images(str(expected), str(actual), tol=TOL)

    if err is not None:
        # Save a visual diff to help debugging
        diffs_dir = Path("results") / "diffs"
        diffs_dir.mkdir(parents=True, exist_ok=True)
        diff_out = diffs_dir / f"{png_name.replace('.png', '')}-diff.png"
        _save_diff(expected, actual, diff_out)
        pytest.fail(f"Image mismatch for {png_name} (RMS tol={TOL}). Details: {err}")
