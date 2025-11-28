# tests/test_image_compare.py
import os
import subprocess
from pathlib import Path
import pytest
from PIL import Image
from matplotlib.testing.compare import compare_images

TOL = float(os.environ.get("IMG_TOL", "0.0"))

@pytest.fixture(scope="session", autouse=True)
def execute_notebooks():
    Path("results").mkdir(exist_ok=True)
    notebooks = [p for p in Path(".").glob("*.ipynb")]
    for nb in notebooks:
        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=300",
            str(nb)
        ], check=True)

def discover_baselines():
    base_dir = Path("baseline")
    return sorted([p.name for p in base_dir.glob("*.png")]) if base_dir.exists() else []

@pytest.mark.parametrize("png_name", discover_baselines())
def test_png_matches_baseline(png_name):
    expected = Path("baseline") / png_name
    actual = Path("results") / png_name

    assert expected.exists(), f"Missing baseline: {expected}"
    assert actual.exists(), f"No output from notebook: {actual}"

    # Check image sizes first (differences often come from bbox/layout/fonts)
    with Image.open(expected) as e_img, Image.open(actual) as a_img:
        assert e_img.size == a_img.size, (
            f"Image sizes differ for {png_name}: "
            f"baseline={e_img.size}, actual={a_img.size}"
        )

    err = compare_images(str(expected), str(actual), tol=TOL)
    assert err is None, f"Image mismatch for {png_name} (tol={TOL}). Details: {err}"