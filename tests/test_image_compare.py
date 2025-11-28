import os
import subprocess
from pathlib import Path
import hashlib
import pytest
from PIL import Image
from matplotlib.testing.compare import compare_images

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

TOL = float(os.environ.get("IMG_TOL", "0.0"))

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

def baselines():
    base = Path("baseline")
    return sorted(p.name for p in base.glob("*.png")) if base.exists() else []

@pytest.mark.parametrize("png_name", baselines())
def test_png_matches_baseline(png_name):
    expected = Path("baseline") / png_name
    actual = Path("results") / png_name

    assert expected.exists(), f"Missing baseline: {expected}"
    assert actual.exists(), f"No output from notebook: {actual}"

    # Quick visibility: confirm we’re comparing the intended files
    print(f"Comparing {png_name}")
    print(f"  baseline: {expected} sha256={sha256(expected)}")
    print(f"  actual:   {actual}   sha256={sha256(actual)}")

    ew, eh = Image.open(expected).size
    aw, ah = Image.open(actual).size
    assert (ew, eh) == (aw, ah), (
        f"Image sizes differ for {png_name}: baseline={(ew, eh)}, actual={(aw, ah)}. "
        "Remove tight bbox and fix rcParams for deterministic size."
    )

    err = compare_images(str(expected), str(actual), tol=TOL)
    assert err is None, f"Image mismatch for {png_name} (tol={TOL}). Details: {err}"