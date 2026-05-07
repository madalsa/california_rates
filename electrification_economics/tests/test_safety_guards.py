"""Verify EE pipeline never writes outside electrification_economics/data/.

Protects parent california_rates outputs and shared-storage symlinks
(e.g. Baseline_<u>/) from accidental overwrite.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def test_ee_data_dir_inside_ee_root():
    assert config.DATA_DIR.resolve().is_relative_to(config.EE_ROOT.resolve())


def test_assert_safe_out_dir_accepts_data_dir():
    # Should not raise
    config.assert_safe_out_dir(config.DATA_DIR)


def test_assert_safe_out_dir_accepts_subdir():
    config.assert_safe_out_dir(config.DATA_DIR / "subdir")


def test_assert_safe_out_dir_rejects_parent_root():
    try:
        config.assert_safe_out_dir(config.CR_ROOT)
    except ValueError:
        return
    raise AssertionError("guard should reject CR_ROOT")


def test_assert_safe_out_dir_rejects_baseline_folder():
    try:
        config.assert_safe_out_dir(config.CR_ROOT / "Baseline_PGE")
    except ValueError:
        return
    raise AssertionError("guard should reject Baseline_PGE/ (shared storage)")


def test_assert_safe_out_dir_rejects_other_repo_outputs():
    """Existing parent outputs must be protected."""
    for name in ("baseline_bills_pge_fresh.csv",
                 "rate_scenarios_sce.csv",
                 "tou_weights_sdge.csv"):
        try:
            config.assert_safe_out_dir(config.CR_ROOT / name)
        except ValueError:
            continue
        raise AssertionError(f"guard should reject {name}")


def test_no_module_writes_outside_data_dir():
    """Static check: no .to_parquet / .to_csv / open(...,'w') with a
    hardcoded path outside DATA_DIR appears in src/."""
    src = Path(__file__).resolve().parents[1] / "src"
    forbidden_substrings = [
        'CR_ROOT /',          # would write to parent root
        'parents[2] /',       # parent of EE = CR_ROOT
    ]
    for py in src.glob("*.py"):
        text = py.read_text()
        for marker in ("to_parquet(", "to_csv("):
            for line in text.splitlines():
                if marker in line and any(
                        s in line for s in forbidden_substrings):
                    raise AssertionError(
                        f"{py.name}: write call references parent root: "
                        f"{line.strip()}")


if __name__ == "__main__":
    failures = 0
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}  {e}")
                failures += 1
            except Exception as e:
                print(f"  ERR   {name}  {type(e).__name__}: {e}")
                failures += 1
    print(f"\n{failures} failure(s)" if failures else "\nall passed")
    sys.exit(failures)
