from pathlib import Path


def test_animation_module_exists():
    repo_root = Path(__file__).resolve().parents[1]

    assert (
        repo_root / "src" / "drill_animation.py"
    ).is_file()