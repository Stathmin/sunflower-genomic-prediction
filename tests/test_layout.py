"""The rework's guard rails.

These tests fail if the old path bootstrap, hand-written EMMAX/BLINK/PLINK, or
hard-coded GWAS loci ever come back.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [
    "while _p != _p.parent",
    "sys.path.insert",
    "KEY_LOCI",
    "def emmax",
    "def blink",
    "def ld_prune",
    "from baker_v4",
    "from blink",
    "from emmax",
]


def test_no_path_hacks_or_custom_tool_reimplementations():
    files = list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "src").rglob("*.py"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN:
            assert forbidden not in text, f"{path.relative_to(ROOT)} contains {forbidden!r}"


def test_obsolete_modules_are_removed():
    assert not (ROOT / "scripts" / "models").exists()
    assert not (ROOT / "tables" / "lines_gwas_panels_loci.csv").exists()


def test_native_install_metadata_exists():
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "uv.lock").is_file()
    assert (ROOT / "pixi.toml").is_file()
