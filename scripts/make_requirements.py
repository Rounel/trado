"""
scripts/make_requirements.py — Regenere requirements.txt depuis pyproject.toml.
Appelee par build.bat.
"""
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOML = ROOT / "pyproject.toml"
REQS = ROOT / "requirements.txt"

with open(TOML, "rb") as f:
    data = tomllib.load(f)

deps = data["project"]["dependencies"]

with open(REQS, "w", encoding="utf-8") as f:
    f.write("# requirements.txt -- genere depuis pyproject.toml\n")
    f.write("# Installe par TRADO.exe au premier lancement\n\n")
    for d in deps:
        f.write(d + "\n")

print(f"requirements.txt mis a jour ({len(deps)} packages)")
