"""Offline verification that the original user-provided sources/pseudocode stay unchanged."""
from pathlib import Path
from hashlib import sha256
import json
import ast

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "references/original_manifest.json").read_text(encoding="utf8"))
    for relative, data in manifest["sources"].items():
        p = ROOT / relative
        if not p.is_file():
            raise ValueError(f"missing original: {p}")
        b = p.read_bytes()
        if len(b) != data["bytes"]:
            raise ValueError(f"original source size changed: {relative}")
        if sha256(b).hexdigest() != data["sha256"]:
            raise ValueError(f"original source modified: {relative}")
    s = (ROOT / "references/original_sources/rolling_meanflow.py").read_text(encoding="utf8")
    d = ast.get_docstring(ast.parse(s), clean=False)
    if sha256(d.encode()).hexdigest() != manifest["original_docstring_sha256"]:
        raise ValueError("original docstring hash changed")
    text = (ROOT / "references/ORIGINAL_ROLLFLOW_ALGORITHM.md").read_text(encoding="utf8")
    if f"```text\n{d}\n```" not in text:
        raise ValueError("the original algorithm block was edited")
    for path in (ROOT / "references/original_sources").glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    print("Original user algorithm and 3 source files: EXACT MATCH")


if __name__ == "__main__":
    verify()
